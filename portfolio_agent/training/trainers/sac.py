"""Soft Actor-Critic trainer for the continuous-allocation strategy.

What this trains
----------------
`strategies/india_sac.py::SACActorNetwork` — a network mapping a feature vector
to an allocation weight in [0, 1]. That module holds inference only and ships
without a checkpoint; this is the loop that produces one.

Why the actor here has an extra head
------------------------------------
SAC optimizes a *stochastic* policy: a squashed Gaussian whose log-standard-
deviation supplies the entropy term the algorithm is named for. The inference
class deliberately keeps only the mean head, because sampling at scoring time
would make two runs of one backtest disagree. So the training actor is a strict
superset — same `net` and `mean_head` parameter names, plus `log_std_head` —
and `inference_state_dict()` drops the extra head so the saved weights load
into the inference class under `strict=True`. The checkpoint therefore cannot
drift from what `IndiaSACStrategy.load()` expects to find.

Why gamma defaults to 0
-----------------------
Discounting assumes the action influences the next state. A price-taking book
allocating one name does not move the market, so the observed state sequence is
exogenous: bootstrapping a value function over it adds estimator variance
without adding signal. With `gamma=0` the critic learns `Q(s, a) = E[r | s, a]`
and the actor maximizes `Q - alpha * log pi` — soft actor-critic applied to a
contextual bandit, which is what this decision actually is. It is left
configurable because the turnover term below does couple consecutive steps.

A caveat stated rather than hidden: the reward is net of turnover, which
depends on the previous allocation, but the state cannot carry that previous
allocation — inference builds its state vector from features alone, and adding
a twelfth input would change `state_dim` and break every existing checkpoint.
The process is therefore mildly partially observed. That is a deliberate
trade against the fixed inference contract, not an oversight.

The reward
----------
The strategy's own docstring specifies "a differential Sortino ratio net of
turnover cost". This implements exactly that, and not a proxy for it:

    R_t = a_t * ret_{t+1} - friction * |a_t - a_{t-1}|

feeds an online Sortino whose per-step increment is the reward. Writing
`A` for the EMA of `R` and `DD2` for the EMA of squared downside,

    D_t = (DD2 * dA - 0.5 * A * dDD2) / DD2^{3/2}

which is Moody & Saffell's differential Sharpe with the second moment replaced
by the downside second moment. Summing `D_t` approximates the Sortino ratio of
the whole path, so a policy maximizing per-step reward is maximizing a
downside-aware ratio rather than raw return — the distinction that stops it
learning "hold maximum size always".

Note the friction term is a function of the *action*: a cost subtracted as a
constant every step cannot penalize turnover, because it shifts every action's
reward identically and leaves the optimum exactly where it was.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field

from portfolio_agent.strategies.india_sac import DEFAULT_SAC_FEATURES, SACActorNetwork
from portfolio_agent.utils.device import get_device

from ..base import BaseTrainer, TrainerConfig, TrainingArtifact, TrainingData
from ..data import prepare_panel
from ..registry import register_trainer

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252

# Bounds on the policy's log standard deviation. Unbounded, the network can
# drive std to zero (a deterministic policy the entropy term then punishes with
# an infinite penalty) or to infinity (pure noise); both show up as NaN losses
# several hundred steps later, far from the cause.
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class SACTrainerConfig(TrainerConfig):
    """Hyperparameters for the SAC trainer."""

    hidden_dim: int = Field(default=256, gt=0, description="Width of both hidden layers.")
    gamma: float = Field(
        default=0.0, ge=0.0, lt=1.0,
        description="Discount factor. 0 treats each decision as a contextual bandit, "
        "which is what a price-taking allocation decision is; see the module docstring.",
    )
    tau: float = Field(
        default=0.005, gt=0.0, le=1.0,
        description="Polyak coefficient for the target critics.",
    )
    buffer_size: int = Field(default=200_000, gt=0, description="Replay buffer capacity.")
    entropy_coef: float = Field(
        default=0.1, gt=0.0,
        description="Initial entropy temperature (alpha). Tuned automatically unless "
        "auto_entropy is false.",
    )
    auto_entropy: bool = Field(
        default=True,
        description="Learn alpha against target_entropy instead of holding it fixed. "
        "A fixed temperature that suits one reward scale suits no other, and the "
        "differential-Sortino reward's scale depends on the data.",
    )
    target_entropy: Optional[float] = Field(
        default=None,
        description="Entropy the tuner aims for. None uses -action_dim = -1.0, the "
        "standard choice for a one-dimensional continuous action.",
    )
    gradient_steps: int = Field(
        default=200, gt=0, description="Gradient steps per epoch."
    )
    rollout_every: int = Field(
        default=1, gt=0,
        description="Re-collect on-policy experience every N epochs. Collecting once "
        "and training forever leaves the buffer full of a randomly-initialized "
        "policy's decisions, and the actor never sees the consequences of its own.",
    )
    friction_cost: float = Field(
        default=0.008, ge=0.0,
        description="Round-trip cost charged on |a_t - a_{t-1}|.",
    )
    sortino_eta: float = Field(
        default=0.01, gt=0.0, lt=1.0,
        description="EMA rate for the online Sortino moments.",
    )
    sortino_warmup: int = Field(
        default=20, ge=0,
        description="Steps used to seed the moments before transitions are emitted. "
        "The first increments of an EMA started at zero are dominated by the "
        "initialization, not by the policy.",
    )
    reward_clip: float = Field(
        default=10.0, gt=0.0,
        description="Bound on the per-step differential reward. The ratio's denominator "
        "is a downside deviation, which is legitimately near zero on a quiet run, "
        "and one unclipped spike is enough to move the weights somewhere every "
        "later batch evaluates to NaN.",
    )
    grad_clip: float = Field(default=1.0, gt=0.0, description="Global gradient-norm cap.")
    min_history: int = Field(
        default=252, gt=0,
        description="Rows a ticker needs after cleaning. The long-lookback features are "
        "NaN until their window fills.",
    )
    warmup_transitions: int = Field(
        default=1000, ge=0,
        description="Transitions required before the first gradient step.",
    )
    features: Optional[List[str]] = Field(
        default=None,
        description="Feature registry names the actor observes. None uses the "
        "strategy's DEFAULT_SAC_FEATURES.",
    )


class SACActorTrainingNetwork(nn.Module):
    """The full stochastic actor: `SACActorNetwork` plus a log-std head.

    Parameter names for `net` and `mean_head` match the inference class exactly,
    so `inference_state_dict()` produces weights that load there under
    `strict=True`. Keeping the two in one file would be tidier; keeping the
    *names* aligned is what actually matters, and the round-trip is asserted in
    the tests.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.log_std_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Deterministic action — identical to what inference computes."""
        return torch.sigmoid(self.mean_head(self.net(state)))

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized sample and its exact log-density.

        Returns:
            `(action, log_prob)` with action in (0, 1). The log-density carries
            the sigmoid's Jacobian correction, `-log(a(1-a))`; omitting it
            silently changes the entropy target and the temperature the tuner
            settles on.
        """
        hidden = self.net(state)
        mean = self.mean_head(hidden)
        log_std = torch.clamp(self.log_std_head(hidden), LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        pre_squash = normal.rsample()
        action = torch.sigmoid(pre_squash)

        log_prob = normal.log_prob(pre_squash)
        # d(sigmoid)/du = a(1-a); the epsilon keeps the log finite when the
        # sample saturates, which it does routinely once the policy sharpens.
        log_prob = log_prob - torch.log(action * (1.0 - action) + 1e-6)
        return action, log_prob

    def inference_state_dict(self) -> Dict[str, torch.Tensor]:
        """Weights with the training-only head removed."""
        return {
            key: value.detach().cpu().clone()
            for key, value in self.state_dict().items()
            if not key.startswith("log_std_head.")
        }


class TwinCritic(nn.Module):
    """Two independent Q networks over (state, action).

    Twin critics with a min over their outputs are SAC's defence against the
    overestimation bias that a single bootstrapped Q accumulates.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = self._build(state_dim, hidden_dim)
        self.q2 = self._build(state_dim, hidden_dim)

    @staticmethod
    def _build(state_dim: int, hidden_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        joint = torch.cat([state, action], dim=-1)
        return self.q1(joint), self.q2(joint)


@dataclass
class ReplayBuffer:
    """Fixed-capacity transition store with a seeded sampler.

    The sampler holds its own `numpy.random.Generator`. Drawing from the global
    `numpy.random` state instead would make a run depend on every other caller
    that happened to touch it, and the platform requires two runs of one
    configuration to agree.
    """

    capacity: int
    state_dim: int
    seed: int = 42

    def __post_init__(self) -> None:
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, 1), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self._size = 0
        self._position = 0
        self._rng = np.random.default_rng(self.seed)

    def __len__(self) -> int:
        return self._size

    def push(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        index = self._position
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = float(done)
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Draw a mini-batch with replacement.

        With replacement on purpose: sampling without it caps the batch at the
        buffer size and raises early in a run, when the buffer is legitimately
        smaller than the batch.
        """
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        idx = self._rng.integers(0, self._size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )


class DifferentialSortino:
    """Online Sortino whose per-step increment is the RL reward.

    Tracks an EMA of the return (`A`) and of the squared downside (`DD2`), and
    returns the first-order change in the ratio contributed by each new
    observation.
    """

    def __init__(self, eta: float = 0.01, clip: float = 10.0):
        self.eta = float(eta)
        self.clip = float(clip)
        self.a = 0.0
        self.dd2 = 0.0

    def warmup(self, portfolio_return: float) -> None:
        """Fold an observation into the moments without emitting a reward."""
        downside = min(portfolio_return, 0.0) ** 2
        self.a += self.eta * (portfolio_return - self.a)
        self.dd2 += self.eta * (downside - self.dd2)

    def update(self, portfolio_return: float) -> float:
        """Fold in an observation and return the differential Sortino increment."""
        downside = min(portfolio_return, 0.0) ** 2
        delta_a = portfolio_return - self.a
        delta_dd2 = downside - self.dd2

        if self.dd2 > 1e-12:
            reward = (self.dd2 * delta_a - 0.5 * self.a * delta_dd2) / (self.dd2 ** 1.5)
        else:
            # No downside observed yet: the ratio is undefined, so fall back to
            # the excess return itself rather than dividing by ~0.
            reward = delta_a

        self.a += self.eta * delta_a
        self.dd2 += self.eta * delta_dd2

        if not math.isfinite(reward):
            return 0.0
        return float(np.clip(reward, -self.clip, self.clip))


def forward_returns(prices: pd.DataFrame, index: pd.Index) -> np.ndarray:
    """Next-step realized return of `close`, aligned to `index`.

    The last element is NaN — there is no observed next price for the final row
    — and callers must not build a transition from it.
    """
    close = prices.reindex(index)["close"].astype(float)
    return (close.shift(-1) / close - 1.0).to_numpy()


@register_trainer("sac")
class SACTrainer(BaseTrainer):
    """Twin-critic, entropy-regularized trainer for `india_sac`."""

    name = "sac"
    strategy_name = "india_sac"

    @classmethod
    def config_model(cls) -> Type[TrainerConfig]:
        return SACTrainerConfig

    def prepare(
        self, app_config: Any, universe: List[str], cfg: TrainerConfig
    ) -> TrainingData:
        assert isinstance(cfg, SACTrainerConfig)
        feature_names = cfg.features or list(DEFAULT_SAC_FEATURES)
        return prepare_panel(
            app_config,
            universe,
            feature_names,
            train_fraction=cfg.train_fraction,
            min_history=cfg.min_history,
        )

    def fit(self, data: TrainingData, cfg: TrainerConfig) -> TrainingArtifact:
        assert isinstance(cfg, SACTrainerConfig)

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        device = get_device(cfg.device, verbose=False)
        state_dim = len(data.feature_names)

        actor = SACActorTrainingNetwork(state_dim, cfg.hidden_dim).to(device)
        critic = TwinCritic(state_dim, cfg.hidden_dim).to(device)
        critic_target = TwinCritic(state_dim, cfg.hidden_dim).to(device)
        critic_target.load_state_dict(critic.state_dict())
        for param in critic_target.parameters():
            param.requires_grad_(False)

        actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.learning_rate)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.learning_rate)

        target_entropy = (
            cfg.target_entropy if cfg.target_entropy is not None else -1.0
        )
        log_alpha = torch.tensor(
            math.log(cfg.entropy_coef), device=device, requires_grad=cfg.auto_entropy
        )
        alpha_opt = (
            torch.optim.Adam([log_alpha], lr=cfg.learning_rate) if cfg.auto_entropy else None
        )

        buffer = ReplayBuffer(cfg.buffer_size, state_dim, seed=cfg.seed)

        history: List[Dict[str, float]] = []
        best_val = -math.inf
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_epoch = 0

        for epoch in range(cfg.epochs):
            if epoch % cfg.rollout_every == 0:
                collected = self._collect(actor, data, cfg, device, buffer)
                logger.debug("epoch %d collected %d transitions", epoch, collected)

            if len(buffer) < max(cfg.warmup_transitions, cfg.batch_size):
                continue

            losses = self._optimize(
                actor, critic, critic_target, actor_opt, critic_opt,
                log_alpha, alpha_opt, buffer, cfg, device, target_entropy,
            )

            val = self.evaluate(actor, data, cfg, device, segment="validation")
            record = {"epoch": epoch + 1, **losses, **val}
            history.append(record)

            if val["val_sortino"] > best_val:
                best_val = val["val_sortino"]
                best_state = actor.inference_state_dict()
                best_epoch = epoch + 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    "epoch %d/%d  critic=%.4f actor=%.4f alpha=%.4f  val_sortino=%.4f",
                    epoch + 1, cfg.epochs, losses["critic_loss"],
                    losses["actor_loss"], losses["alpha"], val["val_sortino"],
                )

        if best_state is None:
            raise RuntimeError(
                "SAC training completed no gradient steps. The replay buffer never "
                f"reached warmup_transitions={cfg.warmup_transitions}; either lower "
                "it or train on more tickers/history."
            )

        train_metrics = self.evaluate(actor, data, cfg, device, segment="train")
        metrics = {
            "best_val_sortino": best_val,
            "best_epoch": best_epoch,
            "epochs_trained": len(history),
            "history": history,
            **{f"train_{k.split('_', 1)[1]}": v for k, v in train_metrics.items()},
            **(history[-1] if history else {}),
        }
        # Report the *selected* checkpoint's score, not the last epoch's.
        metrics["val_sortino"] = best_val

        from ..artifacts import build_metadata

        metadata = build_metadata(
            feature_names=data.feature_names,
            scaler=data.scaler,
            trainer=self.name,
            extra={
                "hidden_dim": cfg.hidden_dim,
                "state_dim": state_dim,
                "strategy": self.strategy_name,
                "training_config": cfg.model_dump(),
                "n_tickers": len(data.tickers),
                "best_epoch": best_epoch,
            },
        )
        return TrainingArtifact(state_dict=best_state, metadata=metadata, metrics=metrics)

    # -- rollout -----------------------------------------------------------

    def _collect(
        self,
        actor: SACActorTrainingNetwork,
        data: TrainingData,
        cfg: SACTrainerConfig,
        device: torch.device,
        buffer: ReplayBuffer,
    ) -> int:
        """Roll the current policy through each ticker's training segment.

        One episode per ticker, walked in date order so the turnover term and
        the Sortino moments both see a real sequence.
        """
        actor.eval()
        pushed = 0

        with torch.no_grad():
            for ticker in data.tickers:
                train_frame, _ = data.split(ticker)
                if len(train_frame) < 3:
                    continue

                # np.array(..., copy) rather than to_numpy(): pandas can hand
                # back a read-only view, and torch.as_tensor on one produces a
                # tensor that shares it, which torch warns about.
                states = np.array(train_frame.to_numpy(dtype=np.float32), copy=True)
                rets = forward_returns(data.prices_by_ticker[ticker], train_frame.index)

                batch = torch.as_tensor(states, device=device)
                actions, _ = actor.sample(batch)
                actions = actions.squeeze(-1).cpu().numpy()

                sortino = DifferentialSortino(cfg.sortino_eta, cfg.reward_clip)
                previous_action = 0.0

                # The final row has no observed next return, so it can start no
                # transition; stop one short of it.
                for t in range(len(states) - 1):
                    ret = rets[t]
                    if not np.isfinite(ret):
                        continue

                    action = float(actions[t])
                    turnover = abs(action - previous_action)
                    portfolio_return = action * float(ret) - cfg.friction_cost * turnover
                    previous_action = action

                    if t < cfg.sortino_warmup:
                        sortino.warmup(portfolio_return)
                        continue

                    reward = sortino.update(portfolio_return)
                    buffer.push(
                        state=states[t],
                        action=action,
                        reward=reward,
                        next_state=states[t + 1],
                        done=(t == len(states) - 2),
                    )
                    pushed += 1

        actor.train()
        return pushed

    # -- optimization ------------------------------------------------------

    def _optimize(
        self,
        actor: SACActorTrainingNetwork,
        critic: TwinCritic,
        critic_target: TwinCritic,
        actor_opt: torch.optim.Optimizer,
        critic_opt: torch.optim.Optimizer,
        log_alpha: torch.Tensor,
        alpha_opt: Optional[torch.optim.Optimizer],
        buffer: ReplayBuffer,
        cfg: SACTrainerConfig,
        device: torch.device,
        target_entropy: float,
    ) -> Dict[str, float]:
        """Run `gradient_steps` SAC updates and report the mean losses."""
        critic_total = 0.0
        actor_total = 0.0

        for _ in range(cfg.gradient_steps):
            states, actions, rewards, next_states, dones = buffer.sample(cfg.batch_size)
            states_t = torch.as_tensor(states, device=device)
            actions_t = torch.as_tensor(actions, device=device)
            rewards_t = torch.as_tensor(rewards, device=device)
            next_states_t = torch.as_tensor(next_states, device=device)
            dones_t = torch.as_tensor(dones, device=device)

            alpha = log_alpha.exp().detach()

            # --- critic: regress onto the soft Bellman target ---
            with torch.no_grad():
                next_actions, next_log_prob = actor.sample(next_states_t)
                target_q1, target_q2 = critic_target(next_states_t, next_actions)
                soft_target = torch.min(target_q1, target_q2) - alpha * next_log_prob
                # gamma=0 collapses this to `rewards_t`, which is the intended
                # bandit case; the term is written out so a non-zero gamma is a
                # config change rather than a code change.
                backup = rewards_t + cfg.gamma * (1.0 - dones_t) * soft_target

            q1, q2 = critic(states_t, actions_t)
            critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)

            critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), cfg.grad_clip)
            critic_opt.step()

            # --- actor: maximize Q - alpha * log pi ---
            sampled_actions, log_prob = actor.sample(states_t)
            q1_pi, q2_pi = critic(states_t, sampled_actions)
            actor_loss = (alpha * log_prob - torch.min(q1_pi, q2_pi)).mean()

            actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), cfg.grad_clip)
            actor_opt.step()

            # --- temperature ---
            if alpha_opt is not None:
                alpha_loss = -(log_alpha * (log_prob.detach() + target_entropy)).mean()
                alpha_opt.zero_grad(set_to_none=True)
                alpha_loss.backward()
                alpha_opt.step()

            with torch.no_grad():
                for param, target_param in zip(
                    critic.parameters(), critic_target.parameters()
                ):
                    target_param.mul_(1.0 - cfg.tau).add_(param, alpha=cfg.tau)

            critic_total += float(critic_loss.item())
            actor_total += float(actor_loss.item())

        return {
            "critic_loss": critic_total / cfg.gradient_steps,
            "actor_loss": actor_total / cfg.gradient_steps,
            "alpha": float(log_alpha.exp().item()),
        }

    # -- evaluation --------------------------------------------------------

    def evaluate(
        self,
        actor: SACActorTrainingNetwork,
        data: TrainingData,
        cfg: SACTrainerConfig,
        device: torch.device,
        *,
        segment: str = "validation",
    ) -> Dict[str, float]:
        """Score the *deterministic* policy on a held-out chronological segment.

        Deterministic because that is what inference runs. Evaluating the
        sampling policy would report a number the backtest can never reproduce.
        """
        actor.eval()
        pooled: List[float] = []

        with torch.no_grad():
            for ticker in data.tickers:
                train_frame, val_frame = data.split(ticker)
                frame = train_frame if segment == "train" else val_frame
                if len(frame) < 3:
                    continue

                block = np.array(frame.to_numpy(dtype=np.float32), copy=True)
                states = torch.as_tensor(block, device=device)
                weights = actor(states).squeeze(-1).cpu().numpy()
                rets = forward_returns(data.prices_by_ticker[ticker], frame.index)

                previous = 0.0
                for t in range(len(weights) - 1):
                    if not np.isfinite(rets[t]):
                        continue
                    action = float(weights[t])
                    net = action * float(rets[t]) - cfg.friction_cost * abs(action - previous)
                    previous = action
                    pooled.append(net)

        actor.train()

        prefix = "train" if segment == "train" else "val"
        if not pooled:
            return {f"{prefix}_sortino": 0.0, f"{prefix}_sharpe": 0.0, f"{prefix}_mean_return": 0.0}

        returns = np.asarray(pooled, dtype=np.float64)
        return {
            f"{prefix}_sortino": _annualized_sortino(returns),
            f"{prefix}_sharpe": _annualized_sharpe(returns),
            f"{prefix}_mean_return": float(returns.mean()),
        }


def _annualized_sharpe(returns: np.ndarray) -> float:
    """Annualized Sharpe of a per-step return series, 0.0 when undefined."""
    std = float(returns.std(ddof=1)) if returns.size > 1 else 0.0
    if std <= 1e-12:
        return 0.0
    return float(returns.mean() / std * math.sqrt(TRADING_DAYS_PER_YEAR))


def _annualized_sortino(returns: np.ndarray) -> float:
    """Annualized Sortino — downside deviation in the denominator.

    A series with no losing step has no downside deviation and an undefined
    ratio; 0.0 is returned rather than an infinity that would win any
    best-checkpoint comparison it entered.
    """
    downside = np.minimum(returns, 0.0)
    dd = float(np.sqrt(np.mean(downside ** 2)))
    if dd <= 1e-12:
        return 0.0
    return float(returns.mean() / dd * math.sqrt(TRADING_DAYS_PER_YEAR))
