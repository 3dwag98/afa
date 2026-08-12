"""Reinforcement learning for the exposure decision.

Read this before using it. The default instinct — a deep network that observes
prices and emits buy/sell — is the single most overfit-prone thing that can be
added to a platform like this, and it will produce a spectacular backtest and
no edge. The reason is arithmetic, not pessimism:

**There is one episode.** Supervised learning here has ~1,250 daily
observations per ticker, which is already thin. Reinforcement learning has far
less than that, because the unit of learning is a *trajectory*, and the market
has produced exactly one — 2021 to 2025 happened once. An agent with a few
thousand parameters can memorise that single path completely. Atari agents
train on millions of episodes precisely because the environment can be re-run;
the market cannot.

**Naive rewards learn leverage, not skill.** Reward the cumulative return and
the growth-optimal policy is "hold the maximum position at all times", which is
not a strategy, it is a beta exposure with extra steps. Any usable reward has
to charge for risk and for turnover in the same units as the return.

**The reward is measured through the same broken instruments.** If costs are
understated, the agent will find and exploit exactly that gap, because that is
what RL does. This is why the environment here charges the platform's real
Indian friction stack rather than a flat basis-point assumption.

So this module deliberately does *not* implement deep RL over prices. It
implements the problem RL is actually suited to on this data:

    Given the regime probabilities and how the book is currently positioned,
    what fraction of capital should be deployed today?

That is a **contextual decision with a small action space and a low-dimensional
state**, which is estimable from one trajectory in a way that a price-level
policy is not. It is also the formulation the literature supports — regime
states from a hidden Markov model, with an RL policy allocating on top of them
(see docs/QUANT_RESEARCH.md section 25 for the regime model this consumes).

The policy is linear-softmax with a handful of parameters, on purpose. A
network here would not be more powerful, it would be more confident.

What this is not
----------------
- Not a stock picker. The strategies choose *what* to hold; this chooses *how
  much*. Those are separable problems and conflating them is how the action
  space explodes past what the data can support.
- Not a substitute for the measurement layer. An RL result is a search result:
  run it through src/performance_stats.py (deflated Sharpe, PBO) before
  believing it, and log the trial.
- Not validated on real Indian data. Nothing in this repository has a
  point-in-time universe yet (docs/REVIEW_STATUS.md, D9), so a good backtest
  here is not evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

TRADING_DAYS_PER_YEAR = 252

# Exposure levels the agent chooses between. Discrete and coarse on purpose: a
# continuous action space needs far more interaction to explore than one
# trajectory provides, and "how much of the book is deployed" does not need
# finer resolution than this to express a regime view.
DEFAULT_EXPOSURE_LEVELS: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass
class EnvironmentConfig:
    """Everything that defines the decision problem.

    The defaults are deliberately conservative. `risk_aversion` and
    `turnover_cost` are the two knobs that decide whether the agent learns a
    strategy or learns leverage, so both are on by default rather than being
    opt-in refinements.
    """

    exposure_levels: Tuple[float, ...] = DEFAULT_EXPOSURE_LEVELS

    # Mean-variance utility: reward = r - (lambda/2) * r^2. With lambda = 0 the
    # optimal policy is always-maximum-exposure, which is why this is not 0.
    risk_aversion: float = 2.0

    # Charged on |change in exposure| every step. Without it the agent
    # rebalances the whole book daily on noise, which is free in simulation and
    # ruinous in an Indian delivery account (~0.8% round trip).
    turnover_cost: float = 0.008

    # Reward scaling. Daily returns are ~0.01, so unscaled rewards produce
    # gradients small enough that the policy barely moves within a realistic
    # number of epochs.
    reward_scale: float = 100.0


class ExposureEnvironment:
    """Sequential exposure decisions over one realized return path.

    **The look-ahead contract, which is the whole game.** At step t the agent
    observes state s_t, built only from information available at the close of
    day t. It chooses an exposure w_t. The reward arrives from r_{t+1}, the
    return the market subsequently delivered. The agent never sees r_{t+1} when
    choosing w_t — a test asserts this by checking that corrupting the future
    leaves earlier observations bit-identical.

    That ordering is why `step()` returns the reward for the action just taken
    rather than for the state just observed: getting this backwards produces an
    agent that appears to predict the market perfectly and is worthless.

    Args:
        strategy_returns: The per-period return of the underlying book at full
            exposure — what the strategy would have earned deployed 100%. The
            agent scales this, it does not pick it.
        regime_probabilities: Optional (T, K) filtered state probabilities from
            src/markov_regime.py. Filtered, never smoothed: smoothed
            probabilities condition on the whole sample and would hand the
            agent the future in its state vector.
        config: Reward and action-space configuration.
    """

    def __init__(
        self,
        strategy_returns: Sequence[float],
        regime_probabilities: Optional[np.ndarray] = None,
        config: Optional[EnvironmentConfig] = None,
    ):
        self.config = config or EnvironmentConfig()
        self.returns = np.asarray(strategy_returns, dtype=float).ravel()
        if not np.all(np.isfinite(self.returns)):
            raise ValueError("strategy_returns must be finite")

        if regime_probabilities is None:
            self.regimes = np.zeros((self.returns.size, 0))
        else:
            self.regimes = np.asarray(regime_probabilities, dtype=float)
            if self.regimes.ndim != 2:
                raise ValueError("regime_probabilities must be 2-D (T, K)")
            if self.regimes.shape[0] != self.returns.size:
                raise ValueError(
                    f"regime_probabilities has {self.regimes.shape[0]} rows but "
                    f"strategy_returns has {self.returns.size}"
                )

        self.n_actions = len(self.config.exposure_levels)
        self.reset()

    @property
    def n_features(self) -> int:
        """Regime probabilities, current exposure, trailing volatility, bias."""
        return self.regimes.shape[1] + 3

    @property
    def n_steps(self) -> int:
        """Decisions available. The last return has no successor to reward it."""
        return max(0, self.returns.size - 1)

    def reset(self) -> np.ndarray:
        """Start a fresh pass over the path, flat."""
        self.t = 0
        self.exposure = 0.0
        return self._observe()

    def _observe(self) -> np.ndarray:
        """State at the current step — strictly backward-looking.

        Trailing volatility uses returns *up to and including* t, never beyond.
        The bias term is explicit so the policy can express an unconditional
        preference without needing one of the real features to carry it.
        """
        parts: List[float] = []
        if self.regimes.shape[1]:
            parts.extend(self.regimes[self.t].tolist())

        window = self.returns[max(0, self.t - 20) : self.t + 1]
        trailing_volatility = (
            float(np.std(window)) * math.sqrt(TRADING_DAYS_PER_YEAR)
            if window.size > 1
            else 0.0
        )

        parts.append(self.exposure)
        parts.append(trailing_volatility)
        parts.append(1.0)  # bias
        return np.asarray(parts, dtype=float)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        """Apply an exposure choice and collect what the market then did.

        Returns:
            (next_state, reward, done, info).
        """
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action {action} outside [0, {self.n_actions})")
        if self.t >= self.n_steps:
            raise RuntimeError("environment is exhausted; call reset()")

        target = float(self.config.exposure_levels[action])
        turnover = abs(target - self.exposure)

        # The return the market delivered *after* the decision.
        realized = self.returns[self.t + 1]
        portfolio_return = target * realized
        cost = self.config.turnover_cost * turnover

        # Mean-variance utility. The quadratic term is what stops the policy
        # collapsing to "always fully invested": without it, more exposure is
        # always better in expectation whenever the drift is positive.
        net = portfolio_return - cost
        reward = net - 0.5 * self.config.risk_aversion * portfolio_return**2
        reward *= self.config.reward_scale

        self.exposure = target
        self.t += 1
        done = self.t >= self.n_steps

        info = {
            "portfolio_return": portfolio_return,
            "net_return": net,
            "turnover": turnover,
            "cost": cost,
            "exposure": target,
        }
        return self._observe(), float(reward), done, info


class LinearSoftmaxPolicy:
    """A softmax policy over exposure levels, linear in the state.

    Deliberately tiny: `n_features x n_actions` parameters, typically under 40.
    A neural policy here would not be more capable, it would be more
    confident — there is one trajectory to learn from, and capacity that
    exceeds the data buys memorisation rather than skill.

    Trained by REINFORCE with a baseline. The baseline (mean return-to-go)
    matters more than usual on this problem: raw returns are noisy and nearly
    zero-mean, so without it the gradient is dominated by the sign of whatever
    the market happened to do rather than by the action's contribution.
    """

    def __init__(self, n_features: int, n_actions: int, seed: int = 0):
        self.n_features = int(n_features)
        self.n_actions = int(n_actions)
        # Zero init means a uniform policy: every exposure equally likely,
        # nothing assumed before any evidence arrives.
        self.weights = np.zeros((self.n_features, self.n_actions))
        self.rng = np.random.default_rng(seed)

    def logits(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=float) @ self.weights

    def probabilities(self, state: np.ndarray) -> np.ndarray:
        """Action distribution, computed with the max subtracted for stability."""
        z = self.logits(state)
        z = z - np.max(z)
        exp = np.exp(z)
        total = exp.sum()
        if not np.isfinite(total) or total <= 0:
            return np.full(self.n_actions, 1.0 / self.n_actions)
        return exp / total

    def sample(self, state: np.ndarray) -> int:
        """Draw an action. Exploration comes from the policy's own entropy."""
        return int(self.rng.choice(self.n_actions, p=self.probabilities(state)))

    def act_greedily(self, state: np.ndarray) -> int:
        """The most likely action — what to use at inference."""
        return int(np.argmax(self.probabilities(state)))

    def gradient(self, state: np.ndarray, action: int) -> np.ndarray:
        """d log pi(a|s) / d weights — the score function.

        For a linear softmax this is the outer product of the state with
        (one-hot(a) - pi(.|s)).
        """
        probabilities = self.probabilities(state)
        indicator = np.zeros(self.n_actions)
        indicator[action] = 1.0
        return np.outer(np.asarray(state, dtype=float), indicator - probabilities)


@dataclass
class TrainingResult:
    """What a training run produced, and enough to judge whether to trust it."""

    policy: LinearSoftmaxPolicy
    episode_rewards: List[float] = field(default_factory=list)
    final_action_distribution: Dict[str, float] = field(default_factory=dict)

    @property
    def improved(self) -> bool:
        """Whether the last fifth of episodes beat the first fifth.

        A weak check, and the only one available in-sample. It catches a run
        that never learned; it says nothing about whether what was learned
        generalizes, which is what evaluate_policy is for.
        """
        if len(self.episode_rewards) < 10:
            return False
        split = max(1, len(self.episode_rewards) // 5)
        return float(np.mean(self.episode_rewards[-split:])) > float(
            np.mean(self.episode_rewards[:split])
        )


def train_policy(
    environment: ExposureEnvironment,
    episodes: int = 200,
    learning_rate: float = 0.01,
    entropy_bonus: float = 0.01,
    discount: float = 0.99,
    seed: int = 0,
) -> TrainingResult:
    """REINFORCE with a return-to-go baseline.

    Args:
        environment: The exposure problem, built on the *training* window only.
        episodes: Passes over the trajectory. Each pass is the same market path
            with different sampled actions — which is the honest description of
            what "more episodes" buys here: more exploration of the action
            space, not more evidence about the market.
        learning_rate: Step size on the policy weights.
        entropy_bonus: Keeps the policy from collapsing to a single action
            early, which on a noisy reward it otherwise does within a few
            dozen episodes and then stops exploring.
        discount: Reward discount. Near 1 because an exposure decision's
            consequences are immediate; the discount is here to keep the
            return-to-go finite rather than to express impatience.
        seed: Seeds action sampling, so a run is reproducible.

    Returns:
        A TrainingResult holding the fitted policy and its learning curve.
    """
    policy = LinearSoftmaxPolicy(environment.n_features, environment.n_actions, seed=seed)
    episode_rewards: List[float] = []
    action_counts = np.zeros(environment.n_actions)

    for _ in range(int(episodes)):
        state = environment.reset()
        states, actions, rewards = [], [], []

        done = environment.n_steps == 0
        while not done:
            action = policy.sample(state)
            next_state, reward, done, _ = environment.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        if not rewards:
            break

        # Discounted return-to-go, then standardized. Standardizing is the
        # baseline: it centres the advantage so actions are compared against
        # what the episode typically did rather than against zero, which on a
        # near-zero-mean reward is the difference between learning and drifting.
        returns_to_go = np.zeros(len(rewards))
        running = 0.0
        for i in range(len(rewards) - 1, -1, -1):
            running = rewards[i] + discount * running
            returns_to_go[i] = running

        advantage = returns_to_go - returns_to_go.mean()
        spread = returns_to_go.std()
        if spread > 1e-12:
            advantage = advantage / spread

        gradient = np.zeros_like(policy.weights)
        for state_t, action_t, advantage_t in zip(states, actions, advantage):
            gradient += policy.gradient(state_t, action_t) * advantage_t

            if entropy_bonus > 0:
                # Entropy gradient, pushing the distribution back toward
                # uniform. Without it the policy commits early to whichever
                # action a noisy first few episodes favoured.
                probabilities = policy.probabilities(state_t)
                log_probabilities = np.log(np.maximum(probabilities, 1e-12))
                entropy_grad = -np.outer(
                    state_t, probabilities * (log_probabilities + 1.0)
                )
                gradient += entropy_bonus * entropy_grad

        policy.weights += learning_rate * gradient / max(1, len(rewards))
        episode_rewards.append(float(np.sum(rewards)))
        for action_t in actions:
            action_counts[action_t] += 1

    total = action_counts.sum()
    distribution = {
        f"{level:.0%}": float(count / total) if total else 0.0
        for level, count in zip(environment.config.exposure_levels, action_counts)
    }
    return TrainingResult(
        policy=policy,
        episode_rewards=episode_rewards,
        final_action_distribution=distribution,
    )


def evaluate_policy(
    policy: LinearSoftmaxPolicy,
    environment: ExposureEnvironment,
    greedy: bool = True,
) -> Dict[str, float]:
    """Run a frozen policy over a held-out window and report what it earned.

    **Run this on a window the policy never trained on.** An RL agent's
    in-sample performance on a single trajectory is close to meaningless: it
    has had many passes over the same path and will have fitted its
    idiosyncrasies. The comparison that matters is against `always_invested`,
    the same book held at full exposure throughout — an agent that cannot beat
    that has learned nothing except to reduce exposure.

    Returns:
        Realized net return, volatility, Sharpe and turnover for the policy and
        for the always-invested baseline.
    """
    state = environment.reset()
    net_returns: List[float] = []
    turnovers: List[float] = []
    exposures: List[float] = []

    done = environment.n_steps == 0
    while not done:
        action = (
            policy.act_greedily(state) if greedy else policy.sample(state)
        )
        state, _, done, info = environment.step(action)
        net_returns.append(info["net_return"])
        turnovers.append(info["turnover"])
        exposures.append(info["exposure"])

    def _sharpe(series: Sequence[float]) -> float:
        arr = np.asarray(series, dtype=float)
        if arr.size < 2:
            return 0.0
        sigma = float(np.std(arr, ddof=1))
        if sigma <= 1e-12:
            return 0.0
        return float(np.mean(arr) / sigma * math.sqrt(TRADING_DAYS_PER_YEAR))

    baseline = environment.returns[1:]

    return {
        "n_steps": float(len(net_returns)),
        "total_return": float(np.sum(net_returns)),
        "sharpe": _sharpe(net_returns),
        "volatility": float(np.std(net_returns, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))
        if len(net_returns) > 1 else 0.0,
        "mean_exposure": float(np.mean(exposures)) if exposures else 0.0,
        "total_turnover": float(np.sum(turnovers)),
        "always_invested_return": float(np.sum(baseline)),
        "always_invested_sharpe": _sharpe(baseline),
    }


def walk_forward_policy(
    strategy_returns: Sequence[float],
    regime_probabilities: Optional[np.ndarray] = None,
    config: Optional[EnvironmentConfig] = None,
    train_fraction: float = 0.6,
    episodes: int = 200,
    seed: int = 0,
) -> Dict[str, object]:
    """Fit on the leading window, evaluate frozen on the rest.

    The only evaluation of an RL policy on this data that means anything. Both
    halves matter: the policy is fitted without seeing the test window, and it
    is evaluated *greedily* so the reported number is the policy rather than a
    lucky draw from it.

    Even so, treat the result as one trial in a search, not as a finding — feed
    it to src/performance_stats.py and log it. An RL agent that beat the
    baseline on one split of one trajectory is exactly the kind of result the
    deflated Sharpe ratio exists to discount.
    """
    returns = np.asarray(strategy_returns, dtype=float).ravel()
    split = int(returns.size * min(max(train_fraction, 0.1), 0.9))
    if split < 30 or returns.size - split < 30:
        return {"skipped": "not enough history to split into train and test"}

    def _slice(start: int, stop: int) -> ExposureEnvironment:
        window = None
        if regime_probabilities is not None:
            window = np.asarray(regime_probabilities)[start:stop]
        return ExposureEnvironment(returns[start:stop], window, config)

    train_environment = _slice(0, split)
    result = train_policy(train_environment, episodes=episodes, seed=seed)

    return {
        "train": evaluate_policy(result.policy, _slice(0, split)),
        "test": evaluate_policy(result.policy, _slice(split, returns.size)),
        "action_distribution": result.final_action_distribution,
        "improved_in_training": result.improved,
        "policy": result.policy,
    }
