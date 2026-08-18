# --- Learned models: supervised LSTM and a SAC allocation policy --------------

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

TRADING_DAYS = 252

MODEL_FEATURES = [
    "return_1d", "return_5d", "return_21d", "rsi_14", "macd",
    "bollinger_pct_b", "atr_pct", "realized_vol_60", "mom_9m_skip1m",
    "volume_ratio_20", "breakout_20",
]


def resolve_device(preference: str = "auto") -> torch.device:
    """Pick a device that actually exists, downgrading rather than failing."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable — using CPU.")
        return torch.device("cpu")
    return torch.device(preference)


def seed_everything(seed: int = 42) -> None:
    """Seed every generator these notebooks draw from.

    Without this a re-run gives different weights, and two 'comparable' runs
    differ by an amount nobody can separate from the change being tested.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Shared: the supervised panel
# =============================================================================


@dataclass
class Standardizer:
    """Per-feature mean/std fitted on training rows only.

    Fitting on the whole history leaks the test period's moments into the
    transform. The leak is small and entirely invisible in the resulting
    metrics, which is what makes it worth being strict about.
    """

    mean: np.ndarray
    std: np.ndarray
    clip: float = 10.0

    @classmethod
    def fit(cls, block: np.ndarray, clip: float = 10.0) -> "Standardizer":
        finite = np.where(np.isfinite(block), block, np.nan)
        mean = np.nanmean(finite, axis=0)
        std = np.nanstd(finite, axis=0)
        std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
        return cls(mean=np.nan_to_num(mean), std=std, clip=clip)

    def transform(self, block: np.ndarray) -> np.ndarray:
        scaled = (block - self.mean) / self.std
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=self.clip, neginf=-self.clip)
        return np.clip(scaled, -self.clip, self.clip).astype(np.float32)

    def to_dict(self) -> Dict[str, object]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "clip": self.clip}


def cross_sectional_rank_target(
    close: pd.DataFrame, horizon: int = 5
) -> pd.DataFrame:
    """Forward return measured as a cross-sectional rank in [-1, 1].

    Predicting the *absolute* forward return spends the model's capacity on the
    market factor, which is nearly unforecastable and — for a long-only book
    with no index hedge — unusable even when forecast correctly. Ranking each
    name against the rest of the universe on the same date leaves the
    idiosyncratic part, which is what choosing between stocks can monetize.
    """
    forward = close.shift(-horizon) / close - 1.0
    ranked = forward.rank(axis=1, pct=True)
    return (2.0 * ranked - 1.0).where(forward.notna())


def build_supervised_panel(
    feature_panel: Dict[str, pd.DataFrame],
    close: pd.DataFrame,
    sequence_length: int = 30,
    horizon: int = 5,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    features: Sequence[str] = tuple(MODEL_FEATURES),
    max_abs_target: float = 1.0,
) -> Dict[str, object]:
    """Windowed sequences, split chronologically across the whole panel.

    The split is by *date*, not by row: every symbol's training window ends
    before any symbol's validation window begins. Splitting by row after
    stacking symbols would put one name's future alongside another's past.
    """
    features = list(features)
    target = cross_sectional_rank_target(close, horizon)

    dates = close.index
    train_end = dates[int(len(dates) * train_fraction)]
    val_end = dates[int(len(dates) * (train_fraction + val_fraction))]

    windows: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}
    labels: Dict[str, List[float]] = {"train": [], "val": [], "test": []}
    keys: Dict[str, List[Tuple[pd.Timestamp, str]]] = {"train": [], "val": [], "test": []}

    for symbol, frame in feature_panel.items():
        if symbol not in target.columns:
            continue
        block = frame.reindex(columns=features)
        y = target[symbol].reindex(block.index)

        usable = block.notna().all(axis=1) & y.notna() & (y.abs() <= max_abs_target)
        values = block.to_numpy(dtype=np.float64)
        y_values = y.to_numpy(dtype=np.float64)
        usable_values = usable.to_numpy()

        for end in range(sequence_length, len(block)):
            if not usable_values[end]:
                continue
            window = values[end - sequence_length:end]
            if not np.isfinite(window).all():
                continue
            date = block.index[end]
            split = "train" if date <= train_end else ("val" if date <= val_end else "test")
            windows[split].append(window)
            labels[split].append(y_values[end])
            keys[split].append((date, symbol))

    if not windows["train"]:
        raise ValueError(
            "no training windows — the panel is too short for "
            f"sequence_length={sequence_length} plus horizon={horizon}"
        )

    train_block = np.concatenate([w for w in windows["train"]], axis=0)
    standardizer = Standardizer.fit(train_block)

    out: Dict[str, object] = {
        "features": features,
        "standardizer": standardizer,
        "sequence_length": sequence_length,
        "horizon": horizon,
        "train_end": train_end,
        "val_end": val_end,
    }
    for split in ("train", "val", "test"):
        if windows[split]:
            stacked = np.stack(windows[split])
            shape = stacked.shape
            scaled = standardizer.transform(stacked.reshape(-1, shape[-1])).reshape(shape)
        else:
            scaled = np.zeros((0, sequence_length, len(features)), dtype=np.float32)
        out[f"X_{split}"] = scaled
        out[f"y_{split}"] = np.asarray(labels[split], dtype=np.float32)
        out[f"keys_{split}"] = keys[split]

    return out


# =============================================================================
# LSTM forecaster
# =============================================================================


class LSTMForecaster(nn.Module):
    """Two-layer LSTM over a feature window, predicting a cross-sectional rank."""

    def __init__(self, n_features: int, hidden_size: int = 64,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # The label is a rank in [-1, 1]; bounding the output to match keeps
            # the loss from being dominated by predictions outside the target's
            # own range, which carry no information.
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


def train_lstm(
    panel: Dict[str, object],
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_size: int = 64,
    patience: int = 6,
    device: str = "auto",
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, object]:
    """Fit the forecaster, selecting the epoch with the best validation loss.

    Early stopping is on validation loss, and the returned weights are the best
    epoch's rather than the last — on this signal-to-noise ratio a model
    reliably keeps improving in-sample long after it has stopped generalizing.
    """
    seed_everything(seed)
    torch_device = resolve_device(device)

    X_train = torch.tensor(panel["X_train"])
    y_train = torch.tensor(panel["y_train"])
    X_val = torch.tensor(panel["X_val"]).to(torch_device)
    y_val = torch.tensor(panel["y_val"]).to(torch_device)

    model = LSTMForecaster(X_train.shape[-1], hidden_size=hidden_size).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator
    )

    history: List[Dict[str, float]] = []
    best_val, best_state, best_epoch, stale = math.inf, None, 0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(torch_device), yb.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(xb)
            seen += len(xb)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val) if len(X_val) else torch.zeros(0, device=torch_device)
            val_loss = float(F.mse_loss(val_pred, y_val).item()) if len(X_val) else math.nan
            ic = (
                _rank_ic(val_pred, y_val, _dates_of(panel.get("keys_val")))
                if len(X_val) > 2 else math.nan
            )

        scheduler.step(val_loss if math.isfinite(val_loss) else 0.0)
        history.append({
            "epoch": epoch,
            "train_loss": total / max(seen, 1),
            "val_loss": val_loss,
            "val_rank_ic": ic,
        })

        if verbose and (epoch == 1 or epoch % 5 == 0):
            print(f"  epoch {epoch:3d}  train {total/max(seen,1):.5f}  "
                  f"val {val_loss:.5f}  rank-IC {ic:+.4f}")

        if math.isfinite(val_loss) and val_loss < best_val - 1e-6:
            best_val, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                if verbose:
                    print(f"  early stop at epoch {epoch} (best was {best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return {
        "model": model,
        "device": torch_device,
        "history": pd.DataFrame(history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "standardizer": panel["standardizer"],
        "features": panel["features"],
        "sequence_length": panel["sequence_length"],
    }


def _dates_of(keys: Optional[Sequence]) -> Optional[List]:
    """The date half of a panel's `(date, symbol)` keys."""
    return None if not keys else [key[0] for key in keys]


#: Names below which a date's cross-section is too thin to rank. Matches
#: `evaluation/metrics.MIN_CROSS_SECTION_NAMES` in the package.
MIN_CROSS_SECTION_NAMES = 5


def _rank_ic(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    dates: Optional[Sequence] = None,
) -> float:
    """Mean per-date Spearman correlation — the metric the ranking is used as.

    **Per date, not pooled**, and the difference is not a detail. Pooling every
    validation observation into one rank correlation measures whether the score
    tracks the market's *level* over time; the model is used to order *one
    day's* cross-section, and those are different questions with different
    answers. On a signal that orders every date perfectly while its level runs
    against the market, the pooled figure is -0.99 and the per-date figure is
    +1.00.

    The package carried the pooled version until T12, where it was driving
    model selection with the wrong sign. This file kept it afterwards, which is
    the drift `test_standalone_agrees.py` now exists to catch.

    Args:
        predictions, targets: Aligned validation tensors.
        dates: The decision date of each observation, from the panel's
            `keys_val`. **Required in practice**: without it there is no
            cross-section to correlate within, so the function refuses rather
            than silently pooling.

    Returns:
        Mean of the per-date correlations, or NaN when no date had a wide
        enough cross-section.
    """
    p = predictions.detach().cpu().numpy().ravel()
    t = targets.detach().cpu().numpy().ravel()
    if len(p) < 3:
        return float("nan")

    if dates is None:
        # Refusing rather than pooling. A pooled number here is not a worse
        # estimate of the same quantity — it is an estimate of a different one.
        return float("nan")

    frame = pd.DataFrame({"date": list(dates), "score": p, "label": t})
    per_date: List[float] = []
    for _date, group in frame.groupby("date", sort=True):
        if len(group) < MIN_CROSS_SECTION_NAMES:
            continue
        sr = group["score"].rank().to_numpy()
        lr = group["label"].rank().to_numpy()
        if sr.std() < 1e-12 or lr.std() < 1e-12:
            continue
        per_date.append(float(np.corrcoef(sr, lr)[0, 1]))

    return float(np.mean(per_date)) if per_date else float("nan")


def lstm_scores(
    trained: Dict[str, object],
    feature_panel: Dict[str, pd.DataFrame],
    close: pd.DataFrame,
    top_fraction: float = 0.25,
    batch_size: int = 1024,
) -> pd.DataFrame:
    """Score every (date, symbol) and keep the top slice per date.

    Predictions are produced for all dates including the training period, so the
    equity curve can be split into in-sample and out-of-sample segments. Only
    the segment after `panel["val_end"]` is evidence of anything.
    """
    model = trained["model"]
    device = trained["device"]
    standardizer: Standardizer = trained["standardizer"]
    features = list(trained["features"])
    sequence_length = int(trained["sequence_length"])

    predictions = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

    model.eval()
    for symbol, frame in feature_panel.items():
        if symbol not in close.columns:
            continue
        block = frame.reindex(columns=features)
        values = block.to_numpy(dtype=np.float64)
        valid = np.isfinite(values).all(axis=1)

        windows, dates = [], []
        for end in range(sequence_length, len(block)):
            if not valid[end - sequence_length:end].all():
                continue
            windows.append(values[end - sequence_length:end])
            dates.append(block.index[end])

        if not windows:
            continue

        stacked = np.stack(windows)
        shape = stacked.shape
        scaled = standardizer.transform(stacked.reshape(-1, shape[-1])).reshape(shape)

        outputs: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(scaled), batch_size):
                chunk = torch.tensor(scaled[start:start + batch_size]).to(device)
                outputs.append(model(chunk).cpu().numpy())

        predictions.loc[dates, symbol] = np.concatenate(outputs)

    ranked = predictions.rank(axis=1, pct=True)
    selected = (ranked >= 1.0 - top_fraction).astype(float)
    return (selected * ranked).fillna(0.0)


# =============================================================================
# SAC allocation policy
# =============================================================================


class SACActor(nn.Module):
    """Squashed-Gaussian policy mapping a feature vector to a weight in [0, 1].

    Sigmoid rather than tanh because the book is long-only and unlevered, so
    there is nothing for a negative half to mean. Inference uses the mean
    action, not a sample: sampling would make two runs of one backtest disagree.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.log_std_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mean_head(self.net(state)))

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(state)
        mean = self.mean_head(hidden)
        log_std = torch.clamp(self.log_std_head(hidden), -5.0, 2.0)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        pre_squash = normal.rsample()
        action = torch.sigmoid(pre_squash)
        # The sigmoid's Jacobian correction. Omitting it silently changes the
        # entropy target and the temperature the tuner settles on.
        log_prob = normal.log_prob(pre_squash) - torch.log(action * (1 - action) + 1e-6)
        return action, log_prob


class TwinCritic(nn.Module):
    """Two Q networks; the min of the pair is SAC's overestimation defence."""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        build = lambda: nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q1, self.q2 = build(), build()

    def forward(self, state, action):
        joint = torch.cat([state, action], dim=-1)
        return self.q1(joint), self.q2(joint)


class DifferentialSortino:
    """Online Sortino whose per-step increment is the reward.

    Moody & Saffell's differential Sharpe with the second moment replaced by the
    downside second moment. Summing the increments approximates the Sortino
    ratio of the whole path, so maximizing per-step reward maximizes a
    downside-aware ratio — the distinction that stops the policy learning
    "hold maximum size always".
    """

    def __init__(self, eta: float = 0.02, clip: float = 10.0):
        self.eta, self.clip = eta, clip
        self.a, self.dd2 = 0.0, 0.0

    def warmup(self, r: float) -> None:
        self.a += self.eta * (r - self.a)
        self.dd2 += self.eta * (min(r, 0.0) ** 2 - self.dd2)

    def update(self, r: float) -> float:
        delta_a = r - self.a
        delta_dd2 = min(r, 0.0) ** 2 - self.dd2
        if self.dd2 > 1e-12:
            reward = (self.dd2 * delta_a - 0.5 * self.a * delta_dd2) / (self.dd2 ** 1.5)
        else:
            reward = delta_a
        self.a += self.eta * delta_a
        self.dd2 += self.eta * delta_dd2
        if not math.isfinite(reward):
            return 0.0
        return float(np.clip(reward, -self.clip, self.clip))


@dataclass
class ReplayBuffer:
    """Fixed-capacity transition store with its own seeded generator.

    Its own generator, not the global numpy state: otherwise a run depends on
    every other caller that happened to draw a random number.
    """

    capacity: int
    state_dim: int
    seed: int = 42

    def __post_init__(self):
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, 1), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self._size, self._pos = 0, 0
        self._rng = np.random.default_rng(self.seed)

    def __len__(self) -> int:
        return self._size

    def push(self, state, action, reward, next_state) -> None:
        i = self._pos
        self.states[i], self.actions[i] = state, action
        self.rewards[i], self.next_states[i] = reward, next_state
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        if self._size == 0:
            raise ValueError("empty replay buffer")
        idx = self._rng.integers(0, self._size, size=batch_size)
        return (self.states[idx], self.actions[idx],
                self.rewards[idx], self.next_states[idx])


def train_sac(
    feature_panel: Dict[str, pd.DataFrame],
    close: pd.DataFrame,
    features: Sequence[str] = tuple(MODEL_FEATURES),
    epochs: int = 40,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    hidden_dim: int = 128,
    gamma: float = 0.0,
    tau: float = 0.005,
    friction_cost: float = 0.008,
    train_fraction: float = 0.7,
    gradient_steps: int = 150,
    buffer_size: int = 200_000,
    device: str = "auto",
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, object]:
    """Train a per-name allocation policy with twin-critic SAC.

    `gamma` defaults to 0 on purpose. Discounting assumes the action influences
    the next state; a price-taking book does not move the market, so the state
    sequence is exogenous and bootstrapping over it adds estimator variance
    without adding signal. With gamma=0 the critic learns Q(s,a) = E[r|s,a] and
    the actor maximizes Q - alpha*log pi — soft actor-critic on what this
    decision actually is, a contextual bandit.

    The reward is the differential Sortino of `a_t * ret_{t+1} - cost*|a_t -
    a_{t-1}|`. The friction term is a function of the *action*: a cost charged
    as a constant every step shifts every action's reward identically and
    therefore cannot penalize turnover at all.
    """
    seed_everything(seed)
    torch_device = resolve_device(device)
    features = list(features)
    state_dim = len(features)

    dates = close.index
    split_date = dates[int(len(dates) * train_fraction)]

    # Standardize on training rows only.
    train_blocks = []
    for symbol, frame in feature_panel.items():
        block = frame.reindex(columns=features)
        block = block[block.index <= split_date].dropna()
        if len(block):
            train_blocks.append(block.to_numpy(dtype=np.float64))
    if not train_blocks:
        raise ValueError("no training rows for the SAC policy")
    standardizer = Standardizer.fit(np.concatenate(train_blocks))

    episodes: Dict[str, Dict[str, np.ndarray]] = {}
    for symbol, frame in feature_panel.items():
        if symbol not in close.columns:
            continue
        block = frame.reindex(columns=features).dropna()
        block = block[block.index <= split_date]
        if len(block) < 60:
            continue
        prices = close[symbol].reindex(block.index)
        forward = (prices.shift(-1) / prices - 1.0).to_numpy()
        episodes[symbol] = {
            "states": standardizer.transform(block.to_numpy(dtype=np.float64)),
            "forward": forward,
        }
    if not episodes:
        raise ValueError("no usable episodes for the SAC policy")

    actor = SACActor(state_dim, hidden_dim).to(torch_device)
    critic = TwinCritic(state_dim, hidden_dim).to(torch_device)
    critic_target = TwinCritic(state_dim, hidden_dim).to(torch_device)
    critic_target.load_state_dict(critic.state_dict())
    for p in critic_target.parameters():
        p.requires_grad_(False)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=learning_rate)
    log_alpha = torch.tensor(math.log(0.1), device=torch_device, requires_grad=True)
    alpha_opt = torch.optim.Adam([log_alpha], lr=learning_rate)
    target_entropy = -1.0

    buffer = ReplayBuffer(buffer_size, state_dim, seed=seed)
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        # Re-collect on-policy experience: training against a buffer collected
        # once means the actor only ever sees a random policy's decisions.
        actor.eval()
        with torch.no_grad():
            for episode in episodes.values():
                states = episode["states"]
                forward = episode["forward"]
                actions, _ = actor.sample(torch.as_tensor(states, device=torch_device))
                actions = actions.squeeze(-1).cpu().numpy()

                sortino = DifferentialSortino()
                previous = 0.0
                for t in range(len(states) - 1):
                    r = forward[t]
                    if not np.isfinite(r):
                        continue
                    a = float(actions[t])
                    net = a * float(r) - friction_cost * abs(a - previous)
                    previous = a
                    if t < 20:
                        sortino.warmup(net)
                        continue
                    buffer.push(states[t], a, sortino.update(net), states[t + 1])
        actor.train()

        if len(buffer) < batch_size:
            continue

        critic_total, actor_total = 0.0, 0.0
        for _ in range(gradient_steps):
            s, a, r, s2 = buffer.sample(batch_size)
            s = torch.as_tensor(s, device=torch_device)
            a = torch.as_tensor(a, device=torch_device)
            r = torch.as_tensor(r, device=torch_device)
            s2 = torch.as_tensor(s2, device=torch_device)
            alpha = log_alpha.exp().detach()

            with torch.no_grad():
                a2, logp2 = actor.sample(s2)
                q1t, q2t = critic_target(s2, a2)
                backup = r + gamma * (torch.min(q1t, q2t) - alpha * logp2)

            q1, q2 = critic(s, a)
            critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
            critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_opt.step()

            new_a, logp = actor.sample(s)
            q1pi, q2pi = critic(s, new_a)
            actor_loss = (alpha * logp - torch.min(q1pi, q2pi)).mean()
            actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_opt.step()

            alpha_loss = -(log_alpha * (logp.detach() + target_entropy)).mean()
            alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            alpha_opt.step()

            with torch.no_grad():
                for p, tp in zip(critic.parameters(), critic_target.parameters()):
                    tp.mul_(1 - tau).add_(p, alpha=tau)

            critic_total += float(critic_loss.item())
            actor_total += float(actor_loss.item())

        history.append({
            "epoch": epoch,
            "critic_loss": critic_total / gradient_steps,
            "actor_loss": actor_total / gradient_steps,
            "alpha": float(log_alpha.exp().item()),
            "buffer": len(buffer),
        })
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"  epoch {epoch:3d}  critic {history[-1]['critic_loss']:.4f}  "
                  f"actor {history[-1]['actor_loss']:+.4f}  alpha {history[-1]['alpha']:.4f}")

    actor.eval()
    return {
        "actor": actor,
        "device": torch_device,
        "standardizer": standardizer,
        "features": features,
        "history": pd.DataFrame(history),
        "split_date": split_date,
    }


def sac_scores(
    trained: Dict[str, object],
    feature_panel: Dict[str, pd.DataFrame],
    close: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Allocation weight per name per date, from the deterministic policy."""
    actor = trained["actor"]
    device = trained["device"]
    standardizer: Standardizer = trained["standardizer"]
    features = list(trained["features"])

    scores = pd.DataFrame(0.0, index=close.index, columns=close.columns)

    actor.eval()
    with torch.no_grad():
        for symbol, frame in feature_panel.items():
            if symbol not in close.columns:
                continue
            block = frame.reindex(columns=features).dropna()
            if block.empty:
                continue
            states = standardizer.transform(block.to_numpy(dtype=np.float64))
            weights = actor(torch.as_tensor(states, device=device)).squeeze(-1).cpu().numpy()
            scores.loc[block.index, symbol] = weights

    # Below the threshold the policy is not asking for a position; keeping those
    # weights would put a token holding in every name in the universe.
    return scores.where(scores >= threshold, 0.0)
