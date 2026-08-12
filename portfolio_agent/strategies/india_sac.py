"""Soft Actor-Critic allocation strategy for Indian equities.

A continuous-action RL policy: instead of emitting a discrete BUY/SELL, the
actor network outputs an allocation weight in [0, 1] for each name, which the
platform's sizing layer then treats as a conviction multiplier. The training
objective this is designed against maximizes a differential Sortino ratio net of
turnover cost, so the policy is rewarded for downside-aware returns rather than
for raw return — but *only inference lives here*. There is no training loop in
this module and no checkpoint ships with the repository.

**This strategy is inert until someone trains it.** `load()` fails, loudly and
by design, when no checkpoint is present. That is the single most important
property of this file: an untrained actor is a randomly-initialized network
whose sigmoid output is an arbitrary number in [0, 1], and roughly 40% of a
universe would clear a 0.60 threshold on noise. A strategy that silently trades
on random weights is far worse than one that refuses to start.

Three unit conventions this file is careful about, because each is a place where
a plausible-looking number would be silently wrong:

- The actor's output is an **allocation weight**, not a probability. It is
  reported as `score` (0-100) and in `component_scores`, never as
  `probability_profit` — that field is read by the live report and the SQLite
  recommendation row as a probability of profit, and the platform's own Monte
  Carlo is what supplies it.
- Stops and targets come from **ATR multiples on the context's risk params**,
  not from fixed percentages, so a filled position's exit plan matches what the
  engine will actually apply (BacktestEngine._exit_levels reads the signal's own
  levels).
- `reward_risk` is reported **net of round-trip friction**, because
  `compliance.min_reward_risk` is documented as a net gate. Reporting a gross
  ratio into a net gate overstates every candidate by roughly the friction
  stack.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .base import TrainableStrategy
from .types import StrategyContext, StrategySignal
from portfolio_agent.config.schema import StrategyConfig
from portfolio_agent.features.scaling import FeatureScaler
from portfolio_agent.utils.device import resolve_device

try:
    from portfolio_agent.src.risk import calculate_stop_target, net_reward_risk
except ImportError:  # running from inside src/ as a flat package
    from src.risk import calculate_stop_target, net_reward_risk

logger = logging.getLogger(__name__)

# Features the actor observes. All eleven exist in the feature registry; the
# long-lookback ones (mom_9m_skip1m, realized_vol_60, traded_value_60) are why
# the default min_history is a full trading year — they are NaN until their
# window fills, and a row with any NaN is not a state the policy can act on.
DEFAULT_SAC_FEATURES: List[str] = [
    "close", "volume_ratio_20", "return_1d", "return_5d",
    "mom_9m_skip1m", "realized_vol_60", "traded_value_60",
    "rsi_14", "macd", "bollinger_pct_b", "atr_14",
]


class SACActorNetwork(nn.Module):
    """The deterministic half of a SAC actor: state -> allocation weight.

    SAC trains a *stochastic* policy — a squashed Gaussian whose log-standard-
    deviation head supplies the entropy term that gives the algorithm its name.
    At inference that head is deliberately dropped and the mean action is used,
    which is the standard evaluation policy and the reason this is reproducible:
    sampling here would make two runs of one backtest disagree, which the
    platform forbids.

    The sigmoid is what makes the action an allocation rather than a signed
    position: this book is long-only and unlevered, so the action space is
    [0, 1] and there is nothing for a tanh's negative half to mean.
    """

    def __init__(self, state_dim: int, action_dim: int = 1, hidden_dim: int = 256):
        super().__init__()
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if action_dim != 1:
            # Guarded rather than silently squeezed: the scoring path reads one
            # weight per name, and a wider head would have its extra columns
            # dropped without anyone noticing.
            raise ValueError(
                f"action_dim must be 1 for a long-only allocation policy, got {action_dim}"
            )
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mean_head(self.net(state)))


class IndiaSACStrategy(TrainableStrategy):
    """Continuous-allocation RL strategy, scored in one batched forward pass."""

    #: Produced by the SAC trainer (training/trainers/sac.py). Training this
    #: strategy is `portfolio-agent train --strategy india_sac`.
    trainer_name = "sac"

    @classmethod
    def training_defaults(cls) -> Dict[str, Any]:
        """Longer than the schema default: an entropy-regularized policy on
        this reward needs more passes than a supervised regression does."""
        return {"epochs": 200, "min_history": 252}

    def __init__(
        self,
        config: StrategyConfig,
        models_dir: str = "models",
        device: str = "cpu",
    ) -> None:
        params = config.params or {}
        self._config = config
        self._models_dir = params.get("models_dir", models_dir)
        self._device = resolve_device(params.get("device", device))

        self._hidden_dim = int(params.get("hidden_dim", 256))
        # Above this the actor is asking for a large enough allocation to call a
        # BUY. Distinct from compliance.target_prob_profit, which gates a
        # probability — this gates a weight.
        self._buy_threshold = float(params.get("action_threshold", 0.60))
        # A held name whose allocation has decayed below this is an exit. Kept
        # asymmetric and configurable rather than mirrored around the buy
        # threshold, so entry and exit conviction can be tuned independently.
        self._exit_threshold = float(params.get("exit_threshold", 0.40))
        if not 0.0 <= self._exit_threshold <= self._buy_threshold <= 1.0:
            raise ValueError(
                f"need 0 <= exit_threshold ({self._exit_threshold}) <= "
                f"action_threshold ({self._buy_threshold}) <= 1"
            )
        self._min_history = int(params.get("min_history", 252))

        self._model_name = params.get("model_name", "india_sac")
        self._checkpoint_path = Path(self._models_dir) / f"{self._model_name}_best.pt"

        self._feature_names: List[str] = list(
            params.get("features", DEFAULT_SAC_FEATURES)
        )
        self._state_dim = len(self._feature_names)

        self._model: Optional[SACActorNetwork] = None
        self._scaler: Optional[FeatureScaler] = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "india_sac"

    @property
    def supports_gpu_batch(self) -> bool:
        """One stacked forward pass over the whole eligible universe."""
        return True

    def required_features(self) -> List[str]:
        return list(self._feature_names)

    def load(self) -> bool:
        """Load the trained actor. Returns False rather than improvising.

        **A missing checkpoint is a failure, not a fallback.** An earlier draft
        of this strategy caught FileNotFoundError, kept the randomly-initialized
        network and returned True. That is the most expensive possible
        behaviour: an untrained sigmoid emits arbitrary values in [0, 1], so a
        large share of any universe clears the threshold, and the platform would
        place trades on noise while reporting a healthy-looking allocation
        score. The caller aborts the run with a clear error instead.
        """
        if not self._checkpoint_path.exists():
            logger.error(
                "SAC checkpoint not found at %s. This strategy ships without one "
                "and cannot be scored untrained — a randomly-initialized actor "
                "produces arbitrary allocations, not neutral ones.",
                self._checkpoint_path,
            )
            return False

        try:
            checkpoint = torch.load(
                self._checkpoint_path, map_location=self._device, weights_only=True
            )
            metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}

            # Feature list travels with the checkpoint when present: a state
            # vector assembled in a different order than training used is not
            # detectable from the weights, and would silently score nonsense.
            recorded_features = metadata.get("feature_names")
            if recorded_features:
                self._feature_names = list(recorded_features)
                self._state_dim = len(self._feature_names)

            self._model = SACActorNetwork(
                state_dim=self._state_dim,
                action_dim=1,
                hidden_dim=int(metadata.get("hidden_dim", self._hidden_dim)),
            )
            state_dict = (
                checkpoint.get("model_state_dict", checkpoint)
                if isinstance(checkpoint, dict) else checkpoint
            )
            self._model.load_state_dict(state_dict)
            self._model = self._model.to(self._device)
            self._model.eval()

            # Same contract MLStrategy uses: the standardization constants ship
            # with the weights so inference applies the transform training used.
            # Absent for checkpoints trained on raw features, which are then
            # scored on raw features rather than silently standardized.
            self._scaler = FeatureScaler.from_dict(metadata.get("feature_scaler"))

            self._loaded = True
            logger.info(
                "Loaded SAC actor from %s (%d features, scaler=%s)",
                self._checkpoint_path, self._state_dim,
                "yes" if self._scaler is not None else "none",
            )
            return True

        except Exception:
            logger.error(
                "Failed to load the SAC actor from %s", self._checkpoint_path,
                exc_info=True,
            )
            self._model = None
            self._loaded = False
            return False

    def score(
        self, symbol: str, features: pd.DataFrame, context: StrategyContext
    ) -> StrategySignal:
        return self.score_batch({symbol: features}, context)[symbol]

    def score_batch(
        self, features_by_symbol: Dict[str, pd.DataFrame], context: StrategyContext
    ) -> Dict[str, StrategySignal]:
        """Score every eligible name in one forward pass.

        Every symbol handed in gets a signal back, so a caller can index the
        result without checking. Names that cannot be scored come back as AVOID
        with the reason in the rationale, which is the same contract
        RuleBasedStrategy's `_empty_signal` provides.

        Deliberately *not* wrapped in a blanket try/except. An earlier draft
        caught everything and returned an all-HOLD dictionary so the engine
        would not crash; the effect is that a genuinely broken model looks
        exactly like a market with no opportunities, forever. Per-ticker
        failures are already handled below by skipping the ticker, and the
        engine's own per-ticker guard handles the rest.
        """
        if not self._loaded and not self.load():
            raise RuntimeError(
                "IndiaSACStrategy.load() failed — refusing to score with an "
                "untrained or unloadable actor. See the log for the cause."
            )

        states: Dict[str, np.ndarray] = {}
        latest_close: Dict[str, float] = {}
        latest_atr: Dict[str, Optional[float]] = {}

        for symbol, features in features_by_symbol.items():
            state = self._state_vector(features)
            if state is None:
                continue
            states[symbol] = state
            row = features.iloc[-1]
            latest_close[symbol] = self._clean(row.get("close")) or 0.0
            latest_atr[symbol] = self._clean(row.get("atr_14"))

        signals: Dict[str, StrategySignal] = {}

        if states:
            symbols = list(states)
            matrix = np.stack([states[s] for s in symbols])
            if self._scaler is not None:
                matrix = self._scaler.transform(matrix)

            batch = torch.tensor(np.asarray(matrix, dtype=np.float32)).to(self._device)
            with torch.no_grad():
                weights = self._model(batch).reshape(-1).cpu().numpy()

            for symbol, weight in zip(symbols, weights):
                signals[symbol] = self._build_signal(
                    symbol=symbol,
                    weight=float(weight),
                    close=latest_close[symbol],
                    atr=latest_atr[symbol],
                    context=context,
                )

        for symbol in features_by_symbol:
            if symbol not in signals:
                signals[symbol] = self._unscoreable(symbol)

        return signals

    @staticmethod
    def _clean(value: Any) -> Optional[float]:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    def _state_vector(self, features: pd.DataFrame) -> Optional[np.ndarray]:
        """The actor's observation: the most recent fully-populated row.

        A single row rather than a window, because the observation is a state in
        the RL sense — the features already encode their own history (a 9-month
        momentum, a 60-day realized volatility). `min_history` is not about the
        model's memory but about the *features*: the long-lookback ones are NaN
        until their windows fill, and a partially-NaN state is not scoreable.
        """
        if not isinstance(features, pd.DataFrame) or features.empty:
            return None
        if len(features) < self._min_history:
            return None
        missing = [f for f in self._feature_names if f not in features.columns]
        if missing:
            return None

        row = features[self._feature_names].iloc[-1]
        values = pd.to_numeric(row, errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            return None
        return values

    def _build_signal(
        self,
        symbol: str,
        weight: float,
        close: float,
        atr: Optional[float],
        context: StrategyContext,
    ) -> StrategySignal:
        risk = context.risk
        stop_price, target_price = calculate_stop_target(
            close, atr, risk.atr_stop_multiplier, risk.atr_target_multiplier
        )

        stop_valid = 0.0 < stop_price < close
        if stop_valid:
            reward_risk = net_reward_risk(
                entry_price=close,
                stop_price=stop_price,
                target_price=target_price,
                buy_cost_pct=risk.buy_cost_pct,
                sell_cost_pct=risk.sell_cost_pct,
            )
        else:
            reward_risk = 0.0

        # The probability of profit is the Monte Carlo's to report, not the
        # actor's. The allocation weight is a different quantity on a different
        # scale, and this field is published to the live Excel report and the
        # SQLite recommendation row as a probability.
        mc_result = getattr(context, "mc_result", None)
        probability_profit = (
            float(mc_result.probability_profit) if mc_result is not None else 0.0
        )

        if not stop_valid:
            signal = "AVOID"
        elif weight >= self._buy_threshold:
            signal = "BUY"
        elif weight <= self._exit_threshold:
            # Only acted on for names already held — the engine filters SELL
            # signals to current holdings — so this reads as "exit if you own
            # it", not as a short.
            signal = "SELL"
        else:
            signal = "HOLD"

        probability_note = (
            f"prob={probability_profit:.2f}" if mc_result is not None
            else "prob=n/a (no MC result in this batch)"
        )
        rationale = "; ".join([
            f"SAC allocation={weight:.3f}",
            f"threshold(buy>={self._buy_threshold:.2f}, exit<={self._exit_threshold:.2f})",
            probability_note,
            f"rr(net {reward_risk:.2f})>={risk.min_reward_risk}",
            "stop<entry:VALID" if stop_valid else "stop>=entry:INVALID",
        ])

        return StrategySignal(
            symbol=symbol,
            signal=signal,
            # 0-100 like every other strategy's score, so the engine's ordering
            # of BUY candidates by descending score keeps its meaning.
            score=round(weight * 100.0, 2),
            trigger="SAC",
            entry_price=close,
            stop_price=stop_price,
            target_price=target_price,
            reward_risk=round(reward_risk, 4),
            probability_profit=round(probability_profit, 6),
            component_scores={"SAC": round(weight, 6)},
            rationale=rationale,
            extra={"sac_allocation_weight": float(weight)},
        )

    def _unscoreable(self, symbol: str) -> StrategySignal:
        """AVOID rather than HOLD: the policy has no opinion, which is not the
        same as an opinion to keep holding."""
        return StrategySignal(
            symbol=symbol, signal="AVOID", score=0.0, trigger="None",
            entry_price=0.0, stop_price=0.0, target_price=0.0,
            reward_risk=0.0, probability_profit=0.0, component_scores={},
            rationale=(
                f"No scoreable state: needs {self._min_history} bars and all of "
                f"{len(self._feature_names)} features finite on the latest row"
            ),
        )
