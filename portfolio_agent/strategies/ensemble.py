"""Ensemble strategies — combine multiple registered strategies into one.

Users can plug together any mix of rule-based and ML strategies into a single
named combination (a "UMA" — a user-made/unified multi-strategy agent),
configured via a YAML file that lists member strategies with weights and a
combination method, mirroring how config/strategies/trend_breakout.yaml
configures the plain rule-based strategy. See
config/strategies/example_uma.yaml for the file shape.

Two combination methods are supported, selected per UMA via the YAML's
`method` field:

- weighted_blend (default): each member's signal is mapped to a -1..1
  strength (BUY=1, WATCH=0.3, HOLD=0, AVOID=-0.3, SELL=-1) and blended by
  weight; score/entry/stop/target/probability are likewise weighted averages.
  Works well for any mix of rule-based and ML members.
- vote: each member casts a BUY/SELL/HOLD-bucketed vote; `vote.mode:
  majority` (>50% agreement) or `vote.mode: unanimous` (all members agree)
  decides the combined signal. More conservative — fewer, higher-conviction
  signals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from .base import BaseStrategy
from .types import StrategyContext, StrategySignal
from portfolio_agent.config.schema import StrategyConfig

_SIGNAL_STRENGTH = {
    "BUY": 1.0,
    "WATCH": 0.3,
    "HOLD": 0.0,
    "AVOID": -0.3,
    "SELL": -1.0,
}

_BUY_LIKE = {"BUY"}
_SELL_LIKE = {"SELL"}


def _strength_to_signal(strength: float) -> str:
    if strength >= 0.6:
        return "BUY"
    if strength >= 0.2:
        return "WATCH"
    if strength <= -0.6:
        return "SELL"
    if strength <= -0.2:
        return "AVOID"
    return "HOLD"


class EnsembleStrategy(BaseStrategy):
    """Combines multiple member strategies into a single "UMA" strategy.

    Not GPU-batched: member strategies (which may include rule-based members
    needing a per-ticker Monte Carlo result) are always scored per-ticker via
    score(), so every member gets a correctly-computed StrategyContext. An ML
    member embedded in a UMA is therefore called once per ticker rather than
    batched across tickers — for maximum ML-inference throughput, use that
    strategy directly (--strategy lstm) instead of wrapping it in a UMA.

    Cross-sectional strategies (momentum, low_volatility) cannot be UMA
    members at all: their signals depend on ranking the *entire* eligible
    universe at once (BaseStrategy.requires_full_batch), which per-ticker
    score() cannot provide. Use them directly instead.
    """

    def __init__(self, config: StrategyConfig):
        self._config = config
        self._yaml_path = config.params.get("yaml_path", config.config_path)
        self._spec = self._load_spec()

        # Lazy import: registry.py imports this module to register
        # EnsembleStrategy, so importing load_strategy at module level here
        # would create a circular import. By the time a UMA is actually
        # instantiated, the registry module has already finished executing.
        from .registry import load_strategy

        self._members: List[BaseStrategy] = []
        self._weights: List[float] = []
        for member_spec in self._spec.get("members", []):
            member_config = StrategyConfig(
                type=member_spec["type"],
                module=config.module,
                config_path=member_spec.get("config_path", config.config_path),
                params=member_spec.get("params", {}),
            )
            member = load_strategy(member_config)
            if member.requires_full_batch:
                raise ValueError(
                    f"UMA member '{member.name}' (type={member_spec['type']!r}) requires the full "
                    f"eligible universe to score correctly (e.g. cross-sectional momentum/low-volatility "
                    f"ranking) and cannot be combined via per-ticker score() the way UMAs blend members "
                    f"today. Use it directly (--strategy {member_spec['type']}) instead of inside a UMA."
                )
            self._members.append(member)
            self._weights.append(float(member_spec.get("weight", 1.0)))

    def _load_spec(self) -> Dict[str, Any]:
        yaml_path = Path(self._yaml_path)
        if not yaml_path.is_absolute():
            if not yaml_path.exists():
                package_root = Path(__file__).parent.parent
                candidate = package_root / self._yaml_path
                if candidate.exists():
                    yaml_path = candidate
                else:
                    yaml_path = package_root.parent / self._yaml_path

        if not yaml_path.exists():
            raise FileNotFoundError(f"UMA (ensemble) YAML file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            spec = yaml.safe_load(f)

        if not spec.get("members"):
            raise ValueError(f"UMA YAML {yaml_path} must define at least one strategy under 'members'")

        return spec

    @property
    def name(self) -> str:
        return self._spec.get("name", "ensemble")

    @property
    def method(self) -> str:
        return self._spec.get("method", "weighted_blend")

    def load(self) -> bool:
        """Load any member strategies that require it (e.g. ML members)."""
        ok = True
        for member in self._members:
            if hasattr(member, "load"):
                ok = member.load() and ok
        return ok

    def required_features(self) -> List[str]:
        names: List[str] = []
        for member in self._members:
            for feature_name in member.required_features():
                if feature_name not in names:
                    names.append(feature_name)
        return names

    def entry_rules(self) -> Dict[str, Any]:
        return {"uma_method": self.method, "members": [m.name for m in self._members], "weights": self._weights}

    def exit_rules(self) -> Dict[str, Any]:
        return {}

    def score(self, symbol: str, features: pd.DataFrame, context: StrategyContext) -> StrategySignal:
        member_signals = [member.score(symbol, features, context) for member in self._members]
        if self.method == "vote":
            return self._combine_vote(symbol, member_signals)
        return self._combine_weighted_blend(symbol, member_signals)

    def _combine_weighted_blend(self, symbol: str, signals: List[StrategySignal]) -> StrategySignal:
        weights = self._weights
        total_weight = sum(weights) or 1.0

        def wavg(values: List[float]) -> float:
            return sum(w * v for w, v in zip(weights, values)) / total_weight

        blended_strength = wavg([_SIGNAL_STRENGTH.get(s.signal, 0.0) for s in signals])
        blended_score = wavg([s.score for s in signals])
        blended_prob = wavg([s.probability_profit for s in signals])
        blended_entry = wavg([s.entry_price for s in signals])
        blended_stop = wavg([s.stop_price for s in signals])
        blended_target = wavg([s.target_price for s in signals])

        signal = _strength_to_signal(blended_strength)

        risk = blended_entry - blended_stop
        reward_risk = (blended_target - blended_entry) / risk if risk > 0 else 0.0

        rationale = "; ".join(
            f"{m.name}(w={w:.2f})={s.signal}/{s.score:.1f}" for m, w, s in zip(self._members, weights, signals)
        )

        return StrategySignal(
            symbol=symbol, signal=signal, score=round(blended_score, 2), trigger="Ensemble:weighted_blend",
            entry_price=blended_entry, stop_price=blended_stop, target_price=blended_target,
            reward_risk=round(reward_risk, 4), probability_profit=round(blended_prob, 6),
            component_scores={m.name: s.score for m, s in zip(self._members, signals)},
            rationale=rationale,
            extra={"member_signals": {m.name: s.signal for m, s in zip(self._members, signals)}},
        )

    def _combine_vote(self, symbol: str, signals: List[StrategySignal]) -> StrategySignal:
        vote_mode = self._spec.get("vote", {}).get("mode", "majority")

        n = len(signals)
        buy_votes = sum(1 for s in signals if s.signal in _BUY_LIKE)
        sell_votes = sum(1 for s in signals if s.signal in _SELL_LIKE)

        if vote_mode == "unanimous":
            signal = "BUY" if buy_votes == n else ("SELL" if sell_votes == n else "HOLD")
        else:
            signal = "BUY" if buy_votes > n / 2 else ("SELL" if sell_votes > n / 2 else "HOLD")

        avg = lambda values: sum(values) / n  # noqa: E731
        avg_score = avg([s.score for s in signals])
        avg_prob = avg([s.probability_profit for s in signals])
        avg_entry = avg([s.entry_price for s in signals])
        avg_stop = avg([s.stop_price for s in signals])
        avg_target = avg([s.target_price for s in signals])

        risk = avg_entry - avg_stop
        reward_risk = (avg_target - avg_entry) / risk if risk > 0 else 0.0

        rationale = (
            f"vote({vote_mode}): BUY={buy_votes}/{n} SELL={sell_votes}/{n} -> {signal}; "
            + "; ".join(f"{m.name}={s.signal}" for m, s in zip(self._members, signals))
        )

        return StrategySignal(
            symbol=symbol, signal=signal, score=round(avg_score, 2), trigger=f"Ensemble:vote:{vote_mode}",
            entry_price=avg_entry, stop_price=avg_stop, target_price=avg_target,
            reward_risk=round(reward_risk, 4), probability_profit=round(avg_prob, 6),
            component_scores={m.name: s.score for m, s in zip(self._members, signals)},
            rationale=rationale,
            extra={"member_signals": {m.name: s.signal for m, s in zip(self._members, signals)}},
        )
