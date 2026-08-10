"""Ensemble strategies — combine multiple registered strategies into one.

Users can plug together any mix of rule-based and ML strategies into a single
named combination (a "UMA" — a user-made/unified multi-strategy agent),
configured via a YAML file that lists member strategies with weights and a
combination method, mirroring how config/strategies/trend_breakout.yaml
configures the plain rule-based strategy. See
config/strategies/example_uma.yaml for the file shape.

Three combination methods are supported, selected per UMA via the YAML's
`method` field:

- trigger: members are converted to ModelVerdicts and arbitrated by
  src/trigger_engine.py. This is the method to use for anything that trades
  real money. The other two both average, and averaging a strong BUY against
  a strong SELL produces a weak BUY — a trade neither member would take
  alone, entered precisely when the models disagree most. The trigger engine
  discounts conviction by the opposing conviction, applies hard vetoes for
  tradability, regime and expected value, and emits a position-size
  multiplier rather than a direction alone.
- weighted_blend (default, retained for compatibility): each member's signal
  is mapped to a -1..1 strength (BUY=1, WATCH=0.3, HOLD=0, AVOID=-0.3,
  SELL=-1) and blended by weight; score/entry/stop/target/probability are
  likewise weighted averages. Cheap and smooth, and wrong in the specific way
  described above whenever members conflict.
- vote: each member casts a BUY/SELL/HOLD-bucketed vote; `vote.mode:
  majority` (>50% agreement) or `vote.mode: unanimous` (all members agree)
  decides the combined signal. More conservative than blending — it cannot
  manufacture a signal out of disagreement — but it throws away conviction
  magnitude, position sizing and the expected-value hurdle.

`weighted_blend` stays the default only so existing UMA files keep behaving
as they did. New configurations should set `method: trigger`; see
config/strategies/uma_meta_orchestrator.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from .base import BaseStrategy
from .types import ModelVerdict, StrategyContext, StrategySignal
from portfolio_agent.config.schema import StrategyConfig
from portfolio_agent.src.trigger_engine import TriggerConfig, TriggerEngine

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

        self._trigger_engine = TriggerEngine(TriggerConfig.from_params(self._spec.get("trigger")))
        # Which members the current regime lets buy. Read from the YAML's
        # `regimes` block by the meta-orchestrator (Phase 4); absent means every
        # member is always permitted, which is how a plain UMA behaves.
        self._regime_map: Dict[str, List[str]] = {
            str(label): [str(name) for name in names]
            for label, names in (self._spec.get("regimes") or {}).items()
        }

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
        rules: Dict[str, Any] = {
            "uma_method": self.method,
            "members": [m.name for m in self._members],
            "weights": self._weights,
        }
        if self.method == "trigger":
            rules["trigger"] = {
                "mode": self._trigger_engine.config.mode,
                "strong_confidence": self._trigger_engine.config.strong_confidence,
                "consensus_confidence": self._trigger_engine.config.consensus_confidence,
                "min_consensus_models": self._trigger_engine.config.min_consensus_models,
                "min_net_ev_pct": self._trigger_engine.config.min_net_ev_pct,
            }
        if self._regime_map:
            rules["regimes"] = self._regime_map
        return rules

    def exit_rules(self) -> Dict[str, Any]:
        return {}

    def permitted_members(self, regime_label: Optional[str]) -> Optional[List[str]]:
        """Member names the given regime allows to buy, or None for "all".

        An unrecognized or missing regime label returns None rather than an
        empty list: not knowing the regime is not evidence that every model
        should be silenced, and standing the whole book down on a lookup miss
        is a far worse failure than trading one sleeve out of season.
        """
        if not self._regime_map or regime_label is None:
            return None
        return self._regime_map.get(regime_label)

    def score(self, symbol: str, features: pd.DataFrame, context: StrategyContext) -> StrategySignal:
        member_signals = [member.score(symbol, features, context) for member in self._members]
        if self.method == "trigger":
            return self._combine_trigger(symbol, member_signals, context)
        if self.method == "vote":
            return self._combine_vote(symbol, member_signals)
        return self._combine_weighted_blend(symbol, member_signals)

    def _combine_trigger(
        self, symbol: str, signals: List[StrategySignal], context: StrategyContext
    ) -> StrategySignal:
        """Arbitrate the members through the trigger engine.

        The emitted signal is either a BUY carrying the engine's size
        multiplier in `extra["position_scale"]` — the same channel the
        cross-sectional strategies already use for volatility targeting, so the
        backtest engine and live orchestrator pick it up with no extra wiring —
        or an AVOID carrying the block reason. There is deliberately no
        in-between: a "weak buy" is the artifact this method exists to remove.

        Entry, stop and target come from the highest-conviction *contributing*
        member rather than from an average of every member's levels. Averaging
        a momentum model's wide ATR stop with a mean-reversion model's tight
        one produces a stop that belongs to neither thesis.
        """
        permitted = self.permitted_members(context.regime_label)
        verdicts = [
            ModelVerdict.from_signal(
                signal,
                model_name=member.name,
                regime_compatible=permitted is None or member.name in permitted,
            )
            for member, signal in zip(self._members, signals)
        ]

        decision = self._trigger_engine.evaluate(verdicts)

        by_name = {member.name: signal for member, signal in zip(self._members, signals)}
        contributing = [by_name[name] for name in decision.contributing_models if name in by_name]
        anchor = (
            max(contributing, key=lambda s: s.score)
            if contributing
            else max(signals, key=lambda s: s.score)
        )

        rationale = "; ".join(
            [decision.reason]
            + [f"{m.name}={s.signal}/{s.score:.1f}" for m, s in zip(self._members, signals)]
        )

        if not decision.allowed:
            return StrategySignal(
                symbol=symbol, signal="AVOID", score=0.0, trigger="Trigger:BLOCK",
                entry_price=anchor.entry_price, stop_price=0.0, target_price=0.0,
                reward_risk=0.0, probability_profit=0.0,
                component_scores={m.name: s.score for m, s in zip(self._members, signals)},
                rationale=rationale,
                extra={
                    "position_scale": 0.0,
                    "trigger_decision": decision.action,
                    "trigger_vetoes": decision.vetoes,
                    "trigger_muted_models": decision.muted_models,
                    "member_signals": {m.name: s.signal for m, s in zip(self._members, signals)},
                },
            )

        return StrategySignal(
            symbol=symbol,
            signal="BUY",
            score=round(decision.effective_confidence * 100, 2),
            trigger=f"Trigger:{decision.fired_rule}",
            entry_price=anchor.entry_price,
            stop_price=anchor.stop_price,
            target_price=anchor.target_price,
            reward_risk=anchor.reward_risk,
            probability_profit=anchor.probability_profit,
            component_scores={m.name: s.score for m, s in zip(self._members, signals)},
            rationale=rationale,
            extra={
                "position_scale": decision.size_multiplier,
                "trigger_decision": decision.action,
                "trigger_rule": decision.fired_rule,
                "trigger_effective_confidence": decision.effective_confidence,
                "trigger_expected_net_ev_pct": decision.expected_net_ev_pct,
                "trigger_contributing_models": decision.contributing_models,
                "trigger_opposing_models": decision.opposing_models,
                "trigger_muted_models": decision.muted_models,
                "member_signals": {m.name: s.signal for m, s in zip(self._members, signals)},
            },
        )

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
