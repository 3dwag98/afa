"""Rule-based strategy implementation for portfolio agent.

This is the single, canonical "Trend + Breakout + Volume + Monte Carlo
Probability" strategy. It replaces three previously-drifted copies of this
logic (src/scoring.py::score_candidate, the old unwired version of this file,
and the crude stub inside BacktestEngine._generate_signals) so the live
orchestrator and the backtest engine make identical decisions from identical
inputs.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from .base import BaseStrategy
from .types import StrategyContext, StrategySignal
from .weighting import combine_rank_composite, combine_weighted
from portfolio_agent.config.schema import StrategyConfig

try:
    from src.risk import calculate_stop_target, net_reward_risk
except ImportError:
    from risk import calculate_stop_target, net_reward_risk


def _clean(value: Any) -> Optional[float]:
    """Convert a possibly-NaN/None pandas scalar to a plain float or None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


class RuleBasedStrategy(BaseStrategy):
    """Rule-based trading strategy configured via YAML.

    Implements the trend/breakout/volume/Monte-Carlo scoring logic that was
    previously hardcoded in src/scoring.py, now driven by configurable
    component weights and ATR multipliers from a YAML rules file.
    """

    def __init__(self, config: StrategyConfig):
        """Initialize the rule-based strategy.

        Args:
            config: StrategyConfig containing the path to the YAML rules file.
        """
        self._config = config
        self._yaml_path = config.params.get("yaml_path", config.config_path)
        self._rules = self._load_rules()

        # "weighted_sum" (the historical default) or "rank_composite". See
        # weighting.combine_rank_composite for why the sum of an ordinal, a
        # binary, a skewed continuous and a near-constant is not a score.
        # Deliberately NOT "scoring": the rules YAML already uses that key for
        # the component weight block.
        mode = config.params.get(
            "scoring_mode", self._rules.get("scoring_mode", "weighted_sum")
        )
        if mode not in ("weighted_sum", "rank_composite"):
            raise ValueError(
                f"Unknown scoring mode {mode!r}; expected 'weighted_sum' or 'rank_composite'"
            )
        self.scoring_mode = mode

    def _load_rules(self) -> Dict[str, Any]:
        """Load rules from the YAML configuration file."""
        yaml_path = Path(self._yaml_path)
        if not yaml_path.is_absolute():
            # First try relative to the current working directory
            if not yaml_path.exists():
                # Then relative to the portfolio_agent package root
                package_root = Path(__file__).parent.parent
                candidate = package_root / self._yaml_path
                if candidate.exists():
                    yaml_path = candidate
                else:
                    # Then relative to the workspace root
                    workspace_root = package_root.parent
                    yaml_path = workspace_root / self._yaml_path

        if not yaml_path.exists():
            raise FileNotFoundError(f"Strategy YAML file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)

    @property
    def name(self) -> str:
        return self._rules.get("name", "rule_based")

    def required_features(self) -> List[str]:
        return ["close", "sma_50", "sma_200", "donchian_upper_20", "volume_ratio_20", "atr_14"]

    def entry_rules(self) -> Dict[str, Any]:
        return self._rules.get("entry", {})

    def exit_rules(self) -> Dict[str, Any]:
        return self._rules.get("exit", {})

    def component_scores(
        self, symbol: str, features: pd.DataFrame, context: StrategyContext
    ) -> Optional[Dict[str, float]]:
        """The four 0-1 sub-scores, without combining them.

        Split out from score() so score_batch() can rank each component across
        the whole cross-section before combining (see
        weighting.combine_rank_composite). Returns None when there is no
        feature data to score.
        """
        if features.empty:
            return None

        latest = features.iloc[-1]
        close = _clean(latest.get("close")) or 0.0
        sma50 = _clean(latest.get("sma_50"))
        sma200 = _clean(latest.get("sma_200"))
        donchian_upper = _clean(latest.get("donchian_upper_20"))
        volume_ratio = _clean(latest.get("volume_ratio_20"))

        if sma200 is None:
            trend_score = 0.0
        elif sma50 is not None and close > sma50 and close > sma200 and sma50 > sma200:
            trend_score = 1.0
        elif close > sma200:
            trend_score = 0.5
        else:
            trend_score = 0.0

        if donchian_upper is None:
            breakout_score = 0.0
        elif close > donchian_upper:
            breakout_score = 1.0
        else:
            breakout_score = 0.0

        volume_score = 0.0 if volume_ratio is None else min(volume_ratio / 2.0, 1.0)

        mc_result = context.mc_for(symbol)
        prob_profit = mc_result.probability_profit if mc_result is not None else 0.0

        return {
            "Trend": trend_score,
            "Breakout": breakout_score,
            "Volume": volume_score,
            "MC_Prob": max(0.0, min(1.0, prob_profit)),
        }

    def score_batch(
        self, features_by_symbol: Dict[str, pd.DataFrame], context: StrategyContext
    ) -> Dict[str, StrategySignal]:
        """Score the whole cross-section, combining components by rank.

        Only used when `scoring_mode` is "rank_composite" (see
        requires_full_batch); otherwise this falls through to the base class's
        per-ticker loop, which is the historical behaviour.
        """
        if self.scoring_mode != "rank_composite":
            return super().score_batch(features_by_symbol, context)

        components = {}
        for symbol, features in features_by_symbol.items():
            scores = self.component_scores(symbol, features, context)
            if scores is not None:
                components[symbol] = scores

        composites = combine_rank_composite(components, context.weights)
        return {
            symbol: self.score(symbol, features, context, composite=composites.get(symbol))
            for symbol, features in features_by_symbol.items()
        }

    @property
    def requires_full_batch(self) -> bool:
        """True under rank-composite scoring: a percentile rank against a
        universe of one is not a rank."""
        return self.scoring_mode == "rank_composite"

    def score(
        self,
        symbol: str,
        features: pd.DataFrame,
        context: StrategyContext,
        composite: Optional[Tuple[float, str]] = None,
    ) -> StrategySignal:
        """Score one ticker.

        Args:
            symbol: Ticker being scored.
            features: Lag-safe feature frame; the last row is the decision row.
            context: Risk params, weights and this round's Monte Carlo results.
            composite: Pre-combined (score, trigger) from score_batch's
                cross-sectional ranking. None combines this ticker's own
                components by weighted sum, which is all a single-ticker
                caller can do.
        """
        if features.empty:
            return StrategySignal(
                symbol=symbol, signal="AVOID", score=0.0, trigger="None",
                entry_price=0.0, stop_price=0.0, target_price=0.0,
                reward_risk=0.0, probability_profit=0.0,
                component_scores={}, rationale="No feature data available",
            )

        latest = features.iloc[-1]
        close = _clean(latest.get("close")) or 0.0
        sma50 = _clean(latest.get("sma_50"))
        sma200 = _clean(latest.get("sma_200"))
        donchian_upper = _clean(latest.get("donchian_upper_20"))
        volume_ratio = _clean(latest.get("volume_ratio_20"))
        atr = _clean(latest.get("atr_14"))

        # 1. Trend score
        if sma200 is None:
            trend_score = 0.0
        elif sma50 is not None and close > sma50 and close > sma200 and sma50 > sma200:
            trend_score = 1.0
        elif close > sma200:
            trend_score = 0.5
        else:
            trend_score = 0.0

        # 2. Breakout score
        if donchian_upper is None:
            breakout_score = 0.0
        elif close > donchian_upper:
            breakout_score = 1.0
        else:
            breakout_score = 0.0

        # 3. Volume score
        volume_score = 0.0 if volume_ratio is None else min(volume_ratio / 2.0, 1.0)

        # 4. Monte Carlo probability-of-profit score
        mc_result = context.mc_for(symbol)
        prob_profit = mc_result.probability_profit if mc_result is not None else 0.0
        mc_score = max(0.0, min(1.0, prob_profit))

        # The gate reads the lower confidence bound, not the point estimate.
        # probability_profit is Phi(mu_hat*sqrt(H)/sigma) to a first
        # approximation, so it inherits the drift's standard error one-for-one
        # and a ticker with no edge at all clears a 0.55 gate 8.5% of the time
        # (src/monte_carlo.py::DriftPrior). Falls back to the point estimate
        # when the simulation did not run, so a stub MonteCarloResult blocks
        # the trade for the reason it always did rather than for a new one.
        gate_prob = prob_profit
        if context.risk.gate_on_probability_lower_bound and mc_result is not None:
            gate_prob = mc_result.probability_profit_gate

        component_scores = {
            "Trend": trend_score,
            "Breakout": breakout_score,
            "Volume": volume_score,
            "MC_Prob": mc_score,
        }
        if composite is not None:
            final_score, trigger = composite
        else:
            final_score, trigger = combine_weighted(component_scores, context.weights)

        # 5. Stop / target from ATR (YAML multipliers, falling back to context defaults)
        exit_rules = self.exit_rules()
        stop_mult = exit_rules.get("stop_loss", {}).get("multiplier", context.risk.atr_stop_multiplier)
        target_mult = exit_rules.get("take_profit", {}).get("multiplier", context.risk.atr_target_multiplier)
        stop_price, target_price = calculate_stop_target(close, atr, stop_mult, target_mult)

        # Reward:risk is measured net of estimated round-trip friction
        # (brokerage, STT, exchange and SEBI charges, GST, stamp duty and
        # slippage) rather than gross, so the min_reward_risk gate below is a
        # statement about money actually kept — see src/risk.py::net_reward_risk.
        stop_valid = stop_price < close
        if stop_valid:
            reward_risk = net_reward_risk(
                entry_price=close,
                stop_price=stop_price,
                target_price=target_price,
                buy_cost_pct=context.risk.buy_cost_pct,
                sell_cost_pct=context.risk.sell_cost_pct,
            )
            gross_reward_risk = (
                (target_price - close) / (close - stop_price) if close != stop_price else 0.0
            )
        else:
            reward_risk = 0.0
            gross_reward_risk = 0.0

        passed_score = final_score >= 60
        passed_prob = gate_prob >= context.risk.target_prob_profit
        passed_rr = reward_risk >= context.risk.min_reward_risk
        passed_price = close >= context.risk.min_price_inr

        rationale = "; ".join([
            f"Score={final_score:.1f}",
            f"trend={trend_score:.1f}",
            f"breakout={breakout_score:.1f}",
            f"volume={volume_score:.2f}",
            f"mc_prob={mc_score:.2f}",
            f"score>=60:{'PASS' if passed_score else 'FAIL'}",
            f"prob({gate_prob:.2f})>={context.risk.target_prob_profit}:{'PASS' if passed_prob else 'FAIL'}",
            f"rr({reward_risk:.2f})>={context.risk.min_reward_risk}:{'PASS' if passed_rr else 'FAIL'}",
            f"price({close:.2f})>={context.risk.min_price_inr}:{'PASS' if passed_price else 'FAIL'}",
            "stop<entry:VALID" if stop_valid else "stop>=entry:INVALID",
        ])

        if not stop_valid:
            signal = "AVOID"
        elif passed_score and passed_prob and passed_rr and passed_price:
            signal = "BUY"
        elif final_score >= 45:
            signal = "WATCH"
        else:
            signal = "AVOID"

        return StrategySignal(
            symbol=symbol,
            signal=signal,
            score=round(final_score, 2),
            trigger=trigger,
            entry_price=close,
            stop_price=stop_price,
            target_price=target_price,
            reward_risk=round(reward_risk, 4),
            probability_profit=round(prob_profit, 6),
            component_scores=component_scores,
            rationale=rationale,
            extra={
                "mc_prob_profit_lower": round(
                    mc_result.probability_profit_gate, 6
                ) if mc_result else 0.0,
                "mc_drift_shrunk": bool(mc_result.drift_shrunk) if mc_result else False,
                "mc_var_95_pct": round(mc_result.var_95, 6) if mc_result else 0.0,
                "mc_cvar_95_pct": round(mc_result.cvar_95, 6) if mc_result else 0.0,
                # Reported alongside the (net) reward_risk so the gap between
                # them is visible: on ATR-tight stops, round-trip friction is a
                # large fraction of the risk being taken, and a strategy whose
                # gross ratio clears the gate but whose net ratio does not is
                # the exact failure this platform is meant to surface.
                "gross_reward_risk": round(gross_reward_risk, 4),
                "round_trip_cost_pct": round(
                    context.risk.buy_cost_pct + context.risk.sell_cost_pct, 6
                ),
            },
        )
