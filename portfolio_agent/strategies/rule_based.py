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
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from .base import BaseStrategy
from .types import StrategyContext, StrategySignal
from .weighting import combine_weighted
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

    def score(self, symbol: str, features: pd.DataFrame, context: StrategyContext) -> StrategySignal:
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
        prob_profit = context.mc_result.probability_profit if context.mc_result is not None else 0.0
        mc_score = max(0.0, min(1.0, prob_profit))

        component_scores = {
            "Trend": trend_score,
            "Breakout": breakout_score,
            "Volume": volume_score,
            "MC_Prob": mc_score,
        }
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
        passed_prob = prob_profit >= context.risk.target_prob_profit
        passed_rr = reward_risk >= context.risk.min_reward_risk
        passed_price = close >= context.risk.min_price_inr

        rationale = "; ".join([
            f"Score={final_score:.1f}",
            f"trend={trend_score:.1f}",
            f"breakout={breakout_score:.1f}",
            f"volume={volume_score:.2f}",
            f"mc_prob={mc_score:.2f}",
            f"score>=60:{'PASS' if passed_score else 'FAIL'}",
            f"prob({prob_profit:.2f})>={context.risk.target_prob_profit}:{'PASS' if passed_prob else 'FAIL'}",
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
                "mc_var_95_pct": round(context.mc_result.var_95, 6) if context.mc_result else 0.0,
                "mc_cvar_95_pct": round(context.mc_result.cvar_95, 6) if context.mc_result else 0.0,
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
