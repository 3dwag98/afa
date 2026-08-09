"""Cross-sectional momentum and low-volatility strategies.

Both rank all eligible tickers against each other in a single score_batch()
call and go long the extreme decile:

- MomentumStrategy: top decile by 9-month (skip 1-month) formation return
  (Jegadeesh-Titman convention).
- LowVolatilityStrategy: bottom decile by trailing 60-day realized volatility
  (the low-volatility anomaly).

See docs/QUANT_RESEARCH.md sections 1 and 2 for the academic basis (with an
emphasis on India-specific studies) and exact formulation.

Unlike strategies/rule_based.py, a ticker's signal here depends on where it
ranks *within the batch*, not on its own history alone. score() (single
ticker) is a thin score_batch()-of-one wrapper for interface compatibility,
but it degenerates to "always top of a universe of one" — real ranking
requires calling score_batch() with the full eligible universe, which is
what the backtest engine and live orchestrator already do for every strategy.

This platform never shorts, so the opposite decile that academic long-short
studies short (low momentum / high volatility) is simply avoided.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import BaseStrategy
from .types import StrategyContext, StrategySignal
from portfolio_agent.config.schema import StrategyConfig

try:
    from src.risk import calculate_stop_target
except ImportError:
    from risk import calculate_stop_target


def _clean(value: Any) -> Optional[float]:
    """Convert a possibly-NaN/None pandas scalar to a plain float or None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _rank_and_select_decile(
    metric_by_symbol: Dict[str, float],
    latest_by_symbol: Dict[str, pd.Series],
    context: StrategyContext,
    top_fraction: float,
    higher_is_better: bool,
    trigger: str,
    component_name: str,
    min_universe: int,
) -> Dict[str, StrategySignal]:
    """Rank symbols by `metric_by_symbol` and go long the extreme decile.

    Shared ranking machinery for MomentumStrategy (higher_is_better=True:
    highest formation return wins) and LowVolatilityStrategy
    (higher_is_better=False: lowest realized volatility wins).

    Args:
        metric_by_symbol: The ranking metric per eligible symbol.
        latest_by_symbol: Each symbol's latest feature row (for close/ATR).
        context: Shared strategy context (risk params, MC result if any).
        top_fraction: Fraction of the universe to select (e.g. 0.1 = top decile).
        higher_is_better: True to select the highest metric values (momentum),
            False to select the lowest (low-volatility).
        trigger: Trigger name recorded on selected signals.
        component_name: Component-score key name.
        min_universe: Minimum eligible tickers required for ranking to be
            considered reliable; below this, every ticker is AVOID.

    Returns:
        Dictionary of symbol -> StrategySignal.
    """
    signals: Dict[str, StrategySignal] = {}

    if len(metric_by_symbol) < min_universe:
        for symbol, latest in latest_by_symbol.items():
            close = _clean(latest.get("close")) or 0.0
            signals[symbol] = StrategySignal(
                symbol=symbol, signal="AVOID", score=0.0, trigger="None",
                entry_price=close, stop_price=0.0, target_price=0.0,
                reward_risk=0.0, probability_profit=0.0,
                component_scores={}, rationale=(
                    f"Universe too small for reliable cross-sectional ranking "
                    f"({len(metric_by_symbol)} < {min_universe} eligible tickers)"
                ),
            )
        return signals

    ranked = sorted(metric_by_symbol.items(), key=lambda kv: kv[1], reverse=higher_is_better)
    n = len(ranked)
    cutoff = max(1, math.ceil(n * top_fraction))
    selected = {symbol for symbol, _ in ranked[:cutoff]}
    rank_position = {symbol: i + 1 for i, (symbol, _) in enumerate(ranked)}

    # Percentile in [0, 1], 1.0 = most favorably ranked (used as the 0-100 score).
    percentile = {
        symbol: 1.0 - ((pos - 1) / (n - 1) if n > 1 else 0.0)
        for symbol, pos in rank_position.items()
    }

    for symbol, metric in metric_by_symbol.items():
        latest = latest_by_symbol[symbol]
        close = _clean(latest.get("close")) or 0.0
        atr = _clean(latest.get("atr_14"))
        stop_price, target_price = calculate_stop_target(
            close, atr, context.risk.atr_stop_multiplier, context.risk.atr_target_multiplier
        )
        stop_valid = stop_price < close
        if stop_valid:
            risk_amount = close - stop_price
            reward_risk = (target_price - close) / risk_amount if risk_amount != 0 else 0.0
        else:
            reward_risk = 0.0

        score = round(percentile[symbol] * 100, 2)
        in_decile = symbol in selected
        passed_price = close >= context.risk.min_price_inr

        if not stop_valid:
            signal = "AVOID"
        elif in_decile and passed_price:
            signal = "BUY"
        elif in_decile:
            signal = "WATCH"
        else:
            signal = "AVOID"

        rationale = "; ".join([
            f"{component_name}={metric:.4f}",
            f"rank={rank_position[symbol]}/{n}",
            f"{'in' if in_decile else 'not in'} top {top_fraction:.0%} decile",
            f"price({close:.2f})>={context.risk.min_price_inr}:{'PASS' if passed_price else 'FAIL'}",
            "stop<entry:VALID" if stop_valid else "stop>=entry:INVALID",
        ])

        signals[symbol] = StrategySignal(
            symbol=symbol,
            signal=signal,
            score=score,
            trigger=trigger if in_decile else "None",
            entry_price=close,
            stop_price=stop_price,
            target_price=target_price,
            reward_risk=round(reward_risk, 4),
            probability_profit=context.mc_result.probability_profit if context.mc_result else 0.0,
            component_scores={component_name: metric},
            rationale=rationale,
        )

    return signals


class MomentumStrategy(BaseStrategy):
    """Cross-sectional momentum: long the top decile by 9-month (skip
    1-month) formation return (docs/QUANT_RESEARCH.md section 1)."""

    def __init__(self, config: StrategyConfig):
        self._config = config
        params = config.params or {}
        self._name = params.get("name", "momentum")
        self._top_fraction = float(params.get("top_percentile", 0.1))
        self._min_universe = int(params.get("min_universe", 5))

    @property
    def name(self) -> str:
        return self._name

    @property
    def requires_full_batch(self) -> bool:
        return True

    def required_features(self) -> List[str]:
        return ["close", "mom_9m_skip1m", "atr_14"]

    def entry_rules(self) -> Dict[str, Any]:
        return {
            "rule": "Long top decile of the eligible universe by 9-month formation "
                    "return, skipping the most recent month",
            "top_percentile": self._top_fraction,
            "long_only": True,
        }

    def score(self, symbol: str, features: pd.DataFrame, context: StrategyContext) -> StrategySignal:
        return self.score_batch({symbol: features}, context)[symbol]

    def score_batch(
        self, features_by_symbol: Dict[str, pd.DataFrame], context: StrategyContext
    ) -> Dict[str, StrategySignal]:
        metric_by_symbol: Dict[str, float] = {}
        latest_by_symbol: Dict[str, pd.Series] = {}

        for symbol, features in features_by_symbol.items():
            if features.empty:
                continue
            latest = features.iloc[-1]
            mom = _clean(latest.get("mom_9m_skip1m"))
            latest_by_symbol[symbol] = latest
            if mom is not None:
                metric_by_symbol[symbol] = mom

        return _rank_and_select_decile(
            metric_by_symbol=metric_by_symbol,
            latest_by_symbol=latest_by_symbol,
            context=context,
            top_fraction=self._top_fraction,
            higher_is_better=True,
            trigger="Momentum",
            component_name="Momentum",
            min_universe=self._min_universe,
        )


class LowVolatilityStrategy(BaseStrategy):
    """Low-volatility anomaly: long the bottom decile by trailing 60-day
    realized volatility (docs/QUANT_RESEARCH.md section 2)."""

    def __init__(self, config: StrategyConfig):
        self._config = config
        params = config.params or {}
        self._name = params.get("name", "low_volatility")
        self._top_fraction = float(params.get("top_percentile", 0.1))
        self._min_universe = int(params.get("min_universe", 5))

    @property
    def name(self) -> str:
        return self._name

    @property
    def requires_full_batch(self) -> bool:
        return True

    def required_features(self) -> List[str]:
        return ["close", "realized_vol_60", "atr_14"]

    def entry_rules(self) -> Dict[str, Any]:
        return {
            "rule": "Long bottom decile of the eligible universe by trailing "
                    "60-day annualized realized volatility",
            "bottom_percentile": self._top_fraction,
            "long_only": True,
        }

    def score(self, symbol: str, features: pd.DataFrame, context: StrategyContext) -> StrategySignal:
        return self.score_batch({symbol: features}, context)[symbol]

    def score_batch(
        self, features_by_symbol: Dict[str, pd.DataFrame], context: StrategyContext
    ) -> Dict[str, StrategySignal]:
        metric_by_symbol: Dict[str, float] = {}
        latest_by_symbol: Dict[str, pd.Series] = {}

        for symbol, features in features_by_symbol.items():
            if features.empty:
                continue
            latest = features.iloc[-1]
            vol = _clean(latest.get("realized_vol_60"))
            latest_by_symbol[symbol] = latest
            if vol is not None:
                metric_by_symbol[symbol] = vol

        return _rank_and_select_decile(
            metric_by_symbol=metric_by_symbol,
            latest_by_symbol=latest_by_symbol,
            context=context,
            top_fraction=self._top_fraction,
            higher_is_better=False,
            trigger="LowVolatility",
            component_name="RealizedVol",
            min_universe=self._min_universe,
        )
