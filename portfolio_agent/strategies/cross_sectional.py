"""Cross-sectional momentum and low-volatility strategies.

Both rank all eligible tickers against each other in a single score_batch()
call and go long the extreme decile:

- MomentumStrategy: top decile by 9-month (skip 1-month) formation return
  (Jegadeesh-Titman convention), with crash protection layered on top.
- LowVolatilityStrategy: bottom decile by trailing 60-day realized volatility
  (the low-volatility anomaly).

See docs/QUANT_RESEARCH.md sections 1, 2 and 12 for the academic basis (with
an emphasis on India-specific studies) and exact formulation.

Unlike strategies/rule_based.py, a ticker's signal here depends on where it
ranks *within the batch*, not on its own history alone. score() (single
ticker) is a thin score_batch()-of-one wrapper for interface compatibility,
but it degenerates to "always top of a universe of one" — real ranking
requires calling score_batch() with the full eligible universe, which is
what the backtest engine and live orchestrator already do for every strategy.

Three risk controls sit between the raw ranking and the emitted signals:

1. **A meaningful universe.** Decile ranking over a handful of names is not
   ranking; `min_universe` defaults to 30 so "top 10%" describes a real
   cross-section rather than a coin flip between three stocks.
2. **Cost-aware reward:risk.** Reported reward:risk is net of estimated
   round-trip friction, and a trade whose target cannot even pay for its own
   costs is never a BUY.
3. **Crash protection** (momentum by default): volatility targeting plus a
   market-regime filter, per src/regime.py.

This platform never shorts, so the opposite decile that academic long-short
studies short (low momentum / high volatility) is simply avoided.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import BaseStrategy
from .types import StrategyContext, StrategySignal
from portfolio_agent.config.schema import StrategyConfig

try:
    from src.risk import calculate_stop_target, net_reward_risk
    from src.liquidity import (
        DEFAULT_MAX_CIRCUIT_LOCK_FRACTION,
        DEFAULT_MAX_ZERO_RETURN_FRACTION,
        DEFAULT_MIN_TRADED_VALUE_INR,
    )
    from src.regime import (
        DEFAULT_CRASH_VOL_MULTIPLE,
        DEFAULT_MAX_SCALE,
        DEFAULT_MIN_SCALE,
        DEFAULT_TARGET_VOLATILITY,
        DEFAULT_TREND_WINDOW,
        DEFAULT_VOL_WINDOW,
        MarketRegime,
        assess_market_regime,
        build_market_proxy,
        neutral_regime,
        volatility_target_scalar,
    )
except ImportError:
    from risk import calculate_stop_target, net_reward_risk
    from liquidity import (
        DEFAULT_MAX_CIRCUIT_LOCK_FRACTION,
        DEFAULT_MAX_ZERO_RETURN_FRACTION,
        DEFAULT_MIN_TRADED_VALUE_INR,
    )
    from regime import (
        DEFAULT_CRASH_VOL_MULTIPLE,
        DEFAULT_MAX_SCALE,
        DEFAULT_MIN_SCALE,
        DEFAULT_TARGET_VOLATILITY,
        DEFAULT_TREND_WINDOW,
        DEFAULT_VOL_WINDOW,
        MarketRegime,
        assess_market_regime,
        build_market_proxy,
        neutral_regime,
        volatility_target_scalar,
    )

# Decile ranking is a statistical statement about a cross-section. Below ~30
# names a "top 10%" selection is 1-3 stocks chosen from a sample far too small
# for the rank to carry information, so the strategy stands aside instead.
DEFAULT_MIN_UNIVERSE = 30


@dataclass
class TradabilityFilter:
    """Liquidity / circuit-lock screen applied before ranking.

    Every ranking formula here assumes a printed close is a price you could
    have transacted at. On NSE/BSE mid- and small-caps that assumption fails
    in two specific, detectable ways (docs/QUANT_RESEARCH.md section 15):

    - A stock locked at its upper circuit prints the return momentum reads as
      strength while offering nothing to buy; locked at the lower circuit, it
      cannot be exited at the modelled stop.
    - A stock that barely trades prints unchanged closes, which suppresses its
      realized variance and walks it into the low-volatility buy decile
      without being remotely low-risk.

    Screened names are reported as AVOID with the failing reason, and are
    removed from the ranking entirely rather than merely being un-buyable:
    leaving an untradeable stock in the cross-section would still shift every
    other name's percentile.

    Tunable through strategy params:

        params:
          liquidity_filter: true
          min_traded_value_inr: 5000000
          max_zero_return_fraction: 0.30
          max_circuit_lock_fraction: 0.10
    """

    enabled: bool = True
    min_traded_value_inr: float = DEFAULT_MIN_TRADED_VALUE_INR
    max_zero_return_fraction: float = DEFAULT_MAX_ZERO_RETURN_FRACTION
    max_circuit_lock_fraction: float = DEFAULT_MAX_CIRCUIT_LOCK_FRACTION

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "TradabilityFilter":
        return cls(
            enabled=bool(params.get("liquidity_filter", True)),
            min_traded_value_inr=float(
                params.get("min_traded_value_inr", DEFAULT_MIN_TRADED_VALUE_INR)
            ),
            max_zero_return_fraction=float(
                params.get("max_zero_return_fraction", DEFAULT_MAX_ZERO_RETURN_FRACTION)
            ),
            max_circuit_lock_fraction=float(
                params.get("max_circuit_lock_fraction", DEFAULT_MAX_CIRCUIT_LOCK_FRACTION)
            ),
        )

    def required_features(self) -> List[str]:
        return [
            "traded_value_60",
            "zero_return_fraction_60",
            "circuit_lock_fraction_60",
            "circuit_locked_today",
        ] if self.enabled else []

    def reject_reason(self, latest: pd.Series) -> Optional[str]:
        """Why this ticker is untradeable today, or None if it is fine.

        A screening statistic that could not be computed (too little history)
        is treated as passing: the filter's job is to exclude names with
        positive evidence of being untradeable, not to reject everything with
        a short cache.
        """
        if not self.enabled:
            return None

        traded_value = _clean(latest.get("traded_value_60"))
        if traded_value is not None and traded_value < self.min_traded_value_inr:
            return (
                f"illiquid: median turnover {traded_value:,.0f} < "
                f"{self.min_traded_value_inr:,.0f} INR/day"
            )

        zero_fraction = _clean(latest.get("zero_return_fraction_60"))
        if zero_fraction is not None and zero_fraction > self.max_zero_return_fraction:
            return (
                f"zombie: {zero_fraction:.0%} of sessions closed unchanged "
                f"(> {self.max_zero_return_fraction:.0%}); low variance reflects "
                f"illiquidity, not stability"
            )

        lock_fraction = _clean(latest.get("circuit_lock_fraction_60"))
        if lock_fraction is not None and lock_fraction > self.max_circuit_lock_fraction:
            return (
                f"circuit-driven: {lock_fraction:.0%} of sessions locked at a limit "
                f"(> {self.max_circuit_lock_fraction:.0%})"
            )

        if _clean(latest.get("circuit_locked_today")):
            return "locked at a circuit limit on the decision date; no fill available"

        return None


@dataclass
class CrashProtection:
    """Volatility-targeting and market-regime settings for a ranked strategy.

    Read from strategy params so a UMA/YAML config can tune or disable each
    control independently:

        params:
          volatility_target: 0.20     # annualized risk budget per position
          regime_filter: true         # stand down in the panic state
          trend_window: 200
          vol_window: 60
          crash_vol_multiple: 1.5
          bear_exposure: 0.0          # >0 to dampen instead of standing down
    """

    volatility_target: float = DEFAULT_TARGET_VOLATILITY
    scale_by_volatility: bool = True
    regime_filter: bool = True
    trend_window: int = DEFAULT_TREND_WINDOW
    vol_window: int = DEFAULT_VOL_WINDOW
    crash_vol_multiple: float = DEFAULT_CRASH_VOL_MULTIPLE
    bear_exposure: float = 0.0
    min_scale: float = DEFAULT_MIN_SCALE
    max_scale: float = DEFAULT_MAX_SCALE

    @classmethod
    def from_params(cls, params: Dict[str, Any], regime_filter_default: bool) -> "CrashProtection":
        """Build from a strategy's `params` block, falling back to defaults."""
        return cls(
            volatility_target=float(params.get("volatility_target", DEFAULT_TARGET_VOLATILITY)),
            scale_by_volatility=bool(params.get("scale_by_volatility", True)),
            regime_filter=bool(params.get("regime_filter", regime_filter_default)),
            trend_window=int(params.get("trend_window", DEFAULT_TREND_WINDOW)),
            vol_window=int(params.get("vol_window", DEFAULT_VOL_WINDOW)),
            crash_vol_multiple=float(params.get("crash_vol_multiple", DEFAULT_CRASH_VOL_MULTIPLE)),
            bear_exposure=float(params.get("bear_exposure", 0.0)),
            min_scale=float(params.get("min_position_scale", DEFAULT_MIN_SCALE)),
            max_scale=float(params.get("max_position_scale", DEFAULT_MAX_SCALE)),
        )


def _clean(value: Any) -> Optional[float]:
    """Convert a possibly-NaN/None pandas scalar to a plain float or None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _assess_regime(
    features_by_symbol: Dict[str, pd.DataFrame],
    protection: CrashProtection,
    benchmark_close: Optional[pd.Series] = None,
) -> MarketRegime:
    """Derive the market regime, preferring a real index over a composite.

    When the caller supplies a benchmark series (the Nifty 50, cached from the
    dataset's indices/ directory), the trend and volatility filters key off it
    — "the market is below its 200-day average" is a statement about the index
    the research actually studied. Without one, the filter falls back to an
    equal-weighted composite of the eligible tickers, which needs no data the
    platform doesn't already have but is only a proxy: it reflects whatever
    happens to be in today's universe, and idiosyncratic noise diversifies out
    of it in a way real index volatility does not.

    Returns a neutral (unscaled) regime when the filter is switched off or
    there isn't enough history to judge.
    """
    if not protection.regime_filter:
        return neutral_regime("regime filter disabled")

    market_close = benchmark_close
    if market_close is None or len(market_close) < protection.trend_window + 1:
        close_by_symbol = {
            symbol: features["close"]
            for symbol, features in features_by_symbol.items()
            if not features.empty and "close" in features.columns
        }
        market_close = build_market_proxy(close_by_symbol)

    return assess_market_regime(
        market_close,
        trend_window=protection.trend_window,
        vol_window=protection.vol_window,
        target_volatility=protection.volatility_target,
        crash_vol_multiple=protection.crash_vol_multiple,
        bear_exposure=protection.bear_exposure,
        min_scale=protection.min_scale,
        max_scale=protection.max_scale,
    )


def _rank_and_select_decile(
    metric_by_symbol: Dict[str, float],
    latest_by_symbol: Dict[str, pd.Series],
    context: StrategyContext,
    top_fraction: float,
    higher_is_better: bool,
    trigger: str,
    component_name: str,
    min_universe: int,
    protection: CrashProtection,
    regime: MarketRegime,
    rejected: Dict[str, str],
) -> Dict[str, StrategySignal]:
    """Rank symbols by `metric_by_symbol` and go long the extreme decile.

    Shared ranking machinery for MomentumStrategy (higher_is_better=True:
    highest formation return wins) and LowVolatilityStrategy
    (higher_is_better=False: lowest realized volatility wins).

    Args:
        metric_by_symbol: The ranking metric per eligible symbol.
        latest_by_symbol: Each symbol's latest feature row (for close/ATR/vol).
        context: Shared strategy context (risk params, MC result if any).
        top_fraction: Fraction of the universe to select (e.g. 0.1 = top decile).
        higher_is_better: True to select the highest metric values (momentum),
            False to select the lowest (low-volatility).
        trigger: Trigger name recorded on selected signals.
        component_name: Component-score key name.
        min_universe: Minimum eligible tickers required for ranking to be
            considered reliable; below this, every ticker is AVOID.
        protection: Volatility-targeting / regime settings.
        regime: The assessed market regime (see _assess_regime).
        rejected: symbol -> reason for symbols the tradability screen removed
            before ranking; each gets an AVOID signal carrying that reason.

    Returns:
        Dictionary of symbol -> StrategySignal. Selected signals carry a
        `position_scale` in `extra`, which the backtest engine and live
        orchestrator multiply into the sized quantity.
    """
    signals: Dict[str, StrategySignal] = {}

    for symbol, reason in rejected.items():
        latest = latest_by_symbol.get(symbol)
        close = _clean(latest.get("close")) if latest is not None else None
        signals[symbol] = StrategySignal(
            symbol=symbol, signal="AVOID", score=0.0, trigger="None",
            entry_price=close or 0.0, stop_price=0.0, target_price=0.0,
            reward_risk=0.0, probability_profit=0.0,
            component_scores={}, rationale=f"Not tradable — {reason}",
            extra={"position_scale": 0.0, "tradability_reject_reason": reason},
        )

    if len(metric_by_symbol) < min_universe:
        for symbol, latest in latest_by_symbol.items():
            if symbol in signals:  # already rejected by the tradability screen
                continue
            close = _clean(latest.get("close")) or 0.0
            signals[symbol] = StrategySignal(
                symbol=symbol, signal="AVOID", score=0.0, trigger="None",
                entry_price=close, stop_price=0.0, target_price=0.0,
                reward_risk=0.0, probability_profit=0.0,
                component_scores={}, rationale=(
                    f"Universe too small for reliable cross-sectional ranking "
                    f"({len(metric_by_symbol)} < {min_universe} eligible tickers)"
                ),
                extra={"position_scale": 0.0},
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
        # Reward:risk net of estimated round-trip friction (brokerage, STT,
        # exchange/SEBI charges, GST, stamp duty, slippage). A gross ratio
        # flatters every trade, and on ATR-tight stops the difference decides
        # whether the edge survives at all.
        reward_risk = net_reward_risk(
            entry_price=close,
            stop_price=stop_price,
            target_price=target_price,
            buy_cost_pct=context.risk.buy_cost_pct,
            sell_cost_pct=context.risk.sell_cost_pct,
        ) if stop_valid else 0.0

        score = round(percentile[symbol] * 100, 2)
        in_decile = symbol in selected
        passed_price = close >= context.risk.min_price_inr
        # net_reward_risk() returns 0.0 when the target cannot clear the
        # round-trip cost, so this doubles as the "trade pays for itself" gate.
        covers_costs = reward_risk > 0.0

        # Per-position volatility targeting: a stock running at twice the risk
        # budget gets half the money, so each holding contributes comparable
        # risk instead of the most volatile names dominating the portfolio.
        stock_vol = _clean(latest.get("realized_vol_60"))
        stock_scalar = (
            volatility_target_scalar(
                stock_vol, protection.volatility_target, protection.min_scale, protection.max_scale
            )
            if protection.scale_by_volatility
            else 1.0
        )
        position_scale = round(max(0.0, regime.exposure_scalar * stock_scalar), 4)
        regime_blocks = regime.blocks_new_entries

        if not stop_valid:
            signal = "AVOID"
        elif in_decile and passed_price and covers_costs and not regime_blocks:
            signal = "BUY"
        elif in_decile:
            # Ranked into the decile but gated by price, costs or the regime:
            # worth watching, not worth buying today.
            signal = "WATCH"
        else:
            signal = "AVOID"

        rationale_parts = [
            f"{component_name}={metric:.4f}",
            f"rank={rank_position[symbol]}/{n}",
            f"{'in' if in_decile else 'not in'} top {top_fraction:.0%} decile",
            f"price({close:.2f})>={context.risk.min_price_inr}:{'PASS' if passed_price else 'FAIL'}",
            f"net_rr({reward_risk:.2f})>0:{'PASS' if covers_costs else 'FAIL'}",
            "stop<entry:VALID" if stop_valid else "stop>=entry:INVALID",
        ]
        if in_decile:
            rationale_parts.append(f"regime={regime.label}")
            rationale_parts.append(f"position_scale={position_scale:.2f}")
            if regime_blocks:
                rationale_parts.append(regime.reason)
        rationale = "; ".join(rationale_parts)

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
            extra={
                "position_scale": position_scale if signal == "BUY" else 0.0,
                "regime": regime.label,
                "regime_exposure_scalar": round(regime.exposure_scalar, 4),
                "volatility_scalar": round(stock_scalar, 4),
                "market_volatility": (
                    round(regime.market_volatility, 6)
                    if regime.market_volatility is not None
                    else None
                ),
            },
        )

    return signals


class MomentumStrategy(BaseStrategy):
    """Cross-sectional momentum: long the top decile by 9-month (skip
    1-month) formation return (docs/QUANT_RESEARCH.md section 1), with
    volatility targeting and a market-regime crash filter (section 12).

    Momentum is the factor most prone to catastrophic drawdown, and its
    crashes are concentrated in a specific, detectable state — a bear market
    rebound with elevated volatility. Both controls are on by default here;
    set `regime_filter: false` / `scale_by_volatility: false` in params to run
    the unprotected academic version.
    """

    def __init__(self, config: StrategyConfig):
        self._config = config
        params = config.params or {}
        self._name = params.get("name", "momentum")
        self._top_fraction = float(params.get("top_percentile", 0.1))
        self._min_universe = int(params.get("min_universe", DEFAULT_MIN_UNIVERSE))
        self._protection = CrashProtection.from_params(params, regime_filter_default=True)
        self._tradability = TradabilityFilter.from_params(params)

    @property
    def name(self) -> str:
        return self._name

    @property
    def requires_full_batch(self) -> bool:
        return True

    def required_features(self) -> List[str]:
        # realized_vol_60 is not part of the ranking metric; it drives the
        # per-position volatility-targeting scalar.
        return [
            "close", "mom_9m_skip1m", "atr_14", "realized_vol_60"
        ] + self._tradability.required_features()

    def entry_rules(self) -> Dict[str, Any]:
        return {
            "rule": "Long top decile of the eligible universe by 9-month formation "
                    "return, skipping the most recent month",
            "top_percentile": self._top_fraction,
            "min_universe": self._min_universe,
            "long_only": True,
            "crash_protection": {
                "volatility_target": self._protection.volatility_target,
                "scale_by_volatility": self._protection.scale_by_volatility,
                "regime_filter": self._protection.regime_filter,
                "trend_window": self._protection.trend_window,
                "crash_vol_multiple": self._protection.crash_vol_multiple,
                "bear_exposure": self._protection.bear_exposure,
            },
            "tradability_filter": {
                "enabled": self._tradability.enabled,
                "min_traded_value_inr": self._tradability.min_traded_value_inr,
                "max_circuit_lock_fraction": self._tradability.max_circuit_lock_fraction,
            },
        }

    def score(self, symbol: str, features: pd.DataFrame, context: StrategyContext) -> StrategySignal:
        return self.score_batch({symbol: features}, context)[symbol]

    def score_batch(
        self, features_by_symbol: Dict[str, pd.DataFrame], context: StrategyContext
    ) -> Dict[str, StrategySignal]:
        metric_by_symbol: Dict[str, float] = {}
        latest_by_symbol: Dict[str, pd.Series] = {}

        rejected: Dict[str, str] = {}

        for symbol, features in features_by_symbol.items():
            if features.empty:
                continue
            latest = features.iloc[-1]
            latest_by_symbol[symbol] = latest

            reason = self._tradability.reject_reason(latest)
            if reason is not None:
                rejected[symbol] = reason
                continue

            mom = _clean(latest.get("mom_9m_skip1m"))
            if mom is not None:
                metric_by_symbol[symbol] = mom

        regime = _assess_regime(features_by_symbol, self._protection, context.benchmark_close)

        return _rank_and_select_decile(
            metric_by_symbol=metric_by_symbol,
            latest_by_symbol=latest_by_symbol,
            context=context,
            top_fraction=self._top_fraction,
            higher_is_better=True,
            trigger="Momentum",
            component_name="Momentum",
            min_universe=self._min_universe,
            protection=self._protection,
            regime=regime,
            rejected=rejected,
        )


class LowVolatilityStrategy(BaseStrategy):
    """Low-volatility anomaly: long the bottom decile by trailing 60-day
    realized volatility (docs/QUANT_RESEARCH.md section 2).

    Volatility targeting applies here too, but the market-regime crash filter
    is off by default: this is the defensive sleeve, and the anomaly's whole
    point is that low-volatility names hold up through the drawdowns that
    momentum crashes in. Set `regime_filter: true` in params to gate it as
    well.
    """

    def __init__(self, config: StrategyConfig):
        self._config = config
        params = config.params or {}
        self._name = params.get("name", "low_volatility")
        self._top_fraction = float(params.get("top_percentile", 0.1))
        self._min_universe = int(params.get("min_universe", DEFAULT_MIN_UNIVERSE))
        self._protection = CrashProtection.from_params(params, regime_filter_default=False)
        self._tradability = TradabilityFilter.from_params(params)

    @property
    def name(self) -> str:
        return self._name

    @property
    def requires_full_batch(self) -> bool:
        return True

    def required_features(self) -> List[str]:
        return ["close", "realized_vol_60", "atr_14"] + self._tradability.required_features()

    def entry_rules(self) -> Dict[str, Any]:
        return {
            "rule": "Long bottom decile of the eligible universe by trailing "
                    "60-day annualized realized volatility",
            "bottom_percentile": self._top_fraction,
            "min_universe": self._min_universe,
            "long_only": True,
            "crash_protection": {
                "volatility_target": self._protection.volatility_target,
                "scale_by_volatility": self._protection.scale_by_volatility,
                "regime_filter": self._protection.regime_filter,
            },
            "tradability_filter": {
                "enabled": self._tradability.enabled,
                "min_traded_value_inr": self._tradability.min_traded_value_inr,
                "max_zero_return_fraction": self._tradability.max_zero_return_fraction,
            },
        }

    def score(self, symbol: str, features: pd.DataFrame, context: StrategyContext) -> StrategySignal:
        return self.score_batch({symbol: features}, context)[symbol]

    def score_batch(
        self, features_by_symbol: Dict[str, pd.DataFrame], context: StrategyContext
    ) -> Dict[str, StrategySignal]:
        metric_by_symbol: Dict[str, float] = {}
        latest_by_symbol: Dict[str, pd.Series] = {}

        rejected: Dict[str, str] = {}

        for symbol, features in features_by_symbol.items():
            if features.empty:
                continue
            latest = features.iloc[-1]
            latest_by_symbol[symbol] = latest

            # This screen matters most here: an illiquid stock's suppressed
            # variance is precisely what would rank it first.
            reason = self._tradability.reject_reason(latest)
            if reason is not None:
                rejected[symbol] = reason
                continue

            vol = _clean(latest.get("realized_vol_60"))
            if vol is not None:
                metric_by_symbol[symbol] = vol

        regime = _assess_regime(features_by_symbol, self._protection, context.benchmark_close)

        return _rank_and_select_decile(
            metric_by_symbol=metric_by_symbol,
            latest_by_symbol=latest_by_symbol,
            context=context,
            top_fraction=self._top_fraction,
            higher_is_better=False,
            trigger="LowVolatility",
            component_name="RealizedVol",
            min_universe=self._min_universe,
            protection=self._protection,
            regime=regime,
            rejected=rejected,
        )
