"""Cross-sectional strategies: rank the eligible universe, hold one decile.

All of them rank every eligible ticker against every other in a single
`score_batch()` call and go long the extreme decile. They divide into two
families by what they rank on.

**Return-based**

- `MomentumStrategy` — top decile by 9-month (skip 1-month) formation return,
  the Jegadeesh-Titman convention, with crash protection layered on top.
- `ResidualMomentumStrategy` — the same formation window measured on the CAPM
  residual and standardized by its own dispersion (Blitz-Huij-Martens). Price
  momentum's return is substantially a bet on whatever the market has been
  rewarding; round two measured this platform's momentum at 58% factor loading.

**Risk-based** — three decompositions of one anomaly, kept separate so the
comparison between them is a command rather than an argument:

- `LowVolatilityStrategy` — bottom decile by *total* realized volatility.
- `IdiosyncraticLowVolatilityStrategy` — by the volatility of the CAPM
  residual, which is the sort the 2025 literature finds survives.
- `BettingAgainstBetaStrategy` — by rolling market beta (Frazzini-Pedersen).

Total volatility mixes beta and idiosyncratic volatility together; the second
and third sort on the halves.

Everything except the ranking metric and its direction is shared. A new
cross-sectional strategy overrides `_formation_metric` and, if it ranks on a
risk measure rather than a return, `higher_metric_is_better` — the tradability
screen, regime assessment, decile selection, volatility targeting and
reward:risk gate all come from `rank_and_select`.

See docs/QUANT_RESEARCH.md sections 1, 2 and 12 for the academic basis, with an
emphasis on India-specific studies. Section 2 gives the *total*-volatility
formula and then the table of all three risk sorts; the residual and beta
formulations live in `features/market_relative.py`, which is where they are
computed.

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
from .registry import register_strategy
from .types import (
    POSITION_SCALE_KEY,
    TRADABILITY_REJECT_KEY,
    StrategyContext,
    StrategySignal,
)
from portfolio_agent.config.schema import StrategyConfig
from portfolio_agent.features.cross_section import build_cross_section, latest_values
from portfolio_agent.features.market_relative import (
    # Aliased because `src.regime` exports a `DEFAULT_VOL_WINDOW` too, and its
    # import below would otherwise shadow this one. Both are 60 today, which is
    # what makes the collision worth naming rather than leaving: they measure
    # different things — one is the residual-estimation window, the other the
    # market's own volatility lookback — so the day either moves, the
    # idiosyncratic sort would silently start using the regime filter's number.
    DEFAULT_VOL_WINDOW as DEFAULT_IDIOSYNCRATIC_WINDOW,
    idiosyncratic_vol_feature,
    market_beta_feature,
)

#: Registry name of the formation measure `ResidualMomentumStrategy` ranks on.
RESIDUAL_MOMENTUM_FEATURE = "residual_momentum_9m_skip1m"

#: Beta estimation window `BettingAgainstBetaStrategy` defaults to. 252
#: sessions rather than the registry's 60: Frazzini & Pedersen estimate
#: correlations on five years and volatilities on one, and a one-year beta
#: is the shortest window on which the ranking is about a stock's exposure
#: rather than about the last quarter's news.
DEFAULT_BAB_BETA_WINDOW = 252

from portfolio_agent.src.risk import calculate_stop_target, net_reward_risk
from portfolio_agent.src.liquidity import (
    DEFAULT_MAX_CIRCUIT_LOCK_FRACTION,
    DEFAULT_MAX_OPERATOR_TRAP_FRACTION,
    DEFAULT_MAX_ZERO_RETURN_FRACTION,
    DEFAULT_MIN_TRADED_VALUE_INR,
)
from portfolio_agent.src.regime import (
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
      cannot be exited at the modelled stop. Both the zero-range lock and the
      weaker "ran up and closed pinned at the limit" operator footprint are
      screened, since only the first has no intraday range and the second is
      the more common of the two.
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
          max_operator_trap_fraction: 0.05
    """

    enabled: bool = True
    min_traded_value_inr: float = DEFAULT_MIN_TRADED_VALUE_INR
    max_zero_return_fraction: float = DEFAULT_MAX_ZERO_RETURN_FRACTION
    max_circuit_lock_fraction: float = DEFAULT_MAX_CIRCUIT_LOCK_FRACTION
    max_operator_trap_fraction: float = DEFAULT_MAX_OPERATOR_TRAP_FRACTION

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
            max_operator_trap_fraction=float(
                params.get("max_operator_trap_fraction", DEFAULT_MAX_OPERATOR_TRAP_FRACTION)
            ),
        )

    def required_features(self) -> List[str]:
        return [
            "traded_value_60",
            "zero_return_fraction_60",
            "circuit_lock_fraction_60",
            "circuit_locked_today",
            "operator_trap_fraction_60",
            "operator_trap_today",
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

        trap_fraction = _clean(latest.get("operator_trap_fraction_60"))
        if trap_fraction is not None and trap_fraction > self.max_operator_trap_fraction:
            return (
                f"operator footprint: {trap_fraction:.0%} of sessions closed pinned at "
                f"an upper circuit (> {self.max_operator_trap_fraction:.0%})"
            )

        if _clean(latest.get("circuit_locked_today")):
            return "locked at a circuit limit on the decision date; no fill available"

        # The weaker single-day footprint, checked last so the more specific
        # reasons above win the message: the stock traded a real range and then
        # shut at its upper limit, so the printed close is not a price anything
        # could have been bought at.
        if _clean(latest.get("operator_trap_today")):
            return (
                "closed pinned at an upper circuit on the decision date; "
                "no offer available at the printed close"
            )

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
    benchmark_ohlcv: Optional[pd.DataFrame] = None,
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
    market_ohlcv = benchmark_ohlcv
    if market_close is None or len(market_close) < protection.trend_window + 1:
        # A composite of the traded universe has no meaningful high/low, so the
        # ADX-based chop test falls back to its close-only proxy rather than
        # being fed a range that does not exist.
        market_ohlcv = None
        close_by_symbol = {
            symbol: features["close"]
            for symbol, features in features_by_symbol.items()
            if not features.empty and "close" in features.columns
        }
        # Only the trailing trend_window + 1 observations can affect either
        # test, and this runs once per scoring round — on a 4,000-name
        # universe over a 5-year backtest, combining full histories instead
        # would be the run's dominant cost for a single float comparison.
        market_close = build_market_proxy(
            close_by_symbol, lookback=protection.trend_window + 1
        )

    return assess_market_regime(
        market_close,
        trend_window=protection.trend_window,
        vol_window=protection.vol_window,
        target_volatility=protection.volatility_target,
        crash_vol_multiple=protection.crash_vol_multiple,
        bear_exposure=protection.bear_exposure,
        min_scale=protection.min_scale,
        max_scale=protection.max_scale,
        market_ohlcv=market_ohlcv,
    )


def rank_and_select(
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

    **Public since T25, and the reason is the point.** Everything a new
    cross-sectional strategy needs is here: the tradability rejections, the
    minimum-universe abstention, the percentile score, the volatility-targeted
    `position_scale`, the reward:risk gate. It was module-private, so the four
    strategies Phase 3 adds — residual momentum, betting-against-beta,
    short-term reversal, pairs — would each have had to either import a
    underscore-prefixed name or reimplement 175 lines of selection logic, and
    the second is how two strategies come to disagree about what "top decile"
    means.

    A new strategy therefore reduces to: compute one number per symbol, and
    call this with `higher_is_better` set the right way.

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
            extra={POSITION_SCALE_KEY: 0.0, TRADABILITY_REJECT_KEY: reason},
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
                extra={POSITION_SCALE_KEY: 0.0},
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
                POSITION_SCALE_KEY: position_scale if signal == "BUY" else 0.0,
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


@register_strategy("momentum")
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
                "max_operator_trap_fraction": self._tradability.max_operator_trap_fraction,
            },
        }

    #: Trigger and component-score label. A subclass ranking on a different
    #: formation measure sets this so the emitted signal says which one, rather
    #: than every momentum variant reporting "Momentum".
    trigger_name = "Momentum"

    @property
    def higher_metric_is_better(self) -> bool:
        """Which end of the formation metric the book goes long.

        True for momentum: the highest formation return wins. A subclass
        ranking on a *risk* measure rather than a return flips it, and having
        it as a property rather than a literal in `score_batch` is what lets
        `BettingAgainstBetaStrategy` reuse everything else unchanged.
        """
        return True

    def _formation_metric(
        self, features_by_symbol: Dict[str, pd.DataFrame], context: StrategyContext
    ) -> Dict[str, float]:
        """The formation return this strategy ranks on, per symbol.

        The one thing a momentum variant changes. Everything around it — the
        tradability screen, the regime assessment, the decile selection, the
        volatility-targeted scale — is identical, which is what makes a
        comparison between two variants a comparison of the formation measure
        and nothing else.
        """
        values: Dict[str, float] = {}
        for symbol, features in features_by_symbol.items():
            if features.empty:
                continue
            value = _clean(features.iloc[-1].get("mom_9m_skip1m"))
            if value is not None:
                values[symbol] = value
        return values

    def score(self, symbol: str, features: pd.DataFrame, context: StrategyContext) -> StrategySignal:
        return self.score_batch({symbol: features}, context)[symbol]

    def score_batch(
        self, features_by_symbol: Dict[str, pd.DataFrame], context: StrategyContext
    ) -> Dict[str, StrategySignal]:
        metric_by_symbol: Dict[str, float] = {}
        latest_by_symbol: Dict[str, pd.Series] = {}

        rejected: Dict[str, str] = {}

        metric = self._formation_metric(features_by_symbol, context)

        for symbol, features in features_by_symbol.items():
            if features.empty:
                continue
            latest = features.iloc[-1]
            latest_by_symbol[symbol] = latest

            reason = self._tradability.reject_reason(latest)
            if reason is not None:
                rejected[symbol] = reason
                continue

            value = _clean(metric.get(symbol))
            if value is not None:
                metric_by_symbol[symbol] = value

        regime = _assess_regime(
            features_by_symbol, self._protection,
            context.benchmark_close, context.benchmark_ohlcv,
        )

        return rank_and_select(
            metric_by_symbol=metric_by_symbol,
            latest_by_symbol=latest_by_symbol,
            context=context,
            top_fraction=self._top_fraction,
            higher_is_better=self.higher_metric_is_better,
            trigger=self.trigger_name,
            component_name=self.trigger_name,
            min_universe=self._min_universe,
            protection=self._protection,
            regime=regime,
            rejected=rejected,
        )


@register_strategy("residual_momentum")
class ResidualMomentumStrategy(MomentumStrategy):
    """Momentum measured on the CAPM residual rather than the raw return.

    Price momentum's returns are substantially a bet on whatever the market
    has been rewarding, which is why it crashes the way it does: the exposure
    that pays during a trend is the same exposure that reverses violently at a
    turn. Round two measured this platform's own momentum at **58% factor
    loading**, so the correction is not hypothetical here.

    Blitz, Huij & Martens (2011) rank on the residual's *information ratio* —
    the mean residual over the formation window divided by its own dispersion —
    and report roughly double the risk-adjusted profit of price momentum with
    materially shallower drawdowns. The standardization is the substance:
    ranking on raw cumulated residuals still puts high-residual-volatility
    names on top, reintroducing exactly the risk exposure residualizing was
    meant to remove.

    Everything except the formation measure is inherited from
    `MomentumStrategy` — the same tradability screen, regime filter,
    volatility targeting and decile selection — so a comparison between the two
    is a comparison of the formation measure and nothing else:

        portfolio-agent compare --strategies momentum,residual_momentum

    This also corrects `docs/QUANT_RESEARCH.md` section 12(c), which recorded
    idiosyncratic momentum as needing section 8's un-ingested factor data. It
    needs a market return and a rolling beta, both of which T14 built.
    """

    trigger_name = "ResidualMomentum"

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        params = config.params or {}
        self._name = params.get("name", "residual_momentum")

    def required_features(self) -> List[str]:
        # `mom_9m_skip1m` is deliberately absent: this strategy does not rank on
        # it, and requesting it would widen every warm-up by 211 rows for a
        # column nothing reads.
        return [
            "close", "atr_14", "realized_vol_60"
        ] + self._tradability.required_features()

    def required_cross_sectional_features(self) -> List[str]:
        return [RESIDUAL_MOMENTUM_FEATURE]

    def entry_rules(self) -> Dict[str, Any]:
        rules = super().entry_rules()
        rules["rule"] = (
            "Long top decile of the eligible universe by the information ratio "
            "of the CAPM residual over a 9-month formation window, skipping "
            "the most recent month"
        )
        rules["formation_metric"] = RESIDUAL_MOMENTUM_FEATURE
        return rules

    def _formation_metric(
        self, features_by_symbol: Dict[str, pd.DataFrame], context: StrategyContext
    ) -> Dict[str, float]:
        """Residual momentum for the batch, via the cross-sectional registry.

        A residual is defined against a cross-section, so below two names there
        is nothing to residualize against and this returns nothing rather than
        a number that would rank identically to raw momentum while being
        labelled otherwise — the same refusal `LowVolatilityStrategy` makes.
        """
        usable = {
            symbol: features
            for symbol, features in features_by_symbol.items()
            if not features.empty and "close" in features.columns
        }
        if len(usable) < 2:
            return {}

        built = build_cross_section(
            usable, [RESIDUAL_MOMENTUM_FEATURE],
            benchmark=context.benchmark_close,
        )
        return latest_values(built.get(RESIDUAL_MOMENTUM_FEATURE, pd.DataFrame()))


@register_strategy("bab")
class BettingAgainstBetaStrategy(MomentumStrategy):
    """Long the low-beta decile — the low-risk anomaly's beta form.

    Frazzini & Pedersen's account is a funding-constraint one: investors who
    want more risk than they can borrow to obtain bid up high-beta stocks
    instead, so beta is overpriced and the security market line is flatter than
    CAPM says. The prediction is that a beta-sorted long-short book earns a
    positive alpha, and the long-only half of it is the low-beta decile.

    **Indian evidence is favourable and specific.** NSE 2001-2016 finds the
    effect positive across capitalizations after controlling for size, value
    and momentum — which matters because the naive worry about a low-beta sort
    is that it is a size bet wearing a different name.

    **And it is conditional.** 2025 Asian work finds the effect concentrated in
    downturns, which is what `evaluation/conditional.py` exists to measure: a
    pooled IC made of a strong down-market number and a flat up-market one
    describes neither state. Evaluate this strategy with the split, not without
    it.

    Ranked lowest-beta-first, which is `higher_is_better=False` — the same
    direction `LowVolatilityStrategy` sorts in, and for a related reason. The
    two are worth comparing directly: beta and idiosyncratic volatility are the
    two halves total volatility mixes together, and T14 found the residual sort
    survives where the total sort does not.

        portfolio-agent compare --strategies bab,low_volatility_idio
    """

    trigger_name = "BettingAgainstBeta"

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        params = config.params or {}
        self._name = params.get("name", "bab")
        self._beta_window = int(params.get("beta_window", DEFAULT_BAB_BETA_WINDOW))
        # Resolved at construction so an unregistered window fails when the
        # strategy is built rather than on the first scored date.
        self._beta_feature = market_beta_feature(self._beta_window)

    def required_features(self) -> List[str]:
        return [
            "close", "atr_14", "realized_vol_60"
        ] + self._tradability.required_features()

    def required_cross_sectional_features(self) -> List[str]:
        return [self._beta_feature]

    @property
    def higher_metric_is_better(self) -> bool:
        return False

    def entry_rules(self) -> Dict[str, Any]:
        rules = super().entry_rules()
        rules["rule"] = (
            f"Long bottom decile of the eligible universe by rolling "
            f"{self._beta_window}-session market beta"
        )
        rules["formation_metric"] = self._beta_feature
        rules["beta_window"] = self._beta_window
        return rules

    def _formation_metric(
        self, features_by_symbol: Dict[str, pd.DataFrame], context: StrategyContext
    ) -> Dict[str, float]:
        """Rolling beta for the batch, via the cross-sectional registry.

        A beta is measured against a cross-section, so below two names there is
        no market to measure against and this returns nothing — the same
        refusal `LowVolatilityStrategy` and `ResidualMomentumStrategy` make.
        """
        usable = {
            symbol: features
            for symbol, features in features_by_symbol.items()
            if not features.empty and "close" in features.columns
        }
        if len(usable) < 2:
            return {}

        built = build_cross_section(
            usable, [self._beta_feature], benchmark=context.benchmark_close,
        )
        return latest_values(built.get(self._beta_feature, pd.DataFrame()))


#: How `LowVolatilityStrategy` measures risk. `total` is trailing realized
#: volatility — the original sort. `idiosyncratic` is the volatility of the
#: CAPM residual, which is the sort the 2025 literature finds survives.
VOLATILITY_SORTS = ("total", "idiosyncratic")


@register_strategy("low_volatility")
class LowVolatilityStrategy(BaseStrategy):
    """Low-volatility anomaly: long the bottom decile by trailing volatility
    (docs/QUANT_RESEARCH.md section 2).

    Two ways to measure that volatility, chosen with the `sort_on` param:

    - ``total`` (default) — trailing 60-day realized volatility. The original
      sort, and the one whose result T05 reported: rank IC +0.061 raw, +0.018
      once beta and size are removed. 71% of the apparent alpha was factor
      loading, which for a volatility screen is close to tautological. It *is*
      a beta bet.
    - ``idiosyncratic`` — volatility of the CAPM residual over the same window.
      Total volatility is beta times market volatility plus the residual, so
      sorting on the total ranks high-beta names and idiosyncratically-wild
      names identically. The 2025 work on the low-risk anomaly finds those two
      sorts behave very differently out of sample: idiosyncratic-volatility
      sorts survive where beta sorts largely do not.

    Registered under both `low_volatility` and `low_volatility_idio` so the two
    can be evaluated against each other without editing a config — which is the
    only way the comparison actually gets made.

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

        sort_on = str(params.get("sort_on", "total")).lower()
        if sort_on not in VOLATILITY_SORTS:
            raise ValueError(
                f"sort_on must be one of {list(VOLATILITY_SORTS)}, got {sort_on!r}"
            )
        self._sort_on = sort_on
        # `idiosyncratic_window`, not `vol_window`. `CrashProtection` already
        # reads `vol_window` for the *market's* volatility lookback, so sharing
        # the key would make one setting move two unrelated windows — the
        # regime filter's view of market stress and the length of the CAPM
        # regression — with no indication that it had.
        self._vol_window = int(
            params.get("idiosyncratic_window", DEFAULT_IDIOSYNCRATIC_WINDOW)
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def requires_full_batch(self) -> bool:
        return True

    def required_features(self) -> List[str]:
        return ["close", "realized_vol_60", "atr_14"] + self._tradability.required_features()

    def entry_rules(self) -> Dict[str, Any]:
        measure = (
            "trailing 60-day annualized realized volatility"
            if self._sort_on == "total"
            else (
                f"annualized volatility of the CAPM residual over "
                f"{self._vol_window} sessions"
            )
        )
        return {
            "rule": f"Long bottom decile of the eligible universe by {measure}",
            "sort_on": self._sort_on,
            "vol_window": self._vol_window,
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

        idiosyncratic = (
            self._idiosyncratic_vol(features_by_symbol, context)
            if self._sort_on == "idiosyncratic"
            else {}
        )

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

            if self._sort_on == "idiosyncratic":
                # No fallback to total volatility for a symbol whose residual
                # could not be estimated. Mixing two different measures into one
                # ranking is the failure this task exists to remove, and a
                # partially-idiosyncratic sort would be harder to notice than a
                # thin one.
                vol = idiosyncratic.get(symbol)
            else:
                vol = _clean(latest.get("realized_vol_60"))
            if vol is not None:
                metric_by_symbol[symbol] = vol

        regime = _assess_regime(
            features_by_symbol, self._protection,
            context.benchmark_close, context.benchmark_ohlcv,
        )

        return rank_and_select(
            metric_by_symbol=metric_by_symbol,
            latest_by_symbol=latest_by_symbol,
            context=context,
            top_fraction=self._top_fraction,
            higher_is_better=False,
            trigger="LowVolatility",
            component_name=(
                "RealizedVol" if self._sort_on == "total" else "IdiosyncraticVol"
            ),
            min_universe=self._min_universe,
            protection=self._protection,
            regime=regime,
            rejected=rejected,
        )

    def required_cross_sectional_features(self) -> List[str]:
        """The residual sort ranks on a feature of the whole cross-section.

        The configured window selects the registry name rather than being
        passed as an argument, matching how every other window in the feature
        layer works (`sma_20`, `realized_vol_60`). An unregistered window
        raises at construction instead of silently ranking on 60 sessions
        under a name that says otherwise.
        """
        if self._sort_on != "idiosyncratic":
            return []
        return [idiosyncratic_vol_feature(self._vol_window)]

    def _idiosyncratic_vol(
        self, features_by_symbol: Dict[str, pd.DataFrame], context: StrategyContext
    ) -> Dict[str, float]:
        """Latest residual volatility per symbol, or an empty map if unusable.

        The market is the cached index when the context carries one and the
        equal-weighted composite of this batch otherwise — the same preference
        order `_assess_regime` uses, for the same reason: a real index is what
        the research studied, and the composite is always available.

        A residual against a one-name "cross-section" is the return itself, so
        below two symbols this returns nothing rather than a number that would
        rank identically to the total-volatility sort while being labelled
        differently.

        Routed through the cross-sectional registry since T24. It used to
        import `idiosyncratic_vol_from_closes` directly and pass `lag=1` by
        hand — the platform's only cross-sectional feature, reached from inside
        a strategy method because no registry could express its shape. The
        decorator owns the lag now, so this cannot drift from the convention
        the per-ticker features follow.
        """
        usable = {
            symbol: features
            for symbol, features in features_by_symbol.items()
            if not features.empty and "close" in features.columns
        }
        if len(usable) < 2:
            return {}

        # `+2` because the window needs its rows *after* the lag and the
        # differencing each consume one.
        if max(len(frame) for frame in usable.values()) < self._vol_window + 2:
            return {}

        built = build_cross_section(
            usable, self.required_cross_sectional_features(),
            benchmark=context.benchmark_close,
        )
        name = self.required_cross_sectional_features()[0]
        return latest_values(built.get(name, pd.DataFrame()))


@register_strategy("low_volatility_idio")
class IdiosyncraticLowVolatilityStrategy(LowVolatilityStrategy):
    """`LowVolatilityStrategy` with the residual sort as its default.

    A subclass rather than a config preset because the registry maps a name to
    a class and constructs it from whatever `StrategyConfig` the caller has.
    Without a distinct class, comparing the two sorts would mean editing a
    config between runs — and a comparison that requires editing a file to
    perform is a comparison that does not get performed. An explicit
    `sort_on` in params still wins, so this is a default and not a lock.
    """

    def __init__(self, config: StrategyConfig):
        params = dict(config.params or {})
        params.setdefault("sort_on", "idiosyncratic")
        params.setdefault("name", "low_volatility_idio")
        super().__init__(config.model_copy(update={"params": params}))
