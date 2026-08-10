"""Momentum crash protection: market-regime detection and volatility targeting.

Motivation (docs/QUANT_RESEARCH.md section 12). Momentum is the factor most
prone to catastrophic, fat-tailed failure. Its crashes are not random: they
cluster in *panic states* — after a bear market, during the rebound, when the
recent losers the strategy is underweight rally hardest and market volatility
is elevated. In Indian equities pure momentum has drawn down 45-55% in 2011,
2018 and the COVID crash, and the Nifty 200 Momentum 30 index's worst drawdown
(-70.25%) is materially deeper than the Nifty's (-55.12%).

The literature offers three fixes; two of them need only OHLCV and are
implemented here:

1. **Constant volatility scaling** — scale exposure by target_vol / realized
   vol so a position's risk contribution stays roughly constant instead of
   ballooning exactly when volatility spikes.
2. **Dynamic scaling / regime filter** — cut momentum exposure outright in the
   panic state (market below its long-run trend, and/or market volatility far
   above normal), which is where the crashes actually happen.
3. Idiosyncratic momentum (ranking on residual rather than total returns)
   needs a factor model and is noted as future work, not implemented here.

Both implemented pieces derive the market state from the traded universe
itself — an equal-weighted composite of the eligible tickers — so no index
feed, VIX feed or other new data source is required. Nifty VIX would be a
better volatility gauge than realized composite volatility if it were
ingested; that is a data gap, recorded in docs/QUANT_RESEARCH.md section 10.

Everything here is deliberately fail-neutral: with too little history to
judge the regime, exposure is left unscaled rather than either blocked or
levered, and the reason is reported on the signal's rationale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# Defaults, all overridable through strategy params.
DEFAULT_TREND_WINDOW = 200        # trading days; the classic long-run trend filter
DEFAULT_VOL_WINDOW = 60           # trading days of realized market volatility
DEFAULT_TARGET_VOLATILITY = 0.20  # 20% annualized — a moderate equity risk budget
DEFAULT_MAX_SCALE = 1.0           # never lever up; scaling only ever cuts exposure
DEFAULT_MIN_SCALE = 0.25          # floor so a vol spike dampens rather than deletes
DEFAULT_CRASH_VOL_MULTIPLE = 1.5  # market vol above 1.5x target = panic state
DEFAULT_BEAR_EXPOSURE = 0.0       # exposure retained in a confirmed downtrend
DEFAULT_ADX_PERIOD = 14           # Wilder's default
DEFAULT_CHOP_ADX = 20.0           # ADX below this = no persistent trend
DEFAULT_CHOP_BAND = 0.02          # within 2% of the 200-day MA = "at" the mean

# Regime classifications used by the meta-orchestrator to decide which models
# are allowed to buy (config/strategies/uma_meta_orchestrator.yaml). Distinct
# from MarketRegime.label, which describes how hard to scale exposure; these
# describe *what kind of market it is*, which is a different question with a
# different answer — a chop and a calm uptrend can call for the same exposure
# scalar while suiting completely different strategies.
BULL_RISK_ON = "BULL_RISK_ON"
BEAR_CRASH_RISK = "BEAR_CRASH_RISK"
SIDEWAYS_CHOP = "SIDEWAYS_CHOP"
NEUTRAL = "NEUTRAL"
UNKNOWN = "UNKNOWN"


@dataclass
class MarketRegime:
    """The market state a cross-sectional strategy is trading into."""

    label: str  # "risk_on" | "elevated_vol" | "crash_risk" | "unknown"
    trend_ok: bool  # composite above its long-run moving average
    market_volatility: Optional[float]  # annualized realized vol of the composite
    vol_scalar: float  # volatility-targeting multiplier in [min_scale, max_scale]
    exposure_scalar: float  # final multiplier applied to position size, in [0, 1]
    reason: str  # human-readable explanation, surfaced on signal rationales
    # Phase 4 classification (BULL_RISK_ON / BEAR_CRASH_RISK / SIDEWAYS_CHOP /
    # NEUTRAL / UNKNOWN). `label` above answers "how much exposure?"; this
    # answers "what kind of market?", and the meta-orchestrator keys its
    # model-to-regime map off it.
    classification: str = UNKNOWN
    trend_distance: Optional[float] = None  # (price / trend_MA - 1), signed
    adx: Optional[float] = None  # trend strength of the benchmark

    @property
    def blocks_new_entries(self) -> bool:
        """Whether the regime is hostile enough to stand aside entirely."""
        return self.exposure_scalar <= 0.0


def neutral_regime(reason: str = "insufficient history to assess market regime") -> MarketRegime:
    """The fail-neutral regime used when the market state cannot be judged.

    Leaves exposure unscaled: with a short cache (or a single feature row, as
    in unit tests) there is no evidence of a panic state, and inventing one
    would silently disable the strategy rather than protect it.
    """
    return MarketRegime(
        label="unknown",
        trend_ok=True,
        market_volatility=None,
        vol_scalar=1.0,
        exposure_scalar=1.0,
        reason=reason,
        classification=UNKNOWN,
    )


def classify_regime(
    trend_distance: Optional[float],
    market_volatility: Optional[float],
    adx: Optional[float],
    target_volatility: float = DEFAULT_TARGET_VOLATILITY,
    crash_vol_multiple: float = DEFAULT_CRASH_VOL_MULTIPLE,
    chop_adx: float = DEFAULT_CHOP_ADX,
    chop_band: float = DEFAULT_CHOP_BAND,
) -> str:
    """Name the market state the way the meta-orchestrator's model map does.

    The three headline definitions are:

    - ``BULL_RISK_ON``     — index above its 200-day average, realized vol
                             below target.
    - ``BEAR_CRASH_RISK``  — index below its 200-day average, *or* realized
                             vol above `crash_vol_multiple` x target.
    - ``SIDEWAYS_CHOP``    — index within `chop_band` of its 200-day average
                             and ADX below `chop_adx`.

    As stated they overlap, so the order of the checks is doing real work and
    is not arbitrary:

    1. **A volatility spike is checked first.** Vol above 1.5x target is
       unambiguous panic regardless of where price sits relative to the
       average, and misreading a crash as a chop would leave mean reversion
       buying into it.
    2. **Chop is checked next**, because it is the most specific condition —
       "near the mean *and* directionless" — and because an index sitting 1%
       below its 200-day average in a calm market is a chop, not a bear. Taking
       the bear branch's "below the MA" clause literally there would mute the
       trend strategies during exactly the drift they handle fine.
    3. **Then the bear branch's trend clause**, then the bull branch.
    4. Anything left — above the average but with volatility between target and
       the crash multiple — is ``NEUTRAL``: not risk-on, not a crash. Naming it
       rather than forcing it into one of the three keeps the map honest, since
       the strategies suited to a calm uptrend are not the ones suited to a
       jittery one.

    Any input that could not be measured yields ``UNKNOWN``, and the
    meta-orchestrator treats that as "permit everything" rather than standing
    the book down on a missing statistic.

    Args:
        trend_distance: Signed distance from the trend average, as a fraction
            (price / MA - 1).
        market_volatility: Annualized realized volatility of the benchmark.
        adx: Benchmark ADX; None disables the chop test.
        target_volatility: Annualized volatility budget.
        crash_vol_multiple: Multiple of target that marks a panic state.
        chop_adx: ADX below which the market is treated as directionless.
        chop_band: Half-width of the "at the mean" band, as a fraction.

    Returns:
        One of the module's regime classification constants.
    """
    if trend_distance is None:
        return UNKNOWN

    if market_volatility is not None and market_volatility > crash_vol_multiple * target_volatility:
        return BEAR_CRASH_RISK

    if adx is not None and adx < chop_adx and abs(trend_distance) <= chop_band:
        return SIDEWAYS_CHOP

    if trend_distance <= 0:
        return BEAR_CRASH_RISK

    if market_volatility is not None and market_volatility < target_volatility:
        return BULL_RISK_ON

    return NEUTRAL


def build_market_proxy(
    close_by_symbol: Dict[str, pd.Series],
    lookback: Optional[int] = None,
) -> Optional[pd.Series]:
    """Build an equal-weighted composite price index from the traded universe.

    Averages **daily returns** across symbols and cumulates the result, rather
    than averaging rebased price levels. The distinction is not cosmetic: real
    universes have ragged start dates, and the eligible set changes daily
    during a backtest. Averaging rebased levels makes a newly-listed ticker
    enter the average at its own base of 1.0 while incumbents sit at 1.4, which
    prints a double-digit synthetic drop in the composite on a day every
    constituent rose — and `assess_market_regime` would read that construction
    artifact as a trend break and as realized volatility, which is exactly what
    drives the crash filter. A return average has no such discontinuity: a
    symbol simply contributes nothing on days it has no return.

    Args:
        close_by_symbol: Per-symbol close price series, indexed by date.
        lookback: Keep only this many trailing observations per symbol before
            combining. The regime tests need at most `trend_window + 1` points,
            and the composite is otherwise rebuilt from every symbol's full
            history on every scoring round.

    Returns:
        The composite index series (starting at 1.0), or None if no usable
        series were supplied.
    """
    returns = []
    for series in close_by_symbol.values():
        if series is None or len(series) < 2:
            continue
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if lookback is not None and len(clean) > lookback + 1:
            clean = clean.iloc[-(lookback + 1):]
        if len(clean) < 2:
            continue
        clean = clean[clean > 0]
        if len(clean) < 2:
            continue
        returns.append(clean.pct_change())

    if not returns:
        return None

    mean_returns = pd.concat(returns, axis=1).mean(axis=1, skipna=True)
    mean_returns = mean_returns.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if len(mean_returns) < 2:
        return None

    composite = (1.0 + mean_returns).cumprod()
    return composite if len(composite) >= 2 else None


def realized_volatility(close: pd.Series, window: int = DEFAULT_VOL_WINDOW) -> Optional[float]:
    """Annualized realized volatility of a price series' trailing `window` returns.

    Returns None when there are not enough observations to fill the window —
    a half-filled window is a noisy estimate, and this feeds a gate that cuts
    real exposure.
    """
    if close is None or len(close) < window + 1:
        return None

    returns = close.pct_change().dropna()
    if len(returns) < window:
        return None

    sigma = float(returns.iloc[-window:].std(ddof=1))
    if not math.isfinite(sigma):
        return None
    return sigma * math.sqrt(TRADING_DAYS_PER_YEAR)


def volatility_target_scalar(
    volatility: Optional[float],
    target_volatility: float = DEFAULT_TARGET_VOLATILITY,
    min_scale: float = DEFAULT_MIN_SCALE,
    max_scale: float = DEFAULT_MAX_SCALE,
) -> float:
    """Constant-volatility-scaling multiplier: target_vol / realized_vol.

    Clamped to [min_scale, max_scale]. max_scale defaults to 1.0, so this can
    only ever reduce a position, never lever one up — the platform's existing
    position caps stay the binding constraint in calm markets.

    Args:
        volatility: Annualized realized volatility, or None if unmeasurable.
        target_volatility: Annualized volatility budget.
        min_scale: Lower clamp, so an extreme vol print dampens rather than
            zeroes a position (the regime filter, not this scalar, is what
            stands the strategy down entirely).
        max_scale: Upper clamp.

    Returns:
        A multiplier in [min_scale, max_scale]; 1.0 when volatility is unknown.
    """
    if volatility is None or volatility <= 0 or target_volatility <= 0:
        return 1.0
    return float(min(max_scale, max(min_scale, target_volatility / volatility)))


def _benchmark_adx(
    market_close: pd.Series,
    market_ohlcv: Optional[pd.DataFrame],
    period: int = DEFAULT_ADX_PERIOD,
) -> Optional[float]:
    """Latest ADX of the benchmark, preferring a real OHLC frame.

    When the caller supplies OHLC the range is real; otherwise the close series
    is passed through as a degenerate frame and calculate_adx falls back to its
    close-only proxy. Returns None when ADX cannot be computed at all, which
    disables the chop test rather than guessing at it.
    """
    try:
        from .indicators import calculate_adx
    except ImportError:  # pragma: no cover - direct-module execution fallback
        from indicators import calculate_adx

    frame = market_ohlcv
    if frame is None or "close" not in getattr(frame, "columns", []):
        frame = pd.DataFrame({"close": market_close})
    elif len(frame) < period + 1:
        return None

    try:
        series = calculate_adx(frame, period=period).dropna()
    except Exception:
        return None

    if series.empty:
        return None
    value = float(series.iloc[-1])
    return value if math.isfinite(value) else None


def assess_market_regime(
    market_close: Optional[pd.Series],
    trend_window: int = DEFAULT_TREND_WINDOW,
    vol_window: int = DEFAULT_VOL_WINDOW,
    target_volatility: float = DEFAULT_TARGET_VOLATILITY,
    crash_vol_multiple: float = DEFAULT_CRASH_VOL_MULTIPLE,
    bear_exposure: float = DEFAULT_BEAR_EXPOSURE,
    min_scale: float = DEFAULT_MIN_SCALE,
    max_scale: float = DEFAULT_MAX_SCALE,
    market_ohlcv: Optional[pd.DataFrame] = None,
    adx_period: int = DEFAULT_ADX_PERIOD,
    chop_adx: float = DEFAULT_CHOP_ADX,
    chop_band: float = DEFAULT_CHOP_BAND,
) -> MarketRegime:
    """Classify the market state and derive an exposure multiplier.

    Three states, ordered by how hostile they are to momentum:

    - **crash_risk** — the composite is below its `trend_window` moving
      average (a confirmed downtrend) *and* realized market volatility is
      above `crash_vol_multiple` x target. This is the panic state in which
      momentum crashes occur; exposure collapses to `bear_exposure`
      (0.0 by default, i.e. no new momentum entries).
    - **elevated_vol** — one of the two conditions holds. Exposure is
      volatility-scaled, and halved again if the downtrend is the trigger.
    - **risk_on** — neither holds; exposure is volatility-scaled only.

    Args:
        market_close: Composite market price series (see build_market_proxy).
        trend_window: Moving-average window defining the long-run trend.
        vol_window: Window for realized market volatility.
        target_volatility: Annualized volatility budget for scaling.
        crash_vol_multiple: Multiple of target volatility that marks a panic state.
        bear_exposure: Exposure retained in the crash state, in [0, 1]. Set
            above 0 to dampen rather than fully stand down.
        min_scale: Lower clamp on the volatility scalar.
        max_scale: Upper clamp on the volatility scalar.
        market_ohlcv: Optional benchmark OHLC frame aligned to market_close.
            Only ADX needs the daily range, and only the SIDEWAYS_CHOP
            classification needs ADX; without it the index falls back to a
            close-only proxy (see indicators.calculate_adx).
        adx_period: Wilder smoothing period for ADX.
        chop_adx: ADX below which the market counts as directionless.
        chop_band: Half-width of the "at the 200-day average" band.

    Returns:
        A MarketRegime. Falls back to neutral_regime() whenever there is not
        enough history to evaluate the trend filter.
    """
    if market_close is None or len(market_close) < trend_window + 1:
        return neutral_regime(
            f"market proxy has <{trend_window + 1} observations; regime filter inactive"
        )

    trend_ma = float(market_close.iloc[-trend_window:].mean())
    last = float(market_close.iloc[-1])
    trend_ok = last > trend_ma
    trend_distance = (last / trend_ma - 1.0) if trend_ma else None

    market_vol = realized_volatility(market_close, vol_window)
    vol_scalar = volatility_target_scalar(market_vol, target_volatility, min_scale, max_scale)

    adx = _benchmark_adx(market_close, market_ohlcv, adx_period)
    classification = classify_regime(
        trend_distance=trend_distance,
        market_volatility=market_vol,
        adx=adx,
        target_volatility=target_volatility,
        crash_vol_multiple=crash_vol_multiple,
        chop_adx=chop_adx,
        chop_band=chop_band,
    )
    tagged = dict(classification=classification, trend_distance=trend_distance, adx=adx)

    vol_spike = market_vol is not None and market_vol > crash_vol_multiple * target_volatility
    vol_text = f"{market_vol:.1%}" if market_vol is not None else "n/a"

    if not trend_ok and vol_spike:
        exposure = min(1.0, max(0.0, bear_exposure))
        return MarketRegime(
            label="crash_risk",
            trend_ok=False,
            market_volatility=market_vol,
            vol_scalar=vol_scalar,
            exposure_scalar=exposure,
            reason=(
                f"crash risk: market {last:.2f} below {trend_window}d MA {trend_ma:.2f} "
                f"and realized vol {vol_text} > {crash_vol_multiple:.1f}x target "
                f"{target_volatility:.0%}; momentum exposure -> {exposure:.0%}"
            ),
            **tagged,
        )

    if not trend_ok:
        # Downtrend without a volatility spike: dampen rather than stand down.
        # The crash literature puts the danger in the *rebound* out of a bear
        # market, so a downtrend alone is a warning, not a stop.
        exposure = max(0.0, min(1.0, vol_scalar * 0.5))
        return MarketRegime(
            label="elevated_vol",
            trend_ok=False,
            market_volatility=market_vol,
            vol_scalar=vol_scalar,
            exposure_scalar=exposure,
            reason=(
                f"downtrend: market {last:.2f} below {trend_window}d MA {trend_ma:.2f}; "
                f"vol {vol_text}; exposure -> {exposure:.0%}"
            ),
            **tagged,
        )

    if vol_spike:
        exposure = max(0.0, min(1.0, vol_scalar))
        return MarketRegime(
            label="elevated_vol",
            trend_ok=True,
            market_volatility=market_vol,
            vol_scalar=vol_scalar,
            exposure_scalar=exposure,
            reason=(
                f"elevated volatility: realized vol {vol_text} > "
                f"{crash_vol_multiple:.1f}x target {target_volatility:.0%}; "
                f"exposure -> {exposure:.0%}"
            ),
            **tagged,
        )

    exposure = max(0.0, min(1.0, vol_scalar))
    return MarketRegime(
        label="risk_on",
        trend_ok=True,
        market_volatility=market_vol,
        vol_scalar=vol_scalar,
        exposure_scalar=exposure,
        reason=(
            f"risk on: market above {trend_window}d MA, vol {vol_text}; "
            f"exposure -> {exposure:.0%}"
        ),
        **tagged,
    )
