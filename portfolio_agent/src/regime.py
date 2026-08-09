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


@dataclass
class MarketRegime:
    """The market state a cross-sectional strategy is trading into."""

    label: str  # "risk_on" | "elevated_vol" | "crash_risk" | "unknown"
    trend_ok: bool  # composite above its long-run moving average
    market_volatility: Optional[float]  # annualized realized vol of the composite
    vol_scalar: float  # volatility-targeting multiplier in [min_scale, max_scale]
    exposure_scalar: float  # final multiplier applied to position size, in [0, 1]
    reason: str  # human-readable explanation, surfaced on signal rationales

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
    )


def build_market_proxy(close_by_symbol: Dict[str, pd.Series]) -> Optional[pd.Series]:
    """Build an equal-weighted composite price index from the traded universe.

    Each ticker's close series is rebased to 1.0 at its first observation and
    the rebased series are averaged, so a ₹5,000 stock does not dominate a ₹50
    one. Symbols are aligned on their shared index and any date is averaged
    over whichever symbols have data there, which keeps the composite usable
    when tickers have ragged histories.

    Args:
        close_by_symbol: Per-symbol close price series, indexed by date.

    Returns:
        The composite index series, or None if no usable series were supplied.
    """
    rebased = []
    for series in close_by_symbol.values():
        if series is None or len(series) < 2:
            continue
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if len(clean) < 2:
            continue
        base = clean.iloc[0]
        if not np.isfinite(base) or base <= 0:
            continue
        rebased.append(clean / base)

    if not rebased:
        return None

    composite = pd.concat(rebased, axis=1).mean(axis=1, skipna=True)
    composite = composite.dropna().sort_index()
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


def assess_market_regime(
    market_close: Optional[pd.Series],
    trend_window: int = DEFAULT_TREND_WINDOW,
    vol_window: int = DEFAULT_VOL_WINDOW,
    target_volatility: float = DEFAULT_TARGET_VOLATILITY,
    crash_vol_multiple: float = DEFAULT_CRASH_VOL_MULTIPLE,
    bear_exposure: float = DEFAULT_BEAR_EXPOSURE,
    min_scale: float = DEFAULT_MIN_SCALE,
    max_scale: float = DEFAULT_MAX_SCALE,
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

    market_vol = realized_volatility(market_close, vol_window)
    vol_scalar = volatility_target_scalar(market_vol, target_volatility, min_scale, max_scale)

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
    )
