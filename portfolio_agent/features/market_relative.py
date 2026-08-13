"""Risk measured against the market rather than in isolation.

Total volatility mixes two things a cross-sectional book should keep apart: how
much a stock moves *with* the market, and how much it moves on its own. Sorting
on the sum ranks high-beta names and idiosyncratically-wild names identically,
and the 2025 work on the low-risk anomaly finds those two sorts behave very
differently out of sample — idiosyncratic-volatility sorts survive where beta
sorts largely do not.

The residual variance has a closed form
---------------------------------------
Fitting a rolling CAPM per date per symbol with an explicit regression would be
a Python loop over (dates x symbols). It is not needed. For an OLS fit on a
window, the residual variance is exactly

    var(residual) = var(r_i) - beta^2 * var(r_m),    beta = cov(r_i, r_m) / var(r_m)

which is the same identity as `var(r_i) * (1 - rho^2)`, and every term on the
right is a `rolling` operation. Same answer, no loop, and the causality is
inherited from `rolling` rather than argued about: a window ends at the row it
labels, so the value on date `t` was estimable on date `t`.

Floating point can push that difference slightly below zero when a stock is
almost perfectly explained by the market, so it is clipped at zero — a negative
variance is arithmetic noise, not a measurement.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

#: Sessions in a year, for annualizing a daily standard deviation. Matches
#: `features/technical.py::realized_vol_60`, which the idiosyncratic version is
#: meant to be directly comparable with.
TRADING_DAYS_PER_YEAR = 252

#: Default estimation window, in sessions. 60 to match `realized_vol_60`: the
#: comparison between total and idiosyncratic volatility is only about the
#: *decomposition* if the window is held fixed.
DEFAULT_VOL_WINDOW = 60


def market_composite(returns: pd.DataFrame) -> pd.Series:
    """Equal-weighted mean return across the cross-section.

    The stand-in for the market when no index is available. `^NSEI` is a price
    index and is frequently not in the cache at all, so this is what the regime
    filter and `evaluation/neutralize.py` already fall back to — stated once
    here so all three mean the same thing by "the market".

    It is a proxy and behaves like one: it reflects whatever happens to be in
    today's universe, and idiosyncratic noise diversifies out of it in a way
    real index volatility does not.
    """
    return returns.mean(axis=1)


def rolling_idiosyncratic_vol(
    returns: pd.DataFrame,
    market: Optional[pd.Series] = None,
    window: int = DEFAULT_VOL_WINDOW,
    min_periods: Optional[int] = None,
    annualize: bool = True,
) -> pd.DataFrame:
    """Volatility of the CAPM residual, per symbol, per date.

    Args:
        returns: Wide (date x symbol) daily returns. Already lagged if the
            caller needs lag safety — this function does not shift, because
            whether a shift is owed depends on how the returns were built.
        market: Market return series aligned to `returns`. Defaults to the
            equal-weighted composite of `returns` itself.
        window: Sessions in the estimation window.
        min_periods: Minimum observations before a value is produced. Defaults
            to half the window, matching `rolling_beta`.
        annualize: Scale by sqrt(252), so the result is comparable with
            `realized_vol_60`.

    Returns:
        Wide (date x symbol) idiosyncratic volatility, NaN until the window
        fills.
    """
    if window < 2:
        raise ValueError(f"window must be at least 2, got {window}")
    if returns.empty:
        return pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    floor = window // 2 if min_periods is None else int(min_periods)
    floor = max(2, floor)

    market_returns = market_composite(returns) if market is None else market
    market_returns = pd.Series(market_returns).reindex(returns.index)

    market_variance = market_returns.rolling(window, min_periods=floor).var()
    # A market with no variance in the window explains nothing, and dividing by
    # it would produce an infinite beta rather than the "residual is the whole
    # return" answer that is actually correct there.
    safe_variance = market_variance.replace(0.0, np.nan)

    residual: dict = {}
    for symbol in returns.columns:
        series = returns[symbol]
        own_variance = series.rolling(window, min_periods=floor).var()
        covariance = series.rolling(window, min_periods=floor).cov(market_returns)
        beta = covariance / safe_variance

        variance = own_variance - beta.pow(2) * market_variance
        # Where the market was flat, beta is undefined and none of the return
        # is explained — the residual is the return itself.
        variance = variance.where(market_variance.notna() & (market_variance > 0), own_variance)
        residual[symbol] = np.sqrt(variance.clip(lower=0.0))

    frame = pd.DataFrame(residual, index=returns.index)
    return frame * np.sqrt(TRADING_DAYS_PER_YEAR) if annualize else frame


def idiosyncratic_vol_from_closes(
    closes: pd.DataFrame,
    market_close: Optional[pd.Series] = None,
    window: int = DEFAULT_VOL_WINDOW,
    lag: int = 1,
) -> pd.DataFrame:
    """`rolling_idiosyncratic_vol` from price levels, with the lag applied.

    The lag is the whole reason this wrapper exists. Every feature in
    `features/technical.py` shifts its input by one bar so a value cannot read
    the session it is used to decide, and `realized_vol_60` does exactly
    `close.shift(1).pct_change()`. An idiosyncratic version that skipped the
    shift would rank on information the decision date does not have, and would
    do it while sitting next to a feature that does not — the mixed-convention
    hazard T10 removed for the indicator modules.

    Args:
        closes: Wide (date x symbol) close prices.
        market_close: Index close prices. Defaults to the equal-weighted
            composite of the cross-section's own returns.
        window: Sessions in the estimation window.
        lag: Bars to shift before differencing. 1 matches the rest of the
            feature layer; 0 is for a caller that has already shifted.

    Returns:
        Wide (date x symbol) annualized idiosyncratic volatility.
    """
    shifted = closes.shift(lag) if lag else closes
    returns = shifted.pct_change()

    market = None
    if market_close is not None:
        market_shifted = pd.Series(market_close).reindex(closes.index)
        if lag:
            market_shifted = market_shifted.shift(lag)
        market = market_shifted.pct_change()

    return rolling_idiosyncratic_vol(returns, market, window=window)
