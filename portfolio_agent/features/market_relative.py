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

from .cross_section import CrossSectionPanel, register_cross_sectional_feature

#: Sessions in a year, for annualizing a daily standard deviation. Matches
#: `features/technical.py::realized_vol_60`, which the idiosyncratic version is
#: meant to be directly comparable with.
TRADING_DAYS_PER_YEAR = 252

#: Default estimation window, in sessions. 60 to match `realized_vol_60`: the
#: comparison between total and idiosyncratic volatility is only about the
#: *decomposition* if the window is held fixed.
DEFAULT_VOL_WINDOW = 60

#: Default beta estimation window, in sessions. Matches the vol window so a
#: beta sort and an idiosyncratic-volatility sort are decompositions of the
#: same 60 sessions rather than two different samples.
DEFAULT_BETA_WINDOW = 60


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


def rolling_beta(
    returns: pd.DataFrame,
    market: Optional[pd.Series] = None,
    window: int = DEFAULT_BETA_WINDOW,
) -> pd.DataFrame:
    """Rolling market beta per symbol.

    Moved here from `evaluation/neutralize.py`, which is where it was written
    and where it did not belong: it is a characteristic of a stock, the
    evaluation layer was importing `market_composite` *from this module* to
    compute it, and `betting-against-beta` needs to rank on it rather than
    neutralize by it. `neutralize.rolling_beta` now delegates, so its callers
    and its tests are unchanged.

    Causal by construction: `rolling` windows end at the row they label, so the
    beta on date `t` was estimable on date `t`.

    Args:
        returns: Wide (date x symbol) daily returns. Already lagged if the
            caller owes a lag — same contract as `rolling_idiosyncratic_vol`.
        market: Market returns. Defaults to the cross-section's own composite.
        window: Sessions in the estimation window.

    Returns:
        Wide (date x symbol) betas, NaN until the window fills.
    """
    if returns.empty:
        return pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    market_returns = market_composite(returns) if market is None else market
    market_returns = pd.Series(market_returns).reindex(returns.index)

    floor = max(2, window // 2)
    market_variance = market_returns.rolling(window, min_periods=floor).var()
    safe_variance = market_variance.replace(0.0, np.nan)

    betas = {
        symbol: returns[symbol].rolling(window, min_periods=floor).cov(market_returns)
        / safe_variance
        for symbol in returns.columns
    }
    return pd.DataFrame(betas, index=returns.index)


# --------------------------------------------------------------------------
# Registered cross-sectional features
#
# The decorator applies the one-bar lag, so the bodies below take the panel's
# frames as already shifted and must not shift again. That is the difference
# between this module before and after T24: the lag used to be re-implemented
# by hand here (`idiosyncratic_vol_from_closes(..., lag=1)`) because there was
# no registry able to express a feature of the whole cross-section, and the
# only caller reached in and imported the function directly.
# --------------------------------------------------------------------------


#: Windows each market-relative feature is registered at. The window lives in
#: the *name*, which is the convention the per-ticker registry already
#: follows — `sma_20`, `sma_50`, `sma_200`, `realized_vol_60` all do exactly
#: this. It matters more here than there: a caller asking for
#: "idiosyncratic volatility" with an out-of-family window would otherwise get
#: the 60-session answer under a name claiming otherwise, and a sort measured
#: over the wrong window is not visibly wrong.
REGISTERED_WINDOWS = (20, 60, 120, 252)


def _panel_returns(panel: CrossSectionPanel) -> tuple:
    """Cross-section and market returns from an already-lagged panel."""
    returns = panel.get("close").pct_change()
    market = panel.benchmark.pct_change() if panel.benchmark is not None else None
    return returns, market


def _register_window_family() -> None:
    """Register both market-relative features at every window in the family.

    A loop rather than eight decorated functions, because the eight would be
    identical but for an integer and the platform has already been bitten once
    by an expression restated in a second place.
    """
    for window in REGISTERED_WINDOWS:

        def make_vol(w: int):
            def feature(panel: CrossSectionPanel) -> pd.DataFrame:
                returns, market = _panel_returns(panel)
                return rolling_idiosyncratic_vol(returns, market, window=w)

            feature.__name__ = f"idiosyncratic_vol_{w}"
            feature.__doc__ = (
                f"Annualized volatility of the CAPM residual over {w} sessions."
            )
            return feature

        def make_beta(w: int):
            def feature(panel: CrossSectionPanel) -> pd.DataFrame:
                returns, market = _panel_returns(panel)
                return rolling_beta(returns, market, window=w)

            feature.__name__ = f"market_beta_{w}"
            feature.__doc__ = (
                f"Rolling CAPM beta over {w} sessions, for sorting rather than "
                f"neutralizing."
            )
            return feature

        register_cross_sectional_feature(
            f"idiosyncratic_vol_{window}", inputs=("close",)
        )(make_vol(window))
        register_cross_sectional_feature(
            f"market_beta_{window}", inputs=("close",)
        )(make_beta(window))


_register_window_family()


def idiosyncratic_vol_feature(window: int) -> str:
    """Registry name for the residual-volatility feature at `window` sessions.

    Raises:
        ValueError: If no feature is registered at that window. Loud, because
            the alternative is ranking on a 60-session residual while the
            config says 120 — a sort measured over the wrong window looks
            exactly like a sort measured over the right one.
    """
    return _family_name("idiosyncratic_vol", window)


def market_beta_feature(window: int) -> str:
    """Registry name for the beta feature at `window` sessions.

    Raises:
        ValueError: If no feature is registered at that window.
    """
    return _family_name("market_beta", window)


def _family_name(stem: str, window: int) -> str:
    window = int(window)
    if window not in REGISTERED_WINDOWS:
        raise ValueError(
            f"No '{stem}' feature registered at a {window}-session window. "
            f"Available: {sorted(REGISTERED_WINDOWS)}. Add the window to "
            f"`REGISTERED_WINDOWS` rather than rounding to a neighbour — the "
            f"window is part of what the feature measures."
        )
    return f"{stem}_{window}"


# --------------------------------------------------------------------------
# Residual momentum
# --------------------------------------------------------------------------

#: Formation window in sessions (~9 months), and the skip (~1 month). Both
#: match `technical.mom_9m_skip1m` exactly, because the whole point of this
#: feature is to be the *same* momentum measured on the residual — a different
#: formation window would make the comparison between them a comparison of two
#: things at once.
RESIDUAL_FORMATION_DAYS = 189
RESIDUAL_SKIP_DAYS = 21

#: Sessions used to estimate the beta the residual is taken against. Longer
#: than the formation window on purpose, and the reason is a trap worth
#: recording: **if beta is fitted with an intercept over exactly the window the
#: residuals are then cumulated over, the cumulative residual is identically
#: zero** — OLS residuals sum to zero by construction. Blitz, Huij & Martens
#: avoid it by estimating over 36 months and forming over 12. Here the beta is
#: estimated on a rolling one-year window ending at each date and applied
#: without an intercept, so the residual carries the alpha rather than having
#: it differenced away.
RESIDUAL_BETA_WINDOW = 252


def residual_returns(
    returns: pd.DataFrame,
    market: Optional[pd.Series] = None,
    beta_window: int = RESIDUAL_BETA_WINDOW,
) -> pd.DataFrame:
    """Daily CAPM residual per symbol, against a rolling beta.

    `resid_t = r_t - beta_t * r_m,t`, where `beta_t` is estimated on the
    trailing `beta_window` ending at `t`. No intercept: the intercept is the
    alpha this feature exists to measure, and subtracting it would remove the
    signal along with the exposure.

    Args:
        returns: Wide (date x symbol) daily returns, already lagged if owed.
        market: Market returns. Defaults to the cross-section's own composite.
        beta_window: Sessions in the rolling beta estimate.

    Returns:
        Wide (date x symbol) residual returns, NaN until the beta window fills.
    """
    if returns.empty:
        return returns

    market_returns = market_composite(returns) if market is None else market
    market_returns = pd.Series(market_returns).reindex(returns.index)

    beta = rolling_beta(returns, market_returns, window=beta_window)
    return returns - beta.mul(market_returns, axis=0)


def residual_momentum(
    returns: pd.DataFrame,
    market: Optional[pd.Series] = None,
    formation: int = RESIDUAL_FORMATION_DAYS,
    skip: int = RESIDUAL_SKIP_DAYS,
    beta_window: int = RESIDUAL_BETA_WINDOW,
) -> pd.DataFrame:
    """Blitz-Huij-Martens residual momentum: the residual's information ratio.

    The ranking statistic is **standardized**, not the raw cumulative residual:

        RESMOM = mean(residual over formation) / std(residual over formation)

    That standardization is the substance of the effect rather than a tidying
    step. Raw cumulated residuals still rank high-residual-volatility names
    highest, which reintroduces exactly the risk exposure residualizing was
    meant to remove; dividing by the residual's own dispersion is what leaves a
    per-unit-of-risk statement. Blitz, Huij & Martens (2011) report roughly
    double the risk-adjusted profit of price momentum with materially shallower
    drawdowns, and attribute the difference to that.

    Round two measured this platform's own momentum at 58% factor loading,
    which is the exposure this removes.

    Args:
        returns: Wide (date x symbol) daily returns, already lagged if owed.
        market: Market returns. Defaults to the cross-section's own composite.
        formation: Sessions in the formation window.
        skip: Sessions skipped before the formation window ends, the
            Jegadeesh-Titman convention that keeps short-term reversal out of a
            momentum signal.
        beta_window: Sessions in the rolling beta estimate.

    Returns:
        Wide (date x symbol) standardized residual momentum.
    """
    residual = residual_returns(returns, market, beta_window=beta_window)
    if residual.empty:
        return residual

    floor = max(2, formation // 2)
    mean = residual.rolling(formation, min_periods=floor).mean()
    dispersion = residual.rolling(formation, min_periods=floor).std()

    # A name whose residual never moved has no risk-adjusted momentum to
    # report, and dividing by zero would rank it first or last depending only
    # on a floating-point sign.
    standardized = mean / dispersion.replace(0.0, np.nan)
    return standardized.shift(skip) if skip else standardized


@register_cross_sectional_feature("residual_momentum_9m_skip1m", inputs=("close",))
def _residual_momentum_9m_skip1m(panel: CrossSectionPanel) -> pd.DataFrame:
    """Standardized CAPM-residual momentum, 9-month formation skipping 1 month."""
    returns, market = _panel_returns(panel)
    return residual_momentum(returns, market)
