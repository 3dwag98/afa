"""GJR-GARCH(1,1) conditional volatility forecasting.

Motivation (see docs/QUANT_RESEARCH.md section 3): the platform's Monte Carlo
simulation (monte_carlo.py) originally assumed *constant* volatility — a
single historical standard deviation applied to every simulated day. Indian
equity research (Nifty 50 / Sensex) consistently finds volatility clustering
and an asymmetric leverage effect (negative shocks raise future volatility
more than positive shocks of the same size). GJR-GARCH(1,1) with Student-t
innovations captures both directly:

    r_t = mu + eps_t,  eps_t = sigma_t * z_t,  z_t ~ Student-t(nu)
    sigma_t^2 = omega + alpha*eps_{t-1}^2 + gamma*eps_{t-1}^2*1[eps_{t-1}<0] + beta*sigma_{t-1}^2

gamma > 0 confirms the leverage effect (negative shocks get extra variance).
Stationarity requires alpha + gamma/2 + beta < 1.

This is an optional enhancement: whenever there isn't enough history to fit
reliably, or the `arch` package isn't installed, or the fit fails to
converge, callers fall back to the existing constant-volatility assumption.

**On cost.** A naive integration fits one MLE per ticker per bar. The
documented backtest is 3,612 tickers x 1,237 trading days = ~4.5 million fits,
and at a realistic 50-200 ms per fit that is 62-248 hours of pure optimizer
time before a single path is simulated. That is why `use_garch_volatility`
defaulted to False: not because the model is optional, but because it could
not terminate. Refit scheduling (see `forecast_volatility`'s
`refit_interval_days`) is what makes it usable — GARCH *parameters* are stable
over weeks, so they are re-estimated on a schedule and the variance
*recursion* is run forward daily from the cached ones, which is arithmetic
rather than optimization.
"""

from __future__ import annotations

import logging
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# GJR-GARCH MLE needs a reasonably long history for stable convergence.
MIN_OBSERVATIONS = 250

# Student-t variance nu/(nu-2) diverges as nu -> 2, so a fitted nu at or below
# this is not usable for generating standardized shocks.
MIN_STUDENT_T_DF = 2.1

# Default bars between refits. GARCH parameters move on the scale of months,
# not days; 21 is one trading month and cuts the fit count ~21x.
DEFAULT_REFIT_INTERVAL_DAYS = 21

# Cached parameter sets are ~7 floats each, so this bound is about memory
# hygiene in a long-lived process rather than a real constraint.
_PARAMETER_CACHE_MAX_ENTRIES = 4096


@dataclass
class GarchForecast:
    """Forecasted daily volatility path from a fitted GJR-GARCH(1,1) model."""

    daily_sigma: np.ndarray  # shape (horizon_days,); forecasted daily return std, decimal (not %)
    leverage_gamma: float  # fitted asymmetry coefficient; > 0 confirms the leverage effect
    persistence: float  # alpha + gamma/2 + beta; must be < 1 for a stationary process
    # Fitted Student-t degrees of freedom (nu). This is the *whole point* of
    # dist="t" and has to reach the simulation: fitting fat-tailed innovations
    # and then drawing Gaussian shocks throws the tail estimate away and leaves
    # VaR/CVaR quietly optimistic. Lower nu = fatter tails; None when the fit
    # did not produce a usable value.
    distribution_df: Optional[float] = None
    # Set when the forecast was built from a gap-aware decomposition
    # (see forecast_volatility_gap_aware): the constant overnight-gap standard
    # deviation folded into daily_sigma, and the session-only GARCH path.
    overnight_sigma: Optional[float] = None
    intraday_sigma: Optional[np.ndarray] = None

    @property
    def gap_aware(self) -> bool:
        """Whether overnight gap risk was modelled separately from the session."""
        return self.overnight_sigma is not None


@dataclass(frozen=True)
class GarchParameters:
    """A fitted GJR-GARCH(1,1) and the recursion state at the end of the fit.

    Everything needed to carry the variance recursion forward past the fitted
    window without refitting. All variance-scale quantities are in percentage
    points squared, matching the units the model is estimated in.
    """

    mu: float  # constant mean, pct
    omega: float  # variance intercept, pct^2
    alpha: float
    gamma: float  # leverage term; > 0 confirms the asymmetry
    beta: float
    distribution_df: Optional[float]  # fitted Student-t nu, when usable
    terminal_variance: float  # sigma^2 of the last fitted observation, pct^2
    terminal_resid: float  # eps of the last fitted observation, pct
    n_fitted: int  # observations the MLE consumed

    @property
    def persistence(self) -> float:
        """alpha + gamma/2 + beta; must be < 1 for a stationary process."""
        return self.alpha + self.gamma / 2.0 + self.beta


# Memoized fits, keyed by (cache_key, anchor, checksum) — see _fit_anchor for
# why the key cannot be a bare "last time we fitted this symbol".
_PARAMETER_CACHE: "OrderedDict[tuple, Optional[GarchParameters]]" = OrderedDict()


def clear_parameter_cache() -> None:
    """Drop every memoized fit. Exposed for tests and long-lived processes."""
    _PARAMETER_CACHE.clear()


def _fit_anchor(n_observations: int, refit_interval_days: int) -> int:
    """Number of observations the fit should consume, given a history length.

    The anchor is a **pure function of the history length**, deliberately, and
    that is the whole design. The obvious scheduling implementation — "refit if
    more than N bars have passed since this symbol was last fitted" — makes the
    result depend on the order calls happen to arrive in. The backtest and the
    orchestrator both dispatch per-ticker scoring to worker processes, so each
    worker would hold its own cache, hit different refit points, and produce
    different volatility paths for the same ticker on the same date. That is a
    determinism break, and `test_parallel_determinism.py` exists because this
    class of bug is otherwise invisible.

    Anchoring to `floor(n / interval) * interval` instead means any process
    seeing the same history computes the same fitted window and filters the
    same 0..interval-1 observations forward. The cache becomes pure
    memoization: it changes how long the answer takes, never what it is.
    """
    if refit_interval_days <= 1:
        return n_observations
    return (n_observations // refit_interval_days) * refit_interval_days


def _fit_garch_parameters(returns_pct: np.ndarray) -> Optional[GarchParameters]:
    """Estimate GJR-GARCH(1,1) by MLE. This is the expensive call."""
    try:
        from arch import arch_model
    except ImportError:
        logger.debug("`arch` package not installed; GARCH volatility forecast unavailable")
        return None

    try:
        model = arch_model(returns_pct, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t")
        result = model.fit(disp="off", show_warning=False)
        params = result.params

        # Student-t variance is nu/(nu-2), which is only finite for nu > 2.
        # A fit that lands at or below that is unusable as a shock
        # distribution, so it is reported as None and the caller draws
        # Gaussian shocks rather than something with undefined variance.
        nu = params.get("nu")
        distribution_df = float(nu) if nu is not None and float(nu) > MIN_STUDENT_T_DF else None

        # The recursion has to resume from where the fit left off, so the last
        # in-sample conditional variance and residual are carried out with the
        # coefficients. Taking them from the fit rather than re-filtering the
        # whole window from an assumed starting variance keeps the handoff
        # exact instead of approximately right.
        conditional_vol = np.asarray(result.conditional_volatility, dtype=float)
        resid = np.asarray(result.resid, dtype=float)
        if conditional_vol.size == 0 or resid.size == 0:
            return None

        fitted = GarchParameters(
            mu=float(params.get("mu", 0.0)),
            omega=float(params.get("omega", 0.0)),
            alpha=float(params.get("alpha[1]", 0.0)),
            gamma=float(params.get("gamma[1]", 0.0)),
            beta=float(params.get("beta[1]", 0.0)),
            distribution_df=distribution_df,
            terminal_variance=float(conditional_vol[-1] ** 2),
            terminal_resid=float(resid[-1]),
            n_fitted=int(returns_pct.size),
        )
        if not np.isfinite(fitted.terminal_variance) or fitted.terminal_variance <= 0:
            return None
        if not np.isfinite(fitted.terminal_resid) or not np.isfinite(fitted.omega):
            return None
        return fitted
    except Exception:
        logger.debug("GJR-GARCH fit failed; falling back to constant volatility", exc_info=True)
        return None


# Distinguishes "not cached" from "cached a failed fit", which is a real
# distinction here: a failure is cached deliberately (see below).
_MISSING = object()


def _cached_fit(
    returns_pct: np.ndarray,
    cache_key: Optional[str],
) -> Optional[GarchParameters]:
    """Fit, or return the memoized fit for this exact window.

    The checksum is over the fitted window itself, so a cache entry can only
    be reused for identical data — a re-download or a corrected bar
    invalidates it rather than silently serving a fit of the old prices.
    """
    if cache_key is None:
        return _fit_garch_parameters(returns_pct)

    checksum = zlib.crc32(np.ascontiguousarray(returns_pct, dtype=np.float64).tobytes())
    key = (cache_key, int(returns_pct.size), checksum)

    cached = _PARAMETER_CACHE.get(key, _MISSING)
    if cached is not _MISSING:
        _PARAMETER_CACHE.move_to_end(key)
        return cached

    fitted = _fit_garch_parameters(returns_pct)
    # A failed fit is cached too. Failures are a property of the window, not
    # of the attempt, so retrying it every bar re-pays the full optimizer cost
    # to be told the same thing.
    _PARAMETER_CACHE[key] = fitted
    if len(_PARAMETER_CACHE) > _PARAMETER_CACHE_MAX_ENTRIES:
        _PARAMETER_CACHE.popitem(last=False)
    return fitted


def _forecast_from_parameters(
    params: GarchParameters,
    trailing_returns_pct: np.ndarray,
    horizon_days: int,
) -> Optional[np.ndarray]:
    """Filter past the fitted window, then project the variance path forward.

    Two distinct steps, both closed-form:

    1. **Filtering.** For each observation after the fit, advance the
       recursion one bar. This is what replaces refitting: new data still
       updates the conditional variance, it just does not re-estimate the
       coefficients.

           sigma^2_t = omega + (alpha + gamma*1[eps_{t-1} < 0]) * eps^2_{t-1}
                       + beta * sigma^2_{t-1}

    2. **Forecasting.** One step ahead the sign of the last residual is known,
       so the leverage term is applied exactly. Beyond that it is unknown and
       symmetric standardized innovations put it below zero half the time, so
       the expectation carries gamma/2 — the same persistence that governs
       stationarity:

           E[sigma^2_{t+h}] = omega + (alpha + gamma/2 + beta) * E[sigma^2_{t+h-1}]

    Returns daily sigma in decimals, or None if the recursion leaves the
    finite positive domain (a non-stationary fit will diverge).
    """
    if horizon_days < 1:
        return None

    variance = params.terminal_variance
    resid = params.terminal_resid

    for observed in trailing_returns_pct:
        leverage = params.gamma if resid < 0 else 0.0
        variance = params.omega + (params.alpha + leverage) * resid ** 2 + params.beta * variance
        resid = float(observed) - params.mu
        if not np.isfinite(variance) or variance <= 0:
            return None

    path = np.empty(horizon_days, dtype=float)
    leverage = params.gamma if resid < 0 else 0.0
    variance = params.omega + (params.alpha + leverage) * resid ** 2 + params.beta * variance
    path[0] = variance
    persistence = params.persistence
    for h in range(1, horizon_days):
        variance = params.omega + persistence * variance
        path[h] = variance

    if not np.all(np.isfinite(path)) or np.any(path <= 0):
        return None
    return np.sqrt(path) / 100.0


def forecast_volatility(
    daily_returns: Sequence[float],
    horizon_days: int,
    refit_interval_days: int = 1,
    cache_key: Optional[str] = None,
) -> Optional[GarchForecast]:
    """Fit a GJR-GARCH(1,1) with Student-t innovations and forecast volatility.

    Args:
        daily_returns: Historical simple daily returns.
        horizon_days: Number of trading days ahead to forecast.
        refit_interval_days: Bars between MLE refits. 1 (the default) refits on
            every call, which is the original behaviour and the only correct
            choice for a one-off forecast. Larger values re-estimate the
            coefficients on a schedule and carry the variance recursion forward
            arithmetically in between — the difference between a backtest that
            terminates and one that does not (see the module docstring).
        cache_key: Identity to memoize the fit under, normally the ticker
            symbol. Required for `refit_interval_days` to save anything: the
            schedule decides *which window* to fit, the cache is what stops it
            being refitted for every bar in that window. Omitted, every call
            fits.

    Returns:
        A GarchForecast, or None if there isn't enough history to fit
        reliably, the `arch` package isn't installed, or the fit didn't
        converge — callers should fall back to a constant-volatility
        assumption in that case rather than treating this as fatal.
    """
    returns = np.asarray(daily_returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) < MIN_OBSERVATIONS:
        return None

    # arch's optimizer converges more reliably on returns scaled to
    # percentage points rather than raw decimals (e.g. 1.2 instead of 0.012).
    returns_pct = returns * 100.0

    anchor = _fit_anchor(len(returns_pct), max(1, int(refit_interval_days)))
    if anchor < MIN_OBSERVATIONS:
        # Rounding the window down landed below the length the fit needs. The
        # alternative — quietly fitting the full history instead — would make
        # the fitted window depend on where the anchor happened to fall, so
        # the same ticker would switch estimators mid-backtest.
        return None

    params = _cached_fit(returns_pct[:anchor], cache_key)
    if params is None:
        return None

    daily_sigma = _forecast_from_parameters(params, returns_pct[anchor:], horizon_days)
    if daily_sigma is None:
        return None

    return GarchForecast(
        daily_sigma=daily_sigma,
        leverage_gamma=params.gamma,
        persistence=params.persistence,
        distribution_df=params.distribution_df,
    )


def forecast_volatility_gap_aware(
    intraday_returns: Sequence[float],
    overnight_returns: Sequence[float],
    horizon_days: int,
    refit_interval_days: int = 1,
    cache_key: Optional[str] = None,
) -> Optional[GarchForecast]:
    """Fit GJR-GARCH to the trading session only, and add gap risk separately.

    Close-to-close returns bundle two processes with different dynamics, and
    GARCH is a model of only one of them (docs/QUANT_RESEARCH.md section 16):

    - The **overnight gap** (open_t / close_{t-1} - 1) reprices global cues,
      FII decisions and policy news while the market is shut. It arrives as a
      single instantaneous jump; there is no within-gap conditional variance
      process for the recursion to track.
    - The **intraday session** (close_t / open_t - 1) is the continuous
      trading that the GARCH recursion actually describes.

    Feeding close-to-close returns to GARCH makes it attribute every gap to
    yesterday's session shock, which inflates alpha and gamma, drags the
    persistence estimate toward (and sometimes past) the stationarity bound,
    and makes the fitted parameters unstable across refits — precisely the
    failure mode Indian equities provoke, since NSE opens after both the US
    close and the Asian session.

    Modelling them apart: sigma_intraday follows GJR-GARCH, sigma_gap is
    estimated as the unconditional standard deviation of the gap series, and
    the two are combined as independent components of the daily move:

        sigma_daily^2 = sigma_intraday^2 + sigma_gap^2

    Independence is the standard simplification and a conservative one here —
    a positive correlation between gaps and the sessions that follow them
    would widen the total, so the combined estimate does not overstate risk.

    Args:
        intraday_returns: Session returns (close/open - 1).
        overnight_returns: Gap returns (open/prev_close - 1), same length.
        horizon_days: Number of trading days ahead to forecast.
        refit_interval_days: Bars between MLE refits of the session model; see
            forecast_volatility. The gap leg is an unconditional standard
            deviation, which is arithmetic, so it is recomputed every call
            regardless — only the session fit is expensive.
        cache_key: Identity to memoize the session fit under. Namespaced away
            from the close-to-close fit for the same symbol, which is a
            different model of different data and must not share an entry.

    Returns:
        A GarchForecast whose daily_sigma includes gap risk, or None when the
        session fit is unavailable — callers should fall back to the
        close-to-close path (or constant volatility) in that case.
    """
    session = forecast_volatility(
        intraday_returns,
        horizon_days,
        refit_interval_days=refit_interval_days,
        cache_key=f"{cache_key}::intraday" if cache_key is not None else None,
    )
    if session is None:
        return None

    gaps = np.asarray(overnight_returns, dtype=float)
    gaps = gaps[np.isfinite(gaps)]
    if len(gaps) < MIN_OBSERVATIONS:
        return None

    gap_sigma = float(np.std(gaps, ddof=1))
    if not np.isfinite(gap_sigma) or gap_sigma < 0:
        return None

    combined = np.sqrt(session.daily_sigma ** 2 + gap_sigma ** 2)
    if not np.all(np.isfinite(combined)) or np.any(combined <= 0):
        return None

    return GarchForecast(
        daily_sigma=combined,
        leverage_gamma=session.leverage_gamma,
        persistence=session.persistence,
        distribution_df=session.distribution_df,
        overnight_sigma=gap_sigma,
        intraday_sigma=session.daily_sigma,
    )
