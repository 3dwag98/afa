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

**Cost.** The MLE fit is the expensive part, and the naive call graph runs one
per ticker per trading day: 3,612 usable tickers over 1,237 trading days is
~4.5 million fits, or 62-248 hours of pure optimizer time at a realistic
50-200 ms each. That is why `use_garch_volatility` defaulted to False — not
because the model is optional, but because the backtest did not terminate with
it on. `forecast_volatility_scheduled()` separates the two halves of the work:

- **Fitting** (omega, alpha, gamma, beta, nu) is an optimization, and the
  parameters are stable over weeks. It runs on a schedule.
- **Filtering and forecasting** is the GJR recursion, which is arithmetic. It
  runs every day, from the cached parameters and the returns realized since
  the fit.

At the default 21-bar refit interval that is ~21x fewer fits for a volatility
path that differs from the per-bar refit only in the parameters' vintage.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# GJR-GARCH MLE needs a reasonably long history for stable convergence.
MIN_OBSERVATIONS = 250

# Student-t variance nu/(nu-2) diverges as nu -> 2, so a fitted nu at or below
# this is not usable for generating standardized shocks.
MIN_STUDENT_T_DF = 2.1

# Refits closer together than this are not worth scheduling — the parameters
# will not have moved and the cache lookup costs more than it saves.
MIN_REFIT_INTERVAL_DAYS = 2

# A fitted persistence at or above 1 is a non-stationary process: the
# unconditional variance omega/(1 - persistence) is undefined and the h-step
# forecast diverges instead of mean-reverting. Such fits are rejected.
MAX_PERSISTENCE = 0.9999


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
    """A fitted GJR-GARCH(1,1) in decimal return space, without any state.

    Separated from GarchForecast because the two have very different lifetimes.
    A forecast is specific to one day and one horizon; the parameters are the
    expensive thing, they move on the scale of weeks, and they are what gets
    cached and reused across a refit interval.

    `arch` fits on percentage returns (1.2 rather than 0.012) because its
    optimizer converges more reliably there, so omega comes back in pct^2 and
    mu in pct. Both are converted here, once, so that everything downstream of
    this dataclass works in the decimal units the rest of the platform uses and
    no caller has to remember which space it is in.
    """

    mu: float  # constant mean of the return process, decimal
    omega: float  # variance intercept, decimal^2
    alpha: float  # ARCH coefficient
    gamma: float  # leverage coefficient; > 0 means negative shocks add variance
    beta: float  # GARCH coefficient
    distribution_df: Optional[float]  # fitted Student-t nu, or None if unusable
    n_observations: int  # length of the window the fit was estimated on

    @property
    def persistence(self) -> float:
        """alpha + gamma/2 + beta; must be < 1 for a stationary process.

        gamma is halved because the leverage term only fires on negative
        shocks, which under a symmetric innovation distribution is half the
        time — so its expected contribution to next period's variance is
        gamma/2, not gamma.
        """
        return self.alpha + self.gamma / 2.0 + self.beta

    @property
    def unconditional_variance(self) -> float:
        """Long-run variance omega/(1 - persistence), the forecast's attractor."""
        slack = 1.0 - self.persistence
        if slack <= 0:
            return float("inf")
        return self.omega / slack


def fit_garch_parameters(daily_returns: Sequence[float]) -> Optional[GarchParameters]:
    """Fit GJR-GARCH(1,1) with Student-t innovations by maximum likelihood.

    This is the expensive half of the model — the part worth scheduling. It
    returns parameters only; turning them into a volatility path is
    forecast_from_parameters()'s job and costs nothing.

    Args:
        daily_returns: Historical simple daily returns.

    Returns:
        Fitted GarchParameters, or None when there isn't enough history, the
        `arch` package isn't installed, the optimizer fails, or the fit lands
        on a non-stationary process.
    """
    try:
        from arch import arch_model
    except ImportError:
        logger.debug("`arch` package not installed; GARCH volatility forecast unavailable")
        return None

    returns = _clean(daily_returns)
    if len(returns) < MIN_OBSERVATIONS:
        return None

    try:
        model = arch_model(returns * 100.0, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t")
        result = model.fit(disp="off", show_warning=False)
        params = result.params

        # omega is a variance in pct^2 and mu a mean in pct; alpha, gamma and
        # beta are ratios of variances and so are scale-free.
        parameters = GarchParameters(
            mu=float(params.get("mu", 0.0)) / 100.0,
            omega=float(params.get("omega", 0.0)) / 10_000.0,
            alpha=float(params.get("alpha[1]", 0.0)),
            gamma=float(params.get("gamma[1]", 0.0)),
            beta=float(params.get("beta[1]", 0.0)),
            distribution_df=_usable_df(params.get("nu")),
            n_observations=len(returns),
        )
    except Exception:
        logger.debug("GJR-GARCH fit failed", exc_info=True)
        return None

    if parameters.omega <= 0 or parameters.persistence >= MAX_PERSISTENCE:
        # Either the variance process has no intercept to revert to, or it is
        # non-stationary and the h-step forecast would diverge rather than
        # mean-revert. Neither is usable; the caller falls back.
        return None
    return parameters


def filter_conditional_variance(
    parameters: GarchParameters,
    daily_returns: Sequence[float],
) -> Optional[Tuple[float, float]]:
    """Run the GJR recursion over a return series and return its end state.

    This is what makes scheduled refitting exact rather than approximate. The
    parameters may be up to `refit_interval_days` old, but the *state* they are
    applied to is current: every return realized since the fit is pushed
    through the recursion, so the sigma^2 the forecast starts from is the one
    conditioned on today's information.

        sigma^2_t = omega + alpha*eps^2_{t-1} + gamma*eps^2_{t-1}*1[eps_{t-1}<0]
                    + beta*sigma^2_{t-1}

    The recursion is seeded at the unconditional variance, which is the
    standard choice and the one `arch` itself uses as a backcast: any error in
    the seed decays geometrically at rate `persistence` and is negligible after
    a few dozen observations.

    Args:
        parameters: Fitted parameters, in decimal space.
        daily_returns: The full return series to filter, oldest first.

    Returns:
        (sigma^2 for the next day, eps for the last observed day), or None if
        the series is empty or the recursion leaves the finite range.
    """
    returns = _clean(daily_returns)
    if len(returns) == 0:
        return None

    residuals = returns - parameters.mu
    squared = residuals * residuals

    sigma2 = parameters.unconditional_variance
    if not np.isfinite(sigma2) or sigma2 <= 0:
        sigma2 = float(np.var(residuals)) or parameters.omega

    # A Python loop, deliberately: the recursion is sequential in sigma^2 and
    # cannot be vectorized. It runs over `refit_interval_days` observations in
    # the scheduled path (21 by default), which is nothing next to one MLE fit.
    for i in range(len(returns)):
        sigma2 = (
            parameters.omega
            + (parameters.alpha + (parameters.gamma if residuals[i] < 0 else 0.0)) * squared[i]
            + parameters.beta * sigma2
        )
        if not math.isfinite(sigma2) or sigma2 <= 0:
            return None

    return float(sigma2), float(residuals[-1])


def forecast_from_parameters(
    parameters: GarchParameters,
    daily_returns: Sequence[float],
    horizon_days: int,
) -> Optional[GarchForecast]:
    """Project the variance path forward from cached parameters. No fitting.

    One step ahead the variance is known exactly, because yesterday's shock is
    observed:

        sigma^2_{T+1} = omega + (alpha + gamma*1[eps_T < 0])*eps^2_T + beta*sigma^2_T

    Beyond that the shock's sign is unknown, so the leverage term contributes
    its expectation — gamma/2 under a symmetric innovation distribution — and
    the recursion becomes the familiar mean-reverting form:

        E[sigma^2_{T+h}] = omega + persistence * E[sigma^2_{T+h-1}]

    which decays toward omega/(1 - persistence) at rate `persistence`.

    Args:
        parameters: Fitted parameters, possibly from an earlier refit.
        daily_returns: Returns through today, used to bring the recursion's
            state up to date (see filter_conditional_variance).
        horizon_days: Number of trading days to forecast.

    Returns:
        A GarchForecast, or None if the state could not be filtered or the
        projected path is not usable.
    """
    if horizon_days <= 0:
        return None

    state = filter_conditional_variance(parameters, daily_returns)
    if state is None:
        return None
    sigma2_next, last_residual = state

    variances = np.empty(horizon_days, dtype=float)
    variances[0] = sigma2_next
    persistence = parameters.persistence
    for h in range(1, horizon_days):
        variances[h] = parameters.omega + persistence * variances[h - 1]

    daily_sigma = np.sqrt(variances)
    if not np.all(np.isfinite(daily_sigma)) or np.any(daily_sigma <= 0):
        return None

    # last_residual is not returned, but computing it is what makes the first
    # forecast step conditional rather than an unconditional guess; naming it
    # keeps that visible at the call site above.
    del last_residual

    return GarchForecast(
        daily_sigma=daily_sigma,
        leverage_gamma=parameters.gamma,
        persistence=persistence,
        distribution_df=parameters.distribution_df,
    )


# Fitted parameters, keyed by (symbol, series kind, fit anchor). Process-local:
# scoring is dispatched to worker processes, each of which sees the same ticker
# on many consecutive days, so a per-process cache captures essentially all of
# the available reuse without any cross-process coordination.
_PARAMETER_CACHE: Dict[Tuple[str, str, int], Optional[GarchParameters]] = {}
_PARAMETER_CACHE_LOCK = threading.Lock()

# Bound on the cache. Each entry is seven floats; the limit exists to stop an
# unbounded universe from growing the dict without end, not to save memory.
MAX_CACHED_PARAMETERS = 20_000


def clear_garch_parameter_cache() -> None:
    """Drop every cached fit. For tests, and for callers changing data sets."""
    with _PARAMETER_CACHE_LOCK:
        _PARAMETER_CACHE.clear()


def garch_parameter_cache_size() -> int:
    """Number of cached fits, including cached failures. For tests."""
    with _PARAMETER_CACHE_LOCK:
        return len(_PARAMETER_CACHE)


def _fit_anchor(n_observations: int, refit_interval_days: int) -> Optional[int]:
    """Length of the window the next scheduled fit should use.

    The anchor is `(n // interval) * interval` — the fit window is *truncated*
    to a multiple of the interval rather than being whatever happened to be
    available when the first call landed. That matters more than it looks:

    - It makes the cached parameters a pure function of (symbol, anchor), so
      the result does not depend on which worker process saw the ticker first
      or in what order the days were dispatched. The platform enforces
      determinism by test (test_parallel_determinism.py), and a cache keyed on
      call order would break it.
    - It makes the refit boundaries the same for every ticker and every run, so
      two backtests over the same window fit on the same days.

    Returns None when there is not enough history to fit at all.
    """
    if n_observations < MIN_OBSERVATIONS:
        return None
    anchor = (n_observations // refit_interval_days) * refit_interval_days
    if anchor < MIN_OBSERVATIONS:
        # Early in the sample the scheduled anchor falls below the fitting
        # minimum. Use everything available rather than declining to fit; the
        # schedule takes over as soon as the next boundary clears the minimum.
        return n_observations
    return anchor


def _cached_parameters(
    symbol: str,
    kind: str,
    returns: np.ndarray,
    refit_interval_days: int,
) -> Optional[GarchParameters]:
    """Fitted parameters for this symbol, refitting only on schedule."""
    anchor = _fit_anchor(len(returns), refit_interval_days)
    if anchor is None:
        return None

    key = (symbol, kind, anchor)
    with _PARAMETER_CACHE_LOCK:
        if key in _PARAMETER_CACHE:
            return _PARAMETER_CACHE[key]

    # Fitted outside the lock: an MLE fit takes 50-200 ms and holding the lock
    # across it would serialize every worker thread behind one ticker. The
    # worst case is two threads fitting the same window concurrently, which
    # wastes one fit and stores the same answer twice.
    parameters = fit_garch_parameters(returns[:anchor])

    with _PARAMETER_CACHE_LOCK:
        if len(_PARAMETER_CACHE) >= MAX_CACHED_PARAMETERS:
            _PARAMETER_CACHE.clear()
        # A failed fit is cached too. Without that, a ticker whose window never
        # converges pays the full optimizer cost every single day — the exact
        # cost this function exists to avoid, concentrated on the worst names.
        _PARAMETER_CACHE[key] = parameters
    return parameters


def forecast_volatility_scheduled(
    daily_returns: Sequence[float],
    horizon_days: int,
    symbol: str,
    refit_interval_days: int,
) -> Optional[GarchForecast]:
    """forecast_volatility(), but refitting the parameters only periodically.

    Identical in output to forecast_volatility() on a refit day, and on other
    days it differs only in that the parameters are up to
    `refit_interval_days - 1` bars old while the conditional variance state is
    fully current. GARCH parameters move on the scale of weeks and the state
    moves daily, so this preserves what actually drives the forecast.

    Args:
        daily_returns: Historical simple daily returns, oldest first, from a
            fixed start (an expanding window — the anchor arithmetic assumes
            observation i means the same thing on consecutive calls).
        horizon_days: Number of trading days ahead to forecast.
        symbol: Cache key. Two different series under one symbol will share
            cached parameters, so callers must not reuse a symbol for
            unrelated data.
        refit_interval_days: Bars between refits. Values below
            MIN_REFIT_INTERVAL_DAYS fit every call, which is
            forecast_volatility()'s behaviour.

    Returns:
        A GarchForecast, or None on the same conditions as forecast_volatility.
    """
    if refit_interval_days < MIN_REFIT_INTERVAL_DAYS:
        return forecast_volatility(daily_returns, horizon_days)

    returns = _clean(daily_returns)
    parameters = _cached_parameters(symbol, "close", returns, refit_interval_days)
    if parameters is None:
        return None
    return forecast_from_parameters(parameters, returns, horizon_days)


def forecast_volatility_gap_aware_scheduled(
    intraday_returns: Sequence[float],
    overnight_returns: Sequence[float],
    horizon_days: int,
    symbol: str,
    refit_interval_days: int,
) -> Optional[GarchForecast]:
    """forecast_volatility_gap_aware(), refitting the session leg on schedule.

    Only the session leg is scheduled, because only the session leg is fitted.
    The gap component is an unconditional standard deviation — one pass over
    the series, no optimizer — so it is recomputed every call and stays fully
    current at no cost.
    """
    if refit_interval_days < MIN_REFIT_INTERVAL_DAYS:
        return forecast_volatility_gap_aware(intraday_returns, overnight_returns, horizon_days)

    session_returns = _clean(intraday_returns)
    parameters = _cached_parameters(symbol, "session", session_returns, refit_interval_days)
    if parameters is None:
        return None

    session = forecast_from_parameters(parameters, session_returns, horizon_days)
    if session is None:
        return None
    return _combine_session_and_gap(session, overnight_returns)


def _clean(values: Sequence[float]) -> np.ndarray:
    """Drop NaN and infinite observations, as a float array."""
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _usable_df(nu) -> Optional[float]:
    """Fitted Student-t nu, or None when its variance nu/(nu-2) is not finite.

    A fit that lands at or below 2 degrees of freedom is unusable as a
    unit-variance shock distribution, so it is reported as None and the caller
    draws Gaussian shocks rather than something with undefined variance.
    """
    if nu is None:
        return None
    value = float(nu)
    return value if value > MIN_STUDENT_T_DF else None


def forecast_volatility(
    daily_returns: Sequence[float],
    horizon_days: int,
) -> Optional[GarchForecast]:
    """Fit a GJR-GARCH(1,1) with Student-t innovations and forecast volatility.

    Args:
        daily_returns: Historical simple daily returns.
        horizon_days: Number of trading days ahead to forecast.

    Returns:
        A GarchForecast, or None if there isn't enough history to fit
        reliably, the `arch` package isn't installed, or the fit didn't
        converge — callers should fall back to a constant-volatility
        assumption in that case rather than treating this as fatal.
    """
    try:
        from arch import arch_model
    except ImportError:
        logger.debug("`arch` package not installed; GARCH volatility forecast unavailable")
        return None

    returns = _clean(daily_returns)
    if len(returns) < MIN_OBSERVATIONS:
        return None

    # arch's optimizer converges more reliably on returns scaled to
    # percentage points rather than raw decimals (e.g. 1.2 instead of 0.012).
    returns_pct = returns * 100.0

    try:
        model = arch_model(returns_pct, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t")
        result = model.fit(disp="off", show_warning=False)

        forecast = result.forecast(horizon=horizon_days, reindex=False)
        variance_pct2 = np.asarray(forecast.variance.values[-1], dtype=float)  # (horizon_days,), in pct^2
        daily_sigma = np.sqrt(variance_pct2) / 100.0

        params = result.params
        alpha = float(params.get("alpha[1]", 0.0))
        gamma = float(params.get("gamma[1]", 0.0))
        beta = float(params.get("beta[1]", 0.0))
        persistence = alpha + gamma / 2.0 + beta

        if not np.all(np.isfinite(daily_sigma)) or np.any(daily_sigma <= 0):
            return None

        return GarchForecast(
            daily_sigma=daily_sigma,
            leverage_gamma=gamma,
            persistence=persistence,
            distribution_df=_usable_df(params.get("nu")),
        )
    except Exception:
        logger.debug("GJR-GARCH fit failed; falling back to constant volatility", exc_info=True)
        return None


def forecast_volatility_gap_aware(
    intraday_returns: Sequence[float],
    overnight_returns: Sequence[float],
    horizon_days: int,
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

    Returns:
        A GarchForecast whose daily_sigma includes gap risk, or None when the
        session fit is unavailable — callers should fall back to the
        close-to-close path (or constant volatility) in that case.
    """
    session = forecast_volatility(intraday_returns, horizon_days)
    if session is None:
        return None
    return _combine_session_and_gap(session, overnight_returns)


def _combine_session_and_gap(
    session: GarchForecast,
    overnight_returns: Sequence[float],
) -> Optional[GarchForecast]:
    """Add unconditional overnight gap variance to a session-only forecast.

    Shared by the per-call and scheduled gap-aware paths so the two cannot
    drift apart in how they combine the legs.
    """
    gaps = _clean(overnight_returns)
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
