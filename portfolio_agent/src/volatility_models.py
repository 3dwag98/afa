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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# GJR-GARCH MLE needs a reasonably long history for stable convergence.
MIN_OBSERVATIONS = 250

# How many new observations may accumulate before the MLE is re-run.
#
# This is the difference between GARCH being usable at this platform's scale
# and not. The call graph is backtest_engine._score_ticker -> MonteCarloSettings
# .run() -> run_monte_carlo_garch() -> forecast_volatility() -> arch_model().fit(),
# i.e. one maximum-likelihood fit per ticker per trading day. For the
# documented backtest -- ~3,600 usable tickers over ~1,240 trading days --
# that is roughly 4.5 MILLION MLE fits, or 60-250 hours of pure optimizer time
# before any simulation runs. `use_garch_volatility: false` was not a
# considered default; it was the only setting under which the backtest
# terminated, which made the most sophisticated component in the repository
# dead code.
#
# GARCH *parameters* are stable over weeks; the conditional variance is not.
# So the parameters are fitted monthly and the recursion -- which is
# arithmetic, not optimization -- is run forward daily from them. That is a
# ~21x reduction in fits for an estimate that barely moves.
GARCH_REFIT_INTERVAL = 21

# Student-t variance nu/(nu-2) diverges as nu -> 2, so a fitted nu at or below
# this is not usable for generating standardized shocks.
MIN_STUDENT_T_DF = 2.1


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
    """A fitted GJR-GARCH(1,1), in percentage-point units.

    Separated from GarchForecast because the two have very different
    lifetimes: these parameters are stable over weeks and are cached, while
    the conditional variance path they generate has to be recomputed from the
    latest bar every single day.
    """

    mu: float           # constant mean, in percentage points
    omega: float        # variance intercept, in pct^2
    alpha: float        # ARCH coefficient
    gamma: float        # leverage (asymmetry) coefficient
    beta: float         # GARCH coefficient
    distribution_df: Optional[float]
    fitted_at_observations: int

    @property
    def persistence(self) -> float:
        """alpha + gamma/2 + beta; must be < 1 for a stationary process."""
        return self.alpha + self.gamma / 2.0 + self.beta


# Fitted parameters by cache key, e.g. a ticker symbol. Process-local: each
# scoring worker keeps its own, which is fine — the cache is an optimization,
# not shared state, and a cold worker simply refits once.
_PARAMETER_CACHE: Dict[str, GarchParameters] = {}


def clear_garch_parameter_cache() -> None:
    """Drop every cached fit. Used by tests and between backtest runs."""
    _PARAMETER_CACHE.clear()


def fit_garch_parameters(returns_pct: np.ndarray) -> Optional[GarchParameters]:
    """Run the GJR-GARCH(1,1) MLE. This is the expensive call.

    Args:
        returns_pct: Clean daily returns in PERCENTAGE POINTS (1.2, not 0.012)
            — arch's optimizer converges far more reliably at that scale.

    Returns:
        Fitted parameters, or None when `arch` is unavailable or the fit did
        not converge.
    """
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

        omega = float(params.get("omega", 0.0))
        if not np.isfinite(omega) or omega <= 0:
            return None

        return GarchParameters(
            mu=float(params.get("mu", 0.0)),
            omega=omega,
            alpha=float(params.get("alpha[1]", 0.0)),
            gamma=float(params.get("gamma[1]", 0.0)),
            beta=float(params.get("beta[1]", 0.0)),
            distribution_df=distribution_df,
            fitted_at_observations=int(len(returns_pct)),
        )
    except Exception:
        logger.debug("GJR-GARCH fit failed; falling back to constant volatility", exc_info=True)
        return None


def forecast_from_parameters(
    parameters: GarchParameters,
    returns_pct: np.ndarray,
    horizon_days: int,
) -> Optional[np.ndarray]:
    """Run the GJR recursion forward from cached parameters. No optimization.

    Two steps, both arithmetic:

    1. **Filter.** Replay the recursion over the whole history to recover
       today's conditional variance and today's residual:

           sigma^2_t = omega + alpha*e^2_{t-1} + gamma*e^2_{t-1}*1[e_{t-1}<0]
                       + beta*sigma^2_{t-1}

       This step is what has to happen every day even when the parameters do
       not: sigma^2_T depends on the latest bar, which is the entire reason
       for using a conditional volatility model.

    2. **Project.** One asymmetric step, then the symmetric recursion:

           E[sigma^2_{T+1}] = omega + (alpha + gamma*1[e_T<0])*e_T^2 + beta*sigma^2_T
           E[sigma^2_{T+h}] = omega + (alpha + gamma/2 + beta) * E[sigma^2_{T+h-1}]

       The gamma/2 for h >= 2 is E[1{e < 0}] under a symmetric innovation
       distribution: beyond one step the sign of the future shock is unknown,
       so the leverage term contributes half its weight.

    Args:
        parameters: A previously fitted model.
        returns_pct: Clean daily returns in percentage points.
        horizon_days: Trading days ahead to forecast.

    Returns:
        Daily sigma path in DECIMALS, shape (horizon_days,), or None when the
        recursion is degenerate.
    """
    if horizon_days < 1 or returns_pct.size < 2:
        return None

    residuals = returns_pct - parameters.mu
    squared = residuals ** 2

    # Seed at the unconditional variance implied by the parameters, falling
    # back to the sample variance when the process is not stationary.
    persistence = parameters.persistence
    if 0 < persistence < 1:
        variance = parameters.omega / (1.0 - persistence)
    else:
        variance = float(np.var(residuals))
    if not np.isfinite(variance) or variance <= 0:
        return None

    for i in range(1, residuals.size):
        leverage = parameters.gamma if residuals[i - 1] < 0 else 0.0
        variance = (
            parameters.omega
            + (parameters.alpha + leverage) * squared[i - 1]
            + parameters.beta * variance
        )
        if not np.isfinite(variance) or variance <= 0:
            return None

    path = np.empty(horizon_days, dtype=float)
    last_leverage = parameters.gamma if residuals[-1] < 0 else 0.0
    variance = (
        parameters.omega
        + (parameters.alpha + last_leverage) * squared[-1]
        + parameters.beta * variance
    )
    path[0] = variance
    for h in range(1, horizon_days):
        variance = parameters.omega + persistence * variance
        path[h] = variance

    if not np.all(np.isfinite(path)) or np.any(path <= 0):
        return None
    return np.sqrt(path) / 100.0


def forecast_volatility(
    daily_returns: Sequence[float],
    horizon_days: int,
    cache_key: Optional[str] = None,
    refit_interval: int = GARCH_REFIT_INTERVAL,
) -> Optional[GarchForecast]:
    """Forecast volatility from a GJR-GARCH(1,1) with Student-t innovations.

    The MLE runs at most once every `refit_interval` new observations per
    `cache_key`; in between, the recursion is run forward from the cached
    parameters (see forecast_from_parameters and GARCH_REFIT_INTERVAL for why
    that distinction is what makes this component usable at all). The
    conditional variance is still recomputed from the latest bar on every
    call — only the *parameter estimation* is scheduled.

    Args:
        daily_returns: Historical simple daily returns.
        horizon_days: Number of trading days ahead to forecast.
        cache_key: Identity to cache the fit under, typically a ticker
            symbol. None disables caching and refits on every call, which is
            the old behaviour and is correct for one-off calls.
        refit_interval: New observations tolerated before a refit.

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

    parameters = None
    if cache_key is not None:
        cached = _PARAMETER_CACHE.get(cache_key)
        if cached is not None and 0 <= len(returns_pct) - cached.fitted_at_observations < refit_interval:
            parameters = cached

    if parameters is None:
        parameters = fit_garch_parameters(returns_pct)
        if parameters is None:
            return None
        if cache_key is not None:
            _PARAMETER_CACHE[cache_key] = parameters

    daily_sigma = forecast_from_parameters(parameters, returns_pct, horizon_days)
    if daily_sigma is None:
        return None

    return GarchForecast(
        daily_sigma=daily_sigma,
        leverage_gamma=parameters.gamma,
        persistence=parameters.persistence,
        distribution_df=parameters.distribution_df,
    )


def forecast_volatility_gap_aware(
    intraday_returns: Sequence[float],
    overnight_returns: Sequence[float],
    horizon_days: int,
    cache_key: Optional[str] = None,
    refit_interval: int = GARCH_REFIT_INTERVAL,
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
        cache_key: Identity to cache the SESSION fit under. Namespaced away
            from the close-to-close fit for the same ticker: they are
            different models of different series and must not share a slot.
        refit_interval: New observations tolerated before a refit.

    Returns:
        A GarchForecast whose daily_sigma includes gap risk, or None when the
        session fit is unavailable — callers should fall back to the
        close-to-close path (or constant volatility) in that case.
    """
    session = forecast_volatility(
        intraday_returns,
        horizon_days,
        cache_key=f"{cache_key}::session" if cache_key else None,
        refit_interval=refit_interval,
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
