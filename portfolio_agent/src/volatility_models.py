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
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# GJR-GARCH MLE needs a reasonably long history for stable convergence.
MIN_OBSERVATIONS = 250


@dataclass
class GarchForecast:
    """Forecasted daily volatility path from a fitted GJR-GARCH(1,1) model."""

    daily_sigma: np.ndarray  # shape (horizon_days,); forecasted daily return std, decimal (not %)
    leverage_gamma: float  # fitted asymmetry coefficient; > 0 confirms the leverage effect
    persistence: float  # alpha + gamma/2 + beta; must be < 1 for a stationary process
    # Set when the forecast was built from a gap-aware decomposition
    # (see forecast_volatility_gap_aware): the constant overnight-gap standard
    # deviation folded into daily_sigma, and the session-only GARCH path.
    overnight_sigma: Optional[float] = None
    intraday_sigma: Optional[np.ndarray] = None

    @property
    def gap_aware(self) -> bool:
        """Whether overnight gap risk was modelled separately from the session."""
        return self.overnight_sigma is not None


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

    returns = np.asarray(daily_returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    returns = returns[~np.isinf(returns)]
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

        return GarchForecast(daily_sigma=daily_sigma, leverage_gamma=gamma, persistence=persistence)
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
        overnight_sigma=gap_sigma,
        intraday_sigma=session.daily_sigma,
    )
