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
