"""Per-symbol forward Monte Carlo simulation feeding strategy scoring.

Simulates a single ticker's forward price path from its historical daily
return distribution (lognormal-shock model) to estimate probability-of-profit
over a horizon — this is an INPUT to RuleBasedStrategy's scoring
(StrategyContext.mc_result), computed fresh per ticker per scoring round.

This is a distinct concern from src/risk_analytics.py::RiskAnalyzer, which
runs a portfolio-level bootstrap resampling of a completed backtest's realized
trade log to report risk-of-ruin as an output metric, not a scoring input.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result model."""
    probability_profit: float = 0.0
    expected_return_pct: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    simulations_count: int = 0
    horizon_days: int = 0


def run_monte_carlo(
    symbol: str,
    daily_returns: list[float],
    horizon_days: int,
    simulations: int,
    seed: int | None = None,
    daily_vol_forecast: Optional[np.ndarray] = None,
) -> MonteCarloResult:
    """Run Monte Carlo simulation on historical returns using log returns.

    Args:
        symbol: Stock ticker symbol (unused in calculation, for identification).
        daily_returns: List of daily returns.
        horizon_days: Number of days to simulate forward.
        simulations: Number of simulation runs.
        seed: Random seed for reproducibility.
        daily_vol_forecast: Optional per-day volatility path (length
            horizon_days, decimal daily std) — e.g. from
            volatility_models.forecast_volatility() — used instead of the
            flat historical standard deviation when provided. See
            run_monte_carlo_garch() for the GARCH-driven convenience wrapper.

    Returns:
        MonteCarloResult with simulation statistics.
    """
    if seed is not None:
        np.random.seed(seed)

    # Convert to numpy array and clean data
    returns_arr = np.array(daily_returns, dtype=float)
    returns_arr = returns_arr[~np.isnan(returns_arr)]
    returns_arr = returns_arr[~np.isinf(returns_arr)]

    # Require at least 30 returns
    if len(returns_arr) < 30:
        return MonteCarloResult(
            probability_profit=0.0,
            expected_return_pct=0.0,
            var_95=0.0,
            cvar_95=0.0,
            simulations_count=0,
            horizon_days=horizon_days
        )

    # Calculate log returns
    # Assuming daily_returns are simple returns, convert to log returns
    # log(1 + r) where r is simple return
    log_returns = np.log1p(returns_arr)

    mu = np.mean(log_returns)

    if daily_vol_forecast is not None:
        sigma_path = np.asarray(daily_vol_forecast, dtype=float)
        if sigma_path.shape[0] != horizon_days:
            raise ValueError(
                f"daily_vol_forecast length ({sigma_path.shape[0]}) must equal horizon_days ({horizon_days})"
            )
        # Day-varying (GARCH-forecasted) volatility path instead of a flat sigma.
        daily_drift_path = mu - 0.5 * sigma_path ** 2
        random_shocks = np.random.normal(0.0, 1.0, size=(simulations, horizon_days)) * sigma_path[None, :]
        path_returns = daily_drift_path[None, :] + random_shocks
        cumulative_returns = path_returns.sum(axis=1)
    else:
        sigma = np.std(log_returns, ddof=0)  # Population std

        # Handle sigma = 0 safely
        if sigma == 0:
            # No volatility, deterministic path
            daily_drift = mu
            cumulative_returns = np.full(simulations, daily_drift * horizon_days)
        else:
            # Simulate cumulative log returns over horizon_days
            daily_drift = mu - 0.5 * sigma ** 2
            random_shocks = np.random.normal(0, sigma, size=(simulations, horizon_days))
            path_returns = daily_drift + random_shocks
            cumulative_returns = path_returns.sum(axis=1)

    # Probability profit = mean(cumulative_returns > 0)
    probability_profit = float(np.mean(cumulative_returns > 0))

    # Expected return pct = mean(exp(cumulative_returns) - 1)
    expected_return_pct = float(np.mean(np.exp(cumulative_returns) - 1))

    # VaR 95 = percentile(cumulative_returns, 5)
    var_95 = float(np.percentile(cumulative_returns, 5))

    # CVaR 95 = mean of returns <= VaR 95
    tail_mask = cumulative_returns <= var_95
    if np.any(tail_mask):
        cvar_95 = float(np.mean(cumulative_returns[tail_mask]))
    else:
        cvar_95 = var_95

    return MonteCarloResult(
        probability_profit=round(probability_profit, 6),
        expected_return_pct=round(expected_return_pct, 6),
        var_95=round(var_95, 6),
        cvar_95=round(cvar_95, 6),
        simulations_count=simulations,
        horizon_days=horizon_days
    )


def run_monte_carlo_garch(
    symbol: str,
    daily_returns: list[float],
    horizon_days: int,
    simulations: int,
    seed: int | None = None,
) -> MonteCarloResult:
    """Like run_monte_carlo(), but forecasts volatility with GJR-GARCH(1,1)
    (see volatility_models.py) instead of assuming a flat historical
    standard deviation, capturing the leverage effect documented for
    Nifty/Sensex returns (docs/QUANT_RESEARCH.md section 3).

    Falls back to run_monte_carlo()'s constant-volatility path whenever
    there isn't enough history to fit GARCH reliably or the fit fails.
    """
    try:
        from .volatility_models import forecast_volatility
    except ImportError:
        from volatility_models import forecast_volatility

    forecast = forecast_volatility(daily_returns, horizon_days)
    daily_vol_forecast = forecast.daily_sigma if forecast is not None else None

    return run_monte_carlo(
        symbol=symbol,
        daily_returns=daily_returns,
        horizon_days=horizon_days,
        simulations=simulations,
        seed=seed,
        daily_vol_forecast=daily_vol_forecast,
    )
