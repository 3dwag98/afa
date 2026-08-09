"""Per-symbol forward Monte Carlo simulation feeding strategy scoring.

Simulates a single ticker's forward price path from its historical daily
return distribution to estimate probability-of-profit over a horizon — this is
an INPUT to RuleBasedStrategy's scoring (StrategyContext.mc_result), computed
fresh per ticker per scoring round.

Three shock-generating processes are available (`simulation.method`), because
the choice materially changes the tail estimates the compliance gate depends
on (docs/QUANT_RESEARCH.md section 14):

- ``gaussian`` — i.i.d. normal shocks around the historical mean/std, the
  classical lognormal model. Fast and smooth, but it assumes stationary
  parameters, thin tails and independent days: it systematically understates
  tail risk, and does so worst in exactly the emerging-market conditions this
  platform trades (circuit limits, retail participation spikes, policy and
  FII-flow shocks).
- ``block_bootstrap`` — the default. Resamples *contiguous blocks* of the
  ticker's own realized returns, so both the empirical fat tails and the
  volatility clustering (a bad day tends to be followed by another) survive
  into the simulated paths. Block lengths are geometric (the stationary
  bootstrap of Politis & Romano) rather than fixed, which avoids the
  artificial periodicity a fixed block length imprints on the paths.
- ``jump_diffusion`` — Merton-style: Gaussian diffusion plus a compound
  Poisson jump component with a negative mean jump, modelling the gap moves
  that continuous diffusion cannot produce.

All three share the same drift/aggregation machinery and honour an optional
GARCH volatility path, so switching methods changes only how shocks are drawn.

This is a distinct concern from src/risk_analytics.py::RiskAnalyzer, which
runs a portfolio-level bootstrap resampling of a completed backtest's realized
trade log to report risk-of-ruin as an output metric, not a scoring input.
"""

import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional

# Minimum realized returns required before the block bootstrap is trusted; a
# resample of a very short history just re-prints the same few days.
MIN_BOOTSTRAP_OBSERVATIONS = 60

SimulationMethod = Literal["gaussian", "block_bootstrap", "jump_diffusion"]


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result model."""
    probability_profit: float = 0.0
    expected_return_pct: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    simulations_count: int = 0
    horizon_days: int = 0
    method: str = "gaussian"


@dataclass(frozen=True)
class MonteCarloSettings:
    """Everything needed to run a forward simulation, in one picklable value.

    Both the backtest engine and the live orchestrator dispatch per-ticker
    scoring to worker processes, and both used to ship these settings as a
    bare positional tuple that had to be unpacked identically in three places.
    Bundling them keeps the two call sites honest as options accumulate.
    """

    horizon_days: int = 20
    simulations: int = 1000
    seed: Optional[int] = 42
    use_garch_volatility: bool = False
    method: SimulationMethod = "gaussian"
    block_size_days: int = 5
    jump_intensity_per_year: float = 12.0
    jump_mean: float = -0.02
    jump_volatility: float = 0.05

    @classmethod
    def from_simulation_config(cls, simulation) -> "MonteCarloSettings":
        """Build from an AppConfig.simulation block."""
        return cls(
            horizon_days=simulation.mc_horizon_days,
            simulations=simulation.mc_simulations,
            seed=simulation.random_seed,
            use_garch_volatility=simulation.use_garch_volatility,
            method=simulation.method,
            block_size_days=simulation.block_size_days,
            jump_intensity_per_year=simulation.jump_intensity_per_year,
            jump_mean=simulation.jump_mean,
            jump_volatility=simulation.jump_volatility,
        )

    def run(self, symbol: str, daily_returns: list[float]) -> MonteCarloResult:
        """Run the configured simulation for one ticker."""
        mc_fn = run_monte_carlo_garch if self.use_garch_volatility else run_monte_carlo
        return mc_fn(
            symbol=symbol,
            daily_returns=daily_returns,
            horizon_days=self.horizon_days,
            simulations=self.simulations,
            seed=self.seed,
            method=self.method,
            block_size_days=self.block_size_days,
            jump_intensity_per_year=self.jump_intensity_per_year,
            jump_mean=self.jump_mean,
            jump_volatility=self.jump_volatility,
        )


def _block_bootstrap_shocks(
    demeaned_log_returns: np.ndarray,
    simulations: int,
    horizon_days: int,
    mean_block_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw shocks by stationary block bootstrap of realized returns.

    Politis & Romano's stationary bootstrap: at each simulated day, either
    continue the current block (probability 1 - 1/L) or jump to a fresh random
    starting point (probability 1/L), giving geometrically distributed block
    lengths with mean L. Contiguous days therefore travel together, which is
    what preserves volatility clustering — resampling days independently would
    destroy exactly the serial dependence that makes a drawdown a drawdown.
    The source series wraps around, so every starting point is equally likely
    and no observation is under-sampled at the edges.

    Args:
        demeaned_log_returns: Historical log returns with the mean removed
            (drift is applied separately by the caller).
        simulations: Number of paths.
        horizon_days: Path length in trading days.
        mean_block_days: Mean block length L.
        rng: Seeded random generator.

    Returns:
        Array of shape (simulations, horizon_days).
    """
    n = len(demeaned_log_returns)
    length = max(1, int(mean_block_days))
    restart_probability = 1.0 / length

    # Walk the index forward one day at a time: continue the current block, or
    # restart at a random point. Vectorized across all paths simultaneously.
    indices = np.empty((simulations, horizon_days), dtype=np.int64)
    current = rng.integers(0, n, size=simulations)
    indices[:, 0] = current
    for day in range(1, horizon_days):
        restart = rng.random(simulations) < restart_probability
        current = np.where(restart, rng.integers(0, n, size=simulations), (current + 1) % n)
        indices[:, day] = current

    return demeaned_log_returns[indices]


def _jump_diffusion_shocks(
    sigma_path: np.ndarray,
    simulations: int,
    horizon_days: int,
    jump_intensity_per_year: float,
    jump_mean: float,
    jump_volatility: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw Gaussian diffusion shocks plus a compound-Poisson jump component.

    Merton's jump-diffusion: alongside the continuous diffusion, each day
    carries N ~ Poisson(lambda/252) jumps of size ~ Normal(jump_mean,
    jump_volatility). Diffusion alone cannot produce a gap; jumps can, which
    is what makes the simulated tail resemble a market with circuit limits,
    overnight policy announcements and sudden flow reversals.

    Args:
        sigma_path: Per-day diffusion volatility, shape (horizon_days,).
        simulations: Number of paths.
        horizon_days: Path length in trading days.
        jump_intensity_per_year: Expected jumps per year (lambda).
        jump_mean: Mean jump size in log-return terms.
        jump_volatility: Standard deviation of jump size.
        rng: Seeded random generator.

    Returns:
        Array of shape (simulations, horizon_days).
    """
    diffusion = rng.normal(0.0, 1.0, size=(simulations, horizon_days)) * sigma_path[None, :]

    daily_intensity = max(0.0, jump_intensity_per_year) / 252.0
    if daily_intensity <= 0:
        return diffusion

    jump_counts = rng.poisson(daily_intensity, size=(simulations, horizon_days))
    # Sum of N i.i.d. Normal(m, s) jumps is Normal(N*m, sqrt(N)*s), so the
    # compound sum is drawn in closed form instead of looped per jump.
    jump_totals = jump_counts * jump_mean + np.sqrt(jump_counts) * jump_volatility * rng.normal(
        0.0, 1.0, size=(simulations, horizon_days)
    )
    return diffusion + jump_totals


def run_monte_carlo(
    symbol: str,
    daily_returns: list[float],
    horizon_days: int,
    simulations: int,
    seed: int | None = None,
    daily_vol_forecast: Optional[np.ndarray] = None,
    method: SimulationMethod = "gaussian",
    block_size_days: int = 5,
    jump_intensity_per_year: float = 12.0,
    jump_mean: float = -0.02,
    jump_volatility: float = 0.05,
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
        method: Shock-generating process — "gaussian", "block_bootstrap" or
            "jump_diffusion" (see module docstring). Falls back to "gaussian"
            when a method's own preconditions aren't met (e.g. too little
            history to bootstrap), so a short series degrades rather than fails.
        block_size_days: Mean block length for method="block_bootstrap".
        jump_intensity_per_year: Expected jumps per year for method="jump_diffusion".
        jump_mean: Mean log jump size for method="jump_diffusion".
        jump_volatility: Std dev of log jump size for method="jump_diffusion".

    Returns:
        MonteCarloResult with simulation statistics.
    """
    rng = np.random.default_rng(seed)

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
            horizon_days=horizon_days,
            method=method,
        )

    # Calculate log returns
    # Assuming daily_returns are simple returns, convert to log returns
    # log(1 + r) where r is simple return
    log_returns = np.log1p(returns_arr)

    mu = np.mean(log_returns)
    sigma = float(np.std(log_returns, ddof=0))  # Population std

    if daily_vol_forecast is not None:
        sigma_path = np.asarray(daily_vol_forecast, dtype=float)
        if sigma_path.shape[0] != horizon_days:
            raise ValueError(
                f"daily_vol_forecast length ({sigma_path.shape[0]}) must equal horizon_days ({horizon_days})"
            )
    else:
        sigma_path = np.full(horizon_days, sigma, dtype=float)

    # A method whose preconditions don't hold silently degrades to the
    # Gaussian path rather than failing the ticker: this feeds live scoring,
    # and a missing MC result is treated as zero probability-of-profit.
    effective_method = method
    if method == "block_bootstrap" and len(log_returns) < MIN_BOOTSTRAP_OBSERVATIONS:
        effective_method = "gaussian"

    if effective_method == "block_bootstrap":
        # Bootstrap the *shape* of the return distribution and re-apply drift
        # explicitly, so the resampled shocks are centred and the drift term
        # stays the same Ito-corrected one every method uses.
        demeaned = log_returns - mu
        shocks = _block_bootstrap_shocks(
            demeaned, simulations, horizon_days, block_size_days, rng
        )
        daily_drift_path = mu - 0.5 * sigma_path ** 2
        cumulative_returns = (daily_drift_path[None, :] + shocks).sum(axis=1)
    elif effective_method == "jump_diffusion":
        shocks = _jump_diffusion_shocks(
            sigma_path, simulations, horizon_days,
            jump_intensity_per_year, jump_mean, jump_volatility, rng,
        )
        # Compensated drift: subtracting the jump component's expected
        # contribution keeps the process's mean return equal to the historical
        # one, so adding jump risk widens the tails without also quietly
        # shifting every expected return downward.
        jump_compensator = (max(0.0, jump_intensity_per_year) / 252.0) * jump_mean
        daily_drift_path = mu - 0.5 * sigma_path ** 2 - jump_compensator
        cumulative_returns = (daily_drift_path[None, :] + shocks).sum(axis=1)
    elif sigma == 0 and daily_vol_forecast is None:
        # No volatility, deterministic path
        cumulative_returns = np.full(simulations, mu * horizon_days)
    else:
        daily_drift_path = mu - 0.5 * sigma_path ** 2
        random_shocks = rng.normal(0.0, 1.0, size=(simulations, horizon_days)) * sigma_path[None, :]
        cumulative_returns = (daily_drift_path[None, :] + random_shocks).sum(axis=1)

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
        horizon_days=horizon_days,
        method=effective_method,
    )


def run_monte_carlo_garch(
    symbol: str,
    daily_returns: list[float],
    horizon_days: int,
    simulations: int,
    seed: int | None = None,
    method: SimulationMethod = "gaussian",
    block_size_days: int = 5,
    jump_intensity_per_year: float = 12.0,
    jump_mean: float = -0.02,
    jump_volatility: float = 0.05,
) -> MonteCarloResult:
    """Like run_monte_carlo(), but forecasts volatility with GJR-GARCH(1,1)
    (see volatility_models.py) instead of assuming a flat historical
    standard deviation, capturing the leverage effect documented for
    Nifty/Sensex returns (docs/QUANT_RESEARCH.md section 3).

    The GARCH volatility path and the shock-generating `method` are
    complementary and compose: GARCH says how large tomorrow's shocks should
    be, `method` says what shape they take. jump_diffusion scales its
    diffusion leg by the GARCH path; block_bootstrap draws from realized
    returns and so carries its own scale, using the GARCH path only for the
    Ito drift correction.

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
        method=method,
        block_size_days=block_size_days,
        jump_intensity_per_year=jump_intensity_per_year,
        jump_mean=jump_mean,
        jump_volatility=jump_volatility,
    )
