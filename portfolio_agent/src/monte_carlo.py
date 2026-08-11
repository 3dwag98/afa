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

import math

import numpy as np
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Iterable, Literal, Optional, Sequence

if TYPE_CHECKING:
    import pandas as pd

# Minimum realized returns required before the block bootstrap is trusted; a
# resample of a very short history just re-prints the same few days.
MIN_BOOTSTRAP_OBSERVATIONS = 60

# Student-t variance nu/(nu-2) is only finite above 2; below this the draw is
# not usable as a unit-variance shock.
MIN_INNOVATION_DF = 2.1

SimulationMethod = Literal["gaussian", "block_bootstrap", "jump_diffusion"]

# Tickers with less history than this contribute nothing usable to the
# cross-sectional drift prior — their sample mean is almost pure noise and
# would inflate the estimated dispersion of true drifts.
MIN_PRIOR_OBSERVATIONS = 250

# One-sided 95% normal quantile, used to turn the posterior standard deviation
# of the drift into the lower confidence bound on probability-of-profit.
_Z_95 = 1.6448536269514722


@dataclass(frozen=True)
class DriftPrior:
    """Cross-sectional prior on the true daily log drift.

    The sample mean of daily returns is the noisiest statistic in finance: its
    standard error is sigma/sqrt(T), which over five years of a 2%/day Indian
    mid-cap is about 0.057%/day — roughly 14% a year. Propagating that number
    forward as if it were the truth is what makes probability-of-profit a
    measurement of estimation error rather than of edge: simulating 20,000
    tickers whose true drift is *exactly zero* puts 8.5% of them over a 0.55
    probability gate, which on a 3,800-name universe is ~322 zero-edge names
    clearing compliance every day. Worse, mu_hat is largest for stocks that
    have already run, so the noise is positively correlated with the momentum
    signal — it confirms rather than checks it.

    The fix is the standard empirical-Bayes one. Treating each ticker's true
    drift as a draw from Normal(mean, tau^2) and its estimate as
    Normal(true, sigma^2/T), the posterior mean is a precision-weighted blend:

        mu_tilde_i = (tau^2 * mu_hat_i + (sigma_i^2/T_i) * mean) / (tau^2 + sigma_i^2/T_i)

    On daily equity data tau^2 << sigma^2/T, so this collapses almost entirely
    onto the universe mean. That is not a failure of the method — it is the
    honest statement that a single ticker's realized mean carries almost no
    information about its forward drift.

    Attributes:
        mean: Universe mean daily log drift (mu_bar).
        tau_squared: Cross-sectional variance of *true* drifts, net of
            estimation noise. Zero means "no evidence that drifts differ at
            all", which shrinks every ticker onto `mean`.
    """

    mean: float = 0.0
    tau_squared: float = 0.0

    def shrink(self, mu_hat: float, sigma: float, n_obs: int) -> float:
        """Posterior-mean drift for one ticker."""
        if n_obs <= 0 or sigma <= 0:
            return self.mean
        estimation_variance = (sigma * sigma) / n_obs
        denominator = self.tau_squared + estimation_variance
        if denominator <= 0:
            return mu_hat
        weight = self.tau_squared / denominator
        return weight * mu_hat + (1.0 - weight) * self.mean

    def posterior_sd(self, sigma: float, n_obs: int) -> float:
        """Standard deviation of the shrunk drift, for the confidence bound.

        Normal-Normal posterior variance is the harmonic-style combination
        tau^2 * se^2 / (tau^2 + se^2), which is smaller than either input:
        shrinkage buys precision by borrowing strength from the panel.
        """
        if n_obs <= 0 or sigma <= 0:
            return 0.0
        estimation_variance = (sigma * sigma) / n_obs
        denominator = self.tau_squared + estimation_variance
        if denominator <= 0:
            return 0.0
        return math.sqrt(self.tau_squared * estimation_variance / denominator)


def estimate_drift_prior(
    panel_log_returns: Iterable[Sequence[float]],
    min_observations: int = MIN_PRIOR_OBSERVATIONS,
) -> Optional[DriftPrior]:
    """Estimate (mu_bar, tau^2) across a panel by method of moments.

    The observed cross-sectional variance of the per-ticker sample means is
    the variance of the true drifts *plus* the average estimation variance:

        Var_i(mu_hat_i) = tau^2 + mean_i(sigma_i^2 / T_i)

    so tau^2 is the first minus the second, floored at zero (a negative
    estimate means the observed dispersion is entirely explained by noise,
    which on daily data is the common case).

    Args:
        panel_log_returns: One sequence of daily log returns per ticker.
        min_observations: Shortest history a ticker may contribute.

    Returns:
        A DriftPrior, or None when fewer than two tickers qualify — callers
        should then leave the drift unshrunk rather than invent a prior from
        a single name.
    """
    sample_means: list[float] = []
    estimation_variances: list[float] = []

    for series in panel_log_returns:
        arr = np.asarray(series, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < min_observations:
            continue
        variance = float(np.var(arr, ddof=1))
        if not math.isfinite(variance) or variance <= 0:
            continue
        sample_means.append(float(np.mean(arr)))
        estimation_variances.append(variance / arr.size)

    if len(sample_means) < 2:
        return None

    observed_dispersion = float(np.var(np.asarray(sample_means), ddof=1))
    average_noise = float(np.mean(np.asarray(estimation_variances)))
    tau_squared = max(0.0, observed_dispersion - average_noise)

    return DriftPrior(mean=float(np.mean(sample_means)), tau_squared=tau_squared)


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
    # Probability-of-profit re-evaluated with the drift pushed to its one-sided
    # 95% lower bound, on the same shock draws. This is the number a gate
    # should read: the point estimate answers "how often does this ticker
    # profit if my drift estimate is exactly right", which it never is.
    #
    # None means "no simulation produced one" — a hand-built or stub result.
    # Read it through probability_profit_gate, never directly, so an absent
    # bound degrades to the point estimate instead of to a hard zero that
    # would silently block every trade.
    probability_profit_lower: Optional[float] = None
    # The drift actually simulated (daily log), after any shrinkage, and its
    # posterior standard deviation. Reported so a run can be audited for how
    # much of its probability came from the prior rather than the ticker.
    drift_daily_log: float = 0.0
    drift_posterior_sd: float = 0.0
    drift_shrunk: bool = False

    @property
    def probability_profit_gate(self) -> float:
        """Probability-of-profit as a compliance gate should read it."""
        if self.probability_profit_lower is None:
            return self.probability_profit
        return self.probability_profit_lower


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
    separate_overnight_gaps: bool = True
    # Cross-sectional drift prior, estimated once per run over the whole
    # universe (see estimate_drift_prior) and attached with
    # dataclasses.replace. None leaves each ticker's raw sample mean in place,
    # which is the pre-shrinkage behaviour and is only correct when there is
    # no panel to borrow strength from.
    drift_prior: Optional[DriftPrior] = None

    def with_drift_prior(self, prior: Optional[DriftPrior]) -> "MonteCarloSettings":
        """Copy carrying `prior`; the dataclass is frozen so workers can pickle it."""
        return replace(self, drift_prior=prior)

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
            separate_overnight_gaps=simulation.separate_overnight_gaps,
        )

    def run(
        self,
        symbol: str,
        daily_returns: list[float],
        ohlcv: Optional["pd.DataFrame"] = None,
    ) -> MonteCarloResult:
        """Run the configured simulation for one ticker.

        Args:
            symbol: Ticker symbol.
            daily_returns: Close-to-close simple returns.
            ohlcv: Optional raw OHLCV frame. When supplied and GARCH volatility
                and separate_overnight_gaps are both on, the close-to-close
                series is decomposed into session and gap legs so the GARCH
                recursion models only the session it is a model of.
        """
        if not self.use_garch_volatility:
            return run_monte_carlo(
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
                drift_prior=self.drift_prior,
            )

        intraday = overnight = None
        if self.separate_overnight_gaps and ohlcv is not None:
            try:
                from .liquidity import split_intraday_and_overnight
            except ImportError:
                from liquidity import split_intraday_and_overnight
            split = split_intraday_and_overnight(ohlcv)
            if split is not None:
                intraday, overnight = split

        return run_monte_carlo_garch(
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
            intraday_returns=intraday,
            overnight_returns=overnight,
            drift_prior=self.drift_prior,
        )


def _standardized_shocks(
    rng: np.random.Generator,
    size: tuple,
    innovation_df: Optional[float] = None,
) -> np.ndarray:
    """Draw unit-variance shocks — Student-t when a fitted nu is available.

    GJR-GARCH is fitted with Student-t innovations precisely because Indian
    equity returns are fat-tailed (docs/QUANT_RESEARCH.md section 3). Drawing
    Gaussian shocks from that fit throws the estimate away and leaves VaR and
    CVaR optimistic exactly where they matter — the 5% tail the compliance
    gate reads.

    Student-t with nu degrees of freedom has variance nu/(nu-2), so the draw
    is divided by sqrt(nu/(nu-2)) to keep unit variance. Without that
    rescaling, switching to t-innovations would silently inflate every
    simulated path's volatility on top of widening its tails, and the two
    effects would be impossible to tell apart.

    Args:
        rng: Seeded generator.
        size: Output shape.
        innovation_df: Fitted degrees of freedom, or None for Gaussian.

    Returns:
        Unit-variance shocks of the requested shape.
    """
    if innovation_df is None or innovation_df <= MIN_INNOVATION_DF:
        return rng.normal(0.0, 1.0, size=size)

    scale = math.sqrt(innovation_df / (innovation_df - 2.0))
    return rng.standard_t(df=innovation_df, size=size) / scale


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
    innovation_df: Optional[float] = None,
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
    diffusion = _standardized_shocks(
        rng, (simulations, horizon_days), innovation_df
    ) * sigma_path[None, :]

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
    innovation_df: Optional[float] = None,
    drift_prior: Optional[DriftPrior] = None,
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
        innovation_df: Student-t degrees of freedom for the shock draw, e.g.
            the nu fitted by GJR-GARCH. None draws Gaussian shocks. Applies to
            the "gaussian" method and to jump_diffusion's diffusion leg;
            block_bootstrap already inherits the empirical tail shape by
            construction and ignores it.
        drift_prior: Cross-sectional prior used to shrink this ticker's sample
            mean drift toward the universe mean (see DriftPrior). None keeps
            the raw sample mean, which propagates its own standard error
            straight into probability_profit.

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

    mu = float(np.mean(log_returns))
    sigma = float(np.std(log_returns, ddof=0))  # Population std

    # Shrink the drift toward the cross-sectional prior before it is
    # propagated over the horizon. Without this the H-day probability of
    # profit is roughly Phi(mu_hat*sqrt(H)/sigma), which inherits the sample
    # mean's standard error one-for-one — see DriftPrior for what that costs.
    drift_posterior_sd = sigma / math.sqrt(len(log_returns)) if sigma > 0 else 0.0
    if drift_prior is not None:
        mu = drift_prior.shrink(mu, sigma, len(log_returns))
        drift_posterior_sd = drift_prior.posterior_sd(sigma, len(log_returns))
    # The one-sided 95% lower drift. Paths are re-scored against this on the
    # *same* shocks, so the gap between the two probabilities is parameter
    # uncertainty alone and carries no extra simulation noise.
    mu_lower = mu - _Z_95 * drift_posterior_sd

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
            innovation_df=innovation_df,
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
        random_shocks = _standardized_shocks(
            rng, (simulations, horizon_days), innovation_df
        ) * sigma_path[None, :]
        cumulative_returns = (daily_drift_path[None, :] + random_shocks).sum(axis=1)

    # Probability profit = mean(cumulative_returns > 0)
    probability_profit = float(np.mean(cumulative_returns > 0))

    # Drift enters every path additively, one mu per simulated day, so moving
    # mu to its lower bound is exactly a parallel shift of the cumulative
    # distribution — no re-simulation required.
    drift_shift = (mu - mu_lower) * horizon_days
    probability_profit_lower = float(np.mean((cumulative_returns - drift_shift) > 0))

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
        probability_profit_lower=round(probability_profit_lower, 6),
        drift_daily_log=round(mu, 8),
        drift_posterior_sd=round(drift_posterior_sd, 8),
        drift_shrunk=drift_prior is not None,
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
    intraday_returns: Optional[list[float]] = None,
    overnight_returns: Optional[list[float]] = None,
    drift_prior: Optional[DriftPrior] = None,
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

    When intraday and overnight return series are supplied, the fit is
    gap-aware: GARCH describes the trading session and overnight gap risk is
    added as a separate component, instead of the recursion attributing every
    global-cue gap to yesterday's session shock (see
    volatility_models.forecast_volatility_gap_aware). Each fallback is
    independent — a failed gap-aware fit drops to close-to-close GARCH, and a
    failed GARCH fit drops to constant volatility.

    Falls back to run_monte_carlo()'s constant-volatility path whenever
    there isn't enough history to fit GARCH reliably or the fit fails.
    """
    try:
        from .volatility_models import forecast_volatility, forecast_volatility_gap_aware
    except ImportError:
        from volatility_models import forecast_volatility, forecast_volatility_gap_aware

    forecast = None
    if intraday_returns is not None and overnight_returns is not None:
        forecast = forecast_volatility_gap_aware(
            intraday_returns, overnight_returns, horizon_days
        )
    if forecast is None:
        forecast = forecast_volatility(daily_returns, horizon_days)
    daily_vol_forecast = forecast.daily_sigma if forecast is not None else None
    # The fitted Student-t nu travels with the volatility path. Fitting
    # fat-tailed innovations and then simulating Gaussian shocks would discard
    # the tail estimate the fit exists to produce.
    innovation_df = forecast.distribution_df if forecast is not None else None

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
        innovation_df=innovation_df,
        drift_prior=drift_prior,
    )
