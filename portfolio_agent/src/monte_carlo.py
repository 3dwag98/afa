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
from typing import TYPE_CHECKING, Iterable, Literal, NamedTuple, Optional, Sequence

if TYPE_CHECKING:
    import pandas as pd

# Minimum realized returns required before the block bootstrap is trusted; a
# resample of a very short history just re-prints the same few days.
MIN_BOOTSTRAP_OBSERVATIONS = 60

# Student-t variance nu/(nu-2) is only finite above 2; below this the draw is
# not usable as a unit-variance shock.
MIN_INNOVATION_DF = 2.1

TRADING_DAYS_PER_YEAR = 252

# Cross-sectional dispersion of *true* annual drifts, in log-return terms —
# the prior standard deviation tau used to shrink each ticker's estimated
# drift (see shrink_drift). 10% a year is deliberately generous: it says a
# genuinely exceptional name might compound 20% a year faster than the market
# and a genuinely poor one 20% slower. This is the *fallback*: when
# use_empirical_drift_prior is on, tau is measured off the cross-section
# instead (see estimate_cross_sectional_drift_prior). Measured on 66 cached NSE
# names it comes out near 5%/year — about half this — so the fixed value
# under-shrinks on that panel.
DEFAULT_PRIOR_ANNUAL_DRIFT_STD = 0.10

SimulationMethod = Literal["gaussian", "block_bootstrap", "jump_diffusion"]

# A cross-section this thin cannot separate true drift dispersion from the
# estimation noise in the sample means — Var(mu_hat) over a handful of names is
# itself mostly noise, so the method of moments would be estimating one noisy
# quantity by subtracting another. Below this, callers keep the fixed prior.
MIN_CROSS_SECTION_FOR_EMPIRICAL_PRIOR = 20


class DriftObservation(NamedTuple):
    """One ticker's contribution to the cross-sectional drift estimate.

    Deliberately just the three sufficient statistics rather than the return
    series: the panel pre-pass runs over the whole active universe, and holding
    3,800 return histories in memory to compute two moments is wasteful.
    """

    sample_mean_log_return: float  # mu_hat_i
    sample_sigma: float  # sigma_i, daily log-return standard deviation
    n_observations: int  # T_i


class CrossSectionalDriftPrior(NamedTuple):
    """Empirical-Bayes prior for true drifts, measured off the cross-section.

    `shrink_drift`'s posterior algebra was always right; what it lacked was a
    defensible tau and mu_0. Hardcoding tau = 10%/year and mu_0 = 0 asserts two
    things about the universe without measuring either. The method of moments
    reads both off the panel instead:

        mu_bar = mean_i(mu_hat_i)
        tau^2  = max(0, Var_i(mu_hat_i) - mean_i(sigma_i^2 / T_i))

    The subtraction is the whole idea. The observed spread of sample means is
    the sum of the true spread and the estimation noise, and the second term is
    exactly the average noise contribution — so what survives is the dispersion
    the panel actually evidences.

    What that comes to in practice, on 66 cached NSE names: tau^2 = 4.15e-8
    (tau ~ 5.1%/year), against a noise floor of 7.91e-7 — so the panel does
    evidence a real spread of true drifts, but a small one, and 95% of the
    observed spread in sample means is estimation noise. On a synthetic panel
    where every true drift is identical the difference lands within a rounding
    error of zero and the max(0, .) floor takes it there, shrinking every name
    to mu_bar. Both outcomes are findings rather than failures.
    """

    prior_mean_log_return: float  # mu_bar, daily log-return units
    prior_variance: float  # tau^2, daily, >= 0
    n_tickers: int
    mean_estimator_variance: float  # mean_i(sigma_i^2 / T_i), the noise floor

    @property
    def shrinkage_intensity(self) -> float:
        """Fraction of the way a ticker of average precision moves to mu_bar.

        1.0 means total shrinkage (no true dispersion found); 0.0 means the
        sample means are trusted as they stand.
        """
        denominator = self.prior_variance + self.mean_estimator_variance
        if denominator <= 0:
            return 1.0
        return float(self.mean_estimator_variance / denominator)


def drift_observation_from_returns(
    daily_returns: Sequence[float] | np.ndarray,
) -> Optional[DriftObservation]:
    """Reduce one ticker's simple returns to its drift sufficient statistics.

    Converts to log returns with the same log1p the simulator uses, so the
    prior is estimated in the units the drift is applied in. Feeding simple
    returns straight in would put mu_bar about sigma^2/2 above the quantity it
    is meant to be a prior for.

    Returns None when the history is too short or degenerate to contribute.
    """
    arr = np.asarray(daily_returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > -1.0]  # log1p is undefined at or below -100%
    if arr.size < 2:
        return None

    log_returns = np.log1p(arr)
    sigma = float(np.std(log_returns, ddof=0))
    if not np.isfinite(sigma) or sigma <= 0:
        return None

    return DriftObservation(
        sample_mean_log_return=float(np.mean(log_returns)),
        sample_sigma=sigma,
        n_observations=int(log_returns.size),
    )


def estimate_cross_sectional_drift_prior(
    observations: Iterable[Optional[DriftObservation]],
) -> Optional[CrossSectionalDriftPrior]:
    """Method-of-moments empirical-Bayes prior over the universe's true drifts.

    Point-in-time by construction: it reads only the sufficient statistics of
    histories the caller already sliced to the as-of date, and introduces no
    randomness. Callers must not hand it returns from beyond the decision date —
    the prior would then carry the future into every ticker's posterior, which
    is a subtler look-ahead than using a future price because it leaks through
    the whole universe at once rather than one name at a time.

    Args:
        observations: Per-ticker statistics; None entries (too little history)
            are skipped.

    Returns:
        The estimated prior, or None when the usable cross-section is smaller
        than MIN_CROSS_SECTION_FOR_EMPIRICAL_PRIOR — in which case the caller
        should fall back to the fixed prior rather than trust two moments
        estimated off a handful of names.
    """
    usable = [
        o for o in observations
        if o is not None and o.n_observations > 0 and o.sample_sigma > 0
    ]
    if len(usable) < MIN_CROSS_SECTION_FOR_EMPIRICAL_PRIOR:
        return None

    sample_means = np.array([o.sample_mean_log_return for o in usable], dtype=float)
    estimator_variances = np.array(
        [(o.sample_sigma ** 2) / o.n_observations for o in usable], dtype=float
    )

    mu_bar = float(np.mean(sample_means))
    # ddof=1: Var(mu_hat) is being compared against the average sampling
    # variance of those same means, and the biased (1/N) form would understate
    # it and over-shrink by a factor of (N-1)/N.
    observed_variance = float(np.var(sample_means, ddof=1))
    noise_floor = float(np.mean(estimator_variances))

    tau_squared = max(0.0, observed_variance - noise_floor)

    return CrossSectionalDriftPrior(
        prior_mean_log_return=mu_bar,
        prior_variance=tau_squared,
        n_tickers=len(usable),
        mean_estimator_variance=noise_floor,
    )


def shrink_drift(
    sample_mean_log_return: float,
    sample_sigma: float,
    n_observations: int,
    prior_annual_drift_std: float = DEFAULT_PRIOR_ANNUAL_DRIFT_STD,
    prior_mean_log_return: float = 0.0,
    prior_variance: Optional[float] = None,
) -> tuple[float, float]:
    """Bayesian posterior for a ticker's daily drift, and its uncertainty.

    The sample mean of daily returns is the noisiest statistic in finance. Its
    standard error is sigma/sqrt(T), which for a 2%/day Indian mid-cap over
    five years is ~0.057%/day — about 14% a year. Propagating that number
    forward as if it were known is what makes probability-of-profit an
    expensive random number generator: simulating tickers whose true drift is
    exactly zero, 16% of them clear a 0.55 probability gate on estimation error
    alone. The rate is 1 - Phi(Phi^-1(gate) * sqrt(T/H)) — note that sigma
    cancels, since the spurious drift a name needs and the standard error of
    mu_hat both scale linearly in it. At T=1250 and a 20-day horizon that is
    16.0%; a 750-day history gives 22.1%, a 2500-day one 8.0%. See
    docs/QUANT_RESEARCH.md section 21. On a 3,800-name universe that is ~610
    zero-edge tickers passing the gate every day.

    Worse, the error is not independent of the rest of the platform. mu_hat is
    large precisely for stocks that have already risen, so an unshrunk drift
    makes the Monte Carlo gate a noisy restatement of the momentum signal it
    is supposed to corroborate.

    Under a Normal(mu_0, tau^2) prior on the true daily drift and a sample mean
    with variance sigma^2/T, the posterior is Normal with

        mu_post  = (tau^2 * mu_hat + (sigma^2/T) * mu_0) / (tau^2 + sigma^2/T)
        var_post = (tau^2 * sigma^2/T) / (tau^2 + sigma^2/T)

    An annual drift is 252 daily drifts, so a prior dispersion expressed per
    year converts to tau = prior_annual_drift_std / 252 per day.

    Args:
        sample_mean_log_return: mu_hat, the realized mean daily log return.
        sample_sigma: Sample standard deviation of daily log returns.
        n_observations: T, the number of returns mu_hat was estimated from.
        prior_annual_drift_std: tau in annualized terms. 0 shrinks all the way
            to the prior mean (no ticker is credited with any drift edge);
            a very large value recovers the raw sample mean.
        prior_mean_log_return: mu_0, the drift a ticker is assumed to have
            before seeing its own history. Defaults to 0 rather than the
            universe mean, which is the conservative choice for a long-only
            book: it declines to credit any name with an edge it has not
            demonstrated beyond the noise. Supply the universe mean (see
            CrossSectionalDriftPrior) to get the James-Stein form instead.
        prior_variance: tau^2 in *daily* units, overriding
            prior_annual_drift_std when supplied. This is the empirical-Bayes
            entry point: estimate_cross_sectional_drift_prior() measures tau^2
            off the panel rather than assuming it, so the amount of shrinkage
            responds to how much true dispersion the universe actually shows.

    Returns:
        (posterior_mean_daily_log_drift, posterior_standard_deviation).
    """
    if n_observations <= 0 or sample_sigma <= 0:
        return prior_mean_log_return, 0.0

    if prior_variance is not None:
        tau_squared = max(0.0, float(prior_variance))
    else:
        tau = max(0.0, float(prior_annual_drift_std)) / TRADING_DAYS_PER_YEAR
        tau_squared = tau * tau
    estimator_variance = (sample_sigma * sample_sigma) / n_observations

    denominator = tau_squared + estimator_variance
    if denominator <= 0:
        return prior_mean_log_return, 0.0

    posterior_mean = (
        tau_squared * sample_mean_log_return + estimator_variance * prior_mean_log_return
    ) / denominator
    posterior_variance = (tau_squared * estimator_variance) / denominator
    return float(posterior_mean), float(math.sqrt(max(0.0, posterior_variance)))


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
    separate_overnight_gaps: bool = True
    prior_annual_drift_std: float = DEFAULT_PRIOR_ANNUAL_DRIFT_STD
    propagate_drift_uncertainty: bool = True
    use_empirical_drift_prior: bool = True
    # Estimated once per decision date over the whole universe and attached
    # with with_drift_prior(); None keeps the fixed prior. A NamedTuple so the
    # settings stay picklable for ProcessPoolExecutor dispatch.
    drift_prior: Optional[CrossSectionalDriftPrior] = None

    def with_drift_prior(
        self, prior: Optional[CrossSectionalDriftPrior]
    ) -> "MonteCarloSettings":
        """Return a copy carrying `prior`, leaving this instance untouched.

        The settings are frozen and shared across worker processes, so the
        per-date prior is attached by making a new value rather than mutating
        one every worker holds a reference to.
        """
        return replace(self, drift_prior=prior)

    def with_drift_prior_from_panel(
        self, returns_by_symbol: Iterable[Sequence[float] | np.ndarray]
    ) -> "MonteCarloSettings":
        """Estimate the empirical-Bayes prior off a panel and attach it.

        **The panel must be point-in-time.** Every return series handed in has
        to be sliced to the same as-of date the signals are being generated
        for. This is a sharper requirement than it looks: a prior contaminated
        with future returns does not leak into one ticker's score, it leaks
        into every ticker's score at once, through mu_bar and tau^2.

        Returns self unchanged when the feature is off or the cross-section is
        too thin to estimate from, so both callers can apply it unconditionally.
        """
        if not self.use_empirical_drift_prior:
            return self
        prior = estimate_cross_sectional_drift_prior(
            drift_observation_from_returns(returns) for returns in returns_by_symbol
        )
        if prior is None:
            return self
        return self.with_drift_prior(prior)

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
            prior_annual_drift_std=simulation.prior_annual_drift_std,
            propagate_drift_uncertainty=simulation.propagate_drift_uncertainty,
            use_empirical_drift_prior=simulation.use_empirical_drift_prior,
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
                prior_annual_drift_std=self.prior_annual_drift_std,
                propagate_drift_uncertainty=self.propagate_drift_uncertainty,
                drift_prior=self.drift_prior,
            )

        intraday = overnight = None
        if self.separate_overnight_gaps and ohlcv is not None:
            from .liquidity import split_intraday_and_overnight
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
            prior_annual_drift_std=self.prior_annual_drift_std,
            propagate_drift_uncertainty=self.propagate_drift_uncertainty,
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
    prior_annual_drift_std: float = DEFAULT_PRIOR_ANNUAL_DRIFT_STD,
    propagate_drift_uncertainty: bool = True,
    drift_prior: Optional[CrossSectionalDriftPrior] = None,
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
        prior_annual_drift_std: Prior dispersion of true annual drifts used to
            shrink this ticker's estimated drift toward zero (see
            shrink_drift). Raising it toward infinity recovers the raw sample
            mean and, with it, the false-positive rate that motivated the
            shrinkage.
        propagate_drift_uncertainty: When True, each simulated path draws its
            own drift from the posterior rather than sharing the posterior
            mean, so what the result reports is the *posterior predictive*
            probability of profit — the probability accounting for the fact
            that the drift is estimated, not known. This is the quantity the
            compliance gate should be reading.
        drift_prior: Empirical-Bayes prior estimated from the cross-section of
            the active universe (see estimate_cross_sectional_drift_prior).
            When supplied it replaces `prior_annual_drift_std` and the zero
            prior mean, giving the James-Stein posterior — each ticker shrunk
            toward the universe mean by however much of the panel's spread is
            noise. Must be estimated from data available as of the decision
            date; see the note in estimate_cross_sectional_drift_prior.

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

    sample_mu = float(np.mean(log_returns))
    sigma = float(np.std(log_returns, ddof=0))  # Population std

    # The drift is estimated, and badly. Shrink it toward zero in proportion to
    # how much of its spread is estimation noise, and carry the residual
    # uncertainty forward instead of discarding it. `mu` from here on is the
    # posterior mean, not the sample mean.
    mu, drift_posterior_sd = shrink_drift(
        sample_mean_log_return=sample_mu,
        sample_sigma=sigma,
        n_observations=len(log_returns),
        prior_annual_drift_std=prior_annual_drift_std,
        prior_mean_log_return=(
            drift_prior.prior_mean_log_return if drift_prior is not None else 0.0
        ),
        prior_variance=drift_prior.prior_variance if drift_prior is not None else None,
    )
    if not propagate_drift_uncertainty:
        drift_posterior_sd = 0.0

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
        # explicitly, so the resampled shocks are centred and the drift term is
        # the same one every method uses. Centring uses the *sample* mean — the
        # shocks are the realized deviations from what actually happened —
        # while the drift re-applied below is the shrunk posterior mean.
        demeaned = log_returns - sample_mu
        shocks = _block_bootstrap_shocks(
            demeaned, simulations, horizon_days, block_size_days, rng
        )
        daily_drift_path = np.full(horizon_days, mu, dtype=float)
    elif effective_method == "jump_diffusion":
        shocks = _jump_diffusion_shocks(
            sigma_path, simulations, horizon_days,
            jump_intensity_per_year, jump_mean, jump_volatility, rng,
            innovation_df=innovation_df,
        )
        # Compensated drift: subtracting the jump component's expected
        # contribution keeps the process's mean return equal to the historical
        # one, so adding jump risk widens the tails without also quietly
        # shifting every expected return downward. This is a different
        # correction from the Ito term removed below and is genuinely needed —
        # it offsets a shift this method's own shocks introduce.
        jump_compensator = (max(0.0, jump_intensity_per_year) / 252.0) * jump_mean
        daily_drift_path = np.full(horizon_days, mu - jump_compensator, dtype=float)
    elif sigma == 0 and daily_vol_forecast is None:
        # No volatility, deterministic path
        shocks = np.zeros((simulations, horizon_days), dtype=float)
        daily_drift_path = np.full(horizon_days, mu, dtype=float)
    else:
        shocks = _standardized_shocks(
            rng, (simulations, horizon_days), innovation_df
        ) * sigma_path[None, :]
        daily_drift_path = np.full(horizon_days, mu, dtype=float)

    # `mu` is applied as-is, with no Ito conversion. It is estimated from
    # np.log1p(returns), so it is *already* a log-space drift — already
    # mu_arith - sigma^2/2, since that is what the log of a lognormal return
    # is. The three branches above used to subtract 0.5*sigma^2 from it again,
    # which drove the simulated log drift to mu_arith - sigma^2 and every path
    # down with it.
    #
    # The error was 0.5*sigma^2*H, proportional to *variance*, which is what
    # made it worse than a level bias: it does not wash out of a cross-sectional
    # ranking, it is a penalty graded by volatility applied hardest to the names
    # it hurts most, and it lands on a hard gate (RuleBasedStrategy tests
    # prob_profit >= compliance.target_prob_profit). In effect it was a second,
    # undocumented low-volatility tilt on every strategy reading mc_result — in
    # a platform that already ships an explicit LowVolatilityStrategy.
    #
    # Sanity check that pins it: a driftless log random walk must be a coin
    # flip over any horizon, whatever its volatility. See
    # docs/QUANT_RESEARCH.md section 14.1.
    cumulative_returns = (daily_drift_path[None, :] + shocks).sum(axis=1)

    # Posterior predictive, not plug-in: every path gets its own draw of the
    # drift from the posterior, so the reported probability of profit accounts
    # for the drift being estimated rather than known. Without this the
    # simulation is confident about the one number it has least right.
    if drift_posterior_sd > 0:
        path_drift = rng.normal(0.0, drift_posterior_sd, size=simulations)
        cumulative_returns = cumulative_returns + path_drift * horizon_days

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
    intraday_returns: Optional[list[float]] = None,
    overnight_returns: Optional[list[float]] = None,
    prior_annual_drift_std: float = DEFAULT_PRIOR_ANNUAL_DRIFT_STD,
    propagate_drift_uncertainty: bool = True,
    drift_prior: Optional[CrossSectionalDriftPrior] = None,
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
    from .volatility_models import forecast_volatility, forecast_volatility_gap_aware

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
        prior_annual_drift_std=prior_annual_drift_std,
        propagate_drift_uncertainty=propagate_drift_uncertainty,
        drift_prior=drift_prior,
    )
