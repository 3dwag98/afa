"""Tests for Monte Carlo simulation module."""

import pytest
import numpy as np
from src.monte_carlo import (
    DriftPrior,
    MonteCarloResult,
    MonteCarloSettings,
    estimate_drift_prior,
    run_monte_carlo,
)


class TestMonteCarlo:
    """Test cases for run_monte_carlo function."""

    def test_deterministic_output_with_seed(self):
        """Test that same seed produces identical results."""
        daily_returns = list(np.random.normal(0.001, 0.02, 100))
        
        result1 = run_monte_carlo(
            symbol="TEST",
            daily_returns=daily_returns,
            horizon_days=20,
            simulations=1000,
            seed=42
        )
        
        result2 = run_monte_carlo(
            symbol="TEST",
            daily_returns=daily_returns,
            horizon_days=20,
            simulations=1000,
            seed=42
        )
        
        assert result1.probability_profit == result2.probability_profit
        assert result1.expected_return_pct == result2.expected_return_pct
        assert result1.var_95 == result2.var_95
        assert result1.cvar_95 == result2.cvar_95
        assert result1.simulations_count == result2.simulations_count

    def test_probability_between_0_and_1(self):
        """Test that probability of profit is between 0 and 1."""
        daily_returns = list(np.random.normal(0.001, 0.02, 100))
        
        result = run_monte_carlo(
            symbol="TEST",
            daily_returns=daily_returns,
            horizon_days=20,
            simulations=1000,
            seed=42
        )
        
        assert 0.0 <= result.probability_profit <= 1.0

    def test_var_less_than_expected_return(self):
        """Test that VaR 95 is usually less than expected return."""
        # Use positive drift returns to make this more likely
        daily_returns = list(np.random.normal(0.002, 0.015, 100))
        
        result = run_monte_carlo(
            symbol="TEST",
            daily_returns=daily_returns,
            horizon_days=20,
            simulations=1000,
            seed=42
        )
        
        # VaR (5th percentile) should typically be lower than expected return
        # This may not always hold for very volatile or negative drift assets
        assert result.var_95 <= result.expected_return_pct

    def test_insufficient_returns_returns_zeroed_result(self):
        """Test that fewer than 30 returns returns zeroed result."""
        # Only 20 returns - less than required 30
        daily_returns = list(np.random.normal(0.001, 0.02, 20))
        
        result = run_monte_carlo(
            symbol="TEST",
            daily_returns=daily_returns,
            horizon_days=20,
            simulations=1000,
            seed=42
        )
        
        assert result.probability_profit == 0.0
        assert result.expected_return_pct == 0.0
        assert result.var_95 == 0.0
        assert result.cvar_95 == 0.0
        assert result.simulations_count == 0
        assert result.horizon_days == 20

    def test_handles_nan_and_inf(self):
        """Test that NaN and inf values are properly removed."""
        daily_returns = [0.01] * 50 + [float('nan')] * 5 + [float('inf')] * 5
        
        result = run_monte_carlo(
            symbol="TEST",
            daily_returns=daily_returns,
            horizon_days=20,
            simulations=1000,
            seed=42
        )
        
        # Should have valid results since we have 50 valid returns
        assert isinstance(result, MonteCarloResult)
        assert result.simulations_count == 1000

    def test_sigma_zero_handling(self):
        """Test that sigma=0 (no volatility) is handled safely."""
        # All returns are identical - zero variance
        daily_returns = [0.001] * 50
        
        result = run_monte_carlo(
            symbol="TEST",
            daily_returns=daily_returns,
            horizon_days=20,
            simulations=1000,
            seed=42
        )
        
        assert isinstance(result, MonteCarloResult)
        assert result.simulations_count == 1000
        # With no volatility, all paths are the same
        assert result.probability_profit in [0.0, 1.0] or abs(result.var_95 - result.cvar_95) < 1e-10

    def test_returns_type_is_monte_carlo_result(self):
        """Test that function returns MonteCarloResult instance."""
        daily_returns = list(np.random.normal(0.001, 0.02, 100))
        
        result = run_monte_carlo(
            symbol="TEST",
            daily_returns=daily_returns,
            horizon_days=20,
            simulations=1000,
            seed=42
        )
        
        assert isinstance(result, MonteCarloResult)

    def test_rounding_to_6_decimals(self):
        """Test that float results are rounded to 6 decimals."""
        daily_returns = list(np.random.normal(0.001, 0.02, 100))
        
        result = run_monte_carlo(
            symbol="TEST",
            daily_returns=daily_returns,
            horizon_days=20,
            simulations=1000,
            seed=42
        )
        
        # Check that values don't have more than 6 decimal places
        def check_decimals(value: float) -> bool:
            str_val = f"{value:.15f}".rstrip('0')
            if '.' in str_val:
                decimals = len(str_val.split('.')[1])
                return decimals <= 6
            return True
        
        assert check_decimals(result.probability_profit)
        assert check_decimals(result.expected_return_pct)
        assert check_decimals(result.var_95)
        assert check_decimals(result.cvar_95)


class TestSimulationMethods:
    """The shock-generating process materially changes the tail estimates the
    compliance gate depends on, so each method is pinned separately."""

    @staticmethod
    def _fat_tailed_returns(n=500, seed=7):
        """Returns with volatility clustering and occasional crash days —
        the shape a Gaussian model cannot reproduce."""
        rng = np.random.default_rng(seed)
        returns = rng.normal(0.0005, 0.01, n)
        # A clustered stretch of high volatility, plus two circuit-limit crashes.
        returns[200:240] = rng.normal(-0.002, 0.05, 40)
        returns[220] = -0.18
        returns[221] = -0.12
        return list(returns)

    @pytest.mark.parametrize("method", ["gaussian", "block_bootstrap", "jump_diffusion"])
    def test_every_method_is_deterministic_under_a_seed(self, method):
        returns = self._fat_tailed_returns()
        kwargs = dict(
            symbol="TEST", daily_returns=returns, horizon_days=20,
            simulations=500, seed=42, method=method,
        )

        first = run_monte_carlo(**kwargs)
        second = run_monte_carlo(**kwargs)

        assert first == second

    @pytest.mark.parametrize("method", ["gaussian", "block_bootstrap", "jump_diffusion"])
    def test_every_method_reports_a_usable_result(self, method):
        result = run_monte_carlo(
            symbol="TEST", daily_returns=self._fat_tailed_returns(),
            horizon_days=20, simulations=500, seed=42, method=method,
        )

        assert 0.0 <= result.probability_profit <= 1.0
        assert result.cvar_95 <= result.var_95
        assert result.simulations_count == 500
        assert result.method == method

    def test_block_bootstrap_produces_a_fatter_left_tail_than_gaussian(self):
        """The whole point: resampling real return blocks keeps the crash days
        and the clustering, which i.i.d. normal draws smooth away."""
        returns = self._fat_tailed_returns()
        common = dict(symbol="TEST", daily_returns=returns, horizon_days=20,
                      simulations=4000, seed=42)

        gaussian = run_monte_carlo(method="gaussian", **common)
        bootstrap = run_monte_carlo(method="block_bootstrap", **common)

        assert bootstrap.cvar_95 < gaussian.cvar_95

    def test_jump_diffusion_produces_a_fatter_left_tail_than_gaussian(self):
        returns = self._fat_tailed_returns()
        common = dict(symbol="TEST", daily_returns=returns, horizon_days=20,
                      simulations=4000, seed=42)

        gaussian = run_monte_carlo(method="gaussian", **common)
        jumpy = run_monte_carlo(
            method="jump_diffusion", jump_intensity_per_year=52.0,
            jump_mean=-0.05, jump_volatility=0.08, **common,
        )

        assert jumpy.cvar_95 < gaussian.cvar_95

    def test_jump_drift_is_compensated_so_jumps_widen_tails_not_shift_the_mean(self):
        """Adding jump risk must not quietly turn every expected return
        negative — otherwise the method choice would double as a return
        forecast, which it is not."""
        returns = self._fat_tailed_returns()
        common = dict(symbol="TEST", daily_returns=returns, horizon_days=20,
                      simulations=8000, seed=11)

        gaussian = run_monte_carlo(method="gaussian", **common)
        jumpy = run_monte_carlo(
            method="jump_diffusion", jump_intensity_per_year=52.0,
            jump_mean=-0.05, jump_volatility=0.08, **common,
        )

        assert jumpy.expected_return_pct == pytest.approx(
            gaussian.expected_return_pct, abs=0.05
        )

    def test_block_bootstrap_falls_back_on_short_history(self):
        """A resample of 40 observations just re-prints the same few days, so
        the method degrades to Gaussian rather than pretending otherwise."""
        short_returns = list(np.random.default_rng(1).normal(0.0, 0.01, 40))

        result = run_monte_carlo(
            symbol="TEST", daily_returns=short_returns, horizon_days=10,
            simulations=200, seed=42, method="block_bootstrap",
        )

        assert result.method == "gaussian"
        assert result.simulations_count == 200

    def test_block_bootstrap_preserves_serial_dependence(self):
        """Contiguous blocks must travel together: a series of strictly
        alternating returns resampled in blocks keeps alternating, while
        independent draws would not."""
        from src.monte_carlo import _block_bootstrap_shocks

        source = np.array([0.01, -0.01] * 50)
        rng = np.random.default_rng(3)

        shocks = _block_bootstrap_shocks(source, simulations=200, horizon_days=30,
                                         mean_block_days=10, rng=rng)

        # Within a block, consecutive draws alternate sign, so consecutive
        # products are negative. With mean block length 10, the large majority
        # of adjacent pairs sit inside a block.
        adjacent_products = shocks[:, :-1] * shocks[:, 1:]
        assert float(np.mean(adjacent_products < 0)) > 0.8

    def test_unseeded_runs_differ(self):
        returns = self._fat_tailed_returns()
        kwargs = dict(symbol="TEST", daily_returns=returns, horizon_days=20,
                      simulations=500, seed=None, method="block_bootstrap")

        assert run_monte_carlo(**kwargs) != run_monte_carlo(**kwargs)


class TestMonteCarloSettings:
    """The settings bundle both worker-dispatch paths ship to their workers."""

    def test_from_simulation_config_round_trips_every_option(self):
        from portfolio_agent.config.schema import AppConfig
        from src.monte_carlo import MonteCarloSettings

        config = AppConfig()
        settings = MonteCarloSettings.from_simulation_config(config.simulation)

        assert settings.horizon_days == config.simulation.mc_horizon_days
        assert settings.simulations == config.simulation.mc_simulations
        assert settings.seed == config.simulation.random_seed
        assert settings.method == config.simulation.method
        assert settings.block_size_days == config.simulation.block_size_days
        assert settings.jump_mean == config.simulation.jump_mean

    def test_run_dispatches_to_the_configured_method(self):
        from src.monte_carlo import MonteCarloSettings

        returns = list(np.random.default_rng(5).normal(0.0005, 0.012, 400))
        settings = MonteCarloSettings(
            horizon_days=20, simulations=300, seed=42, method="block_bootstrap"
        )

        result = settings.run("TEST", returns)

        assert result.method == "block_bootstrap"
        assert result.simulations_count == 300

    def test_settings_are_picklable_for_worker_dispatch(self):
        import pickle
        from src.monte_carlo import MonteCarloSettings

        settings = MonteCarloSettings(method="jump_diffusion", seed=7)

        assert pickle.loads(pickle.dumps(settings)) == settings


class TestStudentTInnovations:
    """GJR-GARCH is fitted with dist='t' precisely because Indian returns are
    fat-tailed. Simulating Gaussian shocks off that fit throws the estimate
    away and leaves VaR/CVaR optimistic exactly where the gate reads them."""

    @staticmethod
    def _returns(n=800, df=4, seed=5):
        return list(np.random.default_rng(seed).standard_t(df=df, size=n) * 0.01)

    def test_shocks_keep_unit_variance_after_rescaling(self):
        """Student-t variance is nu/(nu-2); without dividing it out, switching
        to t-innovations would inflate volatility as well as widening tails,
        and the two effects would be indistinguishable."""
        from src.monte_carlo import _standardized_shocks

        rng = np.random.default_rng(0)
        for df in (3.0, 5.0, 30.0):
            shocks = _standardized_shocks(rng, (200_000,), df)
            assert shocks.std() == pytest.approx(1.0, abs=0.05)

    def test_lower_degrees_of_freedom_means_fatter_tails(self):
        from src.monte_carlo import _standardized_shocks

        rng = np.random.default_rng(1)
        fat = _standardized_shocks(rng, (200_000,), 3.0)
        thin = _standardized_shocks(rng, (200_000,), 50.0)

        def _kurtosis(x):
            return float(((x - x.mean()) ** 4).mean() / x.std() ** 4)

        assert _kurtosis(fat) > _kurtosis(thin)

    def test_none_degrees_of_freedom_draws_gaussian(self):
        from src.monte_carlo import _standardized_shocks

        rng = np.random.default_rng(2)
        shocks = _standardized_shocks(rng, (200_000,), None)

        excess = float(((shocks - shocks.mean()) ** 4).mean() / shocks.std() ** 4 - 3.0)
        assert abs(excess) < 0.2

    def test_degenerate_degrees_of_freedom_falls_back_to_gaussian(self):
        """Below nu = 2 the variance is infinite, so the draw is unusable."""
        from src.monte_carlo import _standardized_shocks

        rng = np.random.default_rng(3)
        shocks = _standardized_shocks(rng, (50_000,), 1.5)

        assert np.isfinite(shocks).all()
        assert shocks.std() == pytest.approx(1.0, abs=0.05)

    def test_t_innovations_widen_the_simulated_tail(self):
        returns = self._returns()
        common = dict(symbol="T", daily_returns=returns, horizon_days=5,
                      simulations=40_000, seed=11, method="gaussian")

        gaussian = run_monte_carlo(**common)
        student_t = run_monte_carlo(innovation_df=4.0, **common)

        assert student_t.cvar_95 < gaussian.cvar_95

    def test_garch_wrapper_passes_its_fitted_nu_to_the_simulation(self, monkeypatch):
        """The regression: the fit produced nu, the dataclass dropped it, and
        the simulation drew normals regardless."""
        from src.volatility_models import GarchForecast
        import src.monte_carlo as mc

        seen = {}
        real_run = mc.run_monte_carlo

        def spy(**kwargs):
            seen["innovation_df"] = kwargs.get("innovation_df")
            return real_run(**kwargs)

        monkeypatch.setattr(mc, "run_monte_carlo", spy)
        monkeypatch.setattr(
            "src.volatility_models.forecast_volatility",
            lambda returns, horizon: GarchForecast(
                daily_sigma=np.full(5, 0.02), leverage_gamma=0.1,
                persistence=0.9, distribution_df=4.2,
            ),
        )

        mc.run_monte_carlo_garch(
            symbol="T", daily_returns=self._returns(), horizon_days=5, simulations=100, seed=1
        )

        assert seen["innovation_df"] == pytest.approx(4.2)


class TestDriftShrinkage:
    """The sample mean of daily returns is the noisiest statistic in finance.
    Propagating it forward unshrunk turns probability-of-profit into a
    measurement of estimation error (src/monte_carlo.py::DriftPrior)."""

    @staticmethod
    def _zero_drift_panel(n_tickers=200, n_obs=1250, sigma=0.02, seed=7):
        rng = np.random.default_rng(seed)
        return [rng.normal(0.0, sigma, n_obs) for _ in range(n_tickers)]

    def test_prior_recovers_near_zero_dispersion_on_a_zero_drift_panel(self):
        """Every ticker here has a TRUE drift of exactly zero, so the observed
        dispersion of sample means is entirely estimation noise and tau^2 must
        come out at (or very near) its zero floor."""
        panel = [np.log1p(r) for r in self._zero_drift_panel()]
        prior = estimate_drift_prior(panel)

        assert prior is not None
        assert abs(prior.mean) < 1e-3
        # What matters is the shrinkage weight tau^2 / (tau^2 + sigma^2/T),
        # i.e. how much of a ticker's own sample mean survives. sigma^2/T is
        # ~3.2e-7 here, and the residual tau^2 is the method-of-moments
        # estimator's own sampling error, so almost nothing should survive.
        estimation_variance = 0.02 ** 2 / 1250
        weight = prior.tau_squared / (prior.tau_squared + estimation_variance)
        assert weight < 0.15

    def test_shrinkage_collapses_the_spread_of_probability_of_profit(self):
        """The headline D2 number: pure-noise tickers must stop clearing the
        compliance gate."""
        panel = self._zero_drift_panel(n_tickers=120)
        prior = estimate_drift_prior([np.log1p(r) for r in panel])

        def probabilities(drift_prior, field):
            return np.array([
                getattr(
                    run_monte_carlo(
                        "T", list(r), horizon_days=20, simulations=400,
                        seed=1000 + i, drift_prior=drift_prior,
                    ),
                    field,
                )
                for i, r in enumerate(panel)
            ])

        unshrunk = probabilities(None, "probability_profit")
        shrunk = probabilities(prior, "probability_profit")
        bounded = probabilities(prior, "probability_profit_lower")

        # Shrinkage removes the drift-estimation component of the spread; what
        # is left is simulation noise at 400 paths.
        assert shrunk.std() < unshrunk.std() / 1.5
        # And the false-positive rate at the default 0.55 gate goes to zero.
        # (The rate scales with sigma*sqrt(H)/sigma_of_mean; the plan's 8.5%
        # figure is a wider process than this fixture, so the assertion is on
        # the direction and the floor, not the magnitude.)
        assert (unshrunk >= 0.55).mean() > 0.01
        assert (shrunk >= 0.55).mean() == 0.0
        assert (bounded >= 0.55).mean() == 0.0

    def test_prior_needs_at_least_two_usable_tickers(self):
        assert estimate_drift_prior([]) is None
        assert estimate_drift_prior([np.zeros(1000)]) is None
        # Too little history to contribute.
        assert estimate_drift_prior([np.ones(10) * 0.001, np.ones(10) * 0.002]) is None

    def test_zero_tau_shrinks_every_ticker_onto_the_prior_mean(self):
        prior = DriftPrior(mean=0.0004, tau_squared=0.0)
        assert prior.shrink(0.01, sigma=0.02, n_obs=1250) == pytest.approx(0.0004)
        assert prior.shrink(-0.01, sigma=0.02, n_obs=1250) == pytest.approx(0.0004)
        # A point-mass prior leaves no posterior uncertainty in the drift.
        assert prior.posterior_sd(sigma=0.02, n_obs=1250) == 0.0

    def test_shrinkage_weight_rises_with_evidence(self):
        """More history means a tighter sample mean, so less is borrowed."""
        prior = DriftPrior(mean=0.0, tau_squared=1e-7)
        short = prior.shrink(0.001, sigma=0.02, n_obs=100)
        long = prior.shrink(0.001, sigma=0.02, n_obs=5000)
        assert 0.0 < short < long < 0.001

    def test_lower_bound_never_exceeds_the_point_estimate(self):
        rng = np.random.default_rng(3)
        returns = list(rng.normal(0.0015, 0.02, 800))
        result = run_monte_carlo("T", returns, horizon_days=20, simulations=2000, seed=5)

        assert result.probability_profit_lower <= result.probability_profit
        assert result.drift_shrunk is False

    def test_settings_carry_the_prior_into_the_simulation(self):
        prior = DriftPrior(mean=0.0, tau_squared=0.0)
        settings = MonteCarloSettings(horizon_days=10, simulations=200, seed=1)
        rng = np.random.default_rng(11)
        returns = list(rng.normal(0.003, 0.02, 600))  # a strong in-sample run-up

        raw = settings.run("T", returns)
        shrunk = settings.with_drift_prior(prior).run("T", returns)

        assert raw.drift_shrunk is False and shrunk.drift_shrunk is True
        # The prior is a point mass at zero drift, so the simulated drift is
        # zero and the run-up stops being extrapolated.
        assert shrunk.drift_daily_log == pytest.approx(0.0, abs=1e-12)
        assert shrunk.probability_profit < raw.probability_profit
        # with_drift_prior copies rather than mutates -- workers pickle these.
        assert settings.drift_prior is None
