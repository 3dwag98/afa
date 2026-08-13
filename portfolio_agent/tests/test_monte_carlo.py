"""Tests for Monte Carlo simulation module."""

import math

import pytest
import numpy as np
from portfolio_agent.src.monte_carlo import run_monte_carlo, MonteCarloResult


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
        from portfolio_agent.src.monte_carlo import _block_bootstrap_shocks

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
        from portfolio_agent.src.monte_carlo import MonteCarloSettings

        config = AppConfig()
        settings = MonteCarloSettings.from_simulation_config(config.simulation)

        assert settings.horizon_days == config.simulation.mc_horizon_days
        assert settings.simulations == config.simulation.mc_simulations
        assert settings.seed == config.simulation.random_seed
        assert settings.method == config.simulation.method
        assert settings.block_size_days == config.simulation.block_size_days
        assert settings.jump_mean == config.simulation.jump_mean

    def test_run_dispatches_to_the_configured_method(self):
        from portfolio_agent.src.monte_carlo import MonteCarloSettings

        returns = list(np.random.default_rng(5).normal(0.0005, 0.012, 400))
        settings = MonteCarloSettings(
            horizon_days=20, simulations=300, seed=42, method="block_bootstrap"
        )

        result = settings.run("TEST", returns)

        assert result.method == "block_bootstrap"
        assert result.simulations_count == 300

    def test_settings_are_picklable_for_worker_dispatch(self):
        import pickle
        from portfolio_agent.src.monte_carlo import MonteCarloSettings

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
        from portfolio_agent.src.monte_carlo import _standardized_shocks

        rng = np.random.default_rng(0)
        for df in (3.0, 5.0, 30.0):
            shocks = _standardized_shocks(rng, (200_000,), df)
            assert shocks.std() == pytest.approx(1.0, abs=0.05)

    def test_lower_degrees_of_freedom_means_fatter_tails(self):
        from portfolio_agent.src.monte_carlo import _standardized_shocks

        rng = np.random.default_rng(1)
        fat = _standardized_shocks(rng, (200_000,), 3.0)
        thin = _standardized_shocks(rng, (200_000,), 50.0)

        def _kurtosis(x):
            return float(((x - x.mean()) ** 4).mean() / x.std() ** 4)

        assert _kurtosis(fat) > _kurtosis(thin)

    def test_none_degrees_of_freedom_draws_gaussian(self):
        from portfolio_agent.src.monte_carlo import _standardized_shocks

        rng = np.random.default_rng(2)
        shocks = _standardized_shocks(rng, (200_000,), None)

        excess = float(((shocks - shocks.mean()) ** 4).mean() / shocks.std() ** 4 - 3.0)
        assert abs(excess) < 0.2

    def test_degenerate_degrees_of_freedom_falls_back_to_gaussian(self):
        """Below nu = 2 the variance is infinite, so the draw is unusable."""
        from portfolio_agent.src.monte_carlo import _standardized_shocks

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
        from portfolio_agent.src.volatility_models import GarchForecast
        import portfolio_agent.src.monte_carlo as mc

        seen = {}
        real_run = mc.run_monte_carlo

        def spy(**kwargs):
            seen["innovation_df"] = kwargs.get("innovation_df")
            return real_run(**kwargs)

        monkeypatch.setattr(mc, "run_monte_carlo", spy)
        monkeypatch.setattr(
            "portfolio_agent.src.volatility_models.forecast_volatility",
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
    """The drift is the noisiest input in the simulation and must be treated
    as estimated rather than known (src/monte_carlo.py::shrink_drift)."""

    def test_posterior_sits_between_the_sample_mean_and_the_prior(self):
        from portfolio_agent.src.monte_carlo import shrink_drift

        sample_mu = 0.001  # 0.1%/day, a very large drift for a daily series
        posterior, sd = shrink_drift(sample_mu, sample_sigma=0.02, n_observations=1250)

        assert 0.0 < posterior < sample_mu
        assert sd > 0.0

    def test_shrinks_harder_when_the_estimate_is_noisier(self):
        """Weight on the sample mean is tau^2 / (tau^2 + sigma^2/T): more
        history, or a quieter series, earns more credibility."""
        from portfolio_agent.src.monte_carlo import shrink_drift

        short, _ = shrink_drift(0.001, sample_sigma=0.02, n_observations=250)
        long, _ = shrink_drift(0.001, sample_sigma=0.02, n_observations=2500)
        noisy, _ = shrink_drift(0.001, sample_sigma=0.06, n_observations=1250)
        quiet, _ = shrink_drift(0.001, sample_sigma=0.01, n_observations=1250)

        assert short < long
        assert noisy < quiet

    def test_zero_prior_dispersion_credits_no_drift_edge(self):
        from portfolio_agent.src.monte_carlo import shrink_drift

        posterior, sd = shrink_drift(
            0.001, sample_sigma=0.02, n_observations=1250, prior_annual_drift_std=0.0
        )
        assert posterior == 0.0
        assert sd == 0.0

    def test_a_wide_prior_recovers_the_raw_sample_mean(self):
        """The old plug-in behaviour has to remain reachable, so the change is
        a defensible default rather than an unremovable opinion."""
        from portfolio_agent.src.monte_carlo import shrink_drift

        posterior, _ = shrink_drift(
            0.001, sample_sigma=0.02, n_observations=1250, prior_annual_drift_std=1e9
        )
        assert posterior == pytest.approx(0.001, rel=1e-6)

    def test_pure_noise_tickers_stop_clearing_the_compliance_gate(self):
        """The defect this exists to fix, measured.

        Every ticker here has a true drift of exactly zero, so the honest
        probability of profit is ~0.5 and none of them should clear the 0.55
        gate. Propagating the unshrunk sample mean, a meaningful fraction do —
        on a 3,800-name universe that is hundreds of zero-edge names passing
        the gate on estimation error every day.
        """
        rng = np.random.default_rng(7)
        histories = [list(rng.normal(0.0, 0.02, size=1250)) for _ in range(120)]

        def share_clearing_gate(**kwargs):
            probs = [
                run_monte_carlo(
                    symbol="X", daily_returns=h, horizon_days=20,
                    simulations=1000, seed=1000 + i, **kwargs,
                ).probability_profit
                for i, h in enumerate(histories)
            ]
            return float(np.mean(np.array(probs) >= 0.55))

        unshrunk = share_clearing_gate(
            prior_annual_drift_std=1e9, propagate_drift_uncertainty=False
        )
        shrunk = share_clearing_gate()

        assert unshrunk > 0.02
        assert shrunk < unshrunk
        # Not zero, and the change is instructive. This used to assert exactly
        # 0.0, which held only because a *second* error was cancelling part of
        # this one: the drift was Ito-corrected twice, costing every path
        # 0.5*sigma^2 per day and pushing borderline names back under the gate.
        # With that removed the fixed 10%/year prior is revealed to leak a
        # little — two bugs were flattering each other. The empirical prior
        # closes the gap (see
        # TestCrossSectionalDriftPrior::test_zero_drift_universe_...), which is
        # why it is the default.
        assert shrunk <= 0.05

    def test_uncertainty_propagation_widens_the_distribution_of_outcomes(self):
        """A posterior predictive is wider than a plug-in, always."""
        rng = np.random.default_rng(11)
        returns = list(rng.normal(0.0008, 0.02, size=1250))

        plug_in = run_monte_carlo(
            symbol="X", daily_returns=returns, horizon_days=20, simulations=8000,
            seed=3, propagate_drift_uncertainty=False,
        )
        predictive = run_monte_carlo(
            symbol="X", daily_returns=returns, horizon_days=20, simulations=8000,
            seed=3, propagate_drift_uncertainty=True,
        )

        # Wider tails: the 5% VaR is further from zero once the drift's own
        # uncertainty is carried into the paths.
        assert predictive.var_95 < plug_in.var_95
        # And the point estimate moves toward the honest 50/50.
        assert abs(predictive.probability_profit - 0.5) <= abs(
            plug_in.probability_profit - 0.5
        )

    def test_settings_carry_the_drift_prior_through_to_the_simulation(self):
        from portfolio_agent.src.monte_carlo import MonteCarloSettings

        returns = list(np.random.default_rng(5).normal(0.002, 0.02, size=1500))
        # Guard the premise: shrinking toward zero only lowers the probability
        # of profit when the realized sample mean was positive to begin with.
        assert np.mean(np.log1p(returns)) > 0

        wide = MonteCarloSettings(
            horizon_days=20, simulations=2000, seed=9,
            prior_annual_drift_std=1e9, propagate_drift_uncertainty=False,
        ).run("X", returns)
        tight = MonteCarloSettings(
            horizon_days=20, simulations=2000, seed=9, prior_annual_drift_std=0.0
        ).run("X", returns)

        assert tight.probability_profit < wide.probability_profit


def _random_walk(rng, log_drift, log_sigma, size):
    """Simple returns whose *log* returns are exactly Normal(log_drift, sigma).

    The distinction matters and is easy to get backwards. Drawing simple
    returns as Normal(0, sigma) does not give a driftless walk — it gives log
    returns with mean -sigma^2/2, i.e. a price that drifts *down*, which
    suppresses the very false-positive rate these tests are trying to measure.
    Both the estimator and the simulator work in log space, so the synthetic
    data is specified there and converted back with expm1.
    """
    return np.expm1(rng.normal(log_drift, log_sigma, size=size))


class TestCrossSectionalDriftPrior:
    """Empirical-Bayes (James-Stein) drift shrinkage.

    `shrink_drift` already had the right posterior algebra, but it was fed a
    *hardcoded* prior: tau = 10% a year, mu_0 = 0. Both numbers were assumed
    rather than measured. The estimator here reads them off the cross-section
    instead, which is what makes the shrinkage adaptive: when the observed
    spread of sample means is no wider than their own standard errors, the
    method of moments returns tau^2 = 0 and every ticker collapses to the
    universe mean, because nothing in the panel evidences a real drift spread.
    """

    def test_method_of_moments_finds_no_dispersion_when_all_spread_is_noise(self):
        """A zero-drift panel has no true dispersion to find.

        Var(mu_hat) and mean(sigma_i^2/T_i) estimate the same quantity when
        every true drift is identical, so their difference is zero up to
        sampling error and tau^2 lands negligible next to the noise floor.

        Asserted as a ratio rather than `== 0.0` deliberately. The difference
        of two nearly-equal moments has sampling error of order
        sqrt(2/N) * noise_floor — about 4.5% of it at N=1000 — so which side of
        zero it falls on is a coin flip in the seed. A test pinning it to
        exactly zero passes on luck; what actually matters is that whatever
        survives is too small to let any drift through, which is what the
        shrinkage intensity measures. The clamp itself is pinned separately in
        test_negative_moment_difference_is_floored_at_zero.
        """
        from portfolio_agent.src.monte_carlo import (
            drift_observation_from_returns,
            estimate_cross_sectional_drift_prior,
        )

        rng = np.random.default_rng(20260811)
        observations = [
            drift_observation_from_returns(_random_walk(rng, 0.0, 0.02, 1250))
            for _ in range(1000)
        ]
        prior = estimate_cross_sectional_drift_prior(observations)

        assert prior is not None
        assert prior.n_tickers == 1000
        # Whatever survives the subtraction is a rounding error on the noise.
        assert prior.prior_variance < 0.10 * prior.mean_estimator_variance
        # So essentially all of every ticker's sample mean is shrunk away.
        assert prior.shrinkage_intensity > 0.90

    def test_negative_moment_difference_is_floored_at_zero(self):
        """tau^2 is a variance and cannot go negative, however the moments fall.

        Constructed rather than sampled so the difference is unambiguously
        negative: identical sample means give Var(mu_hat) = 0 against a
        strictly positive noise floor.
        """
        from portfolio_agent.src.monte_carlo import (
            DriftObservation,
            estimate_cross_sectional_drift_prior,
        )

        observations = [
            DriftObservation(
                sample_mean_log_return=0.0004, sample_sigma=0.02, n_observations=1250
            )
            for _ in range(50)
        ]
        prior = estimate_cross_sectional_drift_prior(observations)

        assert prior is not None
        assert prior.prior_variance == 0.0
        assert prior.shrinkage_intensity == pytest.approx(1.0)
        assert prior.prior_mean_log_return == pytest.approx(0.0004)

    def test_method_of_moments_recovers_a_genuine_drift_spread(self):
        """When true drifts really do differ, tau^2 must not be shrunk away.

        Built so the signal is unmistakable: true daily drifts are drawn with a
        standard deviation of 0.0008, roughly 1.4x each ticker's own standard
        error, so the observed cross-sectional variance is about 3x the noise
        floor and the difference is real rather than sampling slack.
        """
        from portfolio_agent.src.monte_carlo import (
            drift_observation_from_returns,
            estimate_cross_sectional_drift_prior,
        )

        rng = np.random.default_rng(4242)
        true_tau = 0.0008
        true_drifts = rng.normal(0.0, true_tau, size=1000)
        observations = [
            drift_observation_from_returns(_random_walk(rng, d, 0.02, 1250))
            for d in true_drifts
        ]
        prior = estimate_cross_sectional_drift_prior(observations)

        assert prior is not None
        # Recovers the planted dispersion to within 15%.
        assert np.sqrt(prior.prior_variance) == pytest.approx(true_tau, rel=0.15)
        # And therefore shrinks only partially, unlike the pure-noise panel.
        assert 0.2 < prior.shrinkage_intensity < 0.5

    def test_shrinks_toward_the_universe_mean_not_toward_zero(self):
        """The distinguishing property of the James-Stein form.

        A panel whose drifts are all genuinely positive should keep that
        common level: shrinking it to zero would throw away the one thing the
        cross-section actually evidences.
        """
        from portfolio_agent.src.monte_carlo import (
            drift_observation_from_returns,
            estimate_cross_sectional_drift_prior,
            shrink_drift,
        )

        rng = np.random.default_rng(99)
        common = 0.0006
        observations = [
            drift_observation_from_returns(_random_walk(rng, common, 0.02, 1250))
            for _ in range(500)
        ]
        prior = estimate_cross_sectional_drift_prior(observations)

        assert prior is not None
        assert prior.prior_mean_log_return == pytest.approx(common, abs=1.5e-4)

        # An outlier ticker is pulled back to the universe mean, not to zero.
        posterior, _ = shrink_drift(
            sample_mean_log_return=0.004,
            sample_sigma=0.02,
            n_observations=1250,
            prior_variance=prior.prior_variance,
            prior_mean_log_return=prior.prior_mean_log_return,
        )
        assert posterior == pytest.approx(prior.prior_mean_log_return, abs=3e-4)
        assert posterior > 0.0

    def test_estimator_is_deterministic(self):
        from portfolio_agent.src.monte_carlo import (
            drift_observation_from_returns,
            estimate_cross_sectional_drift_prior,
        )

        rng = np.random.default_rng(3)
        histories = [_random_walk(rng, 0.0002, 0.018, 800) for _ in range(50)]
        observations = [drift_observation_from_returns(h) for h in histories]

        first = estimate_cross_sectional_drift_prior(observations)
        second = estimate_cross_sectional_drift_prior(observations)
        assert first == second

    def test_insufficient_cross_section_returns_none(self):
        from portfolio_agent.src.monte_carlo import (
            drift_observation_from_returns,
            estimate_cross_sectional_drift_prior,
        )

        rng = np.random.default_rng(3)
        one = [drift_observation_from_returns(_random_walk(rng, 0.0, 0.02, 400))]
        assert estimate_cross_sectional_drift_prior(one) is None
        assert estimate_cross_sectional_drift_prior([]) is None

    def test_zero_drift_universe_stops_clearing_the_probability_gate(self):
        """The acceptance criterion, measured on a 1,000-ticker universe.

        Every ticker's true drift is zero, so no ticker has an edge and the
        honest pass rate through a 0.55 probability gate is zero. Propagating
        the raw sample mean, ~16% clear it on estimation error alone: with
        T=1250 and sigma=2%/day the standard error of mu_hat is 5.7bp/day, and
        a name needs only 5.6bp/day of *spurious* drift to look like a 55%
        proposition over 20 days. On a 3,800-name universe that is ~640 false
        positives every day. Measured on this panel: 16.9%.

        This figure used to read 9.3%, and the difference was not noise. The
        simulator applied the Ito conversion to a drift already estimated in
        log space, costing every path 0.5*sigma^2 per day and pushing about
        half of these false positives back under the gate — so the drift-noise
        problem looked half as bad as it was, and the threshold a name had to
        clear was 7.6bp/day rather than the honest 5.6bp/day. Two errors were
        flattering each other. Removing the Ito term (see
        TestDriftIsNotItoCorrectedTwice) unmasks the rest.

        The empirical-Bayes prior finds no true dispersion here and shrinks
        essentially all of it away, holding the pass rate at 0.0% both before
        and after that unmasking — which is what makes the two changes safe
        together. The fixed 10%/year prior does not: it went from 0.1% to 0.8%
        once the masking was removed. What the estimated prior adds is that
        nobody had to supply the 10%; it read the absence of dispersion off the
        panel, and would equally have found dispersion had there been any (see
        test_method_of_moments_recovers_a_genuine_drift_spread).
        """
        from portfolio_agent.src.monte_carlo import (
            drift_observation_from_returns,
            estimate_cross_sectional_drift_prior,
        )

        rng = np.random.default_rng(20260812)
        histories = [_random_walk(rng, 0.0, 0.02, 1250) for _ in range(1000)]
        prior = estimate_cross_sectional_drift_prior(
            [drift_observation_from_returns(h) for h in histories]
        )
        assert prior is not None

        def share_clearing_gate(**kwargs):
            probs = [
                run_monte_carlo(
                    symbol="X", daily_returns=list(h), horizon_days=20,
                    simulations=1000, seed=5000 + i, **kwargs,
                ).probability_profit
                for i, h in enumerate(histories)
            ]
            return float(np.mean(np.asarray(probs) >= 0.55))

        raw = share_clearing_gate(
            prior_annual_drift_std=1e9, propagate_drift_uncertainty=False
        )
        james_stein = share_clearing_gate(
            drift_prior=prior, propagate_drift_uncertainty=False
        )

        assert raw == pytest.approx(0.17, abs=0.04)
        assert james_stein < 0.01


class TestDriftIsNotItoCorrectedTwice:
    """The drift is estimated in log space, so it must not be converted again.

    `sample_mu` is the mean of `np.log1p(returns)` — it *is* already
    `mu_arith - sigma^2/2`, because that is what the log of a lognormal
    return is. Subtracting `0.5 * sigma^2` from it a second time drives the
    simulated log drift to `mu_arith - sigma^2` and every path down with it.

    The error is `0.5 * sigma^2 * H`, which is proportional to *variance*. That
    is what makes it worse than a level bias: it does not wash out of a ranking,
    it is a penalty graded by volatility, applied hardest to the names it hurts
    most — and it lands on a hard gate, since RuleBasedStrategy tests
    `prob_profit >= compliance.target_prob_profit`. In effect it applies a
    second, undocumented low-volatility tilt to every strategy that reads
    `mc_result`, in a platform that already has an explicit
    LowVolatilityStrategy.

    One corroborating detail, visible only by reading the code: the
    `sigma == 0` branch applies `mu` with no adjustment at all. It disagreed
    with the other three branches, and it was the one that was right — which is
    what marks the correction as a slip rather than a decision. It is not
    asserted here because `shrink_drift` short-circuits to the prior mean when
    `sample_sigma` is zero, so that branch's drift never reaches the paths.

    Shrinkage is disabled throughout this class so the tests isolate the Ito
    term rather than measuring the prior.
    """

    RAW = dict(prior_annual_drift_std=1e9, propagate_drift_uncertainty=False)

    @staticmethod
    def _driftless(sigma, n=4000, seed=5):
        """Simple returns whose *log* returns have a mean of exactly zero."""
        rng = np.random.default_rng(seed)
        log_returns = rng.normal(0.0, sigma, size=n)
        log_returns -= log_returns.mean()  # exactly zero, not just in expectation
        return list(np.expm1(log_returns))

    @pytest.mark.parametrize("sigma", [0.015, 0.030, 0.045])
    def test_a_driftless_walk_is_a_coin_flip(self, sigma):
        """The cleanest statement of the defect.

        A driftless log random walk is equally likely to finish above or below
        where it started, whatever its volatility. Under the double correction
        the simulated drift is -sigma^2/2, so P(profit) falls to
        Phi(-0.5*sigma*sqrt(H)) — below a half, and further below the more
        volatile the name.
        """
        result = run_monte_carlo(
            symbol="X", daily_returns=self._driftless(sigma), horizon_days=20,
            simulations=60000, seed=11, method="gaussian", **self.RAW,
        )

        assert result.probability_profit == pytest.approx(0.5, abs=0.01)

    def test_the_gate_does_not_penalize_volatility_by_itself(self):
        """Why it is worse than a constant haircut.

        Three names, identical zero log drift, volatility the only difference.
        Any spread in P(profit) between them is the arithmetic, not the data —
        and it is exactly the spread that decides who clears 0.55.
        """
        probabilities = [
            run_monte_carlo(
                symbol="X", daily_returns=self._driftless(sigma), horizon_days=20,
                simulations=60000, seed=11, method="gaussian", **self.RAW,
            ).probability_profit
            for sigma in (0.015, 0.030, 0.045)
        ]

        assert max(probabilities) - min(probabilities) < 0.02

    def test_a_driftless_walk_still_has_a_positive_expected_return(self):
        """Not a contradiction with the coin flip above — lognormal convexity.

        The *median* terminal price is unchanged and the *mean* is above it, by
        exp(H*sigma^2/2). Under the double correction that convexity is exactly
        cancelled and the expected return collapses to zero, which is what
        makes the bug hard to spot: the number looks reasonable.
        """
        sigma, horizon = 0.03, 20
        result = run_monte_carlo(
            symbol="X", daily_returns=self._driftless(sigma), horizon_days=horizon,
            simulations=60000, seed=11, method="gaussian", **self.RAW,
        )

        expected = math.exp(horizon * sigma**2 / 2.0) - 1.0
        assert result.expected_return_pct == pytest.approx(expected, rel=0.15)
        assert result.expected_return_pct > 0.0

    @pytest.mark.parametrize("method", ["gaussian", "block_bootstrap", "jump_diffusion"])
    def test_every_method_treats_the_drift_the_same_way(self, method):
        """All three shared the corrected line, so all three inherited it.

        jump_diffusion carries a negative mean jump, so its probability sits
        below the other two by design; the tolerance covers that while still
        excluding the sigma^2/2 shift.
        """
        result = run_monte_carlo(
            symbol="X", daily_returns=self._driftless(0.02), horizon_days=20,
            simulations=60000, seed=11, method=method,
            jump_intensity_per_year=6.0, jump_mean=-0.02, jump_volatility=0.03,
            **self.RAW,
        )

        assert result.probability_profit == pytest.approx(0.5, abs=0.03)

    def test_the_jump_compensator_survives_the_fix(self):
        """The compensator is a *different* correction and is correct.

        It removes the expected contribution of the jump component so that
        adding jump risk widens the tails without also shifting every expected
        return down. Deleting it along with the Ito term would trade one
        drift bug for another.
        """
        returns = self._driftless(0.02)
        compensated = run_monte_carlo(
            symbol="X", daily_returns=returns, horizon_days=20, simulations=60000,
            seed=11, method="jump_diffusion",
            jump_intensity_per_year=24.0, jump_mean=-0.05, jump_volatility=0.02,
            **self.RAW,
        )
        none = run_monte_carlo(
            symbol="X", daily_returns=returns, horizon_days=20, simulations=60000,
            seed=11, method="jump_diffusion",
            jump_intensity_per_year=0.0, jump_mean=-0.05, jump_volatility=0.02,
            **self.RAW,
        )

        # Heavy negative jumps widen the left tail...
        assert compensated.var_95 < none.var_95
        # ...without dragging the mean down with them.
        assert compensated.expected_return_pct == pytest.approx(
            none.expected_return_pct, abs=0.02
        )
