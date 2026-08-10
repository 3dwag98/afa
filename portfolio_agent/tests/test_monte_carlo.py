"""Tests for Monte Carlo simulation module."""

import pytest
import numpy as np
from src.monte_carlo import run_monte_carlo, MonteCarloResult


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
