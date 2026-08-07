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
