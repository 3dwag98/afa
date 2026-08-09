"""Tests for GJR-GARCH volatility forecasting and its Monte Carlo integration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from src.volatility_models import forecast_volatility, GarchForecast, MIN_OBSERVATIONS
from src.monte_carlo import run_monte_carlo, run_monte_carlo_garch, MonteCarloResult


def _synthetic_returns(n: int, seed: int = 7) -> list[float]:
    rng = np.random.default_rng(seed)
    return list(rng.normal(0.0005, 0.015, n))


class TestForecastVolatility:
    def test_returns_none_below_min_observations(self):
        returns = _synthetic_returns(MIN_OBSERVATIONS - 10)
        assert forecast_volatility(returns, horizon_days=10) is None

    def test_fits_and_forecasts_with_enough_history(self):
        returns = _synthetic_returns(500)
        result = forecast_volatility(returns, horizon_days=15)

        assert result is not None
        assert isinstance(result, GarchForecast)
        assert result.daily_sigma.shape == (15,)
        assert np.all(result.daily_sigma > 0)
        assert np.all(np.isfinite(result.daily_sigma))

    def test_handles_nan_and_inf_in_input(self):
        returns = _synthetic_returns(500) + [float("nan"), float("inf"), float("-inf")]
        result = forecast_volatility(returns, horizon_days=10)
        assert result is not None
        assert result.daily_sigma.shape == (10,)

    def test_none_when_arch_unavailable(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "arch":
                raise ImportError("simulated missing dependency")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        returns = _synthetic_returns(500)
        assert forecast_volatility(returns, horizon_days=10) is None


class TestMonteCarloWithVolForecast:
    def test_daily_vol_forecast_overrides_flat_sigma(self):
        returns = _synthetic_returns(200)
        horizon = 10

        flat_result = run_monte_carlo(
            symbol="TEST", daily_returns=returns, horizon_days=horizon, simulations=2000, seed=42
        )

        # A much larger day-varying vol path should produce a materially
        # different (wider-tailed) result than the flat historical-std path.
        wide_vol_path = np.full(horizon, 0.10)
        wide_result = run_monte_carlo(
            symbol="TEST", daily_returns=returns, horizon_days=horizon, simulations=2000, seed=42,
            daily_vol_forecast=wide_vol_path,
        )

        assert isinstance(wide_result, MonteCarloResult)
        assert wide_result.var_95 < flat_result.var_95

    def test_wrong_length_forecast_raises(self):
        returns = _synthetic_returns(200)
        with pytest.raises(ValueError):
            run_monte_carlo(
                symbol="TEST", daily_returns=returns, horizon_days=10, simulations=100, seed=42,
                daily_vol_forecast=np.full(5, 0.02),
            )

    def test_run_monte_carlo_garch_falls_back_on_short_history(self):
        returns = _synthetic_returns(50)
        result = run_monte_carlo_garch(
            symbol="TEST", daily_returns=returns, horizon_days=10, simulations=500, seed=42
        )
        # Should behave identically to the flat-vol path since GARCH can't fit.
        expected = run_monte_carlo(
            symbol="TEST", daily_returns=returns, horizon_days=10, simulations=500, seed=42
        )
        assert result.probability_profit == expected.probability_profit
        assert result.var_95 == expected.var_95

    def test_run_monte_carlo_garch_with_enough_history(self):
        returns = _synthetic_returns(500)
        result = run_monte_carlo_garch(
            symbol="TEST", daily_returns=returns, horizon_days=10, simulations=500, seed=42
        )
        assert isinstance(result, MonteCarloResult)
        assert result.simulations_count == 500
        assert 0.0 <= result.probability_profit <= 1.0
