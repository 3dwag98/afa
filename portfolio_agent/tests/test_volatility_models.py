"""Tests for GJR-GARCH volatility forecasting and its Monte Carlo integration."""

import sys
from pathlib import Path


import numpy as np
import pytest

from portfolio_agent.src.volatility_models import (
    forecast_volatility,
    forecast_volatility_gap_aware,
    GarchForecast,
    MIN_OBSERVATIONS,
)
from portfolio_agent.src.monte_carlo import run_monte_carlo, run_monte_carlo_garch, MonteCarloResult


try:  # the gap-aware tests need a real GJR-GARCH fit
    import arch  # noqa: F401
    _ARCH_AVAILABLE = True
except ImportError:
    _ARCH_AVAILABLE = False


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


class TestGapAwareForecast:
    """GARCH is a model of the trading session; overnight gaps are jumps that
    arrive while the market is shut, and folding them into the recursion
    distorts the fitted parameters."""

    @staticmethod
    def _series(n=600, session_vol=0.008, gap_vol=0.03, seed=17):
        rng = np.random.default_rng(seed)
        intraday = rng.normal(0.0, session_vol, n)
        overnight = rng.normal(0.0, gap_vol, n)
        return list(intraday), list(overnight)

    def test_returns_none_without_enough_gap_history(self):
        intraday, overnight = self._series(n=600)

        assert forecast_volatility_gap_aware(intraday, overnight[:10], 5) is None

    def test_returns_none_when_the_session_fit_is_unavailable(self):
        intraday, overnight = self._series(n=50)

        assert forecast_volatility_gap_aware(intraday, overnight, 5) is None

    @pytest.mark.skipif(not _ARCH_AVAILABLE, reason="arch package not installed")
    def test_total_volatility_exceeds_the_session_alone(self):
        """Gap risk is added, not substituted: a stock whose sessions are calm
        but which gaps hard overnight is not a low-volatility stock."""
        intraday, overnight = self._series()

        session_only = forecast_volatility(intraday, 5)
        gap_aware = forecast_volatility_gap_aware(intraday, overnight, 5)

        assert session_only is not None and gap_aware is not None
        assert gap_aware.gap_aware is True
        assert np.all(gap_aware.daily_sigma > session_only.daily_sigma)

    @pytest.mark.skipif(not _ARCH_AVAILABLE, reason="arch package not installed")
    def test_components_combine_in_quadrature(self):
        intraday, overnight = self._series()

        gap_aware = forecast_volatility_gap_aware(intraday, overnight, 5)

        assert gap_aware is not None
        expected = np.sqrt(gap_aware.intraday_sigma ** 2 + gap_aware.overnight_sigma ** 2)
        assert np.allclose(gap_aware.daily_sigma, expected)

    @pytest.mark.skipif(not _ARCH_AVAILABLE, reason="arch package not installed")
    def test_close_to_close_fit_is_not_gap_aware(self):
        intraday, overnight = self._series()
        close_to_close = [i + o for i, o in zip(intraday, overnight)]

        forecast = forecast_volatility(close_to_close, 5)

        assert forecast is not None
        assert forecast.gap_aware is False
        assert forecast.overnight_sigma is None
