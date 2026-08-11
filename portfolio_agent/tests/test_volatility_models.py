"""Tests for GJR-GARCH volatility forecasting and its Monte Carlo integration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from src.volatility_models import (
    clear_garch_parameter_cache,
    forecast_volatility,
    forecast_volatility_gap_aware,
    GarchForecast,
    MIN_OBSERVATIONS,
)
from src.monte_carlo import run_monte_carlo, run_monte_carlo_garch, MonteCarloResult


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


class TestGarchRefitScheduling:
    """One MLE per ticker per day is ~4.5 million fits for the documented
    backtest. Parameters are stable over weeks; the conditional variance is
    not. Only the fitting is scheduled."""

    @staticmethod
    def _garch_series(n=1200, seed=0):
        rng = np.random.default_rng(seed)
        returns = np.zeros(n)
        variance = 1.0
        for i in range(1, n):
            previous = returns[i - 1]
            variance = (
                0.05 + (0.06 + (0.08 if previous < 0 else 0.0)) * previous ** 2 + 0.88 * variance
            )
            returns[i] = np.sqrt(variance) * rng.standard_t(6) / np.sqrt(6 / 4)
        return returns / 100.0

    def test_recursion_reproduces_the_optimizer_forecast(self):
        """forecast_from_parameters must be arithmetically identical to what
        arch itself projects, or the whole scheduling idea is a silent
        approximation rather than an optimization."""
        pytest.importorskip("arch")
        from arch import arch_model

        returns = self._garch_series()
        fitted = arch_model(
            returns * 100, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t"
        ).fit(disp="off", show_warning=False)
        reference = np.sqrt(
            np.asarray(fitted.forecast(horizon=20, reindex=False).variance.values[-1])
        ) / 100.0

        clear_garch_parameter_cache()
        ours = forecast_volatility(list(returns), 20)

        assert ours is not None
        assert np.allclose(ours.daily_sigma, reference, rtol=1e-10)

    def test_parameters_are_reused_within_the_refit_interval(self, monkeypatch):
        pytest.importorskip("arch")
        returns = self._garch_series(n=600)

        import portfolio_agent.src.volatility_models as vm

        fits = {"count": 0}
        real_fit = vm.fit_garch_parameters

        def counting_fit(returns_pct):
            fits["count"] += 1
            return real_fit(returns_pct)

        monkeypatch.setattr(vm, "fit_garch_parameters", counting_fit)
        vm.clear_garch_parameter_cache()

        # 42 successive "days" at a 21-day refit interval: 2 fits, not 42.
        for day in range(42):
            vm.forecast_volatility(list(returns[: 500 + day]), 20, cache_key="T")

        assert fits["count"] == 2

    def test_no_cache_key_refits_every_call(self, monkeypatch):
        pytest.importorskip("arch")
        returns = self._garch_series(n=600)

        import portfolio_agent.src.volatility_models as vm

        fits = {"count": 0}
        real_fit = vm.fit_garch_parameters

        def counting_fit(returns_pct):
            fits["count"] += 1
            return real_fit(returns_pct)

        monkeypatch.setattr(vm, "fit_garch_parameters", counting_fit)
        vm.clear_garch_parameter_cache()

        for day in range(5):
            vm.forecast_volatility(list(returns[: 500 + day]), 20)

        assert fits["count"] == 5

    def test_the_variance_path_still_tracks_the_latest_bar(self):
        """The parameters are cached; the conditional variance is not. A fresh
        shock must move the forecast even when no refit happened."""
        pytest.importorskip("arch")
        returns = self._garch_series(n=600)
        clear_garch_parameter_cache()

        calm = forecast_volatility(list(returns), 5, cache_key="T")
        shocked = forecast_volatility(list(returns) + [-0.15], 5, cache_key="T")

        assert calm is not None and shocked is not None
        assert shocked.daily_sigma[0] > calm.daily_sigma[0] * 1.5

    def test_gap_aware_fit_is_namespaced_away_from_close_to_close(self):
        """Two different models of two different series must not share a
        cache slot."""
        pytest.importorskip("arch")
        import portfolio_agent.src.volatility_models as vm

        session = self._garch_series(n=600, seed=1)
        gaps = self._garch_series(n=600, seed=2) * 0.5

        # vm.* consistently: the repo's `src` symlink means
        # src.volatility_models and portfolio_agent.src.volatility_models are
        # distinct module objects with distinct caches.
        vm.clear_garch_parameter_cache()
        vm.forecast_volatility(list(session), 5, cache_key="T")
        vm.forecast_volatility_gap_aware(list(session), list(gaps), 5, cache_key="T")

        assert set(vm._PARAMETER_CACHE) == {"T", "T::session"}
