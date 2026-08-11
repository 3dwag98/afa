"""Tests for GJR-GARCH volatility forecasting and its Monte Carlo integration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from src.volatility_models import (
    _PARAMETER_CACHE,
    clear_parameter_cache,
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


@pytest.mark.skipif(not _ARCH_AVAILABLE, reason="`arch` not installed")
class TestRefitScheduling:
    """Refit scheduling (Phase 0.5 / D5).

    The platform's most sophisticated volatility model was effectively dead
    code: scoring every ticker on every bar of the documented backtest is
    ~4.5 million MLE fits. These pin the three properties that make the
    scheduled path a substitute for it rather than an approximation of it —
    it is exact at the refit boundary, it still consumes every new bar, and it
    does not depend on the order calls arrive in.
    """

    @staticmethod
    def _clustered_returns(n: int, seed: int = 11) -> np.ndarray:
        """A GJR-flavoured series, so the leverage term is actually identified."""
        rng = np.random.default_rng(seed)
        out = np.zeros(n)
        variance = 1e-4
        for i in range(n):
            shock = rng.standard_normal() * np.sqrt(variance)
            out[i] = shock
            variance = 2e-6 + (0.05 + (0.08 if shock < 0 else 0.0)) * shock ** 2 + 0.88 * variance
        return out

    def test_matches_arch_forecast_exactly_at_a_refit_boundary(self):
        """The closed-form recursion must reproduce arch's own analytic forecast.

        This is the load-bearing claim of the whole change. With the history
        length landing exactly on a refit boundary there is nothing to filter
        forward, so any disagreement here is a bug in how the coefficients or
        the terminal state are carried out of the fit — not a modelling
        approximation.
        """
        from arch import arch_model

        returns = self._clustered_returns(1197)  # 1197 = 57 * 21, an exact boundary
        clear_parameter_cache()

        model = arch_model(
            returns * 100.0, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t"
        )
        fitted = model.fit(disp="off", show_warning=False)
        expected = (
            np.sqrt(np.asarray(fitted.forecast(horizon=20, reindex=False).variance.values[-1]))
            / 100.0
        )

        scheduled = forecast_volatility(
            returns, horizon_days=20, refit_interval_days=21, cache_key="EXACT"
        )
        assert scheduled is not None
        np.testing.assert_allclose(scheduled.daily_sigma, expected, rtol=1e-10, atol=1e-14)

    def test_new_bars_still_update_conditional_volatility(self):
        """Scheduling refits must not freeze the forecast between them.

        The point of GARCH is that today's shock changes tomorrow's variance.
        A cache that returned the same path for every bar in the refit window
        would be fast and useless, and would look identical in a timing test.
        """
        returns = self._clustered_returns(1400)
        clear_parameter_cache()

        one_day_ahead = [
            forecast_volatility(
                returns[: 1200 + k], horizon_days=5, refit_interval_days=21, cache_key="LIVE"
            ).daily_sigma[0]
            for k in range(20)  # a full window between refits
        ]

        assert len(set(np.round(one_day_ahead, 12))) == len(one_day_ahead)

    def test_result_does_not_depend_on_call_order(self):
        """Determinism under parallelism, which is the reason for the anchoring.

        Workers hold independent caches and see bars in whatever order the
        scheduler hands them out. Anchoring the fitted window to the history
        length rather than to "when did we last fit this symbol" is what makes
        a per-process cache pure memoization instead of a source of drift.
        """
        returns = self._clustered_returns(1300)
        lengths = list(range(1210, 1240))

        clear_parameter_cache()
        forward = {
            n: forecast_volatility(
                returns[:n], horizon_days=8, refit_interval_days=21, cache_key="ORDER"
            ).daily_sigma
            for n in lengths
        }

        clear_parameter_cache()  # a fresh worker, walking the same bars backwards
        backward = {
            n: forecast_volatility(
                returns[:n], horizon_days=8, refit_interval_days=21, cache_key="ORDER"
            ).daily_sigma
            for n in reversed(lengths)
        }

        for n in lengths:
            np.testing.assert_array_equal(forward[n], backward[n])

    def test_cache_key_is_required_for_the_cache_to_engage(self):
        """Without an identity there is nothing to memoize against, so it refits."""
        returns = self._clustered_returns(1250)
        clear_parameter_cache()

        forecast_volatility(returns, horizon_days=5, refit_interval_days=21, cache_key=None)
        assert len(_PARAMETER_CACHE) == 0

        forecast_volatility(returns, horizon_days=5, refit_interval_days=21, cache_key="KEYED")
        assert len(_PARAMETER_CACHE) == 1

    def test_one_fit_serves_a_whole_refit_window(self):
        """The saving itself: 21 bars of scoring cost one MLE, not 21."""
        returns = self._clustered_returns(1400)
        clear_parameter_cache()

        for k in range(21):
            forecast_volatility(
                returns[: 1197 + k], horizon_days=5, refit_interval_days=21, cache_key="ONE"
            )

        assert len(_PARAMETER_CACHE) == 1

    def test_corrected_data_invalidates_the_cached_fit(self):
        """A cache entry may only serve the window it was fitted on.

        Keying on the symbol alone would let a re-download or a corrected bar
        be scored with a fit of the prices it replaced.
        """
        returns = self._clustered_returns(1197)
        clear_parameter_cache()
        original = forecast_volatility(
            returns, horizon_days=5, refit_interval_days=21, cache_key="STALE"
        )

        corrected = returns.copy()
        corrected[500] *= 6.0  # a bad bar, restated
        revised = forecast_volatility(
            corrected, horizon_days=5, refit_interval_days=21, cache_key="STALE"
        )

        assert len(_PARAMETER_CACHE) == 2
        assert not np.array_equal(original.daily_sigma, revised.daily_sigma)

    def test_history_too_short_for_an_anchored_window_declines(self):
        """Rounding the window down below the fit minimum returns None.

        Silently fitting the full history instead would make the estimator
        switch mid-backtest depending on where the anchor happened to fall.
        """
        returns = self._clustered_returns(MIN_OBSERVATIONS + 1)
        assert (
            forecast_volatility(
                returns, horizon_days=5, refit_interval_days=21, cache_key="SHORT"
            )
            is None
        )

    def test_gap_aware_path_namespaces_its_cache_entry(self):
        """The session fit and the close-to-close fit are different models.

        Same symbol, different data; sharing a cache entry would serve one
        where the other was asked for.
        """
        intraday = self._clustered_returns(1197, seed=2)
        overnight = self._clustered_returns(1197, seed=3) * 0.4
        clear_parameter_cache()

        forecast_volatility(intraday, horizon_days=5, refit_interval_days=21, cache_key="NS")
        gap_aware = forecast_volatility_gap_aware(
            intraday, overnight, horizon_days=5, refit_interval_days=21, cache_key="NS"
        )

        assert gap_aware is not None
        assert gap_aware.gap_aware
        assert len(_PARAMETER_CACHE) == 2
