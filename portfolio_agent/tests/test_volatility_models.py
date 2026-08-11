"""Tests for GJR-GARCH volatility forecasting and its Monte Carlo integration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from src.volatility_models import (
    clear_garch_parameter_cache,
    filter_conditional_variance,
    fit_garch_parameters,
    forecast_from_parameters,
    forecast_volatility,
    forecast_volatility_gap_aware,
    forecast_volatility_gap_aware_scheduled,
    forecast_volatility_scheduled,
    garch_parameter_cache_size,
    GarchForecast,
    GarchParameters,
    MIN_OBSERVATIONS,
    _fit_anchor,
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


# ---------------------------------------------------------------------------
# Scheduled refitting
#
# One MLE fit per ticker per day is ~4.5 million fits for the documented
# backtest, which is why use_garch_volatility was off by default. These tests
# pin the split that makes it affordable: the *fit* runs on a schedule, the
# *recursion* still runs every day.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_parameter_cache():
    clear_garch_parameter_cache()
    yield
    clear_garch_parameter_cache()


def _parameters(**overrides) -> GarchParameters:
    """A stationary GJR-GARCH with a real leverage effect."""
    defaults = dict(
        mu=0.0005, omega=2.0e-6, alpha=0.05, gamma=0.06, beta=0.88,
        distribution_df=6.0, n_observations=500,
    )
    defaults.update(overrides)
    return GarchParameters(**defaults)


class TestFitAnchor:
    """The fit window is truncated to a multiple of the interval so the cached
    parameters are a pure function of (symbol, anchor) — not of which worker
    process happened to see the ticker first."""

    def test_anchor_is_a_multiple_of_the_interval(self):
        assert _fit_anchor(1250, 21) == 1239  # 59 * 21
        assert _fit_anchor(1239, 21) == 1239
        assert _fit_anchor(1260, 21) == 1260

    def test_anchor_is_stable_across_a_refit_interval(self):
        """Every day inside one interval must resolve to the same fit window,
        or the schedule saves nothing."""
        anchors = {_fit_anchor(n, 21) for n in range(1239, 1260)}
        assert anchors == {1239}

    def test_no_anchor_below_the_fitting_minimum(self):
        assert _fit_anchor(MIN_OBSERVATIONS - 1, 21) is None

    def test_falls_back_to_the_full_window_when_the_schedule_undershoots(self):
        """Early in the sample the scheduled boundary sits below the fitting
        minimum; declining to fit there would be worse than fitting on
        everything available."""
        n = MIN_OBSERVATIONS  # 250: the scheduled boundary below it is 231
        assert (n // 21) * 21 < MIN_OBSERVATIONS
        assert _fit_anchor(n, 21) == n


class TestFilterAndForecast:
    """Filtering and forecasting are arithmetic and need no `arch` install —
    which is exactly the point of separating them from the fit."""

    def test_recursion_matches_the_gjr_definition(self):
        params = _parameters()
        returns = [0.01, -0.02, 0.005]

        result = filter_conditional_variance(params, returns)

        assert result is not None
        sigma2, last_residual = result

        expected = params.unconditional_variance
        for r in returns:
            eps = r - params.mu
            leverage = params.gamma if eps < 0 else 0.0
            expected = params.omega + (params.alpha + leverage) * eps ** 2 + params.beta * expected

        assert sigma2 == pytest.approx(expected)
        assert last_residual == pytest.approx(returns[-1] - params.mu)

    def test_negative_shocks_raise_variance_more_than_positive_ones(self):
        """gamma > 0 is the leverage effect, and it has to survive the split
        into cached parameters plus a live recursion."""
        params = _parameters()

        up = filter_conditional_variance(params, [0.03])
        down = filter_conditional_variance(params, [-0.03])

        assert up is not None and down is not None
        assert down[0] > up[0]

    def test_forecast_decays_toward_the_unconditional_variance(self):
        params = _parameters()
        # Start from a shock far above the long-run level.
        forecast = forecast_from_parameters(params, [0.10], horizon_days=200)

        assert forecast is not None
        variances = forecast.daily_sigma ** 2
        assert variances[0] > params.unconditional_variance
        # Monotone decay, converging on the attractor rather than diverging.
        assert np.all(np.diff(variances) < 0)
        assert variances[-1] == pytest.approx(params.unconditional_variance, rel=0.05)

    def test_forecast_carries_the_fitted_tail_and_leverage(self):
        params = _parameters(distribution_df=4.5, gamma=0.07)
        forecast = forecast_from_parameters(params, _synthetic_returns(30), horizon_days=5)

        assert forecast is not None
        assert forecast.distribution_df == 4.5
        assert forecast.leverage_gamma == 0.07
        assert forecast.persistence == pytest.approx(params.persistence)

    def test_state_is_current_even_when_the_parameters_are_not(self):
        """The whole justification for scheduling: stale parameters, live
        state. A calm run and a violent run through the *same* parameters must
        produce different forecasts."""
        params = _parameters()

        calm = forecast_from_parameters(params, [0.001] * 20, horizon_days=5)
        violent = forecast_from_parameters(params, [0.001] * 19 + [-0.12], horizon_days=5)

        assert calm is not None and violent is not None
        assert violent.daily_sigma[0] > 2 * calm.daily_sigma[0]

    def test_non_stationary_parameters_are_rejected_not_projected(self):
        params = _parameters(alpha=0.30, gamma=0.20, beta=0.75)  # persistence > 1
        assert params.persistence > 1.0
        assert not np.isfinite(params.unconditional_variance)

    def test_empty_series_has_no_state(self):
        assert filter_conditional_variance(_parameters(), []) is None

    def test_zero_horizon_is_not_a_forecast(self):
        assert forecast_from_parameters(_parameters(), [0.01], horizon_days=0) is None


class TestScheduledRefits:
    def test_fits_once_per_interval_not_once_per_call(self, monkeypatch):
        import src.volatility_models as vm

        calls = []

        def counting_fit(returns):
            calls.append(len(returns))
            return _parameters(n_observations=len(returns))

        monkeypatch.setattr(vm, "fit_garch_parameters", counting_fit)

        returns = _synthetic_returns(1260)
        # 21 consecutive trading days inside one refit interval.
        for n in range(1239, 1260):
            forecast = forecast_volatility_scheduled(
                returns[:n], horizon_days=5, symbol="ACME.NS", refit_interval_days=21
            )
            assert forecast is not None

        assert calls == [1239], "one fit for the whole interval, on the anchored window"

    def test_the_next_interval_refits(self, monkeypatch):
        import src.volatility_models as vm

        calls = []
        monkeypatch.setattr(
            vm, "fit_garch_parameters",
            lambda returns: (calls.append(len(returns)), _parameters())[1],
        )

        returns = _synthetic_returns(1300)
        for n in (1239, 1259, 1260, 1280):
            forecast_volatility_scheduled(
                returns[:n], horizon_days=5, symbol="ACME.NS", refit_interval_days=21
            )

        assert calls == [1239, 1260]

    def test_symbols_do_not_share_a_fit(self, monkeypatch):
        import src.volatility_models as vm

        calls = []
        monkeypatch.setattr(
            vm, "fit_garch_parameters",
            lambda returns: (calls.append(len(returns)), _parameters())[1],
        )

        returns = _synthetic_returns(1250)
        forecast_volatility_scheduled(returns, 5, symbol="ACME.NS", refit_interval_days=21)
        forecast_volatility_scheduled(returns, 5, symbol="OTHER.NS", refit_interval_days=21)

        assert len(calls) == 2

    def test_a_failed_fit_is_cached_too(self, monkeypatch):
        """Otherwise a ticker that never converges pays the full optimizer cost
        every single day — the exact cost the schedule exists to avoid,
        concentrated on the worst names."""
        import src.volatility_models as vm

        calls = []
        monkeypatch.setattr(
            vm, "fit_garch_parameters",
            lambda returns: (calls.append(len(returns)), None)[1],
        )

        returns = _synthetic_returns(1250)
        for _ in range(5):
            assert forecast_volatility_scheduled(
                returns, 5, symbol="BAD.NS", refit_interval_days=21
            ) is None

        assert len(calls) == 1
        assert garch_parameter_cache_size() == 1

    def test_interval_of_one_bypasses_the_cache(self, monkeypatch):
        """refit_interval_days=1 is documented as reproducing the old
        behaviour, which means going through forecast_volatility()."""
        import src.volatility_models as vm

        called = []
        monkeypatch.setattr(
            vm, "forecast_volatility",
            lambda returns, horizon: called.append(horizon) or None,
        )

        assert forecast_volatility_scheduled(
            _synthetic_returns(500), 5, symbol="ACME.NS", refit_interval_days=1
        ) is None
        assert called == [5]
        assert garch_parameter_cache_size() == 0

    def test_short_history_declines_to_fit(self):
        assert forecast_volatility_scheduled(
            _synthetic_returns(MIN_OBSERVATIONS - 1), 5,
            symbol="TINY.NS", refit_interval_days=21,
        ) is None

    def test_scheduled_gap_aware_adds_gap_variance(self, monkeypatch):
        import src.volatility_models as vm

        monkeypatch.setattr(vm, "fit_garch_parameters", lambda returns: _parameters())

        rng = np.random.default_rng(3)
        intraday = list(rng.normal(0.0, 0.008, 600))
        overnight = list(rng.normal(0.0, 0.03, 600))

        session = forecast_volatility_scheduled(
            intraday, 5, symbol="ACME.NS", refit_interval_days=21
        )
        gap_aware = forecast_volatility_gap_aware_scheduled(
            intraday, overnight, 5, symbol="ACME.NS", refit_interval_days=21
        )

        assert session is not None and gap_aware is not None
        assert gap_aware.gap_aware is True
        assert np.all(gap_aware.daily_sigma > session.daily_sigma)
        assert np.allclose(
            gap_aware.daily_sigma,
            np.sqrt(gap_aware.intraday_sigma ** 2 + gap_aware.overnight_sigma ** 2),
        )

    def test_session_and_close_legs_use_separate_cache_entries(self, monkeypatch):
        """The two legs are different series; sharing a cache slot would apply
        session parameters to close-to-close returns."""
        import src.volatility_models as vm

        monkeypatch.setattr(vm, "fit_garch_parameters", lambda returns: _parameters())

        rng = np.random.default_rng(11)
        intraday = list(rng.normal(0.0, 0.008, 600))
        overnight = list(rng.normal(0.0, 0.03, 600))

        forecast_volatility_scheduled(intraday, 5, symbol="ACME.NS", refit_interval_days=21)
        forecast_volatility_gap_aware_scheduled(
            intraday, overnight, 5, symbol="ACME.NS", refit_interval_days=21
        )

        assert garch_parameter_cache_size() == 2


@pytest.mark.skipif(not _ARCH_AVAILABLE, reason="arch package not installed")
class TestScheduledAgainstTheRealFit:
    def test_matches_the_per_call_path_on_a_refit_boundary(self):
        """On a day where the anchor equals the sample length, the scheduled
        path fits the same window as forecast_volatility() and filters zero
        extra observations, so the two must agree."""
        n = 21 * 30  # 630, an exact multiple of the interval
        returns = _synthetic_returns(n)

        scheduled = forecast_volatility_scheduled(
            returns, horizon_days=10, symbol="ACME.NS", refit_interval_days=21
        )
        per_call = forecast_volatility(returns, horizon_days=10)

        assert scheduled is not None and per_call is not None
        # arch's own forecaster and this recursion are the same arithmetic;
        # the tolerance covers its backcast initialization of sigma^2_0.
        assert np.allclose(scheduled.daily_sigma, per_call.daily_sigma, rtol=0.05)
        assert scheduled.persistence == pytest.approx(per_call.persistence, rel=1e-6)
        assert scheduled.distribution_df == pytest.approx(per_call.distribution_df, rel=1e-6)

    def test_fitted_parameters_reproduce_the_fitted_forecast(self):
        returns = _synthetic_returns(600)

        parameters = fit_garch_parameters(returns)
        assert parameters is not None
        assert 0 < parameters.persistence < 1
        assert parameters.omega > 0

        rebuilt = forecast_from_parameters(parameters, returns, horizon_days=10)
        direct = forecast_volatility(returns, horizon_days=10)

        assert rebuilt is not None and direct is not None
        assert np.allclose(rebuilt.daily_sigma, direct.daily_sigma, rtol=0.05)
