"""Tests for momentum crash protection: market regime detection + vol targeting."""

import numpy as np
import pandas as pd
import pytest

from src.regime import (
    DEFAULT_TREND_WINDOW,
    assess_market_regime,
    build_market_proxy,
    neutral_regime,
    realized_volatility,
    volatility_target_scalar,
)


def _series(values, start="2020-01-01"):
    return pd.Series(values, index=pd.bdate_range(start=start, periods=len(values)))


def _ramp_with_noise(start, end, n=400, noise=0.005, seed=1):
    """A deterministic price ramp plus multiplicative noise.

    Separating trend from volatility keeps these tests unambiguous: a random
    walk with a large daily sigma can finish above its own moving average
    despite a strongly negative drift, which would test the RNG rather than
    the regime logic.
    """
    rng = np.random.default_rng(seed)
    ramp = np.linspace(start, end, n)
    return _series(ramp * (1.0 + rng.normal(0.0, noise, n)))


class TestBuildMarketProxy:
    def test_expensive_stocks_do_not_dominate(self):
        # Both series double. Averaging returns makes the composite double
        # too, regardless of the wildly different price levels.
        cheap = _series([10.0, 20.0, 40.0])
        expensive = _series([5000.0, 10000.0, 20000.0])

        proxy = build_market_proxy({"CHEAP": cheap, "PRICEY": expensive})

        assert proxy is not None
        assert proxy.iloc[-1] / proxy.iloc[0] == pytest.approx(2.0)
        assert len(proxy) == 2  # one point per return

    def test_a_late_arrival_does_not_dent_the_composite(self):
        """Averaging rebased *levels* would drop the composite ~11% on the day
        a shorter-history ticker enters at its own base of 1.0, while both
        constituents rose — and assess_market_regime would read that
        construction artifact as a trend break and as realized volatility."""
        rising = _series(np.linspace(100.0, 190.0, 10))
        latecomer = pd.Series([50.0, 52.5, 55.0, 57.5, 60.0], index=rising.index[-5:])

        proxy = build_market_proxy({"OLD": rising, "NEW": latecomer})

        assert proxy is not None
        assert (proxy.pct_change().dropna() > 0).all()

    def test_handles_ragged_histories(self):
        long_series = _series(np.linspace(100.0, 110.0, 10))
        short_series = pd.Series([50.0, 55.0], index=long_series.index[-2:])

        proxy = build_market_proxy({"LONG": long_series, "SHORT": short_series})

        assert proxy is not None
        assert len(proxy) == 9  # one return per date after the first

    def test_lookback_truncates_the_inputs(self):
        """Only the trailing trend_window + 1 points can affect either test,
        and this runs once per scoring round over the whole universe."""
        series = _series(np.linspace(100.0, 200.0, 500))

        proxy = build_market_proxy({"A": series}, lookback=100)

        assert proxy is not None
        assert len(proxy) == 100

    def test_returns_none_without_usable_series(self):
        assert build_market_proxy({}) is None
        assert build_market_proxy({"A": _series([100.0])}) is None
        assert build_market_proxy({"A": _series([0.0, 0.0])}) is None


class TestRealizedVolatility:
    def test_annualizes_daily_volatility(self):
        rng = np.random.default_rng(3)
        daily_vol = 0.01
        prices = _series(100 * np.exp(np.cumsum(rng.normal(0, daily_vol, 500))))

        vol = realized_volatility(prices, window=250)

        # 1% daily ~= 15.9% annualized; allow sampling error on 250 draws.
        assert vol == pytest.approx(daily_vol * np.sqrt(252), rel=0.20)

    def test_requires_a_full_window(self):
        assert realized_volatility(_series([100.0] * 30), window=60) is None


class TestVolatilityTargetScalar:
    def test_halves_exposure_at_double_target_volatility(self):
        assert volatility_target_scalar(0.40, target_volatility=0.20) == pytest.approx(0.5)

    def test_never_levers_above_one(self):
        # A very calm stock would imply 4x leverage; the cap keeps it at 1.0 so
        # the platform's position limits stay the binding constraint.
        assert volatility_target_scalar(0.05, target_volatility=0.20) == 1.0

    def test_floors_extreme_volatility(self):
        assert volatility_target_scalar(5.0, target_volatility=0.20, min_scale=0.25) == 0.25

    def test_unknown_volatility_is_neutral(self):
        assert volatility_target_scalar(None) == 1.0
        assert volatility_target_scalar(0.0) == 1.0


class TestAssessMarketRegime:
    def test_short_history_is_fail_neutral(self):
        regime = assess_market_regime(_series([100.0] * 50))

        assert regime.label == "unknown"
        assert regime.exposure_scalar == 1.0
        assert regime.blocks_new_entries is False

    def test_rising_calm_market_is_risk_on(self):
        regime = assess_market_regime(_ramp_with_noise(100, 200, noise=0.004))

        assert regime.label == "risk_on"
        assert regime.trend_ok is True
        assert regime.exposure_scalar > 0

    def test_bear_market_with_vol_spike_blocks_new_entries(self):
        """The panic state — downtrend plus elevated volatility — is exactly
        where momentum crashes, so exposure collapses to bear_exposure."""
        falling_and_wild = _ramp_with_noise(200, 100, noise=0.05, seed=11)

        regime = assess_market_regime(falling_and_wild, target_volatility=0.20)

        assert regime.label == "crash_risk"
        assert regime.trend_ok is False
        assert regime.exposure_scalar == 0.0
        assert regime.blocks_new_entries is True
        assert "crash risk" in regime.reason

    def test_bear_exposure_can_dampen_instead_of_standing_down(self):
        falling_and_wild = _ramp_with_noise(200, 100, noise=0.05, seed=11)

        regime = assess_market_regime(
            falling_and_wild, target_volatility=0.20, bear_exposure=0.3
        )

        assert regime.label == "crash_risk"
        assert regime.exposure_scalar == pytest.approx(0.3)
        assert regime.blocks_new_entries is False

    def test_quiet_downtrend_dampens_rather_than_halts(self):
        """A drift lower without a volatility spike is a warning, not a stop:
        the crash literature puts the danger in the rebound, not the decline."""
        falling_calm = _ramp_with_noise(200, 100, noise=0.002, seed=5)

        regime = assess_market_regime(falling_calm, target_volatility=0.20)

        assert regime.trend_ok is False
        assert regime.label == "elevated_vol"
        assert 0.0 < regime.exposure_scalar < 1.0

    def test_uptrend_with_vol_spike_scales_down_but_stays_invested(self):
        rising_wild = _ramp_with_noise(100, 200, noise=0.05, seed=9)

        regime = assess_market_regime(rising_wild, target_volatility=0.20)

        assert regime.trend_ok is True
        assert regime.label == "elevated_vol"
        assert 0.0 < regime.exposure_scalar < 1.0
        assert regime.blocks_new_entries is False

    def test_trend_window_governs_the_history_requirement(self):
        prices = _ramp_with_noise(100, 150, n=120, noise=0.004)

        assert assess_market_regime(prices).label == "unknown"
        assert assess_market_regime(prices, trend_window=100).label != "unknown"
        assert DEFAULT_TREND_WINDOW == 200


class TestNeutralRegime:
    def test_leaves_exposure_untouched(self):
        regime = neutral_regime("because")

        assert regime.exposure_scalar == 1.0
        assert regime.vol_scalar == 1.0
        assert regime.trend_ok is True
        assert regime.blocks_new_entries is False
        assert regime.reason == "because"
