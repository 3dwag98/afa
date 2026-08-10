"""Tests for tradability screening: circuit locks, illiquidity, zombie stocks."""

import numpy as np
import pandas as pd
import pytest

from src.liquidity import (
    DEFAULT_MIN_TRADED_VALUE_INR,
    assess_tradability,
    circuit_locked_days,
    median_traded_value,
    split_intraday_and_overnight,
    zero_return_days,
)


def _ohlcv(closes, volumes=None, ranges=None, opens=None):
    """Build an OHLCV frame from a close series.

    `ranges` gives each session's high-low spread; 0 marks a locked session.
    """
    n = len(closes)
    closes = np.asarray(closes, dtype=float)
    ranges = np.full(n, 1.0) if ranges is None else np.asarray(ranges, dtype=float)
    volumes = np.full(n, 1_000_000.0) if volumes is None else np.asarray(volumes, dtype=float)
    opens = closes if opens is None else np.asarray(opens, dtype=float)
    return pd.DataFrame(
        {
            "open": opens,
            "high": closes + ranges / 2,
            "low": closes - ranges / 2,
            "close": closes,
            "volume": volumes,
        },
        index=pd.bdate_range("2023-01-02", periods=n),
    )


class TestCircuitLockedDays:
    def test_flags_a_zero_range_session_at_the_upper_circuit(self):
        closes = [100.0, 110.0, 121.0]  # two consecutive 10% locks
        df = _ohlcv(closes, ranges=[1.0, 0.0, 0.0])

        locks = circuit_locked_days(df)

        assert list(locks) == [False, True, True]

    def test_flags_lower_circuit_locks_too(self):
        df = _ohlcv([100.0, 90.0], ranges=[1.0, 0.0])

        assert list(circuit_locked_days(df)) == [False, True]

    def test_a_zero_range_day_without_a_big_move_is_not_a_lock(self):
        """An untraded flat day has no range but also no move — that is
        illiquidity, caught by the zombie screen, not a circuit lock."""
        df = _ohlcv([100.0, 100.0], ranges=[1.0, 0.0])

        assert list(circuit_locked_days(df)) == [False, False]

    def test_a_big_move_with_a_real_range_is_not_a_lock(self):
        """A stock that ran 10% but traded through a range was executable."""
        df = _ohlcv([100.0, 110.0], ranges=[1.0, 5.0])

        assert list(circuit_locked_days(df)) == [False, False]

    def test_missing_columns_yield_no_flags(self):
        df = pd.DataFrame({"close": [100.0, 110.0]})

        assert not circuit_locked_days(df).any()


class TestZeroReturnDays:
    def test_flags_unchanged_closes(self):
        df = _ohlcv([100.0, 100.0, 101.0, 101.0])

        assert list(zero_return_days(df)) == [False, True, False, True]


class TestMedianTradedValue:
    def test_uses_the_median_not_the_mean(self):
        """One huge print must not make a dead ticker look liquid."""
        closes = [100.0] * 20
        volumes = [100.0] * 19 + [10_000_000.0]
        df = _ohlcv(closes, volumes=volumes)

        assert median_traded_value(df, window=20) == pytest.approx(10_000.0)

    def test_zero_when_columns_are_missing(self):
        assert median_traded_value(pd.DataFrame({"close": [1.0]})) == 0.0


class TestAssessTradability:
    def _liquid_series(self, n=80, seed=1):
        rng = np.random.default_rng(seed)
        closes = 500 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, n)))
        return _ohlcv(closes, volumes=np.full(n, 500_000.0))

    def test_a_liquid_active_stock_passes(self):
        report = assess_tradability(self._liquid_series())

        assert report.tradable is True
        assert report.reasons == []
        assert report.median_traded_value > DEFAULT_MIN_TRADED_VALUE_INR

    def test_a_zombie_stock_is_rejected(self):
        """The illiquidity illusion: a stock that barely trades prints
        unchanged closes, which suppresses variance and would otherwise rank
        it first in the low-volatility decile."""
        closes = np.repeat(np.arange(100.0, 116.0), 5)  # 80 sessions, 4 in 5 flat
        df = _ohlcv(closes, volumes=np.full(len(closes), 500_000.0))

        report = assess_tradability(df)

        assert report.tradable is False
        assert report.zero_return_fraction > 0.3
        assert any("zombie" in r or "unchanged" in r for r in report.reasons)

    def test_an_illiquid_stock_is_rejected_on_turnover(self):
        rng = np.random.default_rng(2)
        closes = 20 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 80)))
        df = _ohlcv(closes, volumes=np.full(80, 100.0))  # ~₹2,000/day

        report = assess_tradability(df)

        assert report.tradable is False
        assert any("illiquid" in r for r in report.reasons)

    def test_a_serially_circuit_locked_stock_is_rejected(self):
        """The operator-driven pump: momentum reads the printed run as
        strength, but there is no offer to lift."""
        rng = np.random.default_rng(3)
        closes = [100.0]
        ranges = [1.0]
        for _ in range(50):  # ordinary trading first
            closes.append(closes[-1] * float(np.exp(rng.normal(0, 0.01))))
            ranges.append(1.0)
        for _ in range(15):  # then 15 consecutive upper circuits
            closes.append(closes[-1] * 1.10)
            ranges.append(0.0)
        for _ in range(14):  # and back to trading, so the run is not "today"
            closes.append(closes[-1] * float(np.exp(rng.normal(0, 0.01))))
            ranges.append(1.0)
        df = _ohlcv(closes, ranges=ranges, volumes=np.full(len(closes), 500_000.0))

        report = assess_tradability(df)

        assert report.tradable is False
        assert report.locked_today is False
        assert any("circuit-driven" in r for r in report.reasons)

    def test_an_old_circuit_run_outside_the_window_does_not_disqualify(self):
        """The screen looks at the trailing window: a lock streak from a year
        ago is history, not a live tradability problem."""
        rng = np.random.default_rng(9)
        closes = [100.0]
        ranges = [1.0]
        for _ in range(10):
            closes.append(closes[-1] * 1.10)
            ranges.append(0.0)
        for _ in range(80):
            closes.append(closes[-1] * float(np.exp(rng.normal(0, 0.01))))
            ranges.append(1.0)
        df = _ohlcv(closes, ranges=ranges, volumes=np.full(len(closes), 500_000.0))

        assert assess_tradability(df).tradable is True

    def test_locked_on_the_decision_date_blocks_the_trade(self):
        rng = np.random.default_rng(4)
        closes = list(500 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 79))))
        ranges = [1.0] * 79
        closes.append(closes[-1] * 1.10)  # locks at the upper circuit today
        ranges.append(0.0)
        df = _ohlcv(closes, ranges=ranges, volumes=np.full(80, 500_000.0))

        report = assess_tradability(df)

        assert report.locked_today is True
        assert report.tradable is False
        assert any("no fill available" in r for r in report.reasons)

    def test_empty_history_is_untradable(self):
        report = assess_tradability(pd.DataFrame())

        assert report.tradable is False
        assert report.reasons == ["no price history"]

    def test_thresholds_are_tunable(self):
        rng = np.random.default_rng(2)
        closes = 20 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 80)))
        df = _ohlcv(closes, volumes=np.full(80, 100.0))

        assert assess_tradability(df).tradable is False
        assert assess_tradability(df, min_traded_value_inr=0.0).tradable is True


class TestSplitIntradayAndOvernight:
    def test_decomposes_the_close_to_close_move(self):
        # Day 2 gaps up 10% at the open, then gives back ~5% intraday.
        df = pd.DataFrame({
            "open": [100.0, 110.0],
            "high": [101.0, 111.0],
            "low": [99.0, 104.0],
            "close": [100.0, 104.5],
            "volume": [1e6, 1e6],
        }, index=pd.bdate_range("2023-01-02", periods=2))
        # Extra rows so dropping the first (gapless) row still leaves usable data.
        extra = pd.DataFrame({
            "open": [104.0, 105.0],
            "high": [106.0, 107.0],
            "low": [103.0, 104.0],
            "close": [105.0, 106.0],
            "volume": [1e6, 1e6],
        }, index=pd.bdate_range("2023-01-04", periods=2))
        df = pd.concat([df, extra])

        split = split_intraday_and_overnight(df)

        assert split is not None
        intraday, overnight = split
        assert overnight[0] == pytest.approx(0.10)
        assert intraday[0] == pytest.approx(104.5 / 110.0 - 1.0)

    def test_returns_none_without_open_prices(self):
        df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})

        assert split_intraday_and_overnight(df) is None

    def test_returns_none_on_too_little_history(self):
        df = _ohlcv([100.0, 101.0])


        assert split_intraday_and_overnight(df) is None
