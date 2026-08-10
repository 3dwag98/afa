"""Tests for tradability screening: circuit locks, illiquidity, zombie stocks."""

import numpy as np
import pandas as pd
import pytest

from src.liquidity import (
    circuit_locked_days,
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
