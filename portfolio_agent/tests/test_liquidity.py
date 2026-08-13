"""Tests for tradability screening: circuit locks, illiquidity, zombie stocks."""

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.src.liquidity import (
    circuit_locked_days,
    lower_circuit_locked_days,
    operator_trap_days,
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


class TestDynamicCircuitBands:
    """Exchanges impose ad-hoc 1% and 2% ASM/GSM bands on volatile or
    operator-suspected scrips, not just the 5/10/20% defaults."""

    @pytest.mark.parametrize("band", [0.01, 0.02, 0.05, 0.10, 0.20])
    def test_every_statutory_band_is_detected_in_both_directions(self, band):
        up = _ohlcv([100.0, 100.0 * (1 + band)], ranges=[1.0, 0.0])
        down = _ohlcv([100.0, 100.0 * (1 - band)], ranges=[1.0, 0.0])

        assert list(circuit_locked_days(up)) == [False, True]
        assert list(circuit_locked_days(down)) == [False, True]

    def test_a_two_percent_lock_is_no_longer_missed(self):
        """The regression: a 4.5% floor waved 2% locks straight through to the
        momentum ranking, so the strategy would BUY a stock with no offer."""
        df = _ohlcv([100.0, 102.0], ranges=[1.0, 0.0])

        assert bool(circuit_locked_days(df).iloc[-1]) is True

    def test_tick_rounding_around_a_band_still_counts(self):
        df = _ohlcv([100.0, 101.8], ranges=[1.0, 0.0])  # 1.8%, just under 2%

        assert bool(circuit_locked_days(df).iloc[-1]) is True

    def test_a_move_between_bands_is_not_a_lock(self):
        """Band matching, not a floor: 3.5% is not a statutory limit, so a
        zero-range day there is thin trading, not a lock."""
        df = _ohlcv([100.0, 103.5], ranges=[1.0, 0.0])

        assert bool(circuit_locked_days(df).iloc[-1]) is False

    def test_moves_beyond_the_widest_band_still_count(self):
        df = _ohlcv([100.0, 135.0], ranges=[1.0, 0.0])

        assert bool(circuit_locked_days(df).iloc[-1]) is True

    def test_bands_are_configurable(self):
        df = _ohlcv([100.0, 102.0], ranges=[1.0, 0.0])

        assert bool(circuit_locked_days(df, bands=(0.10, 0.20)).iloc[-1]) is False


def _ohlcv_closing_at(closes, closes_at="high", locked=None):
    """OHLCV where flagged sessions close at their session high (or low).

    `locked[i]` True means session i traded a real intraday range and then shut
    pinned at the limit — the operator footprint the zero-range detector misses.
    """
    n = len(closes)
    closes = np.asarray(closes, dtype=float)
    locked = np.zeros(n, dtype=bool) if locked is None else np.asarray(locked, dtype=bool)

    high = np.where(locked & (closes_at == "high"), closes, closes + 1.0)
    low = np.where(locked & (closes_at == "low"), closes, closes - 1.0)
    return pd.DataFrame(
        {
            "open": closes,
            "high": high,
            "low": low,
            "close": closes,
            "volume": np.full(n, 1_000_000.0),
        },
        index=pd.bdate_range("2023-01-02", periods=n),
    )


class TestOnePercentBand:
    """ASM/GSM impose 1% bands on exactly the names most likely to be
    operator-driven, so it is the band that matters most."""

    def test_a_one_percent_lock_is_detected(self):
        df = _ohlcv([100.0, 101.0], ranges=[1.0, 0.0])

        assert bool(circuit_locked_days(df).iloc[-1]) is True

    def test_tick_rounding_around_the_one_percent_band_still_counts(self):
        df = _ohlcv([100.0, 100.9], ranges=[1.0, 0.0])

        assert bool(circuit_locked_days(df).iloc[-1]) is True

    def test_the_tolerance_narrows_with_the_band(self):
        """A flat 40bp window around a 1% band would accept anything from 0.6%
        to 1.4% — half the quiet days on the tape. 0.6% is not a limit move."""
        df = _ohlcv([100.0, 100.6], ranges=[1.0, 0.0])

        assert bool(circuit_locked_days(df).iloc[-1]) is False

    def test_wider_bands_keep_the_absolute_tolerance(self):
        """The relative cap only binds below 2%; the ladder already in use is
        unchanged."""
        df = _ohlcv([100.0, 104.6], ranges=[1.0, 0.0])  # 4.6%, within 40bp of 5%

        assert bool(circuit_locked_days(df).iloc[-1]) is True


class TestOperatorTrapDays:
    def test_flags_a_session_that_ran_up_and_locked_at_the_close(self):
        """The whole point: this session traded a real range, so the zero-range
        detector waves it through, yet it shut with no offer left."""
        df = _ohlcv_closing_at([100.0, 102.0], locked=[False, True])

        assert bool(circuit_locked_days(df).iloc[-1]) is False
        assert bool(operator_trap_days(df).iloc[-1]) is True

    def test_a_close_at_the_high_without_a_band_move_is_not_a_trap(self):
        df = _ohlcv_closing_at([100.0, 103.5], locked=[False, True])

        assert bool(operator_trap_days(df).iloc[-1]) is False

    def test_a_band_move_that_did_not_close_at_the_high_is_not_a_trap(self):
        df = _ohlcv_closing_at([100.0, 102.0], locked=[False, False])

        assert bool(operator_trap_days(df).iloc[-1]) is False

    def test_a_lower_circuit_close_is_not_an_upper_trap(self):
        """Asymmetric on purpose: a down-lock is an exit that cannot be taken,
        not an entry that cannot be filled."""
        df = _ohlcv_closing_at([100.0, 98.0], closes_at="low", locked=[False, True])

        assert bool(operator_trap_days(df).iloc[-1]) is False
        assert bool(lower_circuit_locked_days(df).iloc[-1]) is True

    def test_missing_columns_yield_no_flags(self):
        df = pd.DataFrame({"close": [100.0, 102.0]})

        assert not operator_trap_days(df).any()
        assert not lower_circuit_locked_days(df).any()
