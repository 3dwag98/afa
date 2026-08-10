"""Tests for indicators module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.indicators import calculate_adx, calculate_indicators, calculate_all_indicators
from src.models import IndicatorSnapshot


def generate_synthetic_data(rows: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data.
    
    Args:
        rows: Number of rows to generate.
        seed: Random seed for reproducibility.
    
    Returns:
        DataFrame with open, high, low, close, volume columns.
    """
    np.random.seed(seed)
    
    # Generate dates
    end_date = datetime.now()
    dates = [end_date - timedelta(days=i) for i in range(rows)]
    dates.reverse()
    
    # Generate prices using random walk
    close = 100 + np.cumsum(np.random.randn(rows) * 2)
    
    # Ensure close prices are positive
    close = np.abs(close) + 1
    
    # Generate OHLC from close
    daily_range = np.abs(np.random.randn(rows)) * 2
    high = close + daily_range
    low = close - daily_range
    open_price = low + np.random.rand(rows) * (high - low)
    
    # Generate volume
    volume = np.random.randint(1000000, 10000000, size=rows)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    return df


class TestCalculateIndicators:
    """Tests for calculate_indicators function."""
    
    def test_sma_values_not_nan_after_warmup(self):
        """Test that SMA values are not NaN after warmup period."""
        df = generate_synthetic_data(300)
        
        snapshot = calculate_indicators("TEST", df)
        
        # After 300 rows, all SMAs should have valid values
        assert snapshot.sma20 is not None
        assert snapshot.sma50 is not None
        assert snapshot.sma200 is not None
        
        # Verify they are floats
        assert isinstance(snapshot.sma20, float)
        assert isinstance(snapshot.sma50, float)
        assert isinstance(snapshot.sma200, float)
    
    def test_donchian_upper_equals_rolling_max_of_high(self):
        """Test that Donchian upper equals rolling max of high."""
        df = generate_synthetic_data(300)
        
        snapshot = calculate_indicators("TEST", df)
        
        # Calculate expected Donchian upper manually
        expected_donchian = df['high'].rolling(window=20).max().iloc[-1]
        
        assert snapshot.donchian_upper_20 is not None
        assert abs(snapshot.donchian_upper_20 - expected_donchian) < 1e-6
    
    def test_atr_is_positive(self):
        """Test that ATR is positive."""
        df = generate_synthetic_data(300)
        
        snapshot = calculate_indicators("TEST", df)
        
        assert snapshot.atr14 is not None
        assert snapshot.atr14 > 0
    
    def test_short_data_returns_none_for_sma200(self):
        """Test that short data returns None for SMA200."""
        # Generate only 100 rows (not enough for SMA200)
        df = generate_synthetic_data(100)
        
        snapshot = calculate_indicators("TEST", df)
        
        # SMA20 and SMA50 should be available
        assert snapshot.sma20 is not None
        assert snapshot.sma50 is not None
        
        # SMA200 should be None (not enough data)
        assert snapshot.sma200 is None
    
    def test_volume_ratio_with_zero_volume(self):
        """Test that volume_ratio is None when volume is zero."""
        df = generate_synthetic_data(300)
        # Set latest volume to 0
        df.loc[df.index[-1], 'volume'] = 0
        
        snapshot = calculate_indicators("TEST", df)
        
        assert snapshot.volume_ratio is None
    
    def test_daily_log_return(self):
        """Test daily log return calculation."""
        df = generate_synthetic_data(300)
        
        snapshot = calculate_indicators("TEST", df)
        
        assert snapshot.daily_log_return is not None
        
        # Manually calculate expected log return
        expected_log_return = np.log(df['close'].iloc[-1] / df['close'].iloc[-2])
        
        assert abs(snapshot.daily_log_return - expected_log_return) < 1e-6
    
    def test_prev_donchian_upper_20(self):
        """Test prev Donchian upper is shifted by 1 day."""
        df = generate_synthetic_data(300)
        
        snapshot = calculate_indicators("TEST", df)
        
        # Calculate expected prev Donchian
        donchian_series = df['high'].rolling(window=20).max()
        expected_prev = donchian_series.shift(1).iloc[-1]
        
        assert snapshot.prev_donchian_upper_20 is not None
        assert abs(snapshot.prev_donchian_upper_20 - expected_prev) < 1e-6
    
    def test_does_not_mutate_input_df(self):
        """Test that input DataFrame is not mutated."""
        df = generate_synthetic_data(300)
        
        # Store original values
        original_close = df['close'].copy()
        original_high = df['high'].copy()
        
        calculate_indicators("TEST", df)
        
        # Check that original values are unchanged
        pd.testing.assert_series_equal(df['close'], original_close)
        pd.testing.assert_series_equal(df['high'], original_high)
    
    def test_returns_indicator_snapshot(self):
        """Test that function returns IndicatorSnapshot model."""
        df = generate_synthetic_data(300)
        
        snapshot = calculate_indicators("TEST", df)
        
        assert isinstance(snapshot, IndicatorSnapshot)
        assert snapshot.symbol == "TEST"


class TestCalculateAllIndicators:
    """Tests for calculate_all_indicators function."""
    
    def test_calculates_for_multiple_tickers(self):
        """Test calculation for multiple tickers."""
        data = {
            "AAPL": generate_synthetic_data(300),
            "GOOGL": generate_synthetic_data(300),
            "MSFT": generate_synthetic_data(300)
        }
        
        results = calculate_all_indicators(data)
        
        assert len(results) == 3
        
        symbols = [r.symbol for r in results]
        assert "AAPL" in symbols
        assert "GOOGL" in symbols
        assert "MSFT" in symbols
    
    def test_skips_invalid_tickers_missing_columns(self):
        """Test that invalid tickers with missing columns are skipped."""
        data = {
            "VALID": generate_synthetic_data(300),
            "INVALID": pd.DataFrame({'open': [1, 2, 3]})  # Missing required columns
        }
        
        results = calculate_all_indicators(data)
        
        assert len(results) == 1
        assert results[0].symbol == "VALID"
    
    def test_skips_empty_dataframe(self):
        """Test that empty DataFrames are skipped."""
        data = {
            "VALID": generate_synthetic_data(300),
            "EMPTY": pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        }
        
        results = calculate_all_indicators(data)
        
        assert len(results) == 1
        assert results[0].symbol == "VALID"
    
    def test_returns_list_of_snapshots(self):
        """Test that function returns list of IndicatorSnapshots."""
        data = {
            "TICKER": generate_synthetic_data(300)
        }
        
        results = calculate_all_indicators(data)
        
        assert isinstance(results, list)
        assert all(isinstance(r, IndicatorSnapshot) for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestCalculateADX:
    """ADX is what separates a chop from a trend at the same distance from the
    moving average, so the regime map depends on it being right."""

    @staticmethod
    def _frame(closes, width=1.0):
        closes = np.asarray(closes, dtype=float)
        return pd.DataFrame(
            {
                'open': closes,
                'high': closes + width,
                'low': closes - width,
                'close': closes,
            },
            index=pd.bdate_range("2022-01-03", periods=len(closes)),
        )

    def test_a_clean_trend_scores_high(self):
        rising = self._frame([100 + i for i in range(120)])

        assert calculate_adx(rising).dropna().iloc[-1] > 40

    def test_an_oscillating_market_scores_low(self):
        chop = self._frame([100 + (2 if i % 2 else -2) for i in range(120)])

        assert calculate_adx(chop).dropna().iloc[-1] < 25

    def test_direction_does_not_change_the_strength(self):
        """ADX measures persistence, not sign: a clean downtrend is as strong
        a trend as a clean uptrend."""
        up = calculate_adx(self._frame([100 + i for i in range(120)])).dropna().iloc[-1]
        down = calculate_adx(self._frame([100 - i * 0.5 for i in range(120)])).dropna().iloc[-1]

        assert down == pytest.approx(up, rel=0.05)

    def test_a_close_only_frame_still_produces_a_reading(self):
        """Index series are often cached as closes alone; the formulas stay
        well defined with high = low = close."""
        closes = pd.DataFrame({'close': [100 + i for i in range(120)]})

        value = calculate_adx(closes).dropna().iloc[-1]

        assert 0 <= value <= 100

    def test_a_flat_series_is_unmeasurable_rather_than_zero(self):
        """Zero range carries no directional information at all, and 0/0
        reported as 0 would read as a confident 'no trend'."""
        flat = pd.DataFrame({'close': [100.0] * 120})

        assert calculate_adx(flat).dropna().empty

    def test_values_stay_within_bounds(self):
        rng = np.random.default_rng(9)
        closes = 100 + np.cumsum(rng.normal(0, 1.0, 300))
        frame = self._frame(closes, width=1.5)

        values = calculate_adx(frame).dropna()

        assert values.min() >= 0
        assert values.max() <= 100

    def test_too_little_history_returns_nan(self):
        assert calculate_adx(pd.DataFrame({'close': [100.0]})).isna().all()

    def test_a_frame_without_a_close_column_returns_nan(self):
        assert calculate_adx(pd.DataFrame({'high': [1.0, 2.0]})).isna().all()
