"""Tests for data_ingestion module."""

import pandas as pd
import pytest

from src.data_ingestion import (
    generate_synthetic_ohlcv,
    validate_ohlcv,
)


class TestGenerateSyntheticOhlcv:
    """Tests for generate_synthetic_ohlcv function."""

    def test_creates_valid_ohlcv(self):
        """Test that synthetic generator creates valid OHLCV data."""
        df = generate_synthetic_ohlcv("TEST.NS", days=100, seed=42)

        # Check it's a DataFrame
        assert isinstance(df, pd.DataFrame)

        # Check required columns exist and are lowercase
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Check we have approximately the right number of rows (business days)
        assert len(df) >= 50, f"Expected at least 50 rows, got {len(df)}"

        # Check close prices are positive
        assert (df["close"] > 0).all(), "Close prices must be positive"

        # Check high >= low
        assert (df["high"] >= df["low"]).all(), "High must be >= low"

        # Check volume is positive
        assert (df["volume"] > 0).all(), "Volume must be positive"

        # Check index is datetime
        assert isinstance(df.index, pd.DatetimeIndex), "Index should be DatetimeIndex"

        # Check no duplicate dates
        assert not df.index.duplicated().any(), "No duplicate dates allowed"

    def test_deterministic_with_seed(self):
        """Test that same seed produces same output."""
        df1 = generate_synthetic_ohlcv("TEST.NS", days=100, seed=42)
        df2 = generate_synthetic_ohlcv("TEST.NS", days=100, seed=42)

        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different output."""
        df1 = generate_synthetic_ohlcv("TEST.NS", days=100, seed=42)
        df2 = generate_synthetic_ohlcv("TEST.NS", days=100, seed=123)

        # They should be different
        assert not df1["close"].equals(df2["close"])


class TestValidateOhlcv:
    """Tests for validate_ohlcv function."""

    def test_rejects_empty_data(self):
        """Test that validate_ohlcv rejects empty DataFrame."""
        df = pd.DataFrame()
        assert validate_ohlcv(df) is False

    def test_rejects_insufficient_rows(self):
        """Test that validate_ohlcv rejects DataFrame with too few rows."""
        df = generate_synthetic_ohlcv("TEST.NS", days=30, seed=42)  # ~21 business days
        assert validate_ohlcv(df, min_rows=50) is False

    def test_accepts_valid_synthetic_data(self):
        """Test that validate_ohlcv accepts valid synthetic data."""
        df = generate_synthetic_ohlcv("TEST.NS", days=200, seed=42)
        assert validate_ohlcv(df, min_rows=50) is True

    def test_rejects_negative_close(self):
        """Test that validate_ohlcv rejects data with negative close prices."""
        df = generate_synthetic_ohlcv("TEST.NS", days=100, seed=42)
        df.loc[df.index[0], "close"] = -100
        assert validate_ohlcv(df) is False

    def test_rejects_high_less_than_low(self):
        """Test that validate_ohlcv rejects data where high < low."""
        df = generate_synthetic_ohlcv("TEST.NS", days=100, seed=42)
        df.loc[df.index[0], "high"] = df.loc[df.index[0], "low"] - 1
        assert validate_ohlcv(df) is False

    def test_rejects_duplicate_dates(self):
        """Test that validate_ohlcv rejects data with duplicate dates."""
        df = generate_synthetic_ohlcv("TEST.NS", days=100, seed=42)
        # Create duplicate by resetting index and adding duplicate row
        df_reset = df.reset_index()
        df_dup = pd.concat([df_reset, df_reset.iloc[[0]]], ignore_index=True)
        df_dup = df_dup.set_index("Date")
        assert validate_ohlcv(df_dup) is False

    def test_accepts_min_rows_exactly(self):
        """Test that validate_ohlcv accepts data with exactly min_rows."""
        df = generate_synthetic_ohlcv("TEST.NS", days=70, seed=42)
        # Trim to exactly 50 rows
        df_trimmed = df.head(50)
        assert validate_ohlcv(df_trimmed, min_rows=50) is True
