"""Tests for src/data_store.py."""

import os
import shutil
from pathlib import Path
import pandas as pd
import pytest
from datetime import datetime, timedelta

# Import from src
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_store import (
    DataStore, 
    DATA_DIR, 
    _ticker_filename, 
    _extract_ticker_df,
    _fill_missing_days,
    generate_data_quality_report
)


@pytest.fixture
def clean_data_dir():
    """Clean up DATA_DIR before and after tests."""
    # Clean before test
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Clean after test
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)


class TestTickerFilename:
    """Test the _ticker_filename helper."""
    
    def test_standard_ticker(self):
        assert _ticker_filename("RELIANCE.NS") == "RELIANCE.NS.parquet"
    
    def test_ticker_with_slash(self):
        assert _ticker_filename("TEST/NS") == "TEST_NS.parquet"
    
    def test_ticker_with_colon(self):
        assert _ticker_filename("TEST:NS") == "TEST_NS.parquet"
    
    def test_ticker_with_backslash(self):
        assert _ticker_filename("TEST\\NS") == "TEST_NS.parquet"


class TestExtractTickerDF:
    """Test the _extract_ticker_df helper."""
    
    def test_extract_ticker_df_multiindex(self):
        """Test extraction from MultiIndex columns (multi-ticker download)."""
        # Build a fake MultiIndex-column DataFrame mimicking yfinance output
        dates = pd.date_range("2023-01-01", periods=5)
        
        # Create MultiIndex columns: (ticker, metric)
        tickers = ["AAA.NS", "BBB.NS"]
        metrics = ["Open", "High", "Low", "Close", "Volume"]
        
        multi_columns = pd.MultiIndex.from_product([tickers, metrics])
        
        # Create sample data
        data = {
            ("AAA.NS", "Open"): [100.0, 101.0, 102.0, 103.0, 104.0],
            ("AAA.NS", "High"): [105.0, 106.0, 107.0, 108.0, 109.0],
            ("AAA.NS", "Low"): [99.0, 100.0, 101.0, 102.0, 103.0],
            ("AAA.NS", "Close"): [103.0, 104.0, 105.0, 106.0, 107.0],
            ("AAA.NS", "Volume"): [1000, 1100, 1200, 1300, 1400],
            ("BBB.NS", "Open"): [200.0, 201.0, 202.0, 203.0, 204.0],
            ("BBB.NS", "High"): [205.0, 206.0, 207.0, 208.0, 209.0],
            ("BBB.NS", "Low"): [199.0, 200.0, 201.0, 202.0, 203.0],
            ("BBB.NS", "Close"): [203.0, 204.0, 205.0, 206.0, 207.0],
            ("BBB.NS", "Volume"): [2000, 2100, 2200, 2300, 2400],
        }
        
        raw = pd.DataFrame(data, index=dates)
        raw.columns = multi_columns
        
        # Test extraction for AAA.NS
        df_aaa = _extract_ticker_df(raw, "AAA.NS", is_single=False)
        assert df_aaa is not None
        assert len(df_aaa) == 5
        assert isinstance(df_aaa.columns, pd.Index)  # Not MultiIndex
        assert "close" in df_aaa.columns  # Lowercase
        assert "volume" in df_aaa.columns
        
        # Test extraction for BBB.NS
        df_bbb = _extract_ticker_df(raw, "BBB.NS", is_single=False)
        assert df_bbb is not None
        assert len(df_bbb) == 5
        
        # Test extraction for non-existent ticker
        df_ccc = _extract_ticker_df(raw, "CCC.NS", is_single=False)
        assert df_ccc is None
    
    def test_extract_ticker_df_single(self):
        """Test extraction from flat columns (single-ticker download)."""
        dates = pd.date_range("2023-01-01", periods=5)
        
        # Flat column DataFrame (single ticker)
        df = pd.DataFrame({
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [105.0, 106.0, 107.0, 108.0, 109.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "Close": [103.0, 104.0, 105.0, 106.0, 107.0],
            "Volume": [1000, 1100, 1200, 1300, 1400],
        }, index=dates)
        
        result = _extract_ticker_df(df, "AAA.NS", is_single=True)
        assert result is not None
        assert len(result) == 5
        assert "close" in result.columns  # Lowercase
    
    def test_extract_ticker_df_empty(self):
        """Test extraction returns None for empty input."""
        assert _extract_ticker_df(None, "AAA.NS", is_single=True) is None
        assert _extract_ticker_df(pd.DataFrame(), "AAA.NS", is_single=True) is None


class TestFillMissingDays:
    """Test the _fill_missing_days helper."""
    
    def test_forward_fill_missing_days(self):
        """Test forward-filling of missing business days."""
        # Create a DataFrame with business-day data but remove one weekday in the middle
        # Simulating a holiday gap
        dates = pd.bdate_range("2023-01-02", periods=10)  # Mon-Fri only
        df = pd.DataFrame({
            "open": [100.0 + i for i in range(10)],
            "high": [105.0 + i for i in range(10)],
            "low": [99.0 + i for i in range(10)],
            "close": [103.0 + i for i in range(10)],
            "volume": [1000 + i * 100 for i in range(10)],
        }, index=dates)
        
        # Remove a day in the middle (simulate holiday) - e.g., remove index 4 (Jan 6)
        gap_date = dates[4]
        df_with_gap = df.drop(gap_date)
        
        # Call _fill_missing_days over the full range
        start_date = "2023-01-02"
        end_date = "2023-01-13"
        result = _fill_missing_days(df_with_gap, start_date, end_date, ffill_limit=3)
        
        # Assert the resulting index contains the gap date
        assert gap_date in result.index
        
        # Assert 'close' on the gap date equals the previous day's close (forward-filled)
        prev_close = df.loc[dates[3], 'close']  # Day before gap
        assert result.loc[gap_date, 'close'] == prev_close
        
        # Assert 'volume' on the gap date is 0 (NOT forward-filled)
        assert result.loc[gap_date, 'volume'] == 0
        
    def test_ffill_limit_respected(self):
        """Test that forward-fill limit is respected for consecutive gaps."""
        # Create data with a large gap (> ffill_limit)
        dates = pd.bdate_range("2023-01-02", periods=20)
        df = pd.DataFrame({
            "open": [100.0 + i for i in range(len(dates))],
            "high": [105.0 + i for i in range(len(dates))],
            "low": [99.0 + i for i in range(len(dates))],
            "close": [103.0 + i for i in range(len(dates))],
            "volume": [1000 + i * 100 for i in range(len(dates))],
        }, index=dates)
        
        # Remove 5 consecutive days (more than ffill_limit=3)
        gap_dates = dates[3:8]  # 5 days gap
        df_with_large_gap = df.drop(gap_dates)
        
        start_date = "2023-01-02"
        end_date = "2023-01-27"
        result = _fill_missing_days(df_with_large_gap, start_date, end_date, ffill_limit=3)
        
        # First 3 gap days should be filled, remaining should be NaN
        # Gap starts at dates[3], so dates[3], dates[4], dates[5] should be filled
        # dates[6], dates[7] should remain NaN
        assert result.loc[dates[3], 'close'] == df.loc[dates[2], 'close']  # Filled
        assert result.loc[dates[4], 'close'] == df.loc[dates[2], 'close']  # Filled
        assert result.loc[dates[5], 'close'] == df.loc[dates[2], 'close']  # Filled (limit reached)
        assert pd.isna(result.loc[dates[6], 'close'])  # Beyond limit, NaN
        assert pd.isna(result.loc[dates[7], 'close'])  # Beyond limit, NaN
        
        # Volume should be 0 for all gap days
        assert result.loc[dates[3], 'volume'] == 0
        assert result.loc[dates[4], 'volume'] == 0
        assert result.loc[dates[5], 'volume'] == 0
        assert result.loc[dates[6], 'volume'] == 0
        assert result.loc[dates[7], 'volume'] == 0
    
    def test_volume_not_forward_filled(self):
        """Test that volume column is never forward-filled."""
        dates = pd.bdate_range("2023-01-02", periods=5)
        df = pd.DataFrame({
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [105.0, 106.0, 107.0, 108.0, 109.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [103.0, 104.0, 105.0, 106.0, 107.0],
            "volume": [1000, 1100, 1200, 1300, 1400],
        }, index=dates)
        
        # Remove day at index 2
        gap_date = dates[2]
        df_with_gap = df.drop(gap_date)
        
        start_date = "2023-01-02"
        end_date = "2023-01-06"
        result = _fill_missing_days(df_with_gap, start_date, end_date, ffill_limit=3)
        
        # Volume on gap date should be 0, not forward-filled
        assert result.loc[gap_date, 'volume'] == 0
        # But price should be forward-filled
        assert result.loc[gap_date, 'close'] == df.loc[dates[1], 'close']


class TestSaveLoadRoundtrip:
    """Test save and load roundtrip."""
    
    def test_save_load_roundtrip(self, clean_data_dir):
        """Test that saved data can be loaded correctly."""
        # Create synthetic DataFrame
        dates = pd.date_range("2023-01-01", periods=10)
        df = pd.DataFrame({
            "open": [100.0 + i for i in range(10)],
            "high": [105.0 + i for i in range(10)],
            "low": [99.0 + i for i in range(10)],
            "close": [103.0 + i for i in range(10)],
            "volume": [1000 + i * 100 for i in range(10)],
        }, index=dates)
        
        # Save
        ds = DataStore()
        path = ds.save_ticker_data("TEST.NS", df)
        
        assert path.exists()
        
        # Load
        loaded = ds.load_ticker_data("TEST.NS")
        
        assert loaded is not None
        assert len(loaded) == len(df)
        assert list(loaded.columns) == list(df.columns)


class TestDataStore:
    """Test DataStore class methods."""
    
    def test_batch_download_and_cache_mocked(self, clean_data_dir, monkeypatch):
        """Test batch download with mocked yfinance."""
        # Mock yfinance.download to return fake data
        def mock_download(tickers, start, end, **kwargs):
            dates = pd.date_range(start, end, freq='D')
            is_single = len(tickers) == 1 if isinstance(tickers, list) else True
            
            if is_single:
                # Single ticker: flat columns
                return pd.DataFrame({
                    "Open": [100.0] * len(dates),
                    "High": [105.0] * len(dates),
                    "Low": [99.0] * len(dates),
                    "Close": [103.0] * len(dates),
                    "Volume": [1000] * len(dates),
                }, index=dates)
            else:
                # Multiple tickers: MultiIndex columns
                metrics = ["Open", "High", "Low", "Close", "Volume"]
                multi_columns = pd.MultiIndex.from_product([tickers, metrics])
                
                data = {}
                for ticker in tickers:
                    for metric in metrics:
                        base = 100.0 if metric == "Open" else (
                            105.0 if metric == "High" else (
                                99.0 if metric == "Low" else (
                                    103.0 if metric == "Close" else 1000
                                )
                            )
                        )
                        data[(ticker, metric)] = [base] * len(dates)
                
                df = pd.DataFrame(data, index=dates)
                df.columns = multi_columns
                return df
        
        monkeypatch.setattr("yfinance.download", mock_download)
        
        ds = DataStore()
        tickers = ["AAPL.NS", "GOOG.NS", "MSFT.NS"]
        start_date = "2023-01-01"
        end_date = "2023-01-10"
        
        stats = ds.batch_download_and_cache(
            tickers, 
            start_date, 
            end_date, 
            chunk_size=2,
            skip_existing=False
        )
        
        assert stats['total'] == 3
        assert stats['downloaded'] == 3
        assert stats['failed'] == 0
        
        # Verify files were created
        for ticker in tickers:
            path = DATA_DIR / _ticker_filename(ticker)
            assert path.exists()
    
    def test_load_ticker_data_with_dates(self, clean_data_dir):
        """Test loading with date filtering."""
        # Create and save data using business days only (to match _fill_missing_days behavior)
        dates = pd.bdate_range("2023-01-01", periods=30)
        df = pd.DataFrame({
            "open": [100.0 + i for i in range(30)],
            "high": [105.0 + i for i in range(30)],
            "low": [99.0 + i for i in range(30)],
            "close": [103.0 + i for i in range(30)],
            "volume": [1000 + i * 100 for i in range(30)],
        }, index=dates)
        
        ds = DataStore()
        ds.save_ticker_data("TEST.NS", df)
        
        # Load with date filter
        loaded = ds.load_ticker_data("TEST.NS", start_date="2023-01-10", end_date="2023-01-20")
        
        assert loaded is not None
        # Business days between Jan 10 and Jan 20: 9 days (excludes weekends)
        assert len(loaded) == 9
        assert loaded.index.min() >= pd.Timestamp("2023-01-10")
        assert loaded.index.max() <= pd.Timestamp("2023-01-20")
    
    def test_load_nonexistent_returns_none(self, clean_data_dir):
        """Test that loading nonexistent file returns None."""
        ds = DataStore()
        result = ds.load_ticker_data("NONEXISTENT.NS")
        assert result is None


class TestIntegration:
    """Integration tests for download -> save -> load workflow."""
    
    def test_download_save_load_workflow(self, clean_data_dir, monkeypatch):
        """Test complete workflow: download, save, and load."""
        # Mock yfinance
        def mock_download(tickers, start, end, **kwargs):
            dates = pd.date_range(start, end, freq='D')
            is_single = len(tickers) == 1 if isinstance(tickers, list) else True
            
            if is_single:
                return pd.DataFrame({
                    "Open": [100.0 + i for i in range(len(dates))],
                    "High": [105.0 + i for i in range(len(dates))],
                    "Low": [99.0 + i for i in range(len(dates))],
                    "Close": [103.0 + i for i in range(len(dates))],
                    "Volume": [1000 + i * 100 for i in range(len(dates))],
                }, index=dates)
            else:
                metrics = ["Open", "High", "Low", "Close", "Volume"]
                multi_columns = pd.MultiIndex.from_product([tickers, metrics])
                
                data = {}
                for ticker in tickers:
                    for idx, metric in enumerate(metrics):
                        base = [100.0, 105.0, 99.0, 103.0, 1000][idx]
                        data[(ticker, metric)] = [base + i for i in range(len(dates))]
                
                df = pd.DataFrame(data, index=dates)
                df.columns = multi_columns
                return df
        
        monkeypatch.setattr("yfinance.download", mock_download)
        
        ds = DataStore()
        tickers = ["TICK1.NS", "TICK2.NS"]
        start_date = "2023-01-01"
        end_date = "2023-01-10"
        
        # Download and cache
        stats = ds.batch_download_and_cache(tickers, start_date, end_date, skip_existing=False)
        assert stats['downloaded'] == 2
        
        # Load back
        for ticker in tickers:
            loaded = ds.load_ticker_data(ticker)
            assert loaded is not None
            assert len(loaded) == 10
            assert 'close' in loaded.columns
            assert 'volume' in loaded.columns


class TestDataQualityReport:
    """Test generate_data_quality_report function."""
    
    def test_data_quality_report_good_data(self):
        """Test report with good data - should pass."""
        # Use recent dates so data is not stale
        end_date = pd.Timestamp.now() - pd.Timedelta(days=1)
        start_date = end_date - pd.Timedelta(days=20)
        dates = pd.bdate_range(start=start_date, end=end_date)
        
        df = pd.DataFrame({
            "open": [100.0 + i for i in range(len(dates))],
            "high": [105.0 + i for i in range(len(dates))],
            "low": [99.0 + i for i in range(len(dates))],
            "close": [103.0 + i for i in range(len(dates))],
            "volume": [1000 + i * 100 for i in range(len(dates))],
        }, index=dates)
        
        report = generate_data_quality_report(df, "TEST.NS")
        
        # Check all required keys exist
        required_keys = [
            "ticker", "rows", "start_date", "end_date", "missing_values",
            "total_missing", "duplicate_dates", "zero_volume_days",
            "date_gaps", "days_out_of_range", "is_stale", "passed"
        ]
        for key in required_keys:
            assert key in report
        
        assert report["ticker"] == "TEST.NS"
        assert report["rows"] == len(dates)
        assert report["total_missing"] == 0
        assert report["is_stale"] is False
        assert report["passed"] is True
    
    def test_data_quality_report_with_missing_values(self):
        """Test report with >10% missing close values - should fail."""
        dates = pd.bdate_range("2023-01-02", periods=10)
        df = pd.DataFrame({
            "open": [100.0 + i for i in range(10)],
            "high": [105.0 + i for i in range(10)],
            "low": [99.0 + i for i in range(10)],
            "close": [103.0 + i for i in range(10)],
            "volume": [1000 + i * 100 for i in range(10)],
        }, index=dates)
        
        # Introduce >10% missing values in close (2 out of 10 = 20%)
        df.loc[dates[0], 'close'] = None
        df.loc[dates[1], 'close'] = None
        
        report = generate_data_quality_report(df, "TEST.NS")
        
        assert report["total_missing"] >= 2
        assert report["passed"] is False
    
    def test_data_quality_report_empty_dataframe(self):
        """Test report with empty DataFrame - should fail."""
        df = pd.DataFrame()
        
        report = generate_data_quality_report(df, "EMPTY.NS")
        
        assert report["rows"] == 0
        assert report["passed"] is False
        assert report["start_date"] is None
        assert report["end_date"] is None
    
    def test_data_quality_report_stale_data(self):
        """Test report with data ending 10 days ago - should be stale."""
        # Create data ending 10 days ago
        end_date = pd.Timestamp.now() - pd.Timedelta(days=10)
        start_date = end_date - pd.Timedelta(days=20)
        dates = pd.bdate_range(start=start_date, end=end_date)
        
        df = pd.DataFrame({
            "open": [100.0 + i for i in range(len(dates))],
            "high": [105.0 + i for i in range(len(dates))],
            "low": [99.0 + i for i in range(len(dates))],
            "close": [103.0 + i for i in range(len(dates))],
            "volume": [1000 + i * 100 for i in range(len(dates))],
        }, index=dates)
        
        report = generate_data_quality_report(df, "STALE.NS", stale_threshold_days=5)
        
        assert report["is_stale"] is True
    
    def test_data_quality_report_none_input(self):
        """Test report with None input - should fail."""
        report = generate_data_quality_report(None, "NONE.NS")
        
        assert report["rows"] == 0
        assert report["passed"] is False
        assert report["start_date"] is None
        assert report["end_date"] is None
    
    def test_data_quality_report_all_keys_exist(self):
        """Test that all required keys exist in the returned dict."""
        dates = pd.bdate_range("2023-01-02", periods=5)
        df = pd.DataFrame({
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [105.0, 106.0, 107.0, 108.0, 109.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [103.0, 104.0, 105.0, 106.0, 107.0],
            "volume": [1000, 1100, 1200, 1300, 1400],
        }, index=dates)
        
        report = generate_data_quality_report(df, "KEYS.NS")
        
        required_keys = [
            "ticker", "rows", "start_date", "end_date", "missing_values",
            "total_missing", "duplicate_dates", "zero_volume_days",
            "date_gaps", "days_out_of_range", "is_stale", "passed"
        ]
        
        for key in required_keys:
            assert key in report, f"Missing required key: {key}"
