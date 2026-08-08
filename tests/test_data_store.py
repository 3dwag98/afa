"""Tests for src/data_store.py at workspace root level."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import pytest

from data_store import (
    DATA_DIR,
    get_ticker_data,
    batch_download_and_cache,
    load_ticker_data,
    generate_data_quality_report,
)


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    """Redirect DATA_DIR to a temporary directory."""
    test_dir = tmp_path / "market_data"
    test_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("data_store.DATA_DIR", test_dir)
    return test_dir


class TestIntegrationWorkflow:
    """Integration tests for the end-to-end workflow."""

    def test_integration_workflow(self, tmp_path, monkeypatch):
        """Test complete workflow: download -> cache -> load -> forward-fill -> quality report.
        
        This test is hermetic - no real network calls are made.
        """
        # Redirect DATA_DIR to tmp_path
        test_dir = tmp_path / "market_data"
        test_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("data_store.DATA_DIR", test_dir)
        
        # Track how many times mock_download is called
        call_count = [0]
        
        # Mock yf.download to return synthetic data
        def mock_download(tickers, start, end, **kwargs):
            call_count[0] += 1
            dates = pd.bdate_range(start, end)
            is_single = len(tickers) == 1 if isinstance(tickers, list) else True
            
            if is_single:
                # Single ticker: flat columns
                return pd.DataFrame({
                    "Open": [100.0 + i for i in range(len(dates))],
                    "High": [105.0 + i for i in range(len(dates))],
                    "Low": [99.0 + i for i in range(len(dates))],
                    "Close": [103.0 + i for i in range(len(dates))],
                    "Volume": [1000 + i * 100 for i in range(len(dates))],
                }, index=dates)
            else:
                # Multiple tickers: MultiIndex columns
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
        
        # Define date range
        start_date = "2023-01-02"
        end_date = "2023-01-13"
        
        # First call: should download and cache
        df1 = get_ticker_data("TEST.NS", start_date, end_date, force_refresh=False)
        
        # Assert returned object is a DataFrame (NOT None)
        assert df1 is not None
        assert isinstance(df1, pd.DataFrame)
        
        # Assert it has rows
        assert len(df1) > 0
        
        # Assert expected columns exist
        expected_cols = {"open", "high", "low", "close", "volume"}
        assert expected_cols.issubset(set(df1.columns))
        
        # Second call with force_refresh=False: should load from cache
        df2 = get_ticker_data("TEST.NS", start_date, end_date, force_refresh=False)
        
        # Assert the stub download was called only once
        assert call_count[0] == 1, f"Expected 1 download call, got {call_count[0]}"
        
        # Assert second result is also a DataFrame
        assert df2 is not None
        assert isinstance(df2, pd.DataFrame)
        assert len(df2) > 0
        
        # Call generate_data_quality_report on the result
        report = generate_data_quality_report(df1, "TEST.NS")
        
        # Assert report returns a dict with "passed" key
        assert isinstance(report, dict)
        assert "passed" in report
        assert isinstance(report["passed"], bool)


class TestGetTickerData:
    """Tests for get_ticker_data convenience function."""

    def test_get_ticker_data_returns_dataframe(self, tmp_path, monkeypatch):
        """Test that get_ticker_data returns a DataFrame when successful."""
        # Redirect DATA_DIR
        test_dir = tmp_path / "market_data"
        test_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("data_store.DATA_DIR", test_dir)
        
        # Mock download
        def mock_download(tickers, start, end, **kwargs):
            dates = pd.bdate_range(start, end)
            return pd.DataFrame({
                "Open": [100.0] * len(dates),
                "High": [105.0] * len(dates),
                "Low": [99.0] * len(dates),
                "Close": [103.0] * len(dates),
                "Volume": [1000] * len(dates),
            }, index=dates)
        
        monkeypatch.setattr("yfinance.download", mock_download)
        
        df = get_ticker_data("AAPL.NS", "2023-01-02", "2023-01-10")
        
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_batch_download_and_cache_returns_bool(self, tmp_path, monkeypatch):
        """Test that batch_download_and_cache returns True/False."""
        # Redirect DATA_DIR
        test_dir = tmp_path / "market_data"
        test_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("data_store.DATA_DIR", test_dir)
        
        # Mock download
        def mock_download(tickers, start, end, **kwargs):
            dates = pd.bdate_range(start, end)
            return pd.DataFrame({
                "Open": [100.0] * len(dates),
                "High": [105.0] * len(dates),
                "Low": [99.0] * len(dates),
                "Close": [103.0] * len(dates),
                "Volume": [1000] * len(dates),
            }, index=dates)
        
        monkeypatch.setattr("yfinance.download", mock_download)
        
        result = batch_download_and_cache(["TEST.NS"], "2023-01-02", "2023-01-10")
        
        assert isinstance(result, bool)
        assert result is True
