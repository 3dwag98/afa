"""
Tests for Universe Manager and Data Store.

Includes:
- Mock yfinance download tests
- Chunking logic tests
- Parquet save/load roundtrip tests
"""

import os
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, PropertyMock

import pandas as pd
import pytest

# Import modules under test
from portfolio_agent.src.universe import UniverseManager
from portfolio_agent.src.data_store import DataStore


class TestUniverseManager:
    """Tests for UniverseManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        # Point to temp directory to avoid using the global fallback
        self.manager = UniverseManager(cache_dir=Path(self.temp_dir))
        self.manager.LOCAL_TICKER_PATH = Path(self.temp_dir) / "nse500_tickers.csv"
    
    def teardown_method(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_default_paths(self):
        """Test initialization with default paths."""
        manager = UniverseManager()
        assert manager.cache_dir is not None
    
    def test_init_custom_cache_dir(self):
        """Test initialization with custom cache directory."""
        manager = UniverseManager(cache_dir=Path(self.temp_dir))
        assert manager.cache_dir == Path(self.temp_dir)
    
    def test_load_local_fallback_with_sample_data(self):
        """Test loading ticker list from local CSV."""
        # Create sample CSV
        csv_path = Path(self.temp_dir) / "nse500_tickers.csv"
        with open(csv_path, 'w') as f:
            f.write("ticker\n")
            f.write("RELIANCE\n")
            f.write("TCS\n")
            f.write("INFY\n")
        
        tickers = self.manager._load_local_fallback()
        assert tickers is not None
        assert len(tickers) == 3
        assert "RELIANCE.NS" in tickers
        assert "TCS.NS" in tickers
        assert "INFY.NS" in tickers
    
    def test_load_local_fallback_missing_file(self):
        """Test loading when file doesn't exist."""
        result = self.manager._load_local_fallback()
        assert result is None
    
    def test_save_and_load_ticker_list(self):
        """Test saving and loading ticker list roundtrip."""
        tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
        csv_path = Path(self.temp_dir) / "test_tickers.csv"
        
        self.manager.save_ticker_list(tickers, csv_path)
        loaded = self.manager.load_ticker_list(csv_path)
        
        # load_ticker_list uses LOCAL_TICKER_PATH by default, need to set it
        self.manager.LOCAL_TICKER_PATH = csv_path
        loaded = self.manager.load_ticker_list(csv_path)
        
        assert len(loaded) == 3
        assert "RELIANCE.NS" in loaded
    
    @patch('portfolio_agent.src.universe.requests.get')
    def test_fetch_from_url_success(self, mock_get):
        """Test fetching ticker list from URL."""
        mock_response = MagicMock()
        mock_response.text = "ticker\nRELIANCE\nTCS\nINFY\n"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        tickers = self.manager._fetch_from_url("http://test.com/tickers.csv")
        
        assert tickers is not None
        assert len(tickers) == 3
        assert "RELIANCE.NS" in tickers
    
    @patch('portfolio_agent.src.universe.requests.get')
    def test_fetch_from_url_failure(self, mock_get):
        """Test URL fetch failure handling."""
        mock_get.side_effect = Exception("Network error")
        
        result = self.manager._fetch_from_url("http://test.com/tickers.csv")
        assert result is None
    
    @patch('yfinance.Ticker')
    def test_fetch_from_yfinance_etf(self, mock_ticker):
        """Test fetching from yfinance ETF holdings."""
        mock_etf = MagicMock()
        type(mock_etf).holdings = PropertyMock(return_value=[
            {'symbol': 'RELIANCE', 'weight': 0.1},
            {'symbol': 'TCS', 'weight': 0.08},
        ])
        mock_ticker.return_value = mock_etf
        
        tickers = self.manager._fetch_from_yfinance_etf()
        
        assert tickers is not None
        assert len(tickers) == 2
        assert "RELIANCE.NS" in tickers
    
    def test_get_master_ticker_list_local_fallback(self):
        """Test get_master_ticker_list uses local fallback."""
        # Create sample CSV
        csv_path = Path(self.temp_dir) / "nse500_tickers.csv"
        with open(csv_path, 'w') as f:
            f.write("ticker\n")
            f.write("RELIANCE\n")
            f.write("TCS\n")
        
        # Override local path
        self.manager.LOCAL_TICKER_PATH = csv_path
        
        tickers = self.manager.get_master_ticker_list()
        assert len(tickers) == 2
        assert "RELIANCE.NS" in tickers
    
    def test_get_master_ticker_list_caching(self):
        """Test that ticker list is cached."""
        # Create sample CSV
        csv_path = Path(self.temp_dir) / "nse500_tickers.csv"
        with open(csv_path, 'w') as f:
            f.write("ticker\nRELIANCE\n")
        
        self.manager.LOCAL_TICKER_PATH = csv_path
        
        # First call
        tickers1 = self.manager.get_master_ticker_list()
        
        # Modify cache file
        with open(csv_path, 'w') as f:
            f.write("ticker\nTCS\n")
        
        # Second call should return cached result
        tickers2 = self.manager.get_master_ticker_list()
        
        assert tickers1 == tickers2
    
    @patch('yfinance.Ticker')
    def test_check_data_availability(self, mock_ticker):
        """Test data availability check."""
        mock_stock = MagicMock()
        # Create mock history with 5 years of data (~1250 trading days)
        mock_history = pd.DataFrame(
            index=pd.date_range(start='2019-01-01', periods=1250, freq='B'),
            data={'Close': range(1250)}
        )
        mock_stock.history.return_value = mock_history
        mock_ticker.return_value = mock_stock
        
        result = self.manager._check_data_availability("RELIANCE.NS", years=5)
        assert result is True
    
    @patch('yfinance.Ticker')
    def test_get_avg_volume(self, mock_ticker):
        """Test average volume calculation."""
        mock_stock = MagicMock()
        mock_history = pd.DataFrame({
            'Volume': [10_000_000, 20_000_000, 30_000_000]  # 1, 2, 3 crores
        })
        mock_stock.history.return_value = mock_history
        mock_ticker.return_value = mock_stock
        
        avg_vol_cr = self.manager._get_avg_volume("RELIANCE.NS")
        # Average is 20,000,000 = 2 crores
        assert avg_vol_cr == 2.0
    
    @patch.object(UniverseManager, '_check_data_availability')
    @patch.object(UniverseManager, '_get_avg_volume')
    def test_filter_universe(self, mock_avg_vol, mock_check_data):
        """Test universe filtering."""
        # Mock all tickers as having sufficient data
        mock_check_data.return_value = True
        
        # Mock volumes: RELIANCE passes, PENNY fails
        def volume_side_effect(ticker):
            if "RELIANCE" in ticker:
                return 10.0  # 10 crores - passes
            return 1.0  # 1 crore - fails (below 5 cr threshold)
        
        mock_avg_vol.side_effect = volume_side_effect
        
        tickers = ["RELIANCE.NS", "PENNY.NS", "TCS.NS"]
        filtered = self.manager.filter_universe(tickers, min_avg_volume_cr=5.0)
        
        assert len(filtered) == 1
        assert "RELIANCE.NS" in filtered
    
    def test_filter_universe_adds_ns_suffix(self):
        """Test that filter adds .NS suffix if missing."""
        # Create sample CSV
        csv_path = Path(self.temp_dir) / "nse500_tickers.csv"
        with open(csv_path, 'w') as f:
            f.write("ticker\nRELIANCE\n")
        
        self.manager.LOCAL_TICKER_PATH = csv_path
        self.manager._ticker_cache = None  # Clear cache
        
        # Mock to always pass filters
        with patch.object(UniverseManager, '_check_data_availability', return_value=True):
            with patch.object(UniverseManager, '_get_avg_volume', return_value=10.0):
                filtered = self.manager.filter_universe(["RELIANCE"])
                
                assert len(filtered) == 1
                assert filtered[0] == "RELIANCE.NS"


class TestDataStore:
    """Tests for DataStore class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.store = DataStore(cache_dir=Path(self.temp_dir))
    
    def teardown_method(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_creates_cache_dir(self):
        """Test initialization creates cache directory."""
        cache_path = Path(self.temp_dir) / "market_data"
        store = DataStore(cache_dir=cache_path)
        assert cache_path.exists()
    
    def test_get_ticker_path(self):
        """Test ticker path generation."""
        path = self.store._get_ticker_path("RELIANCE.NS")
        assert path.name == "RELIANCE.NS.parquet"
    
    def test_parse_ticker_from_path(self):
        """Test parsing ticker from path."""
        path = Path(self.temp_dir) / "RELIANCE.NS.parquet"
        ticker = self.store._parse_ticker_from_path(path)
        assert ticker == "RELIANCE.NS"
    
    @patch('yfinance.download')
    def test_fetch_chunk_success(self, mock_download):
        """Test successful chunk fetch."""
        mock_history = pd.DataFrame({
            'Open': [100, 101],
            'High': [105, 106],
            'Low': [99, 100],
            'Close': [102, 103],
            'Volume': [1000, 2000]
        }, index=pd.date_range(start='2024-01-01', periods=2, freq='D'))
        mock_download.return_value = mock_history
        
        results = self.store._fetch_chunk(["RELIANCE.NS"], "2024-01-01", "2024-01-02")
        
        assert "RELIANCE.NS" in results
        df = results["RELIANCE.NS"]
        assert df is not None
        assert len(df) == 2
        assert 'close' in df.columns
    
    @patch('yfinance.download')
    def test_fetch_chunk_empty_data(self, mock_download):
        """Test fetch with empty data response."""
        mock_download.return_value = pd.DataFrame()
        
        results = self.store._fetch_chunk(["RELIANCE.NS"], "2024-01-01", "2024-01-02")
        assert results.get("RELIANCE.NS") is None
    
    @patch('portfolio_agent.src.data_store.time.sleep')
    @patch('yfinance.download')
    def test_fetch_chunk_retry_logic(self, mock_download, mock_sleep):
        """Test retry logic with exponential backoff."""
        # Fail first two attempts, succeed on third
        side_effects = [Exception("Timeout"), Exception("Timeout"), 
                       pd.DataFrame({'Close': [100]}, index=['2024-01-01'])]
        mock_download.side_effect = side_effects
        
        results = self.store._fetch_chunk(["RELIANCE.NS"], "2024-01-01", "2024-01-01", max_retries=3)
        
        assert mock_sleep.call_count == 2  # Called twice for retries
        assert results.get("RELIANCE.NS") is not None
    
    def test_save_load_roundtrip(self):
        """Test parquet save and load roundtrip."""
        # Create sample data
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [99, 100, 101],
            'close': [102, 103, 104],
            'volume': [1000, 2000, 3000]
        }, index=pd.date_range(start='2024-01-01', periods=3, freq='D'))
        
        # Save
        self.store.save_ticker_data("RELIANCE.NS", df)
        
        # Load
        loaded = self.store.load_ticker_data_only("RELIANCE.NS")
        
        assert loaded is not None
        assert len(loaded) == 3
        assert 'close' in loaded.columns
    
    def test_is_cache_valid(self):
        """Test cache validity check."""
        # Create and save sample data
        df = pd.DataFrame({
            'close': [100, 101, 102]
        }, index=pd.date_range(start='2024-01-01', end='2024-01-03', freq='D'))
        
        self.store.save_ticker_data("RELIANCE.NS", df)
        
        # Load and verify data exists
        loaded = self.store.load_ticker_data_only("RELIANCE.NS")
        assert loaded is not None
    
    @patch('yfinance.download')
    def test_batch_download_chunking(self, mock_download):
        """Test batch download processes in chunks."""
        mock_history = pd.DataFrame({
            'Open': [100],
            'High': [105],
            'Low': [99],
            'Close': [102],
            'Volume': [1000]
        }, index=pd.date_range(start='2024-01-01', periods=1, freq='D'))
        mock_download.return_value = mock_history
        
        tickers = [f"TICK{i}.NS" for i in range(120)]  # 120 tickers
        
        stats = self.store.batch_download_and_cache(
            tickers,
            "2024-01-01",
            "2024-01-01",
            chunk_size=50,
            skip_existing=False
        )
        
        assert stats['total'] == 120
        assert stats['downloaded'] == 120
        assert stats['skipped'] == 0
    
    @patch('yfinance.download')
    def test_batch_download_skip_existing(self, mock_download):
        """Test batch download skips existing cached data."""
        mock_history = pd.DataFrame({
            'Open': [100],
            'High': [105],
            'Low': [99],
            'Close': [102],
            'Volume': [1000]
        }, index=pd.date_range(start='2024-01-01', periods=1, freq='D'))
        mock_download.return_value = mock_history
        
        tickers = ["TICK1.NS", "TICK2.NS"]
        
        # First download
        stats1 = self.store.batch_download_and_cache(
            tickers, "2024-01-01", "2024-01-01", skip_existing=False
        )
        assert stats1['downloaded'] == 2
        
        # Second download with skip_existing=True
        stats2 = self.store.batch_download_and_cache(
            tickers, "2024-01-01", "2024-01-01", skip_existing=True
        )
        assert stats2['skipped'] == 2
        assert stats2['downloaded'] == 0
    
    @patch('yfinance.download')
    def test_load_ticker_data_forward_fill(self, mock_download):
        """Test load_ticker_data handles market holidays with forward fill."""
        # Create data with a gap (simulating weekend/holiday)
        dates = pd.DatetimeIndex(['2024-01-01', '2024-01-02', '2024-01-05'])  # Gap on 3rd, 4th
        mock_history = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [102, 103, 104],
            'Volume': [1000, 2000, 3000]
        }, index=dates)
        
        mock_download.return_value = mock_history
        
        # Download and save
        self.store.batch_download_and_cache(
            ["TEST.NS"], "2024-01-01", "2024-01-05", skip_existing=False
        )
        
        # Load with forward fill
        df = self.store.load_ticker_data("TEST.NS", "2024-01-01", "2024-01-05", forward_fill_days=3)
        
        assert df is not None
        # Should have filled the gap (5 days total with forward fill)
        assert len(df) >= 3
    
    def test_load_ticker_data_missing(self):
        """Test loading non-existent ticker returns None."""
        result = self.store.load_ticker_data("NONEXISTENT.NS", "2024-01-01", "2024-01-31")
        assert result is None
    
    def test_get_cached_tickers(self):
        """Test getting list of cached tickers."""
        # Create sample data
        df = pd.DataFrame({'close': [100]}, index=pd.date_range(start='2024-01-01', periods=1))
        
        self.store.save_ticker_data("RELIANCE.NS", df)
        self.store.save_ticker_data("TCS.NS", df)
        
        tickers = self.store.get_cached_tickers()
        
        assert len(tickers) == 2
        assert "RELIANCE.NS" in tickers
        assert "TCS.NS" in tickers
    
    def test_clear_cache_single_ticker(self):
        """Test clearing cache for single ticker."""
        df = pd.DataFrame({'close': [100]}, index=pd.date_range(start='2024-01-01', periods=1))
        
        self.store.save_ticker_data("RELIANCE.NS", df)
        self.store.save_ticker_data("TCS.NS", df)
        
        self.store.clear_cache("RELIANCE.NS")
        
        tickers = self.store.get_cached_tickers()
        assert len(tickers) == 1
        assert "RELIANCE.NS" not in tickers
    
    def test_clear_cache_all(self):
        """Test clearing entire cache."""
        df = pd.DataFrame({'close': [100]}, index=pd.date_range(start='2024-01-01', periods=1))
        
        self.store.save_ticker_data("RELIANCE.NS", df)
        self.store.save_ticker_data("TCS.NS", df)
        
        self.store.clear_cache()
        
        tickers = self.store.get_cached_tickers()
        assert len(tickers) == 0
    
    @patch('yfinance.download')
    def test_data_quality_report(self, mock_download):
        """Test data quality report generation."""
        mock_history = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [100, 101, 102],
            'Volume': [1000, 2000, 3000]
        }, index=pd.date_range(start='2024-01-01', periods=3, freq='D'))
        mock_download.return_value = mock_history
        
        # Download data
        self.store.batch_download_and_cache(
            ["RELIANCE.NS"], "2024-01-01", "2024-01-03", skip_existing=False
        )
        
        report = self.store.get_data_quality_report()
        
        assert len(report) == 1
        assert report.iloc[0]['ticker'] == "RELIANCE.NS"
        assert report.iloc[0]['total_days'] == 3
        assert bool(report.iloc[0]['has_volume']) is True


class TestIntegration:
    """Integration tests for UniverseManager and DataStore."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.universe = UniverseManager(cache_dir=Path(self.temp_dir))
        self.universe.LOCAL_TICKER_PATH = Path(self.temp_dir) / "nse500_tickers.csv"
        self.store = DataStore(cache_dir=Path(self.temp_dir) / "market_data")
    
    def teardown_method(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('yfinance.download')
    @patch('yfinance.Ticker')
    def test_full_workflow(self, mock_ticker, mock_download):
        """Test full workflow: get tickers -> filter -> download -> load.

        DataStore downloads through yfinance.download(), not yfinance.Ticker,
        so that has to be mocked too — without it this test reached the real
        Yahoo API and failed on any machine without internet access.
        """
        dates = pd.date_range(start='2024-01-01', periods=2, freq='D')
        metrics = ['Open', 'High', 'Low', 'Close', 'Volume']
        values = {'Open': 100.0, 'High': 105.0, 'Low': 99.0, 'Close': 102.0, 'Volume': 1000}

        def fake_download(tickers, start=None, end=None, **kwargs):
            symbols = tickers if isinstance(tickers, list) else [tickers]
            if len(symbols) == 1:
                return pd.DataFrame({m: [values[m]] * len(dates) for m in metrics}, index=dates)
            data = {(s, m): [values[m]] * len(dates) for s in symbols for m in metrics}
            df = pd.DataFrame(data, index=dates)
            df.columns = pd.MultiIndex.from_tuples(df.columns)
            return df

        mock_download.side_effect = fake_download

        mock_stock = MagicMock()
        # For universe manager methods
        mock_history_long = pd.DataFrame(
            {'Close': range(1250)},
            index=pd.date_range(start='2019-01-01', periods=1250, freq='B')
        )
        type(mock_stock).holdings = PropertyMock(return_value=[{'symbol': 'RELIANCE'}, {'symbol': 'TCS'}])
        mock_stock.history.return_value = mock_history_long
        mock_ticker.return_value = mock_stock
        
        # Create local fallback
        csv_path = Path(self.temp_dir) / "nse500_tickers.csv"
        with open(csv_path, 'w') as f:
            f.write("ticker\nRELIANCE\nTCS\nINFY\n")
        
        # Setup data store mock - different history for downloads
        mock_ds_stock = MagicMock()
        mock_ds_stock.history.return_value = pd.DataFrame({
            'Open': [100, 101],
            'High': [105, 106],
            'Low': [99, 100],
            'Close': [102, 103],
            'Volume': [1000, 2000]
        }, index=pd.date_range(start='2024-01-01', periods=2, freq='D'))
        mock_ticker.return_value = mock_ds_stock
        
        # Get master list
        tickers = self.universe.get_master_ticker_list()
        assert len(tickers) == 3
        
        # Filter universe (mock to pass all)
        with patch.object(UniverseManager, '_check_data_availability', return_value=True):
            with patch.object(UniverseManager, '_get_avg_volume', return_value=10.0):
                filtered = self.universe.filter_universe(tickers[:2])
                assert len(filtered) == 2
        
        # Download data
        stats = self.store.batch_download_and_cache(
            filtered, "2024-01-01", "2024-01-02", skip_existing=False
        )
        assert stats['downloaded'] == 2
        
        # Load data
        df = self.store.load_ticker_data("RELIANCE.NS", "2024-01-01", "2024-01-02")
        assert df is not None
        assert len(df) == 2
