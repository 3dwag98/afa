"""
Universe Manager for Indian Equity Market (NSE/BSE).

Handles ticker list retrieval and filtering for the Nifty 500 universe.
"""

import os
import csv
from typing import List, Optional
from pathlib import Path

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class UniverseManager:
    """
    Manages the universe of Indian equities for backtesting.
    
    Provides methods to:
    - Get master ticker list (Nifty 500)
    - Filter tickers based on liquidity and data availability
    """
    
    # Public CSV URL for Nifty 500 constituents (GitHub Gist or similar)
    NIFTY_500_CSV_URL = "https://raw.githubusercontent.com/amitkumarjha/nse-data/main/nifty500_tickers.csv"
    
    # Fallback local path
    LOCAL_TICKER_PATH = Path(__file__).parent.parent / "data" / "nse500_tickers.csv"
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the UniverseManager.
        
        Args:
            cache_dir: Directory for caching ticker lists. Defaults to data/ directory.
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / "data"
        self.cache_dir = cache_dir
        self._ticker_cache: Optional[List[str]] = None
    
    def _fetch_from_url(self, url: str) -> Optional[List[str]]:
        """
        Fetch ticker list from a public URL.
        
        Args:
            url: URL to fetch CSV from.
            
        Returns:
            List of ticker symbols or None if fetch fails.
        """
        if not REQUESTS_AVAILABLE:
            return None
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            tickers = []
            lines = response.text.strip().split('\n')
            
            # Skip header if present
            start_idx = 0
            if lines and ('ticker' in lines[0].lower() or 'symbol' in lines[0].lower()):
                start_idx = 1
            
            for line in lines[start_idx:]:
                line = line.strip()
                if line:
                    # Handle CSV format
                    parts = line.split(',')
                    if parts:
                        ticker = parts[0].strip()
                        # Ensure NSE suffix
                        if not ticker.endswith('.NS'):
                            ticker = f"{ticker}.NS"
                        tickers.append(ticker.upper())
            
            return tickers if tickers else None
            
        except (requests.RequestException, Exception):
            return None
    
    def _fetch_from_yfinance_etf(self) -> Optional[List[str]]:
        """
        Fetch holdings from a broad market ETF as fallback.
        
        Uses NIFTYBEES (Nifty 50 ETF) as a proxy for liquid stocks.
        
        Returns:
            List of ticker symbols or None if fetch fails.
        """
        if not YFINANCE_AVAILABLE:
            return None
        
        try:
            # Try to get Nifty 50 ETF holdings
            etf = yf.Ticker("NIFTYBEES.NS")
            holdings = etf.holdings
            
            if holdings is not None and len(holdings) > 0:
                tickers = []
                for holding in holdings:
                    symbol = holding.get('symbol', '')
                    if symbol:
                        if not symbol.endswith('.NS'):
                            symbol = f"{symbol}.NS"
                        tickers.append(symbol.upper())
                return tickers
        except Exception:
            pass
        
        return None
    
    def _load_local_fallback(self) -> Optional[List[str]]:
        """
        Load ticker list from local CSV file.
        
        Returns:
            List of ticker symbols or None if file doesn't exist.
        """
        if not self.LOCAL_TICKER_PATH.exists():
            return None
        
        try:
            tickers = []
            with open(self.LOCAL_TICKER_PATH, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    # Skip header
                    if i == 0 and ('ticker' in row[0].lower() or 'symbol' in row[0].lower()):
                        continue
                    if row:
                        ticker = row[0].strip()
                        if ticker:
                            if not ticker.endswith('.NS'):
                                ticker = f"{ticker}.NS"
                            tickers.append(ticker.upper())
            return tickers if tickers else None
        except Exception:
            return None
    
    def get_master_ticker_list(self) -> List[str]:
        """
        Get the master list of Nifty 500 tickers.
        
        Attempts sources in order:
        1. Public CSV URL (GitHub gist with NSE data)
        2. yfinance ETF holdings (NIFTYBEES)
        3. Local fallback CSV file
        
        Returns:
            List of ticker symbols with .NS suffix (e.g., RELIANCE.NS)
            
        Raises:
            RuntimeError: If all sources fail to provide ticker list.
        """
        # Return cached result if available
        if self._ticker_cache is not None:
            return self._ticker_cache.copy()
        
        tickers = None
        
        # Try URL first
        tickers = self._fetch_from_url(self.NIFTY_500_CSV_URL)
        
        # Try yfinance ETF holdings
        if tickers is None:
            tickers = self._fetch_from_yfinance_etf()
        
        # Try local fallback
        if tickers is None:
            tickers = self._load_local_fallback()
        
        if tickers is None:
            raise RuntimeError(
                "Failed to fetch ticker list from all sources. "
                "Please ensure data/nse500_tickers.csv exists."
            )
        
        # Cache and return
        self._ticker_cache = tickers
        return tickers.copy()
    
    def _check_data_availability(self, ticker: str, years: int = 5) -> bool:
        """
        Check if a ticker has sufficient historical data.
        
        Args:
            ticker: Ticker symbol with .NS suffix.
            years: Minimum years of data required.
            
        Returns:
            True if sufficient data is available, False otherwise.
        """
        if not YFINANCE_AVAILABLE:
            # Assume available if yfinance not installed
            return True
        
        try:
            stock = yf.Ticker(ticker)
            history = stock.history(period=f"{years}y")
            
            # Check if we have roughly 250 trading days per year
            min_days = years * 200  # Conservative estimate
            return len(history) >= min_days
        except Exception:
            return False
    
    def _get_avg_volume(self, ticker: str, period: str = "1y") -> float:
        """
        Get average daily volume for a ticker.
        
        Args:
            ticker: Ticker symbol with .NS suffix.
            period: Period for volume calculation.
            
        Returns:
            Average daily volume in crores (1 crore = 10 million).
        """
        if not YFINANCE_AVAILABLE:
            return float('inf')  # Assume liquid if yfinance not installed
        
        try:
            stock = yf.Ticker(ticker)
            history = stock.history(period=period)
            
            if len(history) == 0:
                return 0.0
            
            avg_volume = history['Volume'].mean()
            # Convert to crores (1 crore = 10,000,000)
            avg_volume_cr = avg_volume / 10_000_000
            return avg_volume_cr
        except Exception:
            return 0.0
    
    def filter_universe(
        self,
        tickers: Optional[List[str]] = None,
        min_avg_volume_cr: float = 5.0,
        min_years_data: int = 5
    ) -> List[str]:
        """
        Filter universe to liquid stocks with sufficient history.
        
        Filters out:
        - Penny stocks (low volume)
        - Illiquid stocks (below volume threshold)
        - Stocks without sufficient historical data
        
        Args:
            tickers: List of tickers to filter. If None, uses master list.
            min_avg_volume_cr: Minimum average daily volume in crores.
            min_years_data: Minimum years of historical data required.
            
        Returns:
            Clean list of NSE tickers (e.g., RELIANCE.NS)
        """
        if tickers is None:
            tickers = self.get_master_ticker_list()
        
        filtered = []
        
        for ticker in tickers:
            # Ensure proper format
            if not ticker.endswith('.NS'):
                ticker = f"{ticker}.NS"
            ticker = ticker.upper()
            
            # Check data availability
            if not self._check_data_availability(ticker, min_years_data):
                continue
            
            # Check volume/liquidity
            avg_vol_cr = self._get_avg_volume(ticker)
            if avg_vol_cr < min_avg_volume_cr:
                continue
            
            filtered.append(ticker)
        
        return filtered
    
    def save_ticker_list(self, tickers: List[str], filepath: Optional[Path] = None) -> None:
        """
        Save ticker list to CSV for offline use.
        
        Args:
            tickers: List of ticker symbols.
            filepath: Path to save CSV. Defaults to local fallback path.
        """
        if filepath is None:
            filepath = self.LOCAL_TICKER_PATH
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['ticker'])
            for ticker in tickers:
                writer.writerow([ticker])
    
    def load_ticker_list(self, filepath: Optional[Path] = None) -> List[str]:
        """
        Load ticker list from CSV file.
        
        Args:
            filepath: Path to CSV file. Defaults to local fallback path.
            
        Returns:
            List of ticker symbols.
        """
        if filepath is None:
            filepath = self.LOCAL_TICKER_PATH
        
        filepath = Path(filepath)
        return self._load_local_fallback() or []
