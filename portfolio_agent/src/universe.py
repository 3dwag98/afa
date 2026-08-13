"""
Universe Manager for Indian Equity Market (NSE/BSE).

Handles ticker list retrieval and filtering for the Nifty 500 universe.
"""

import os
import csv
import logging
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

logger = logging.getLogger(__name__)


def discover_available_tickers(data_dir: str = "data/market_data") -> list[str]:
    """
    Discover all tickers for which historical data is available locally.

    Thin wrapper over data_store.get_cached_tickers() (the single canonical
    parquet-cache scanning implementation).

    Args:
        data_dir: Directory containing parquet files. Defaults to "data/market_data".

    Returns:
        Sorted list of unique ticker symbols. Empty list if no parquet files exist.
    """
    from .data_store import get_cached_tickers

    return get_cached_tickers(Path(data_dir))


def select_universe(
    tickers: list[str],
    max_tickers: int | None = None,
    selection: str = "alphabetical",
    seed: int = 42,
    purpose: str = "backtest",
) -> list[str]:
    """Choose `max_tickers` from the available cache.

    **Why this is not just `tickers[:n]`.** The cache is scanned in sorted
    filename order, so an alphabetical truncation returns whatever happens to
    sit at the front of the alphabet — the same few hundred names every time,
    for training and for backtesting alike. Two consequences, both bad:

    - It is not a sample of the market. It is a sample of the alphabet, and
      every cross-sectional claim (momentum deciles, low-volatility ranks) is
      then made about that slice rather than about Indian equities.
    - Training and backtesting see the *identical* names, so a model is
      evaluated on the very tickers it was fitted on. That is not out-of-sample
      in the cross-sectional dimension, however carefully the dates are split.

    `selection="random"` draws a seeded random sample instead, and `purpose`
    offsets the seed so the training and backtest draws are different samples
    of the same cache. Seeded rather than truly random: two runs of the same
    config must produce the same universe or nothing is reproducible, and the
    platform's determinism tests would fail immediately.

    Note what this does *not* fix. A random sample of an alphabetical cache is
    still not a point-in-time index membership — the names present are the
    names that survived to be downloaded, so survivorship bias is untouched.
    See docs/REVIEW_STATUS.md (D9); this makes the sample less arbitrary, not
    correct.

    Args:
        tickers: Available tickers, in cache order.
        max_tickers: How many to return. None or <= 0 returns all of them.
        selection: "alphabetical" (the historical behaviour) or "random".
        seed: Base seed for the random draw.
        purpose: Offsets the seed so different consumers draw different
            samples. "train" and "backtest" are the meaningful values.

    Returns:
        The selected tickers, always sorted, so downstream ordering is stable
        regardless of how they were drawn.
    """
    if max_tickers is None or max_tickers <= 0 or max_tickers >= len(tickers):
        return list(tickers)

    if selection == "random":
        import random

        # zlib.crc32 rather than hash(): PYTHONHASHSEED randomizes str hashing
        # per process, so hash(purpose) would draw a different universe on
        # every invocation — the exact non-reproducibility this seeds against.
        import zlib

        offset = zlib.crc32(purpose.encode("utf-8"))
        rng = random.Random((int(seed) + offset) & 0xFFFFFFFF)
        return sorted(rng.sample(list(tickers), max_tickers))

    return list(tickers[:max_tickers])


#: The draw used by everything that *measures* a strategy rather than fits one.
#:
#: `purpose` offsets the RNG so training and measurement sample different names
#: — a model must not be scored on the tickers it was fitted on. But that split
#: is two-way, not three-way, and `evaluate` had silently fallen through to
#: `"train"`: it scored the *training* draw while `backtest` traded a different
#: one. At `universe_size=50` on a 400-name cache the two shared **6 names**,
#: so an IC and an equity curve for "the same strategy" described essentially
#: different markets, and the platform printed them side by side.
#:
#: Named rather than spelled out at each call site, because the bug was a
#: default argument nobody passed, and a shared constant is the thing a reader
#: notices is missing.
MEASUREMENT_PURPOSE = "backtest"

#: The draw a model is fitted on. Deliberately different from the above.
TRAINING_PURPOSE = "train"


def resolve_backtest_universe(
    force_full_download: bool = False,
    max_tickers: int | None = None,
    selection: str = "alphabetical",
    seed: int = 42,
    purpose: str = "backtest",
) -> list[str]:
    """
    Resolve the universe of tickers for backtesting.

    This function:
    1. Discovers locally available tickers
    2. If none found or force_full_download=True, downloads data for all master tickers
    3. Selects max_tickers of them (see select_universe)

    Args:
        force_full_download: If True, forces download of full master list.
        max_tickers: Maximum number of tickers to return (for quick tests).
                     None means use ALL available.
        selection: "alphabetical" or "random" — see select_universe.
        seed: Base seed for a random draw.
        purpose: "train" or "backtest"; offsets the seed so the two draw
                 different samples of the same cache.

    Returns:
        List of ticker symbols with available data. Never returns None.
    """
    # Import here to avoid circular imports
    import pandas as pd
    from portfolio_agent.src.data_store import batch_download_and_cache
    
    # Step 1: Discover available tickers
    tickers = discover_available_tickers()
    
    # Step 2: If empty OR force_full_download, download from master list
    if not tickers or force_full_download:
        try:
            # Get master ticker list
            manager = UniverseManager()
            master_list = manager.get_master_ticker_list()
            
            if master_list:
                # Calculate date range (5 years of history)
                end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
                start_date = (pd.Timestamp.now() - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
                
                # Download data for all tickers
                batch_download_and_cache(
                    tickers=master_list,
                    start_date=start_date,
                    end_date=end_date,
                    skip_existing=False
                )
                
                # Re-discover tickers that actually got valid data
                tickers = discover_available_tickers()
        except Exception as e:
            logger.warning(f"Download failed: {e}. Falling back to locally discovered tickers.")
            # Fall back to whatever was discovered locally (may be empty)
            pass
    
    # Step 3: Select from what is available
    available = len(tickers)
    tickers = select_universe(
        tickers,
        max_tickers=max_tickers,
        selection=selection,
        seed=seed,
        purpose=purpose,
    )

    logger.info(
        "Resolved universe: %d of %d cached tickers (%s selection, purpose=%s)",
        len(tickers), available, selection, purpose,
    )

    return tickers


class UniverseManager:
    """
    Manages the universe of Indian equities for backtesting.
    
    Provides methods to:
    - Get master ticker list (Nifty 500)
    - Filter tickers based on liquidity and data availability
    """
    
    # Public CSV URL for Nifty 500 constituents (GitHub Gist or similar)
    NIFTY_500_CSV_URL = "https://raw.githubusercontent.com/amitkumarjha/nse-data/main/nifty500_tickers.csv"
    
    # Fallback local path (cwd-relative, matching data_store.DATA_DIR's convention)
    LOCAL_TICKER_PATH = Path("data") / "nse500_tickers.csv"

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the UniverseManager.

        Args:
            cache_dir: Directory for caching ticker lists. Defaults to data/ directory
                (resolved relative to the current working directory).
        """
        if cache_dir is None:
            cache_dir = Path("data")
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
