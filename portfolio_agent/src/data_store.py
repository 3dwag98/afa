"""
Data Store for high-performance market data storage and retrieval.

Uses pandas and parquet for efficient local caching of OHLCV data.
"""

import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta

import pandas as pd

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Single source of truth for cache directory
DATA_DIR = Path("data") / "market_data"

logger = logging.getLogger(__name__)


def _ticker_filename(ticker: str) -> str:
    """
    Generate a safe filename for a ticker.
    
    Args:
        ticker: Ticker symbol (e.g., RELIANCE.NS).
        
    Returns:
        Safe filename string.
    """
    safe = ticker.replace("/", "_").replace("\\", "_").replace(":", "_")
    return f"{safe}.parquet"


def load_ticker_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Load ticker data from local parquet cache (module-level convenience function).
    
    Args:
        ticker: Ticker symbol.
        start_date: Optional start date (YYYY-MM-DD).
        end_date: Optional end date (YYYY-MM-DD).
        
    Returns:
        DataFrame with OHLCV data, or None if file doesn't exist or is empty.
    """
    # Build path
    path = DATA_DIR / _ticker_filename(ticker)
    
    if not path.exists():
        return None
    
    try:
        df = pd.read_parquet(path)
        
        if df.empty:
            return None
        
        # Restore datetime index
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        elif 'index' in df.columns:
            df['index'] = pd.to_datetime(df['index'])
            df = df.set_index('index')
        
        # Ensure index is datetime and sorted ascending
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                return None
        
        df = df.sort_index()
        
        # Filter by date range if provided
        if start_date is not None:
            df = df[df.index >= pd.to_datetime(start_date)]
        
        if end_date is not None:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        # After filtering, check if empty
        if df.empty:
            return None
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading {ticker}: {e}")
        return None


def read_cached_bars(
    ticker: str, cache_dir: Optional[Path] = None
) -> Optional[pd.DataFrame]:
    """Read one ticker's bars exactly as stored, from any cache directory.

    Distinct from the two loaders beside it in ways that matter to a caller
    checking data quality:

    * `load_ticker_data` reads only the module-level `DATA_DIR`, so it cannot
      be pointed at a second store.
    * `DataStore.load_ticker_data` *forward-fills* up to three missing days to
      paper over holidays. That is the right default for a strategy that wants
      a continuous series and exactly wrong here — a gap detector reading
      through a gap filler can never report a gap.

    Args:
        ticker: Ticker symbol, in the cache's own spelling.
        cache_dir: Directory to read. Defaults to `DATA_DIR`.

    Returns:
        Date-indexed bars sorted ascending, or None when the file is absent,
        empty, or has no usable date column.
    """
    return DataStore(cache_dir=cache_dir)._load_raw_ticker_data(ticker)


def batch_download_and_cache(
    tickers: List[str],
    start_date: str,
    end_date: str,
    chunk_size: int = 50,
    skip_existing: bool = True,
    max_workers: Optional[int] = None,
) -> bool:
    """
    Download and cache data for multiple tickers (module-level convenience function).

    Args:
        tickers: List of ticker symbols.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        chunk_size: Number of tickers per batch.
        skip_existing: Skip tickers that already have valid cached data.
        max_workers: Concurrent chunk downloads (default: DataStore's default).

    Returns:
        True on success, False on failure.
    """
    ds = DataStore(cache_dir=DATA_DIR)
    ds.chunk_size = chunk_size

    stats = ds.batch_download_and_cache(
        tickers, start_date, end_date,
        chunk_size=chunk_size,
        skip_existing=skip_existing,
        max_workers=max_workers,
    )
    
    # Return True if all tickers were successfully downloaded or skipped
    return stats['failed'] == 0


def get_cached_tickers(cache_dir: Optional[Path] = None) -> List[str]:
    """
    Discover all tickers with cached parquet data (module-level convenience
    function; the single canonical implementation used by both DataStore and
    universe.py's discover_available_tickers()).

    Symbols beginning with "^" are excluded. Those are indices (^NSEI, ^BSESN,
    ...) cached alongside equities so the market-regime filter can read them —
    they are not instruments this platform trades, and letting one into the
    universe would rank the Nifty against individual stocks by momentum, queue
    orders against it, and pollute the equal-weighted composite the regime
    filter builds from the universe.

    Args:
        cache_dir: Directory to scan. Defaults to DATA_DIR (data/market_data,
            resolved relative to the current working directory).

    Returns:
        Sorted list of unique tradeable ticker symbols. Empty if none cached.
    """
    directory = Path(cache_dir) if cache_dir is not None else DATA_DIR
    directory.mkdir(parents=True, exist_ok=True)
    tickers = {
        path.stem for path in directory.glob("*.parquet") if not path.stem.startswith("^")
    }
    return sorted(tickers)


def generate_synthetic_ohlcv(
    ticker: str, days: int = 500, seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data using a random walk. Used only as a fallback
    when neither cached nor freshly-downloaded real data is available (e.g.
    offline development, or config.data.allow_synthetic_fallback in tests).

    Args:
        ticker: Stock ticker symbol (unused in calculation, kept for API symmetry).
        days: Number of days of data to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with synthetic OHLCV data.
    """
    import numpy as np

    np.random.seed(seed)

    end_date = datetime(2024, 1, 1)
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq="B")

    returns = np.random.normal(0.0005, 0.02, len(dates))
    close_prices = 100 * np.cumprod(1 + returns)

    daily_vol = np.abs(np.random.normal(0.01, 0.005, len(dates)))
    high_prices = close_prices * (1 + daily_vol)
    low_prices = close_prices * (1 - daily_vol)
    open_prices = low_prices + np.random.uniform(0, 1, len(dates)) * (high_prices - low_prices)
    high_prices = np.maximum(high_prices, low_prices)
    volume = np.random.randint(100000, 10000000, len(dates)).astype(float)

    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def fetch_and_cache(
    config,
    tickers: List[str],
    start_date: str,
    end_date: str,
    skip_existing: bool = True,
) -> bool:
    """Populate the parquet cache from whichever source config.data.source names.

    The single place that decides between the Hub dataset and yfinance, so
    every caller (the live agent's missing-ticker top-up, the CLI's
    download-data command) picks the same source without repeating the branch.
    A failed HuggingFace ingest falls back to yfinance rather than leaving the
    run with no data at all — the fallback is logged loudly, because silently
    switching sources mid-experiment is exactly how two "identical" backtests
    end up disagreeing.

    Args:
        config: AppConfig instance.
        tickers: Tickers to fetch.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        skip_existing: Skip tickers whose cached data already covers the range
            (yfinance path only; the Hub path reads one file for everything, so
            per-ticker skipping saves nothing).

    Returns:
        True when every requested ticker was cached or skipped.
    """
    source = getattr(config.data, "source", "yfinance")

    if source == "huggingface":
        try:
            from .hf_dataset import sync_hf_to_cache
        except ImportError:  # pragma: no cover - script-style import path
            from hf_dataset import sync_hf_to_cache

        try:
            written = sync_hf_to_cache(
                dataset_id=config.data.hf_dataset_id,
                revision=config.data.hf_revision,
                asset_dir=config.data.hf_asset_dir,
                adjust_prices=config.data.hf_adjust_prices,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
            )
            missing = sorted(set(t.upper() for t in tickers) - set(written))
            if missing:
                logger.warning(
                    "%d of %d requested tickers are absent from %s (e.g. %s)",
                    len(missing), len(tickers), config.data.hf_dataset_id, missing[:5],
                )
            return not missing
        except Exception:
            logger.warning(
                "HuggingFace ingest from %s failed; falling back to yfinance for this fetch",
                config.data.hf_dataset_id, exc_info=True,
            )

    return batch_download_and_cache(
        tickers,
        start_date=start_date,
        end_date=end_date,
        skip_existing=skip_existing,
        max_workers=getattr(config.data, "download_workers", None),
    )


def load_or_fetch_data(
    config,
    force_refresh: bool = False,
    use_auto_discovery: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Load cached ticker data (via the per-ticker parquet cache), downloading
    anything missing, and falling back to synthetic data only if configured.

    Supersedes the old src/data_ingestion.py::load_or_fetch_data(), which
    maintained a second, independent combined-file cache
    (data/raw/ohlcv_data.parquet) that never interoperated with this module's
    per-ticker parquet cache — that split-brain cache is gone.

    Args:
        config: AppConfig instance.
        force_refresh: If True, ignore the cache and re-download everything.
        use_auto_discovery: If True, auto-discover all cached tickers when
            config.data.tickers is empty (which takes precedence when set).

    Returns:
        Dictionary mapping ticker to DataFrame.
    """
    tickers_to_use = list(config.data.tickers)

    if use_auto_discovery and not tickers_to_use:
        discovered = get_cached_tickers()
        if discovered:
            tickers_to_use = discovered
            logger.info(f"Auto-discovered {len(tickers_to_use)} tickers from cache")
        else:
            logger.warning("No cached tickers found and config.data.tickers is empty")

    result: Dict[str, pd.DataFrame] = {}
    missing: List[str] = []

    for ticker in tickers_to_use:
        df = None if force_refresh else load_ticker_data(ticker)
        if df is not None and len(df) > 0:
            result[ticker] = df
        else:
            missing.append(ticker)

    if missing:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=config.data.default_history_years * 365)
        logger.info(f"Fetching fresh data for {len(missing)} tickers")
        fetch_and_cache(
            config,
            missing,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            skip_existing=False,
        )
        for ticker in missing:
            df = load_ticker_data(ticker)
            if df is not None and len(df) > 0:
                result[ticker] = df

    if not result:
        if not config.data.allow_synthetic_fallback:
            # Loudly, and as an exception. Returning an empty dict let every
            # caller decide for itself what "no data" meant, and the usual
            # decision was to carry on and produce an empty-looking but
            # otherwise normal result.
            raise RuntimeError(
                f"No market data available for any of {len(tickers_to_use)} ticker(s) "
                f"and synthetic fallback is off.\n"
                f"  Run `portfolio-agent download-data` to populate the cache, then "
                f"`portfolio-agent data status` to confirm what arrived.\n"
                f"  Set data.allow_synthetic_fallback=true only for offline plumbing "
                f"tests — it substitutes random-walk bars, and any number computed "
                f"from them describes a random-number generator."
            )
        logger.warning(
            "No real data available; generating SYNTHETIC random-walk bars for %d "
            "ticker(s). Every number downstream of this describes a random-number "
            "generator, not a market.",
            len(tickers_to_use),
        )
        for ticker in tickers_to_use:
            result[ticker] = generate_synthetic_ohlcv(
                ticker, days=config.data.min_history_days, seed=config.simulation.random_seed
            )

    return result


def get_ticker_data(
    ticker: str,
    start_date: str,
    end_date: str,
    force_refresh: bool = False
) -> Optional[pd.DataFrame]:
    """
    Get ticker data with automatic caching and forward-fill.
    
    Convenience function that:
    1. Tries to load from cache (unless force_refresh=True)
    2. Downloads and caches if not available
    3. Forward-fills missing business days
    
    Args:
        ticker: Ticker symbol.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        force_refresh: If True, forces re-download even if cached.
        
    Returns:
        DataFrame with OHLCV data and forward-filled dates, or None on failure.
    """
    # Normalize ticker
    if not ticker.endswith('.NS'):
        ticker = f"{ticker}.NS"
    ticker = ticker.upper()
    
    # 1. Try cache first unless force_refresh
    if not force_refresh:
        df = load_ticker_data(ticker, start_date, end_date)
        if df is not None and len(df) > 0:
            return _fill_missing_days(df, start_date, end_date)
    
    # 2. Download and cache
    ok = batch_download_and_cache([ticker], start_date, end_date, skip_existing=False)
    if not ok:
        return None
    
    # 3. Load back
    df = load_ticker_data(ticker, start_date, end_date)
    if df is None:
        return None
    
    return _fill_missing_days(df, start_date, end_date)


def _extract_ticker_df(raw: pd.DataFrame, ticker: str, is_single: bool) -> Optional[pd.DataFrame]:
    """
    Extract DataFrame for a single ticker from yfinance output.
    
    Handles both single-ticker (flat columns) and multi-ticker (MultiIndex columns)
    outputs from yfinance.download().
    
    Args:
        raw: Raw DataFrame from yfinance.download().
        ticker: Ticker symbol to extract.
        is_single: True if downloading a single ticker (flat columns expected).
        
    Returns:
        Clean DataFrame with flat column names, or None if extraction fails.
    """
    if raw is None or raw.empty:
        return None
    
    if is_single:
        df = raw.copy()
    else:
        # Multi-ticker download: columns are MultiIndex (ticker, metric)
        if not isinstance(raw.columns, pd.MultiIndex):
            # Unexpected format, try to handle gracefully
            df = raw.copy()
        elif ticker not in raw.columns.get_level_values(0):
            return None
        else:
            df = raw[ticker].copy()
    
    # Flatten if still MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Lowercase all column names
    df.columns = [str(c).lower() for c in df.columns]
    
    # Drop fully-empty rows
    df = df.dropna(how="all")
    
    return df


def generate_data_quality_report(
    df: Optional[pd.DataFrame], 
    ticker: str,
    stale_threshold_days: int = 5
) -> Dict[str, Any]:
    """
    Generate a comprehensive data quality report for a ticker's DataFrame.
    
    Args:
        df: DataFrame with OHLCV data and DatetimeIndex. Can be None or empty.
        ticker: Ticker symbol for the report.
        stale_threshold_days: Number of days after which data is considered stale.
        
    Returns:
        Dictionary with quality metrics using exactly these keys:
        - ticker: str
        - rows: int
        - start_date: str or None (ISO date string)
        - end_date: str or None
        - missing_values: dict (column -> count of NaN)
        - total_missing: int
        - duplicate_dates: int
        - zero_volume_days: int
        - date_gaps: int (business-day gaps > 1 day between consecutive rows)
        - days_out_of_range: int (rows outside expected range, default 0)
        - is_stale: bool (True if end_date older than stale_threshold_days)
        - passed: bool (overall quality gate)
    """
    # Handle empty/None input
    if df is None or df.empty:
        return {
            "ticker": ticker,
            "rows": 0,
            "start_date": None,
            "end_date": None,
            "missing_values": {},
            "total_missing": 0,
            "duplicate_dates": 0,
            "zero_volume_days": 0,
            "date_gaps": 0,
            "days_out_of_range": 0,
            "is_stale": True,
            "passed": False
        }
    
    # Ensure we have a copy to work with
    df = df.copy()
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    df = df.sort_index()
    
    # Calculate basic stats
    rows = len(df)
    start_date = df.index.min()
    end_date = df.index.max()
    
    # Convert dates to ISO strings
    start_date_str = start_date.strftime('%Y-%m-%d') if pd.notna(start_date) else None
    end_date_str = end_date.strftime('%Y-%m-%d') if pd.notna(end_date) else None
    
    # Count missing values per column
    missing_values = {}
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        if missing_count > 0:
            missing_values[col] = missing_count
    
    total_missing = sum(missing_values.values())
    
    # Count duplicate dates
    duplicate_dates = int(df.index.duplicated().sum())
    
    # Count zero volume days (handle missing 'volume' column gracefully)
    if 'volume' in df.columns:
        zero_volume_days = int((df['volume'] == 0).sum())
    else:
        zero_volume_days = 0
    
    # Calculate date gaps: count diffs greater than 3 calendar days
    # This tolerates weekends but catches longer gaps
    if len(df) > 1:
        index_series = pd.Series(df.index)
        diffs = index_series.diff().dt.days
        date_gaps = int((diffs > 3).sum())
    else:
        date_gaps = 0
    
    # Days out of range (default 0 as no expected range provided)
    days_out_of_range = 0
    
    # Check if data is stale
    last_date = pd.Timestamp(end_date).normalize()
    today = pd.Timestamp.now().normalize()
    is_stale = (today - last_date) > pd.Timedelta(days=stale_threshold_days)
    
    # Calculate passed status
    # passed = (rows > 0) and (total_missing / max(rows,1) < 0.10) and (not is_stale)
    missing_ratio = total_missing / max(rows, 1)
    passed = (rows > 0) and (missing_ratio < 0.10) and (not is_stale)
    
    return {
        "ticker": ticker,
        "rows": rows,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "missing_values": missing_values,
        "total_missing": total_missing,
        "duplicate_dates": duplicate_dates,
        "zero_volume_days": zero_volume_days,
        "date_gaps": date_gaps,
        "days_out_of_range": days_out_of_range,
        "is_stale": is_stale,
        "passed": passed
    }


def _fill_missing_days(
    df: pd.DataFrame, 
    start_date: str, 
    end_date: str, 
    ffill_limit: int = 3
) -> pd.DataFrame:
    """
    Fill missing business days in the DataFrame using forward-fill.
    
    Creates a complete business-day date range and forward-fills price columns
    while leaving volume as 0 for filled days. This handles market holidays
    and weekends properly.
    
    Args:
        df: DataFrame with OHLCV data and DatetimeIndex.
        start_date: Start date for the complete range (YYYY-MM-DD).
        end_date: End date for the complete range (YYYY-MM-DD).
        ffill_limit: Maximum number of consecutive days to forward-fill.
        
    Returns:
        DataFrame with complete business-day index and forward-filled prices.
    """
    df = df.copy()
    
    # Ensure index is datetime
    df.index = pd.to_datetime(df.index)
    
    # Remove duplicate indices, keep last
    df = df[~df.index.duplicated(keep='last')]
    
    # Sort by index
    df = df.sort_index()
    
    # Create full business-day range (Mon-Fri)
    # Market holidays will appear as NaN rows to fill
    full_range = pd.bdate_range(start=start_date, end=end_date)
    
    # Reindex to full business-day range
    df = df.reindex(full_range)
    
    # Forward-fill PRICE columns only, with limit to avoid filling long suspensions
    price_cols = [c for c in df.columns if c.lower() != 'volume']
    df[price_cols] = df[price_cols].ffill(limit=ffill_limit)
    
    # Volume: never forward-fill. Fill remaining NaN with 0.
    if 'volume' in df.columns:
        df['volume'] = df['volume'].fillna(0)
    
    # Set index name
    df.index.name = 'date'
    
    return df


class DataStore:
    """
    High-performance data store for market data.
    
    Features:
    - Batch download with chunking to avoid API rate limits
    - Parquet-based local storage for fast I/O
    - Retry logic with exponential backoff
    - Forward-fill for handling market holidays
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the DataStore.
        
        Args:
            cache_dir: Directory for storing parquet files.
                      Defaults to DATA_DIR (data/market_data, resolved
                      relative to the current working directory).
        """
        # Must default to the same DATA_DIR the module-level helpers read from.
        # It used to default to `<package>/data/market_data` instead, so a
        # DataStore() built without an explicit cache_dir wrote parquet files
        # into the source tree where load_ticker_data() would never find them.
        if cache_dir is None:
            cache_dir = DATA_DIR
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.chunk_size = 50  # Tickers per batch
        self.max_retries = 3
        self.base_delay = 1.0  # Base delay in seconds for backoff
        self.download_workers = 4  # Concurrent chunk downloads (network-bound)
    
    def _get_ticker_path(self, ticker: str) -> Path:
        """
        Get the parquet file path for a ticker.
        
        Args:
            ticker: Ticker symbol (e.g., RELIANCE.NS).
            
        Returns:
            Path to the parquet file.
        """
        return self.cache_dir / _ticker_filename(ticker)
    
    def _parse_ticker_from_path(self, path: Path) -> str:
        """
        Parse ticker symbol from parquet filename.
        
        Args:
            path: Path to parquet file.
            
        Returns:
            Ticker symbol (e.g., RELIANCE.NS).
        """
        filename = path.stem
        # Reverse the safe transformation
        return filename.replace('_NS', '.NS').replace('_', '.')
    
    def has_ticker_data(self, ticker: str, min_bytes: int = 1) -> bool:
        """Whether this ticker already has a non-empty parquet file cached.

        Used to skip work that has already been done — most importantly by the
        HuggingFace sync, which without it re-downloads every symbol on every
        invocation.

        `min_bytes` guards the case that makes a naive `exists()` check worse
        than no check at all: a run interrupted mid-write leaves a zero-byte
        file, and treating that as "cached" would permanently skip a ticker
        that never actually downloaded.
        """
        path = self._get_ticker_path(ticker)
        try:
            return path.is_file() and path.stat().st_size >= max(1, int(min_bytes))
        except OSError:
            return False

    def save_ticker_data(self, ticker: str, df: pd.DataFrame) -> Path:
        """
        Save ticker DataFrame to parquet file.

        Args:
            ticker: Ticker symbol.
            df: DataFrame with OHLCV data.

        Returns:
            Path to saved parquet file.
        """
        # Ensure directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        
        # Lowercase all column names
        df.columns = [c.lower() for c in df.columns]
        
        # Drop fully-empty rows
        df = df.dropna(how="all")
        
        # Reset index to include date as column
        df_to_save = df.reset_index()
        

        path = self._get_ticker_path(ticker)
        
        # Use pyarrow engine if available, otherwise fastparquet
        try:
            df_to_save.to_parquet(path, engine='pyarrow', index=False)
        except ImportError:
            try:
                df_to_save.to_parquet(path, engine='fastparquet', index=False)
            except ImportError:
                # Fallback to default engine
                df_to_save.to_parquet(path, index=False)
        
        return path
    
    # NOTE: a first, shadowed copy of load_ticker_data() used to sit here. Two
    # definitions of the same method in one class means only the last one is
    # ever callable — the dead copy was removed so the forward-filling version
    # below is unambiguously the implementation.

    def _fetch_chunk(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        max_retries: int = 3
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Download data for a chunk of tickers with retry logic.
        
        Uses yfinance.download with group_by='ticker' to fetch multiple
        tickers efficiently, then extracts individual DataFrames.
        
        Args:
            tickers: List of ticker symbols.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            max_retries: Maximum retry attempts with exponential backoff.
            
        Returns:
            Dictionary mapping ticker to DataFrame (or None if failed).
        """
        results: Dict[str, Optional[pd.DataFrame]] = {}
        
        if not YFINANCE_AVAILABLE:
            for ticker in tickers:
                results[ticker] = None
            return results
        
        is_single = len(tickers) == 1
        delay = self.base_delay
        
        for attempt in range(max_retries):
            try:
                # Download with group_by='ticker' for consistent MultiIndex output
                raw = yf.download(
                    tickers,
                    start=start_date,
                    end=end_date,
                    group_by='ticker',
                    auto_adjust=True,
                    progress=False,
                    threads=False
                )
                
                # Extract each ticker's DataFrame
                for ticker in tickers:
                    df = _extract_ticker_df(raw, ticker, is_single)
                    
                    if df is not None and not df.empty:
                        # Check required columns
                        if 'close' not in df.columns:
                            logger.warning(f"Ticker {ticker} missing 'close' column, skipping.")
                            results[ticker] = None
                            continue
                        
                        # Add volume as 0 if missing
                        if 'volume' not in df.columns:
                            df['volume'] = 0
                        
                        results[ticker] = df
                    else:
                        results[ticker] = None
                
                # Successfully downloaded, return results
                return results
                
            except Exception as e:
                logger.warning(f"Chunk download attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    # All retries exhausted, return None for all tickers
                    logger.error(f"Failed to download chunk after {max_retries} attempts: {e}")
                    for ticker in tickers:
                        results[ticker] = None
                    return results
        
        return results
    
    def _is_cache_valid(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> bool:
        """
        Check if cached data covers the requested date range.
        
        Args:
            ticker: Ticker symbol.
            start_date: Requested start date.
            end_date: Requested end date.
            
        Returns:
            True if cache is valid, False otherwise.
        """
        df = self._load_raw_ticker_data(ticker)
        
        if df is None or len(df) == 0:
            return False
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            return False
        
        # Check date range coverage
        cache_start = pd.Timestamp(df.index.min())
        cache_end = pd.Timestamp(df.index.max())
        
        req_start = pd.Timestamp(start_date)
        req_end = pd.Timestamp(end_date)
        
        return cache_start <= req_start and cache_end >= req_end
    
    def batch_download_and_cache(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        chunk_size: Optional[int] = None,
        skip_existing: bool = True,
        max_workers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Download and cache data for multiple tickers in batches.

        Downloads data in chunks to avoid API rate limits. Chunks are fetched
        concurrently on a small thread pool — downloading is network-bound, so
        threads (not processes) are the right tool and the GIL is released
        while waiting on the socket. Parquet writes and statistics stay on the
        calling thread, keeping file I/O serialized and the returned stats
        deterministic regardless of which chunk lands first.

        Args:
            tickers: List of ticker symbols.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            chunk_size: Number of tickers per batch. Defaults to self.chunk_size.
            skip_existing: Skip tickers that already have valid cached data.
            max_workers: Concurrent chunk downloads. Defaults to
                self.download_workers. Use 1 for strictly sequential downloads.

        Returns:
            Dictionary with download statistics:
            - total: Total number of tickers
            - downloaded: Number successfully downloaded
            - skipped: Number skipped (already cached)
            - failed: Number that failed to download
            - errors: List of failed ticker symbols
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        if max_workers is None:
            max_workers = self.download_workers

        stats = {
            'total': len(tickers),
            'downloaded': 0,
            'skipped': 0,
            'failed': 0,
            'errors': []
        }

        # Build the chunks up front, resolving what actually needs fetching.
        chunks: List[List[str]] = []
        for i in range(0, len(tickers), chunk_size):
            normalized_chunk = []
            for ticker in tickers[i:i + chunk_size]:
                if not ticker.endswith('.NS'):
                    ticker = f"{ticker}.NS"
                normalized_chunk.append(ticker.upper())

            to_download = []
            for ticker in normalized_chunk:
                if skip_existing and self._is_cache_valid(ticker, start_date, end_date):
                    stats['skipped'] += 1
                else:
                    to_download.append(ticker)

            if to_download:
                chunks.append(to_download)

        if not chunks:
            print(f"Download complete: 0 downloaded, {stats['skipped']} skipped, 0 failed")
            return stats

        total_chunks = len(chunks)
        workers = max(1, min(max_workers, total_chunks))

        def _record(chunk_index: int, results: Dict[str, Optional[pd.DataFrame]]) -> None:
            print(f"Processing chunk {chunk_index + 1}/{total_chunks}")
            for ticker, df in results.items():
                if df is not None and not df.empty:
                    self.save_ticker_data(ticker, df)
                    stats['downloaded'] += 1
                else:
                    stats['failed'] += 1
                    stats['errors'].append(ticker)

        if workers == 1:
            for index, chunk in enumerate(chunks):
                _record(index, self._fetch_chunk(chunk, start_date, end_date))
                if index + 1 < total_chunks:
                    time.sleep(1.0)  # be polite to the provider
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(self._fetch_chunk, chunk, start_date, end_date)
                    for chunk in chunks
                ]
                # Consumed in submission order so stats['errors'] is stable.
                for index, future in enumerate(futures):
                    try:
                        _record(index, future.result())
                    except Exception as e:
                        logger.error(f"Chunk {index + 1} failed: {e}")
                        stats['failed'] += len(chunks[index])
                        stats['errors'].extend(chunks[index])

        print(f"Download complete: {stats['downloaded']} downloaded, "
              f"{stats['skipped']} skipped, {stats['failed']} failed")

        return stats
    
    def load_ticker_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        forward_fill_days: int = 3
    ) -> Optional[pd.DataFrame]:
        """
        Load ticker data from local parquet cache.
        
        Handles missing days using forward-fill to account for market holidays.
        Uses business-day frequency and only forward-fills price columns (not volume).
        
        Args:
            ticker: Ticker symbol.
            start_date: Start date (YYYY-MM-DD). If None, loads all data.
            end_date: End date (YYYY-MM-DD). If None, loads all data.
            forward_fill_days: Maximum days to forward-fill for missing data.
            
        Returns:
            Clean pandas DataFrame with OHLCV data, or None if data unavailable.
        """
        # Ensure proper format
        if not ticker.endswith('.NS'):
            ticker = f"{ticker}.NS"
        ticker = ticker.upper()
        
        # Load from cache (without date filtering first)
        df = self._load_raw_ticker_data(ticker)
        
        if df is None or len(df) == 0:
            return None
        
        # Filter by date range if provided
        if start_date is not None or end_date is not None:
            if start_date is not None:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date is not None:
                df = df[df.index <= pd.to_datetime(end_date)]
            
            if len(df) == 0:
                return None
            
            # Use the _fill_missing_days helper for proper business-day handling
            df = _fill_missing_days(df, start_date, end_date, ffill_limit=forward_fill_days)
            
            # Drop any leading rows where 'close' is still NaN (before first real data point)
            if 'close' in df.columns:
                df = df[df['close'].notna()]
            
            # If after reindex + ffill the 'close' column is entirely NaN, return None
            if df.empty or ('close' in df.columns and df['close'].isna().all()):
                return None
            
            # Restore original column types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Add ticker column if not present
            if 'ticker' not in df.columns:
                df['ticker'] = ticker
            
            return df
        
        return df
    
    def _load_raw_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load raw ticker data from local parquet cache without any processing.
        
        This is used internally by load_ticker_data.
        
        Args:
            ticker: Ticker symbol.
            
        Returns:
            DataFrame with OHLCV data, or None if file doesn't exist.
        """
        # Build path using the same helper
        path = self.cache_dir / _ticker_filename(ticker)
        
        if not path.exists():
            return None
        
        try:
            df = pd.read_parquet(path)
            
            if df.empty:
                return None
            
            # Restore datetime index
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            elif 'index' in df.columns:
                df['index'] = pd.to_datetime(df['index'])
                df = df.set_index('index')
            
            # Ensure index is datetime and sorted ascending
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    return None
            
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {ticker}: {e}")
            return None
    
    def load_ticker_data_only(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load ticker data from local parquet cache without date filtering.
        
        This is a simple wrapper for backward compatibility.
        
        Args:
            ticker: Ticker symbol.
            
        Returns:
            DataFrame with OHLCV data, or None if file doesn't exist.
        """
        return self.load_ticker_data(ticker, start_date=None, end_date=None)
    
    def get_cached_tickers(self) -> List[str]:
        """
        Get list of all tickers with cached data.

        Returns:
            List of ticker symbols.
        """
        return get_cached_tickers(self.cache_dir)
    
    def clear_cache(self, ticker: Optional[str] = None) -> None:
        """
        Clear cached data.
        
        Args:
            ticker: Specific ticker to clear. If None, clears all cache.
        """
        if ticker is None:
            # Clear all
            for path in self.cache_dir.glob("*.parquet"):
                path.unlink()
        else:
            # Clear specific ticker
            if not ticker.endswith('.NS'):
                ticker = f"{ticker}.NS"
            ticker = ticker.upper()
            path = self.cache_dir / _ticker_filename(ticker)
            if path.exists():
                path.unlink()
    
    def get_data_quality_report(
        self,
        tickers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate a data quality report for cached tickers.
        
        Args:
            tickers: List of tickers to check. If None, checks all cached.
            
        Returns:
            DataFrame with quality metrics per ticker.
        """
        if tickers is None:
            tickers = self.get_cached_tickers()
        
        reports = []
        for ticker in tickers:
            df = self.load_ticker_data_only(ticker)
            
            if df is None:
                continue
            
            report = {
                'ticker': ticker,
                'start_date': df.index.min(),
                'end_date': df.index.max(),
                'total_days': len(df),
                'missing_days': None,  # Would need comparison to trading calendar
                'has_volume': bool('volume' in df.columns and df['volume'].notna().any())
            }
            reports.append(report)
        
        return pd.DataFrame(reports)
