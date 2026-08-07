"""Data ingestion module for fetching market data."""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# Configure logging
logger = logging.getLogger(__name__)


def _setup_logging(log_file: str = "logs/agent.log") -> None:
    """Setup logging to file and console."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)


def generate_synthetic_ohlcv(
    ticker: str, days: int = 500, seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic OHLCV data using random walk.
    
    Args:
        ticker: Stock ticker symbol.
        days: Number of days of data to generate.
        seed: Random seed for reproducibility.
    
    Returns:
        DataFrame with synthetic OHLCV data.
    """
    np.random.seed(seed)
    
    # Generate dates - use fixed end date for determinism
    end_date = datetime(2024, 1, 1)
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq="B")  # Business days
    
    # Random walk for close prices starting at 100
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Mean daily return ~0.05%, vol ~2%
    close_prices = 100 * np.cumprod(1 + returns)
    
    # Generate high, low based on close
    daily_vol = np.abs(np.random.normal(0.01, 0.005, len(dates)))  # Daily volatility
    high_prices = close_prices * (1 + daily_vol)
    low_prices = close_prices * (1 - daily_vol)
    
    # Open is between low and high
    open_prices = low_prices + np.random.uniform(0, 1, len(dates)) * (high_prices - low_prices)
    
    # Ensure high >= low
    high_prices = np.maximum(high_prices, low_prices)
    
    # Volume: random positive values
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
    df.index.name = "Date"
    
    return df


def fetch_ohlcv(tickers: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV data for Indian tickers using yfinance.
    
    Args:
        tickers: List of ticker symbols (e.g., ['RELIANCE.NS', 'TCS.NS']).
        period: Period of data to fetch (default "2y").
    
    Returns:
        Dictionary mapping ticker to DataFrame with columns:
        open, high, low, close, volume.
    """
    _setup_logging()
    
    try:
        # Download all tickers at once with group_by="ticker"
        data = yf.download(tickers, period=period, group_by="ticker", auto_adjust=True)
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        return {}
    
    result = {}
    
    if len(tickers) == 1:
        # Single ticker case - data might not be multi-indexed
        df = data.copy()
        if df.empty:
            logger.warning(f"No data received for ticker: {tickers[0]}")
            return {}
        
        # Normalize column names to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Sort by date ascending
        df = df.sort_index()
        
        # Drop rows where close is NaN
        df = df.dropna(subset=["close"])
        
        if df.empty:
            logger.warning(f"No valid data for ticker: {tickers[0]}")
            return {}
        
        result[tickers[0]] = df
    else:
        # Multiple tickers - data is multi-indexed by ticker
        for ticker in tickers:
            try:
                df = data[ticker].copy()
                
                if df.empty:
                    logger.warning(f"No data received for ticker: {ticker}")
                    continue
                
                # Flatten column names if multi-level
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0].lower() for col in df.columns]
                else:
                    df.columns = [col.lower() for col in df.columns]
                
                # Sort by date ascending
                df = df.sort_index()
                
                # Drop rows where close is NaN
                df = df.dropna(subset=["close"])
                
                if df.empty:
                    logger.warning(f"No valid data for ticker: {ticker}")
                    continue
                
                result[ticker] = df
            except Exception as e:
                logger.warning(f"Error processing ticker {ticker}: {e}")
                continue
    
    return result


def validate_ohlcv(df: pd.DataFrame, min_rows: int = 50) -> bool:
    """Validate OHLCV data meets requirements.
    
    Args:
        df: DataFrame to validate.
        min_rows: Minimum required rows.
    
    Returns:
        True if valid, False otherwise.
    """
    # Must have at least min_rows rows
    if len(df) < min_rows:
        return False
    
    # Close must be positive
    if "close" not in df.columns or not (df["close"] > 0).all():
        return False
    
    # High >= low
    if "high" not in df.columns or "low" not in df.columns:
        return False
    if not (df["high"] >= df["low"]).all():
        return False
    
    # No duplicate dates
    if df.index.duplicated().any():
        return False
    
    return True


def load_or_fetch_data(config, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Load cached data or fetch fresh data from yfinance.
    
    Args:
        config: AppConfig instance with configuration.
        force_refresh: If True, ignore cache and fetch fresh data.
    
    Returns:
        Dictionary mapping ticker to DataFrame.
    """
    _setup_logging(config.log_file)
    
    # Determine cache directory
    base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cache_dir = base_dir / "data" / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Check cache
    cache_file = cache_dir / "ohlcv_data.parquet"
    
    if not force_refresh and cache_file.exists():
        logger.info(f"Loading cached data from {cache_file}")
        try:
            # Load parquet and split by ticker if it has ticker column
            cached_df = pd.read_parquet(cache_file)
            if "ticker" in cached_df.columns:
                result = {}
                for ticker in cached_df["ticker"].unique():
                    ticker_df = cached_df[cached_df["ticker"] == ticker].drop(
                        columns=["ticker"]
                    )
                    if validate_ohlcv(ticker_df):
                        result[ticker] = ticker_df
                if result:
                    logger.info(f"Loaded {len(result)} tickers from cache")
                    return result
        except Exception as e:
            logger.warning(f"Error loading cache: {e}")
    
    # Fetch fresh data
    logger.info(f"Fetching fresh data for tickers: {config.tickers}")
    data = fetch_ohlcv(config.tickers)
    
    if not data:
        logger.warning("No data fetched from yfinance")
        
        # Check if synthetic fallback is allowed
        allow_synthetic = getattr(config, "allow_synthetic_fallback", False)
        if allow_synthetic:
            logger.info("Generating synthetic data as fallback")
            for ticker in config.tickers:
                data[ticker] = generate_synthetic_ohlcv(
                    ticker, days=config.min_history_days, seed=config.random_seed
                )
        else:
            logger.error("yfinance failed and synthetic fallback is disabled")
            return {}
    
    # Cache the data
    logger.info(f"Caching data to {cache_file}")
    try:
        # Combine all dataframes with ticker column for storage
        dfs_with_ticker = []
        for ticker, df in data.items():
            df_copy = df.copy()
            df_copy["ticker"] = ticker
            dfs_with_ticker.append(df_copy)
        
        if dfs_with_ticker:
            combined_df = pd.concat(dfs_with_ticker, ignore_index=False)
            combined_df.to_parquet(cache_file)
    except Exception as e:
        logger.warning(f"Error caching data: {e}")
    
    return data
