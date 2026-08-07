"""Data ingestion module for fetching market data."""

import pandas as pd
from typing import List, Optional
from datetime import datetime, timedelta


def fetch_historical_data(ticker: str, days: int = 250) -> pd.DataFrame:
    """Fetch historical price data for a ticker.

    Args:
        ticker: Stock ticker symbol (e.g., 'RELIANCE.NS').
        days: Number of days of historical data to fetch.

    Returns:
        DataFrame with OHLCV data.
    """
    # Placeholder - to be implemented with yfinance
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    try:
        import yfinance as yf
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            return pd.DataFrame()
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


def fetch_multiple_tickers(tickers: List[str], days: int = 250) -> dict:
    """Fetch historical data for multiple tickers.

    Args:
        tickers: List of ticker symbols.
        days: Number of days of historical data.

    Returns:
        Dictionary mapping ticker to DataFrame.
    """
    data = {}
    for ticker in tickers:
        df = fetch_historical_data(ticker, days)
        if not df.empty:
            data[ticker] = df
    return data


def validate_data(df: pd.DataFrame, min_rows: int = 100) -> bool:
    """Validate that fetched data meets minimum requirements.

    Args:
        df: DataFrame to validate.
        min_rows: Minimum required rows.

    Returns:
        True if valid, False otherwise.
    """
    if df.empty:
        return False
    if len(df) < min_rows:
        return False
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        return False
    return True
