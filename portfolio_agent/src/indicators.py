"""Technical indicators module."""

import pandas as pd
import numpy as np
from typing import Dict, Any


def calculate_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Simple Moving Average.

    Args:
        df: DataFrame with 'Close' column.
        period: SMA period.

    Returns:
        Series with SMA values.
    """
    return df['Close'].rolling(window=period).mean()


def calculate_ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Exponential Moving Average.

    Args:
        df: DataFrame with 'Close' column.
        period: EMA period.

    Returns:
        Series with EMA values.
    """
    return df['Close'].ewm(span=period, adjust=False).mean()


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index.

    Args:
        df: DataFrame with 'Close' column.
        period: RSI period.

    Returns:
        Series with RSI values.
    """
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(df: pd.DataFrame, fast: int = 12, 
                   slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """Calculate MACD indicator.

    Args:
        df: DataFrame with 'Close' column.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal line period.

    Returns:
        Dictionary with MACD, signal, and histogram series.
    """
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, 
                               std_dev: float = 2.0) -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands.

    Args:
        df: DataFrame with 'Close' column.
        period: Rolling window period.
        std_dev: Number of standard deviations.

    Returns:
        Dictionary with upper, middle, and lower band series.
    """
    middle = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    return {
        'upper': upper,
        'middle': middle,
        'lower': lower
    }


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range.

    Args:
        df: DataFrame with 'High', 'Low', 'Close' columns.
        period: ATR period.

    Returns:
        Series with ATR values.
    """
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to a DataFrame.

    Args:
        df: DataFrame with OHLCV data.

    Returns:
        DataFrame with added indicator columns.
    """
    result = df.copy()

    # Moving averages
    result['SMA_20'] = calculate_sma(df, 20)
    result['SMA_50'] = calculate_sma(df, 50)
    result['EMA_20'] = calculate_ema(df, 20)

    # RSI
    result['RSI'] = calculate_rsi(df, 14)

    # MACD
    macd_data = calculate_macd(df)
    result['MACD'] = macd_data['macd']
    result['MACD_Signal'] = macd_data['signal']
    result['MACD_Hist'] = macd_data['histogram']

    # Bollinger Bands
    bb_data = calculate_bollinger_bands(df)
    result['BB_Upper'] = bb_data['upper']
    result['BB_Middle'] = bb_data['middle']
    result['BB_Lower'] = bb_data['lower']

    # ATR
    result['ATR'] = calculate_atr(df)

    return result
