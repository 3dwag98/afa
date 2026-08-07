"""Technical indicators module."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging

# Use absolute imports for CLI execution
try:
    from .models import IndicatorSnapshot
except ImportError:
    from models import IndicatorSnapshot

logger = logging.getLogger(__name__)


def calculate_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Simple Moving Average.

    Args:
        df: DataFrame with 'close' column.
        period: SMA period.

    Returns:
        Series with SMA values.
    """
    return df['close'].rolling(window=period).mean()


def calculate_ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Exponential Moving Average.

    Args:
        df: DataFrame with 'close' column.
        period: EMA period.

    Returns:
        Series with EMA values.
    """
    return df['close'].ewm(span=period, adjust=False).mean()


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index.

    Args:
        df: DataFrame with 'close' column.
        period: RSI period.

    Returns:
        Series with RSI values.
    """
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(df: pd.DataFrame, fast: int = 12, 
                   slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """Calculate MACD indicator.

    Args:
        df: DataFrame with 'close' column.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal line period.

    Returns:
        Dictionary with MACD, signal, and histogram series.
    """
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
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
        df: DataFrame with 'close' column.
        period: Rolling window period.
        std_dev: Number of standard deviations.

    Returns:
        Dictionary with upper, middle, and lower band series.
    """
    middle = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
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
        df: DataFrame with 'high', 'low', 'close' columns.
        period: ATR period.

    Returns:
        Series with ATR values.
    """
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
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


def calculate_indicators(symbol: str, df: pd.DataFrame) -> IndicatorSnapshot:
    """Calculate technical indicators for a ticker DataFrame.

    Args:
        symbol: Ticker symbol.
        df: DataFrame with columns: open, high, low, close, volume.

    Returns:
        IndicatorSnapshot with latest indicator values.
    """
    # Work on a copy to avoid mutating input
    df = df.copy()
    
    # Calculate SMA20, SMA50, SMA200
    sma20_series = df['close'].rolling(window=20).mean()
    sma50_series = df['close'].rolling(window=50).mean()
    sma200_series = df['close'].rolling(window=200).mean()
    
    # Calculate Donchian Upper 20 (20-day rolling max of high)
    donchian_upper_20_series = df['high'].rolling(window=20).max()
    prev_donchian_upper_20_series = donchian_upper_20_series.shift(1)
    
    # Calculate Avg Volume 20
    avg_volume_20_series = df['volume'].rolling(window=20).mean()
    
    # Calculate Volume Ratio = latest volume / avg volume 20
    volume_ratio_series = df['volume'] / avg_volume_20_series
    
    # Calculate ATR14
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr14_series = true_range.rolling(window=14).mean()
    
    # Calculate daily log returns = ln(close / previous_close)
    log_returns = np.log(df['close'] / df['close'].shift(1))
    
    # Get latest row values
    latest_idx = len(df) - 1
    
    # Extract values, handling None cases
    sma20 = float(sma20_series.iloc[latest_idx]) if not pd.isna(sma20_series.iloc[latest_idx]) else None
    sma50 = float(sma50_series.iloc[latest_idx]) if not pd.isna(sma50_series.iloc[latest_idx]) else None
    sma200_val = sma200_series.iloc[latest_idx]
    sma200 = float(sma200_val) if not pd.isna(sma200_val) else None
    
    donchian_upper_20_val = donchian_upper_20_series.iloc[latest_idx]
    donchian_upper_20 = float(donchian_upper_20_val) if not pd.isna(donchian_upper_20_val) else None
    
    prev_donchian_upper_20_val = prev_donchian_upper_20_series.iloc[latest_idx]
    prev_donchian_upper_20 = float(prev_donchian_upper_20_val) if not pd.isna(prev_donchian_upper_20_val) else None
    
    avg_volume_20_val = avg_volume_20_series.iloc[latest_idx]
    avg_volume_20 = float(avg_volume_20_val) if not pd.isna(avg_volume_20_val) else None
    
    # Volume ratio: None if volume missing or zero
    latest_volume = df['volume'].iloc[latest_idx]
    if pd.isna(latest_volume) or latest_volume == 0:
        volume_ratio = None
    else:
        volume_ratio_val = volume_ratio_series.iloc[latest_idx]
        volume_ratio = float(volume_ratio_val) if not pd.isna(volume_ratio_val) else None
    
    # ATR14: None if cannot be computed
    atr14_val = atr14_series.iloc[latest_idx]
    atr14 = float(atr14_val) if not pd.isna(atr14_val) else None
    
    # Daily log return
    log_return_val = log_returns.iloc[latest_idx]
    daily_log_return = float(log_return_val) if not pd.isna(log_return_val) else None
    
    return IndicatorSnapshot(
        symbol=symbol,
        sma20=sma20,
        sma50=sma50,
        sma200=sma200,
        donchian_upper_20=donchian_upper_20,
        prev_donchian_upper_20=prev_donchian_upper_20,
        avg_volume_20=avg_volume_20,
        volume_ratio=volume_ratio,
        atr14=atr14,
        daily_log_return=daily_log_return
    )


def calculate_all_indicators(data: Dict[str, pd.DataFrame]) -> List[IndicatorSnapshot]:
    """Calculate indicators for all tickers in the data dictionary.

    Args:
        data: Dictionary mapping ticker symbols to DataFrames.

    Returns:
        List of IndicatorSnapshot objects.
    """
    results = []
    
    for symbol, df in data.items():
        try:
            # Validate required columns
            required_columns = {'open', 'high', 'low', 'close', 'volume'}
            if not required_columns.issubset(set(df.columns)):
                logger.warning(f"Skipping ticker {symbol}: missing required columns")
                continue
            
            # Skip if DataFrame is empty
            if df.empty:
                logger.warning(f"Skipping ticker {symbol}: empty DataFrame")
                continue
            
            snapshot = calculate_indicators(symbol, df)
            results.append(snapshot)
            
        except Exception as e:
            logger.warning(f"Skipping ticker {symbol}: error computing indicators - {e}")
            continue
    
    return results
