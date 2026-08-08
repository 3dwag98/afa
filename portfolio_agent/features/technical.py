"""Technical indicators as registered features.

CRITICAL: All features use .shift(1) or rolling windows ending at t-1
to ensure NO look-ahead bias. Features computed at time t should only
use data available up to time t-1.
"""

import pandas as pd
import numpy as np
from typing import Callable

from .registry import register_feature


@register_feature('sma_20')
def sma_20(df: pd.DataFrame) -> pd.Series:
    """Calculate 20-period Simple Moving Average.
    
    Uses close price shifted by 1 to avoid look-ahead bias.
    
    Args:
        df: DataFrame with 'close' column.
    
    Returns:
        Series with SMA_20 values (lagged by 1 period).
    """
    return df['close'].shift(1).rolling(window=20).mean()


@register_feature('sma_50')
def sma_50(df: pd.DataFrame) -> pd.Series:
    """Calculate 50-period Simple Moving Average.
    
    Uses close price shifted by 1 to avoid look-ahead bias.
    
    Args:
        df: DataFrame with 'close' column.
    
    Returns:
        Series with SMA_50 values (lagged by 1 period).
    """
    return df['close'].shift(1).rolling(window=50).mean()


@register_feature('sma_200')
def sma_200(df: pd.DataFrame) -> pd.Series:
    """Calculate 200-period Simple Moving Average.
    
    Uses close price shifted by 1 to avoid look-ahead bias.
    
    Args:
        df: DataFrame with 'close' column.
    
    Returns:
        Series with SMA_200 values (lagged by 1 period).
    """
    return df['close'].shift(1).rolling(window=200).mean()


@register_feature('donchian_upper_20')
def donchian_upper_20(df: pd.DataFrame) -> pd.Series:
    """Calculate 20-period Donchian Channel Upper Band.
    
    The upper band is the highest high over the past 20 periods,
    using data shifted by 1 to avoid look-ahead bias.
    
    Args:
        df: DataFrame with 'high' column.
    
    Returns:
        Series with Donchian upper band values (lagged by 1 period).
    """
    return df['high'].shift(1).rolling(window=20).max()


@register_feature('atr_14')
def atr_14(df: pd.DataFrame) -> pd.Series:
    """Calculate 14-period Average True Range.
    
    Uses shifted OHLC data to avoid look-ahead bias.
    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns.
    
    Returns:
        Series with ATR_14 values (lagged by 1 period).
    """
    # Shift all prices by 1 to ensure no look-ahead
    high = df['high'].shift(1)
    low = df['low'].shift(1)
    close = df['close'].shift(1)
    
    high_low = high - low
    high_close = np.abs(high - close.shift(1))
    low_close = np.abs(low - close.shift(1))
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=14).mean()


@register_feature('rsi_14')
def rsi_14(df: pd.DataFrame) -> pd.Series:
    """Calculate 14-period Relative Strength Index.
    
    Uses shifted close prices to avoid look-ahead bias.
    RSI = 100 - (100 / (1 + RS)) where RS = avg_gain / avg_loss
    
    Args:
        df: DataFrame with 'close' column.
    
    Returns:
        Series with RSI_14 values (lagged by 1 period).
    """
    # Use shifted close to avoid look-ahead
    close_shifted = df['close'].shift(1)
    delta = close_shifted.diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


@register_feature('macd')
def macd(df: pd.DataFrame) -> pd.Series:
    """Calculate MACD (Moving Average Convergence Divergence).
    
    Uses shifted close prices to avoid look-ahead bias.
    MACD = EMA_12 - EMA_26 (using lagged data)
    
    Args:
        df: DataFrame with 'close' column.
    
    Returns:
        Series with MACD values (lagged by 1 period).
    """
    # Use shifted close to avoid look-ahead
    close_shifted = df['close'].shift(1)
    
    ema_fast = close_shifted.ewm(span=12, adjust=False).mean()
    ema_slow = close_shifted.ewm(span=26, adjust=False).mean()
    
    return ema_fast - ema_slow


@register_feature('bollinger_pct_b')
def bollinger_pct_b(df: pd.DataFrame) -> pd.Series:
    """Calculate Bollinger Bands %B indicator.
    
    %B = (close - lower_band) / (upper_band - lower_band)
    Uses shifted data to avoid look-ahead bias.
    
    Args:
        df: DataFrame with 'close' column.
    
    Returns:
        Series with Bollinger %B values (lagged by 1 period).
    """
    # Use shifted close to avoid look-ahead
    close_shifted = df['close'].shift(1)
    
    middle = close_shifted.rolling(window=20).mean()
    std = close_shifted.rolling(window=20).std()
    
    upper = middle + (std * 2.0)
    lower = middle - (std * 2.0)
    
    # %B calculation
    pct_b = (close_shifted - lower) / (upper - lower)
    return pct_b


@register_feature('return_1d')
def return_1d(df: pd.DataFrame) -> pd.Series:
    """Calculate 1-day simple return.
    
    Uses only past data: (close_t-1 - close_t-2) / close_t-2
    This ensures the return at time t reflects what was known at t-1.
    
    Args:
        df: DataFrame with 'close' column.
    
    Returns:
        Series with 1-day returns (lagged by 1 period).
    """
    # Shift by 1 first, then calculate return
    # This gives us the return that was observable at time t-1
    close_shifted = df['close'].shift(1)
    return close_shifted.pct_change()


@register_feature('return_5d')
def return_5d(df: pd.DataFrame) -> pd.Series:
    """Calculate 5-day simple return.
    
    Uses only past data: (close_t-1 - close_t-6) / close_t-6
    This ensures the return at time t reflects what was known at t-1.
    
    Args:
        df: DataFrame with 'close' column.
    
    Returns:
        Series with 5-day returns (lagged by 1 period).
    """
    # Shift by 1 first, then calculate 5-period return
    close_shifted = df['close'].shift(1)
    return close_shifted.pct_change(periods=5)
