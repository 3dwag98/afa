"""Technical indicators as registered features.

CRITICAL: All features use .shift(1) or rolling windows ending at t-1
to ensure NO look-ahead bias. Features computed at time t should only
use data available up to time t-1.
"""

import pandas as pd
import numpy as np
from typing import Callable

from .registry import register_feature
from portfolio_agent.src.liquidity import circuit_locked_days, zero_return_days


@register_feature('close')
def close(df: pd.DataFrame) -> pd.Series:
    """Pass through the raw close price.

    Not lagged: the current day's close is the reference/entry price known at
    decision time, not a look-ahead concern.

    Args:
        df: DataFrame with 'close' column.

    Returns:
        The 'close' column unchanged.
    """
    return df['close']


@register_feature('volume_ratio_20')
def volume_ratio_20(df: pd.DataFrame) -> pd.Series:
    """Calculate volume strength as 5-day average volume over 20-day average.

    Uses volume shifted by 1 to avoid look-ahead bias.

    Args:
        df: DataFrame with 'volume' column.

    Returns:
        Series with volume ratio values (lagged by 1 period).
    """
    volume_shifted = df['volume'].shift(1)
    avg_5d = volume_shifted.rolling(window=5).mean()
    avg_20d = volume_shifted.rolling(window=20).mean()
    return avg_5d / avg_20d


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


@register_feature('mom_9m_skip1m')
def mom_9m_skip1m(df: pd.DataFrame) -> pd.Series:
    """Cross-sectional momentum formation return (Jegadeesh-Titman convention).

    MOM(t) = P(t - 1mo) / P(t - 1mo - 9mo) - 1, skipping the most recent
    month to avoid short-term reversal contamination. India-specific studies
    support 6-12 month formation windows; this platform defaults to 9 months
    (see docs/QUANT_RESEARCH.md section 1). Approximates months as 21 trading
    days. Already look-ahead safe: the most recent price used is 21 trading
    days in the past.

    Args:
        df: DataFrame with 'close' column.

    Returns:
        Series with the formation-period return (lagged by the 1-month skip).
    """
    skip_days = 21
    formation_days = 189  # ~9 months
    close = df['close']
    return close.shift(skip_days) / close.shift(skip_days + formation_days) - 1


@register_feature('realized_vol_60')
def realized_vol_60(df: pd.DataFrame) -> pd.Series:
    """Trailing 60-day annualized realized volatility of daily returns.

    Used by the low-volatility anomaly strategy (docs/QUANT_RESEARCH.md
    section 2): stocks are ranked ascending by this metric and the platform
    goes long the lowest-volatility decile. Uses close shifted by 1 to avoid
    look-ahead bias.

    Args:
        df: DataFrame with 'close' column.

    Returns:
        Series with annualized realized volatility (lagged by 1 period).
    """
    close_shifted = df['close'].shift(1)
    daily_returns = close_shifted.pct_change()
    return daily_returns.rolling(window=60).std() * np.sqrt(252)


@register_feature('traded_value_60')
def traded_value_60(df: pd.DataFrame) -> pd.Series:
    """Median daily turnover (close x volume) over the trailing 60 sessions.

    The liquidity screen behind the low-volatility strategy's "illiquidity
    illusion" guard: a stock whose variance is low because nothing trades is
    not a low-risk stock (docs/QUANT_RESEARCH.md section 15). Median rather
    than mean, so one operator print cannot make a dead ticker look liquid.
    Uses data shifted by 1 to avoid look-ahead bias.

    Args:
        df: DataFrame with 'close' and 'volume' columns.

    Returns:
        Series with median rupee turnover (lagged by 1 period).
    """
    traded_value = (df['close'] * df['volume']).shift(1)
    return traded_value.rolling(window=60, min_periods=20).median()


@register_feature('zero_return_fraction_60')
def zero_return_fraction_60(df: pd.DataFrame) -> pd.Series:
    """Share of the last 60 sessions that closed exactly unchanged.

    An illiquid ticker that did not trade carries yesterday's close forward,
    printing r = 0 rather than a small return — which mechanically suppresses
    realized variance and pushes the stock into the low-volatility buy decile
    for entirely the wrong reason. Uses data shifted by 1 to avoid look-ahead
    bias.

    Args:
        df: DataFrame with 'close' column.

    Returns:
        Series with the unchanged-close fraction in [0, 1] (lagged by 1 period).
    """
    flat = zero_return_days(df).astype(float).shift(1)
    return flat.rolling(window=60, min_periods=20).mean()


@register_feature('circuit_lock_fraction_60')
def circuit_lock_fraction_60(df: pd.DataFrame) -> pd.Series:
    """Share of the last 60 sessions that locked at a circuit limit.

    A stock pinned at its upper circuit prints a large return the momentum
    formula reads as strength, while offering no liquidity to buy; the same
    stock locked at the lower circuit cannot be exited at the modelled stop.
    Uses data shifted by 1 to avoid look-ahead bias.

    Args:
        df: DataFrame with 'high', 'low' and 'close' columns.

    Returns:
        Series with the circuit-locked fraction in [0, 1] (lagged by 1 period).
    """
    locked = circuit_locked_days(df).astype(float).shift(1)
    return locked.rolling(window=60, min_periods=20).mean()


@register_feature('circuit_locked_today')
def circuit_locked_today(df: pd.DataFrame) -> pd.Series:
    """1.0 when the most recent completed session locked at a circuit limit.

    Distinct from the 60-day fraction above: that measures whether a stock is
    structurally operator-driven, this answers "can this order actually be
    filled tomorrow?". Uses data shifted by 1 to avoid look-ahead bias.

    Args:
        df: DataFrame with 'high', 'low' and 'close' columns.

    Returns:
        Series of 0.0/1.0 flags (lagged by 1 period).
    """
    return circuit_locked_days(df).astype(float).shift(1)
