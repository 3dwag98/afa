"""Stock scoring module."""

import pandas as pd
from typing import Dict, Any


def calculate_technical_score(df: pd.DataFrame) -> float:
    """Calculate technical analysis score.

    Args:
        df: DataFrame with indicator columns.

    Returns:
        Score between 0 and 1.
    """
    if df.empty or len(df) < 50:
        return 0.5

    latest = df.iloc[-1]
    score = 0.0
    max_score = 5.0

    # RSI scoring (30-70 range is neutral)
    rsi = latest.get('RSI', 50)
    if 40 <= rsi <= 60:
        score += 1.0
    elif rsi < 30:  # Oversold - potential buy
        score += 2.0
    elif rsi > 70:  # Overbought - potential sell
        score += 0.0
    else:
        score += 0.5

    # MACD scoring
    macd_hist = latest.get('MACD_Hist', 0)
    if macd_hist > 0:
        score += 1.5
    elif macd_hist < 0:
        score += 0.5
    else:
        score += 1.0

    # Price vs SMA scoring
    close = latest.get('Close', 0)
    sma_20 = latest.get('SMA_20', close)
    sma_50 = latest.get('SMA_50', close)

    if close > sma_20 > sma_50:  # Uptrend
        score += 1.5
    elif close < sma_20 < sma_50:  # Downtrend
        score += 0.0
    else:
        score += 0.75

    # Bollinger Bands position
    bb_lower = latest.get('BB_Lower', close * 0.95)
    bb_upper = latest.get('BB_Upper', close * 1.05)

    if close <= bb_lower:  # Near lower band - potential bounce
        score += 1.0
    elif close >= bb_upper:  # Near upper band - potential pullback
        score += 0.5
    else:
        score += 0.75

    return min(score / max_score, 1.0)


def calculate_momentum_score(df: pd.DataFrame, lookback: int = 20) -> float:
    """Calculate momentum score based on price change.

    Args:
        df: DataFrame with 'Close' column.
        lookback: Number of days for momentum calculation.

    Returns:
        Score between 0 and 1.
    """
    if len(df) < lookback:
        return 0.5

    returns = df['Close'].pct_change().iloc[-lookback:]
    cumulative_return = (1 + returns).prod() - 1

    # Normalize to 0-1 scale
    # Assume +/- 20% over lookback period as extremes
    normalized = (cumulative_return + 0.2) / 0.4
    return max(0.0, min(1.0, normalized))


def calculate_volume_score(df: pd.DataFrame, period: int = 20) -> float:
    """Calculate volume strength score.

    Args:
        df: DataFrame with 'Volume' column.
        period: Period for average volume calculation.

    Returns:
        Score between 0 and 1.
    """
    if len(df) < period:
        return 0.5

    avg_volume = df['Volume'].rolling(window=period).mean().iloc[-1]
    recent_volume = df['Volume'].iloc[-5:].mean()

    if avg_volume == 0:
        return 0.5

    volume_ratio = recent_volume / avg_volume

    # Higher volume indicates stronger conviction
    if volume_ratio > 1.5:
        return 1.0
    elif volume_ratio > 1.0:
        return 0.75
    elif volume_ratio > 0.8:
        return 0.5
    else:
        return 0.25


def calculate_combined_score(df: pd.DataFrame, 
                             weights: Dict[str, float] = None) -> float:
    """Calculate combined score from multiple factors.

    Args:
        df: DataFrame with price and indicator data.
        weights: Optional weights for each score component.

    Returns:
        Combined score between 0 and 1.
    """
    if weights is None:
        weights = {
            'technical': 0.4,
            'momentum': 0.35,
            'volume': 0.25
        }

    tech_score = calculate_technical_score(df)
    mom_score = calculate_momentum_score(df)
    vol_score = calculate_volume_score(df)

    combined = (
        weights['technical'] * tech_score +
        weights['momentum'] * mom_score +
        weights['volume'] * vol_score
    )

    return max(0.0, min(1.0, combined))
