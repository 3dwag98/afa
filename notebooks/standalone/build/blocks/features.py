# --- Features: the indicator set the strategies below are defined on ----------

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI. Uses an EWM with alpha=1/window, which is Wilder smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram — the part that carries the signal, not the raw line."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    return line - line.ewm(span=signal, adjust=False).mean()


def bollinger_pct_b(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """Position within the Bollinger band: 0 at the lower band, 1 at the upper."""
    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    width = (2 * n_std * std).replace(0.0, np.nan)
    return (close - (mid - n_std * std)) / width


def true_range(frame: pd.DataFrame) -> pd.Series:
    """True range, which unlike high-low accounts for the overnight gap.

    On Indian equities the gap is where a large share of the move happens —
    a circuit-limited open is entirely gap — so high-low alone understates
    realized range and every ATR-derived stop built on it is too tight.
    """
    prev_close = frame["close"].shift(1)
    return pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    return true_range(frame).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def realized_vol(close: pd.Series, window: int = 60) -> pd.Series:
    """Annualized realized volatility of daily log returns."""
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(window, min_periods=window).std() * np.sqrt(TRADING_DAYS)


def momentum_skip(close: pd.Series, lookback: int = 252, skip: int = 21) -> pd.Series:
    """Total return over `lookback` sessions, excluding the most recent `skip`.

    The skip is not decoration. One-month returns reverse on average, so a
    momentum signal that includes the latest month is partly buying the very
    thing that is about to mean-revert; Jegadeesh-Titman skip the month for
    exactly this reason.
    """
    return close.shift(skip) / close.shift(lookback) - 1.0


def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Today's volume against its trailing average — the confirmation term."""
    average = volume.rolling(window, min_periods=window).mean()
    return volume / average.replace(0.0, np.nan)


def traded_value(frame: pd.DataFrame, window: int = 60) -> pd.Series:
    """Average daily traded value: the liquidity measure that matters for sizing."""
    return (frame["close"] * frame["volume"]).rolling(window, min_periods=window).mean()


def rolling_high(close: pd.Series, window: int = 20) -> pd.Series:
    """Trailing high *excluding today*, so a breakout test is not self-referential."""
    return close.shift(1).rolling(window, min_periods=window).max()


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Every feature the strategies in these notebooks read, for one symbol.

    Returns a frame indexed like the input. Rows are left NaN until each
    window has filled — they are dropped where a strategy needs them, rather
    than back-filled, because a back-filled indicator is a look-ahead.
    """
    close, volume = frame["close"], frame["volume"]
    features = pd.DataFrame(index=frame.index)

    features["close"] = close
    features["return_1d"] = close.pct_change()
    features["return_5d"] = close.pct_change(5)
    features["return_21d"] = close.pct_change(21)

    features["sma_20"] = sma(close, 20)
    features["sma_50"] = sma(close, 50)
    features["sma_200"] = sma(close, 200)

    features["rsi_14"] = rsi(close, 14)
    features["macd"] = macd(close)
    features["bollinger_pct_b"] = bollinger_pct_b(close)
    features["atr_14"] = atr(frame, 14)
    features["atr_pct"] = features["atr_14"] / close

    features["realized_vol_60"] = realized_vol(close, 60)
    features["mom_9m_skip1m"] = momentum_skip(close, 189, 21)
    features["mom_12m_skip1m"] = momentum_skip(close, 252, 21)

    features["volume_ratio_20"] = volume_ratio(volume, 20)
    features["traded_value_60"] = traded_value(frame, 60)
    features["high_20"] = rolling_high(close, 20)
    features["breakout_20"] = close / features["high_20"] - 1.0

    return features


def build_feature_panel(panel: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Features for every symbol in a panel."""
    return {symbol: build_features(frame) for symbol, frame in panel.items()}


def cross_section(
    feature_panel: Dict[str, pd.DataFrame], column: str
) -> pd.DataFrame:
    """One feature as a wide dates x symbols matrix, for cross-sectional ranking."""
    return pd.DataFrame(
        {symbol: features[column] for symbol, features in feature_panel.items()}
    ).sort_index()


def zscore_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardize each date across symbols.

    Cross-sectional rather than time-series: the question a model choosing
    between stocks is asked is "is this high relative to what else I could buy
    today", not "is this high for this stock historically". It also strips the
    market factor out by construction, and cannot leak across dates since it
    fits no state.
    """
    mean = frame.mean(axis=1)
    std = frame.std(axis=1).replace(0.0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0)


def rank_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-date cross-sectional rank scaled to [-1, 1].

    More robust than a z-score on Indian data, where a circuit-limited print
    dominates the cross-sectional mean but moves a ranking by one place.
    """
    ranked = frame.rank(axis=1, pct=True)
    return 2.0 * ranked - 1.0
