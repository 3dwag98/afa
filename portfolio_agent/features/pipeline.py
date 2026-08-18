"""Feature pipeline for building feature matrices from OHLCV data."""

from functools import lru_cache

import pandas as pd
import numpy as np
from typing import Optional, Sequence

from .registry import get_feature, list_features

#: Rows a feature is allowed to need before `warmup_rows` gives up on it. Every
#: registered feature warms up far inside this; the cap only bounds the probe.
_WARMUP_PROBE_ROWS = 600


def warmup_rows(feature_names: Sequence[str]) -> int:
    """Rows of history before every named feature is defined.

    **Measured, not declared.** Four modules carried a minimum-history
    threshold and all four disagreed — 20 in the backtest, 252 in the
    evaluation harness, 252 again in the trainer panel builder, and
    `data.min_history_days` in the supervised path. They were guessing at one
    number: how much history the *longest lookback among the requested
    features* needs. That is a property of the request, so it is computed from
    the request.

    Each feature is built once on a synthetic probe series and the first row
    where it is defined is its warm-up; the answer for a set is the largest. A
    feature added tomorrow with a three-year lookback raises this
    automatically, where a constant would silently start scoring NaNs.

    The cost this replaces is not theoretical. At the backtest's 20-row
    threshold, six of the ten features `momentum` requires are NaN — including
    `mom_9m_skip1m`, the value it ranks on — so the opening months of every
    backtest ranked the universe on undefined numbers while the harness refused
    those same dates.

    Raises:
        KeyError: If a name is not registered — surfaced here rather than as a
            failure part-way through a run.
    """
    return max((_warmup_for(name) for name in feature_names), default=0)


@lru_cache(maxsize=None)
def _warmup_for(name: str) -> int:
    """First row at which one feature is defined, on a synthetic probe series.

    Cached because the probe costs a feature evaluation and the answer is a
    property of the code, not of the data. A geometric random walk rather than
    a straight line: a constant series makes several features degenerate (zero
    true range, zero variance) and would report them as never warming up.
    """
    rng = np.random.default_rng(0)
    n = _WARMUP_PROBE_ROWS
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    probe = pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, n)),
            "high": close * (1 + np.abs(rng.normal(0, 0.006, n))),
            "low": close * (1 - np.abs(rng.normal(0, 0.006, n))),
            "close": close,
            "volume": rng.integers(1e5, 1e6, n).astype(float),
        },
        index=pd.bdate_range("2020-01-01", periods=n),
    )

    series = pd.Series(get_feature(name)(probe)).reset_index(drop=True)
    defined = series.notna()
    if not defined.any():
        # A feature that never resolves on 600 clean rows will not resolve on a
        # ticker's cache either; nothing is gained by pretending otherwise.
        return _WARMUP_PROBE_ROWS
    # +1 because a value at position p means p+1 rows were needed to produce it.
    return int(defined.idxmax()) + 1


def build_features(
    df: pd.DataFrame, 
    feature_names: list[str],
    normalize: bool = False,
    normalize_window: int = 252
) -> pd.DataFrame:
    """Build a feature matrix from OHLCV DataFrame.
    
    Args:
        df: DataFrame with OHLCV columns (open, high, low, close, volume).
            Should be indexed by date or have a datetime index.
        feature_names: List of feature names to compute. Features are fetched
            from the registry and applied to the DataFrame.
        normalize: If True, apply rolling z-score normalization using only
            past data (no look-ahead bias).
        normalize_window: Window size for rolling z-score normalization.
            Default is 252 (approximately 1 trading year).
    
    Returns:
        DataFrame with computed features as columns. Each feature column
        is named according to the feature name. The returned DataFrame
        has the same index as the input DataFrame.
    
    Raises:
        KeyError: If any feature name is not found in the registry.
        ValueError: If required columns are missing from the input DataFrame.
    
    Example:
        >>> from portfolio_agent.features.technical import sma_20, rsi_14
        >>> from portfolio_agent.features.pipeline import build_features
        >>> df = pd.DataFrame({...})  # OHLCV data
        >>> features = build_features(df, ['sma_20', 'rsi_14'])
    """
    # Validate required columns
    required_columns = {'open', 'high', 'low', 'close', 'volume'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"DataFrame must have OHLCV columns."
        )
    
    # Build feature dictionary
    feature_dict = {}
    
    for feature_name in feature_names:
        # Get feature function from registry
        feature_func = get_feature(feature_name)
        
        # Apply feature function to DataFrame
        try:
            feature_series = feature_func(df)
            feature_dict[feature_name] = feature_series
        except Exception as e:
            raise ValueError(
                f"Error computing feature '{feature_name}': {e}"
            )
    
    # Create feature DataFrame
    feature_df = pd.DataFrame(feature_dict, index=df.index)
    
    # Apply normalization if requested
    if normalize:
        feature_df = _normalize_features(feature_df, normalize_window)
    
    return feature_df


def _normalize_features(
    df: pd.DataFrame, 
    window: int = 252
) -> pd.DataFrame:
    """Apply rolling z-score normalization using only past data.
    
    For each feature, computes:
        normalized = (value - rolling_mean) / rolling_std
    
    The rolling window uses only past data (window ends at t-1) to avoid
    look-ahead bias.
    
    Args:
        df: DataFrame with feature columns to normalize.
        window: Rolling window size for computing mean and std.
    
    Returns:
        DataFrame with normalized feature values.
    """
    normalized_df = pd.DataFrame(index=df.index)
    
    for col in df.columns:
        # Shift by 1 to ensure we only use past data for normalization
        # This prevents look-ahead bias
        series = df[col].shift(1)
        
        rolling_mean = series.rolling(window=window, min_periods=1).mean()
        rolling_std = series.rolling(window=window, min_periods=1).std()
        
        # Z-score normalization
        normalized = (series - rolling_mean) / (rolling_std + 1e-8)  # epsilon to avoid division by zero
        
        normalized_df[col] = normalized
    
    return normalized_df


def get_available_features() -> list[str]:
    """Get list of all available feature names.
    
    Returns:
        List of registered feature names.
    """
    return list_features()


def validate_feature_names(feature_names: list[str]) -> tuple[list[str], list[str]]:
    """Validate feature names against the registry.
    
    Args:
        feature_names: List of feature names to validate.
    
    Returns:
        Tuple of (valid_names, invalid_names).
    """
    valid_names = []
    invalid_names = []
    
    for name in feature_names:
        try:
            get_feature(name)
            valid_names.append(name)
        except KeyError:
            invalid_names.append(name)
    
    return valid_names, invalid_names
