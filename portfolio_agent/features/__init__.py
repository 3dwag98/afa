"""Features package initialization."""

from .registry import register_feature, get_feature, list_features, is_feature_registered
from .technical import (
    sma_20,
    sma_50,
    sma_200,
    donchian_upper_20,
    atr_14,
    rsi_14,
    macd,
    bollinger_pct_b,
    return_1d,
    return_5d,
)
from .pipeline import build_features, get_available_features, validate_feature_names

__all__ = [
    # Registry functions
    'register_feature',
    'get_feature',
    'list_features',
    'is_feature_registered',
    
    # Technical indicators
    'sma_20',
    'sma_50',
    'sma_200',
    'donchian_upper_20',
    'atr_14',
    'rsi_14',
    'macd',
    'bollinger_pct_b',
    'return_1d',
    'return_5d',
    
    # Pipeline functions
    'build_features',
    'get_available_features',
    'validate_feature_names',
]
