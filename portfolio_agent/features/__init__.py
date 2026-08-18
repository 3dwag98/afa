"""Features package initialization."""

from .registry import register_feature, get_feature, list_features, is_feature_registered
from .cross_section import (
    CrossSectionPanel,
    build_cross_section,
    get_cross_sectional_feature,
    is_cross_sectional_feature,
    latest_values,
    list_cross_sectional_features,
    panel_from_frames,
    register_cross_sectional_feature,
)
# Imported for its side effect as much as its names: registration happens at
# import time, and this module not being imported here is why the platform's
# only cross-sectional feature lived outside every registry and was reached by
# importing it directly inside a strategy method.
from . import characteristics  # noqa: F401
from . import cointegration  # noqa: F401
from . import market_relative  # noqa: F401
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

    # Cross-sectional registry
    'CrossSectionPanel',
    'register_cross_sectional_feature',
    'get_cross_sectional_feature',
    'list_cross_sectional_features',
    'is_cross_sectional_feature',
    'build_cross_section',
    'panel_from_frames',
    'latest_values',

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
