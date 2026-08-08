"""Tests for the features module.

Ensures no look-ahead bias in feature calculations.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from portfolio_agent.features.registry import (
    register_feature, 
    get_feature, 
    list_features,
    is_feature_registered,
    _FEATURE_REGISTRY
)
from portfolio_agent.features.technical import (
    sma_20, sma_50, sma_200,
    donchian_upper_20,
    atr_14,
    rsi_14,
    macd,
    bollinger_pct_b,
    return_1d,
    return_5d,
)
from portfolio_agent.features.pipeline import (
    build_features,
    get_available_features,
    validate_feature_names,
    _normalize_features,
)


def generate_synthetic_ohlcv(rows: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data.
    
    Args:
        rows: Number of rows to generate.
        seed: Random seed for reproducibility.
    
    Returns:
        DataFrame with open, high, low, close, volume columns.
    """
    np.random.seed(seed)
    
    # Generate dates
    end_date = datetime.now()
    dates = [end_date - timedelta(days=i) for i in range(rows)]
    dates.reverse()
    
    # Generate prices using random walk
    close = 100 + np.cumsum(np.random.randn(rows) * 2)
    close = np.abs(close) + 1  # Ensure positive
    
    # Generate OHLC from close
    daily_range = np.abs(np.random.randn(rows)) * 2
    high = close + daily_range
    low = close - daily_range
    open_price = low + np.random.rand(rows) * (high - low)
    
    # Generate volume
    volume = np.random.randint(1000000, 10000000, size=rows)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=pd.DatetimeIndex(dates))
    
    return df


class TestRegistry:
    """Tests for the feature registry."""
    
    def test_register_feature_decorator(self):
        """Test that @register_feature decorator works."""
        # The technical indicators should already be registered
        assert is_feature_registered('sma_20')
        assert is_feature_registered('rsi_14')
        assert is_feature_registered('macd')
    
    def test_get_feature_returns_callable(self):
        """Test that get_feature returns a callable function."""
        func = get_feature('sma_20')
        assert callable(func)
    
    def test_get_feature_raises_keyerror_for_unknown(self):
        """Test that get_feature raises KeyError for unknown feature."""
        with pytest.raises(KeyError):
            get_feature('nonexistent_feature')
    
    def test_list_features_returns_all_registered(self):
        """Test that list_features returns all registered features."""
        features = list_features()
        assert 'sma_20' in features
        assert 'sma_50' in features
        assert 'sma_200' in features
        assert 'donchian_upper_20' in features
        assert 'atr_14' in features
        assert 'rsi_14' in features
        assert 'macd' in features
        assert 'bollinger_pct_b' in features
        assert 'return_1d' in features
        assert 'return_5d' in features
    
    def test_custom_feature_registration(self):
        """Test registering a custom feature."""
        # Clear any existing test feature
        if 'test_custom' in _FEATURE_REGISTRY:
            del _FEATURE_REGISTRY['test_custom']
        
        @register_feature('test_custom')
        def custom_feature(df: pd.DataFrame) -> pd.Series:
            return df['close'].shift(1)
        
        assert is_feature_registered('test_custom')
        # Check that we can retrieve and call the function (not exact equality due to wrapper)
        retrieved = get_feature('test_custom')
        assert callable(retrieved)
        
        # Clean up
        del _FEATURE_REGISTRY['test_custom']


class TestNoLookAheadBias:
    """Tests to ensure no look-ahead bias in feature calculations."""
    
    def test_sma_20_no_lookahead(self):
        """Test SMA_20 uses only past data (no look-ahead)."""
        df = generate_synthetic_ohlcv(300)
        
        result = sma_20(df)
        
        # At time t, SMA should use data up to t-1
        # So result.iloc[i] should equal mean of close[0:i] shifted by 1
        # Verify by checking that result at index i doesn't use close[i]
        
        # Manual calculation with proper lag
        expected = df['close'].shift(1).rolling(20).mean()
        
        pd.testing.assert_series_equal(result, expected)
        
        # Verify that result at time t doesn't depend on close at time t
        # by checking that modifying close[t] doesn't affect result[t]
        df_modified = df.copy()
        df_modified.loc[df.index[-1], 'close'] = df_modified['close'].iloc[-1] * 2
        
        result_original = sma_20(df)
        result_modified = sma_20(df_modified)
        
        # The last value should be the same since we only changed current close
        assert result_original.iloc[-1] == result_modified.iloc[-1]
    
    def test_rsi_14_no_lookahead(self):
        """Test RSI_14 uses only past data."""
        df = generate_synthetic_ohlcv(300)
        
        result = rsi_14(df)
        
        # Manual calculation with proper lag
        close_shifted = df['close'].shift(1)
        delta = close_shifted.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        expected = 100 - (100 / (1 + rs))
        
        pd.testing.assert_series_equal(result, expected)
    
    def test_macd_no_lookahead(self):
        """Test MACD uses only past data."""
        df = generate_synthetic_ohlcv(300)
        
        result = macd(df)
        
        # Manual calculation with proper lag
        close_shifted = df['close'].shift(1)
        ema_fast = close_shifted.ewm(span=12, adjust=False).mean()
        ema_slow = close_shifted.ewm(span=26, adjust=False).mean()
        expected = ema_fast - ema_slow
        
        pd.testing.assert_series_equal(result, expected)
    
    def test_atr_14_no_lookahead(self):
        """Test ATR_14 uses only past data."""
        df = generate_synthetic_ohlcv(300)
        
        result = atr_14(df)
        
        # Verify result is not NaN after warmup period
        assert not pd.isna(result.iloc[-1])
        
        # Check that result at time t doesn't change when we modify OHLC at time t
        df_modified = df.copy()
        df_modified.loc[df.index[-1], 'high'] = df_modified['high'].iloc[-1] * 2
        df_modified.loc[df.index[-1], 'low'] = df_modified['low'].iloc[-1] * 0.5
        df_modified.loc[df.index[-1], 'close'] = df_modified['close'].iloc[-1] * 1.5
        
        result_original = atr_14(df)
        result_modified = atr_14(df_modified)
        
        # Should be identical since we only changed current values
        assert result_original.iloc[-1] == result_modified.iloc[-1]
    
    def test_return_1d_no_lookahead(self):
        """Test return_1d uses only past data."""
        df = generate_synthetic_ohlcv(300)
        
        result = return_1d(df)
        
        # Expected: pct_change of shifted close
        expected = df['close'].shift(1).pct_change()
        
        pd.testing.assert_series_equal(result, expected)
    
    def test_donchian_upper_20_no_lookahead(self):
        """Test Donchian upper uses only past high values."""
        df = generate_synthetic_ohlcv(300)
        
        result = donchian_upper_20(df)
        
        # Expected: rolling max of shifted high
        expected = df['high'].shift(1).rolling(20).max()
        
        pd.testing.assert_series_equal(result, expected)
        
        # Modifying current high shouldn't affect current Donchian value
        df_modified = df.copy()
        df_modified.loc[df.index[-1], 'high'] = df_modified['high'].iloc[-1] * 10
        
        result_original = donchian_upper_20(df)
        result_modified = donchian_upper_20(df_modified)
        
        assert result_original.iloc[-1] == result_modified.iloc[-1]


class TestFeatureValues:
    """Tests for correct feature value calculations."""
    
    def test_sma_values_after_warmup(self):
        """Test that SMA values are valid after warmup period."""
        df = generate_synthetic_ohlcv(300)
        
        sma20_result = sma_20(df)
        sma50_result = sma_50(df)
        sma200_result = sma_200(df)
        
        # After 300 rows, all SMAs should have valid values
        assert not pd.isna(sma20_result.iloc[-1])
        assert not pd.isna(sma50_result.iloc[-1])
        assert not pd.isna(sma200_result.iloc[-1])
    
    def test_rsi_in_valid_range(self):
        """Test that RSI values are in valid range [0, 100]."""
        df = generate_synthetic_ohlcv(300)
        
        result = rsi_14(df)
        
        # Get non-NaN values
        valid_values = result.dropna()
        
        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()
    
    def test_bollinger_pct_b_reasonable_range(self):
        """Test that Bollinger %B is in reasonable range."""
        df = generate_synthetic_ohlcv(300)
        
        result = bollinger_pct_b(df)
        
        # %B can go outside [0, 1] but should be finite
        valid_values = result.dropna()
        assert np.isfinite(valid_values).all()
    
    def test_atr_positive(self):
        """Test that ATR is always positive."""
        df = generate_synthetic_ohlcv(300)
        
        result = atr_14(df)
        
        valid_values = result.dropna()
        assert (valid_values >= 0).all()


class TestPipeline:
    """Tests for the feature pipeline."""
    
    def test_build_features_single_feature(self):
        """Test building features with a single feature."""
        df = generate_synthetic_ohlcv(300)
        
        result = build_features(df, ['sma_20'])
        
        assert isinstance(result, pd.DataFrame)
        assert 'sma_20' in result.columns
        assert len(result) == len(df)
    
    def test_build_features_multiple_features(self):
        """Test building features with multiple features."""
        df = generate_synthetic_ohlcv(300)
        
        feature_names = ['sma_20', 'rsi_14', 'macd', 'atr_14']
        result = build_features(df, feature_names)
        
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == feature_names
        assert len(result) == len(df)
    
    def test_build_features_with_normalization(self):
        """Test building features with normalization."""
        df = generate_synthetic_ohlcv(300)
        
        result = build_features(df, ['sma_20', 'rsi_14'], normalize=True)
        
        assert isinstance(result, pd.DataFrame)
        assert 'sma_20' in result.columns
        assert 'rsi_14' in result.columns
        
        # Normalized values should have mean ~0 and std ~1 (approximately)
        # Note: due to shifting, first values will be NaN
        valid_sma = result['sma_20'].dropna()
        if len(valid_sma) > 100:  # Need enough data
            assert abs(valid_sma.mean()) < 2  # Roughly centered
            assert valid_sma.std() > 0.5  # Has some variance
    
    def test_build_features_preserves_index(self):
        """Test that build_features preserves the DataFrame index."""
        df = generate_synthetic_ohlcv(300)
        
        result = build_features(df, ['sma_20'])
        
        pd.testing.assert_index_equal(result.index, df.index)
    
    def test_build_features_raises_on_missing_columns(self):
        """Test that build_features raises error for missing columns."""
        df = pd.DataFrame({'close': [1, 2, 3]})  # Missing OHLCV columns
        
        with pytest.raises(ValueError, match="Missing required columns"):
            build_features(df, ['sma_20'])
    
    def test_build_features_raises_on_unknown_feature(self):
        """Test that build_features raises error for unknown feature."""
        df = generate_synthetic_ohlcv(300)
        
        with pytest.raises(KeyError):
            build_features(df, ['nonexistent_feature'])
    
    def test_get_available_features(self):
        """Test getting available features."""
        features = get_available_features()
        
        assert isinstance(features, list)
        assert 'sma_20' in features
        assert 'rsi_14' in features
    
    def test_validate_feature_names(self):
        """Test validating feature names."""
        valid, invalid = validate_feature_names(['sma_20', 'rsi_14', 'nonexistent'])
        
        assert 'sma_20' in valid
        assert 'rsi_14' in valid
        assert 'nonexistent' in invalid


class TestNormalizeFeatures:
    """Tests for feature normalization."""
    
    def test_normalize_no_lookahead(self):
        """Test that normalization doesn't introduce look-ahead bias."""
        df = pd.DataFrame({
            'feature1': list(range(100)),
            'feature2': list(range(100, 200))
        })
        
        normalized = _normalize_features(df, window=20)
        
        # Verify that normalized value at time t doesn't depend on value at time t
        # by checking that modifying value at t doesn't affect normalized value at t
        df_modified = df.copy()
        df_modified.loc[df.index[-1], 'feature1'] = 999999
        
        normalized_original = _normalize_features(df, window=20)
        normalized_modified = _normalize_features(df_modified, window=20)
        
        # The last normalized value should be different because we're normalizing
        # the shifted series, but the shift happens BEFORE normalization
        # Actually, since we shift before normalizing, changing the last value
        # shouldn't affect the normalized last value
        assert normalized_original['feature1'].iloc[-1] == normalized_modified['feature1'].iloc[-1]
    
    def test_normalize_handles_zero_std(self):
        """Test that normalization handles zero standard deviation."""
        df = pd.DataFrame({
            'constant': [1.0] * 100  # Constant value = zero std
        })
        
        normalized = _normalize_features(df, window=20)
        
        # Should not raise division by zero error
        assert normalized is not None
        # Values should be finite (not inf) - NaN is acceptable for early periods
        # where rolling std is 0 or NaN
        non_nan_values = normalized['constant'].dropna()
        if len(non_nan_values) > 0:
            assert np.isfinite(non_nan_values).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
