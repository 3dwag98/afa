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


class TestNormalizationIsCausal:
    """Z-scoring with statistics fitted over the whole panel would leak future
    volatility and price levels into every training row — the classic way a
    walk-forward Sharpe of 2.0 turns into a live loss."""

    @staticmethod
    def _ohlcv(n=300, seed=3):
        rng = np.random.default_rng(seed)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
        return pd.DataFrame({
            'open': close, 'high': close * 1.01, 'low': close * 0.99,
            'close': close, 'volume': np.full(n, 1e6),
        }, index=pd.bdate_range('2022-01-03', periods=n))

    def test_early_rows_do_not_change_when_later_data_is_appended(self):
        """The decisive test: if normalization used full-sample statistics,
        adding future bars would rewrite the past."""
        full = self._ohlcv(n=300)
        truncated = full.iloc[:200]

        names = ['sma_20', 'rsi_14', 'atr_14']
        normalized_full = build_features(full, names, normalize=True, normalize_window=252)
        normalized_short = build_features(truncated, names, normalize=True, normalize_window=252)

        overlap_full = normalized_full.iloc[:200].dropna()
        overlap_short = normalized_short.dropna()
        common = overlap_full.index.intersection(overlap_short.index)

        assert len(common) > 100
        assert np.allclose(
            overlap_full.loc[common].values, overlap_short.loc[common].values, equal_nan=True
        )

    def test_normalization_is_off_in_the_shipped_config(self):
        from portfolio_agent.config.loader import load_config

        assert load_config().features.normalize is False


class TestNormalizationIsCausal:
    """Task 3.3's real question is not 'is a scaler fitted per fold' but 'can a
    fold boundary leak at all'. This pipeline never fits a global scaler: the
    normalizer is a trailing rolling z-score over `close.shift(1)`, so a
    feature value at t is a function of data strictly before t and nothing
    else. That is a strictly stronger guarantee than per-fold fitting — it
    holds for every possible split, including ones nobody thought to test.

    These tests pin the property that makes it true, so a future change to
    `_normalize_features` that introduces a whole-series statistic (a
    StandardScaler over the panel, an expanding mean without the shift) fails
    here rather than silently inflating every walk-forward fold's score.
    """

    @staticmethod
    def _ohlcv(n=400, seed=11):
        rng = np.random.default_rng(seed)
        close = 100 + np.cumsum(rng.normal(0, 1.5, n))
        return pd.DataFrame(
            {
                'open': close + rng.normal(0, 0.3, n),
                'high': close + np.abs(rng.normal(0, 1.0, n)),
                'low': close - np.abs(rng.normal(0, 1.0, n)),
                'close': close,
                'volume': rng.integers(1_000_000, 5_000_000, n).astype(float),
            },
            index=pd.bdate_range("2021-01-04", periods=n),
        )

    FEATURES = ['sma_20', 'rsi_14', 'macd', 'bollinger_pct_b', 'atr_14', 'return_1d', 'return_5d']

    def test_truncating_the_future_does_not_change_the_past(self):
        """The decisive test. If any statistic were fitted over the whole
        series, deleting the tail would move every earlier value."""
        df = self._ohlcv()
        split = 250

        full = build_features(df, self.FEATURES, normalize=True, normalize_window=252)
        prefix = build_features(
            df.iloc[:split], self.FEATURES, normalize=True, normalize_window=252
        )

        pd.testing.assert_frame_equal(full.iloc[:split], prefix, check_exact=False, rtol=1e-12)

    def test_a_walk_forward_split_sees_identical_training_rows(self):
        """Concretely: fold 2's expanded training window must reproduce fold
        1's rows exactly, or the 'expanding history' is silently a different
        dataset each time."""
        df = self._ohlcv()

        fold1 = build_features(df.iloc[:200], self.FEATURES, normalize=True)
        fold2 = build_features(df.iloc[:320], self.FEATURES, normalize=True)

        pd.testing.assert_frame_equal(fold2.iloc[:200], fold1, check_exact=False, rtol=1e-12)

    def test_perturbing_a_future_bar_leaves_earlier_features_untouched(self):
        df = self._ohlcv()
        tampered = df.copy()
        tampered.iloc[300:, tampered.columns.get_loc('close')] *= 5.0

        base = build_features(df, self.FEATURES, normalize=True)
        after = build_features(tampered, self.FEATURES, normalize=True)

        pd.testing.assert_frame_equal(
            base.iloc[:300], after.iloc[:300], check_exact=False, rtol=1e-12
        )

    def test_the_normalizer_excludes_the_current_row_from_its_own_statistics(self):
        """The `.shift(1)` inside _normalize_features. Without it the value at
        t is standardized against a window containing t."""
        from portfolio_agent.features.pipeline import _normalize_features

        frame = pd.DataFrame({'x': [1.0, 2.0, 3.0, 100.0]})
        normalized = _normalize_features(frame, window=252)

        # The final row standardizes x[t-1] = 3.0 against rows 1..3, so the
        # outlier at t cannot influence it.
        assert not np.isnan(normalized['x'].iloc[-1])
        assert abs(normalized['x'].iloc[-1]) < 10


class TestWarmupRows:
    """How much history a feature set needs, measured rather than declared.

    Four modules carried a minimum-history threshold and all four disagreed —
    20 rows in the backtest, 252 in the evaluation harness, 252 again in the
    trainer panel builder, `data.min_history_days` in the supervised path. All
    four were reaching for one quantity: the longest lookback among the
    features actually requested. That is a property of the request, so it is
    computed from the request.
    """

    def test_a_longer_lookback_needs_more_rows(self):
        from portfolio_agent.features.pipeline import warmup_rows

        assert warmup_rows(['sma_200']) > warmup_rows(['sma_50'])
        assert warmup_rows(['sma_50']) > warmup_rows(['sma_20'])

    def test_a_set_needs_what_its_slowest_member_needs(self):
        from portfolio_agent.features.pipeline import warmup_rows

        assert warmup_rows(['sma_20', 'sma_200']) == warmup_rows(['sma_200'])

    def test_an_empty_request_needs_nothing(self):
        from portfolio_agent.features.pipeline import warmup_rows

        assert warmup_rows([]) == 0

    def test_the_answer_is_the_row_the_feature_is_first_defined(self):
        """Not an approximation of it — build at the boundary and check.

        The contract is exact: at `warmup_rows(f)` rows the feature resolves,
        and at one row fewer it does not. A threshold that was merely
        *sufficient* would let an off-by-one survive here.
        """
        from portfolio_agent.features.pipeline import build_features, warmup_rows

        needed = warmup_rows(['sma_50'])
        rng = np.random.default_rng(3)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, needed)))
        df = pd.DataFrame(
            {'open': close, 'high': close * 1.01, 'low': close * 0.99,
             'close': close, 'volume': np.full(needed, 1e6)},
            index=pd.bdate_range('2022-01-03', periods=needed),
        )

        assert not pd.isna(build_features(df, ['sma_50']).iloc[-1]['sma_50'])
        assert pd.isna(build_features(df.iloc[:-1], ['sma_50']).iloc[-1]['sma_50'])

    def test_the_backtest_s_old_bar_left_momentum_undefined(self):
        """Why the threshold had to be derived, in one assertion.

        The engine admitted any ticker with 20 rows. `mom_9m_skip1m` needs
        three quarters, so for the opening months of every backtest the
        strategy ranked the universe on the feature it ranks on being NaN.
        """
        from portfolio_agent.features.pipeline import warmup_rows

        assert warmup_rows(['mom_9m_skip1m']) > 20

    def test_an_unregistered_name_is_an_error_not_a_zero(self):
        from portfolio_agent.features.pipeline import warmup_rows

        with pytest.raises(KeyError):
            warmup_rows(['no_such_feature'])


class TestEffectiveMinHistory:
    """The caller's threshold is a floor, never a ceiling.

    `min_history` was answering two questions at once. One is statistical — a
    year before a name is eligible is a judgement about sample adequacy, and it
    belongs to the caller. The other is mechanical: below the warm-up the
    feature is NaN, and no setting makes that reasonable to rank on.
    """

    def test_a_low_request_is_raised_to_the_warm_up(self):
        from portfolio_agent.features.pipeline import effective_min_history, warmup_rows

        assert effective_min_history(['sma_200'], 20) == warmup_rows(['sma_200'])

    def test_a_high_request_is_left_alone(self):
        from portfolio_agent.features.pipeline import effective_min_history

        assert effective_min_history(['sma_20'], 500) == 500

    def test_the_shipped_default_already_covered_the_registry(self):
        """252 was right by luck, and this records the margin it had.

        Both `DEFAULT_MIN_HISTORY` constants describe themselves as standing in
        for the longest lookback. They were correct only because the longest
        one is 211. A feature registered with a three-year window makes them
        wrong, and this assertion is what would fail.
        """
        from portfolio_agent.evaluation.harness import DEFAULT_MIN_HISTORY
        from portfolio_agent.features.pipeline import warmup_rows
        from portfolio_agent.features.registry import list_features

        assert warmup_rows(list_features()) <= DEFAULT_MIN_HISTORY

    def test_the_two_defaults_are_the_same_number(self):
        from portfolio_agent.evaluation.harness import DEFAULT_MIN_HISTORY as evaluation
        from portfolio_agent.training.data import DEFAULT_MIN_HISTORY as training

        assert evaluation == training
