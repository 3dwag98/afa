"""Tests for input standardization (features/scaling.py).

The defect these pin down: training printed `nan` for every epoch on a CUDA
GPU. Half the model's inputs are price levels, an Indian large-cap trades
above ₹1,00,000, and fp16 — which automatic mixed precision casts to — cannot
represent anything above 65504. The first autocast matmul turned those
features into `inf`, the loss into NaN, and every subsequent epoch was dead.
"""

import numpy as np
import pytest

from portfolio_agent.features.scaling import DEFAULT_CLIP, FeatureScaler


FP16_MAX = 65504.0


def _price_level_panel(seed: int = 0, n: int = 500) -> np.ndarray:
    """A feature block shaped like the real one: price levels beside returns."""
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(50_000, 160_000, n),   # sma_20 for a high-priced name
        rng.uniform(50_000, 160_000, n),   # sma_50
        rng.uniform(0, 100, n),            # rsi_14
        rng.normal(0, 500, n),             # macd
        rng.normal(0, 0.02, n),            # return_1d
    ])


class TestFp16Overflow:
    """The mechanism behind the NaN, and the proof the scaler removes it."""

    def test_raw_price_features_overflow_half_precision(self):
        panel = _price_level_panel()
        assert panel.max() > FP16_MAX

        with np.errstate(over="ignore"):
            as_half = panel.astype(np.float16)

        assert not np.isfinite(as_half).all(), (
            "this is the bug: casting raw price-level features to fp16 yields inf"
        )

    def test_standardized_features_survive_half_precision(self):
        panel = _price_level_panel()

        scaled = FeatureScaler.fit(panel).transform(panel)

        assert np.isfinite(scaled.astype(np.float16)).all()
        assert np.abs(scaled).max() <= DEFAULT_CLIP


class TestFeatureScaler:
    def test_standardizes_each_feature_independently(self):
        panel = _price_level_panel()

        scaled = FeatureScaler.fit(panel).transform(panel)

        # Every column lands on roughly zero mean / unit variance regardless of
        # the units it arrived in.
        assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-2)
        assert np.allclose(scaled.std(axis=0), 1.0, atol=1e-2)

    def test_output_is_float32(self):
        """The dataset builds float32 tensors; returning float64 would force a
        silent per-batch copy."""
        assert FeatureScaler.fit(_price_level_panel()).transform(
            _price_level_panel()
        ).dtype == np.float32

    def test_non_finite_inputs_do_not_poison_the_statistics(self):
        """One inf in a column used to make its mean inf and every standardized
        value in it NaN — turning a single bad bar into a dead feature."""
        panel = _price_level_panel()
        panel[7, 0] = np.inf
        panel[9, 1] = np.nan

        scaler = FeatureScaler.fit(panel)

        assert np.isfinite(scaler.mean).all()
        assert np.isfinite(scaler.std).all()

    def test_transform_output_is_always_finite(self):
        panel = _price_level_panel()
        panel[3, 2] = np.inf
        panel[4, 2] = -np.inf
        panel[5, 2] = np.nan

        scaled = FeatureScaler.fit(panel).transform(panel)

        assert np.isfinite(scaled).all()

    def test_constant_feature_does_not_divide_by_zero(self):
        panel = np.column_stack([np.full(100, 42.0), np.arange(100.0)])

        scaled = FeatureScaler.fit(panel).transform(panel)

        assert np.isfinite(scaled).all()
        assert np.allclose(scaled[:, 0], 0.0)

    def test_extreme_outliers_are_clipped_not_dropped(self):
        panel = _price_level_panel()
        panel[0, 0] = 1e12

        scaled = FeatureScaler.fit(panel).transform(panel)

        assert len(scaled) == len(panel)
        assert scaled[0, 0] == pytest.approx(DEFAULT_CLIP)

    def test_rejects_a_block_whose_width_does_not_match(self):
        scaler = FeatureScaler.fit(_price_level_panel())

        with pytest.raises(ValueError, match="fitted on 5 features"):
            scaler.transform(np.zeros((10, 3)))

    def test_cannot_fit_on_zero_rows(self):
        with pytest.raises(ValueError, match="zero rows"):
            FeatureScaler.fit(np.zeros((0, 4)))


class TestSerialization:
    """The scaler ships inside the checkpoint metadata, so it has to survive
    a JSON round trip exactly — inference applying a *different* transform
    from training is worse than applying none."""

    def test_round_trips_through_a_dict(self):
        panel = _price_level_panel()
        original = FeatureScaler.fit(panel)

        restored = FeatureScaler.from_dict(original.to_dict())

        assert np.allclose(original.transform(panel), restored.transform(panel))

    def test_missing_payload_means_no_scaler(self):
        """Checkpoints written before standardization existed were fitted on
        raw features and must keep being scored on raw features."""
        assert FeatureScaler.from_dict(None) is None
        assert FeatureScaler.from_dict({}) is None
        assert FeatureScaler.from_dict({"clip": 10.0}) is None
