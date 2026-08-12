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


class TestCrossSectionalFeatureScaling:
    """Task 4.1's second half: standardize across the universe, per date.

    The global scaler is fitted on training rows only, so it never leaked — but
    it answers the wrong question. A z-score against a five-year pooled mean
    says "this RSI is high for this stock over the sample"; what a
    cross-sectional model needs is "this RSI is high *relative to what else is
    available to buy today*". The pooled version also silently encodes the
    market factor into every feature: on a day the whole market gapped down,
    every name's return feature is low against the pooled mean, and the model
    reads a market state it cannot act on.

    The cross-sectional form cannot leak by construction — the transform for
    date t reads only rows dated t — which is a stronger guarantee than "we
    remembered to fit on the training split".
    """

    @staticmethod
    def _panel(n_dates=12, tickers=("A", "B", "C", "D", "E", "F"), seed=3):
        import pandas as pd

        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2024-01-01", periods=n_dates)
        return {
            ticker: pd.DataFrame(
                {
                    "close": rng.uniform(50, 5000, n_dates),
                    "rsi_14": rng.uniform(10, 90, n_dates),
                    "target_return_5d": rng.normal(0, 0.02, n_dates),
                },
                index=dates,
            )
            for ticker in tickers
        }

    def _scaled(self, panel, **kwargs):
        from portfolio_agent.features.scaling import apply_cross_sectional_scaling

        return apply_cross_sectional_scaling(
            panel, feature_columns=["close", "rsi_14"], **kwargs
        )

    def test_each_feature_is_mean_zero_and_unit_variance_on_every_date(self):
        import pandas as pd

        scaled = self._scaled(self._panel())

        for column in ("close", "rsi_14"):
            by_date = pd.DataFrame(
                {ticker: frame[column] for ticker, frame in scaled.items()}
            )
            for date, row in by_date.iterrows():
                assert float(row.mean()) == pytest.approx(0.0, abs=1e-12), (column, date)
                assert float(row.std(ddof=0)) == pytest.approx(1.0, abs=1e-12), (column, date)

    def test_uses_no_information_from_any_other_date(self):
        """The acceptance criterion, tested directly rather than by inspection.

        Rewrite every row after the first date — arbitrarily, including the
        future the model must not see — and the first date's scaled values must
        be bit-for-bit identical. Nothing about a fitted-on-train-split scaler
        can be checked this sharply; this one can, because it fits nothing.
        """
        panel = self._panel()
        scaled_before = self._scaled(panel)

        corrupted = {
            ticker: frame.copy() for ticker, frame in panel.items()
        }
        for frame in corrupted.values():
            frame.iloc[1:, :] = frame.iloc[1:, :] * 1000.0 + 7.0
        scaled_after = self._scaled(corrupted)

        for ticker in panel:
            for column in ("close", "rsi_14"):
                assert scaled_after[ticker][column].iloc[0] == (
                    scaled_before[ticker][column].iloc[0]
                )

    def test_the_target_column_is_left_alone(self):
        """Scaling the label would change what the model is being asked to
        predict — and the label already has its own cross-sectional
        transform."""
        panel = self._panel()
        scaled = self._scaled(panel)

        for ticker in panel:
            np.testing.assert_array_equal(
                scaled[ticker]["target_return_5d"].to_numpy(),
                panel[ticker]["target_return_5d"].to_numpy(),
            )

    def test_a_constant_feature_becomes_zero_not_infinity(self):
        """Zero cross-sectional dispersion means the feature separates nobody
        today. Dividing by that spread would turn a column carrying no
        information into the one that dominates the input."""
        import pandas as pd

        dates = pd.bdate_range("2024-01-01", periods=6)
        panel = {
            ticker: pd.DataFrame(
                {"flat": np.full(6, 42.0), "target_x": np.arange(6, dtype=float)},
                index=dates,
            )
            for ticker in ("A", "B", "C", "D", "E")
        }
        from portfolio_agent.features.scaling import apply_cross_sectional_scaling

        scaled = apply_cross_sectional_scaling(panel, feature_columns=["flat"])

        for ticker in panel:
            assert np.all(scaled[ticker]["flat"].to_numpy() == 0.0)

    def test_removes_the_market_factor_from_a_common_move(self):
        """The reason this beats the pooled scaler.

        Every name gaps down together. Pooled against a five-year mean, that
        day's return feature reads as extreme for everyone; cross-sectionally
        it reads as ordinary for everyone, which is the truth — nothing
        separated them.
        """
        import pandas as pd

        dates = pd.bdate_range("2024-01-01", periods=3)
        panel = {
            ticker: pd.DataFrame(
                {"return_1d": np.array([0.001, -0.090 + 0.001 * i, 0.002])},
                index=dates,
            )
            for i, ticker in enumerate(("A", "B", "C", "D", "E"))
        }
        from portfolio_agent.features.scaling import apply_cross_sectional_scaling

        scaled = apply_cross_sectional_scaling(panel, feature_columns=["return_1d"])
        crash_day = np.array([scaled[t]["return_1d"].iloc[1] for t in panel])

        assert float(np.mean(crash_day)) == pytest.approx(0.0, abs=1e-12)
        assert float(np.std(crash_day, ddof=0)) == pytest.approx(1.0, abs=1e-12)

    def test_clips_extreme_cross_sectional_outliers(self):
        import pandas as pd

        dates = pd.bdate_range("2024-01-01", periods=1)
        values = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0, "E": 1.0, "F": 1e9}
        panel = {
            ticker: pd.DataFrame({"x": [value]}, index=dates)
            for ticker, value in values.items()
        }
        from portfolio_agent.features.scaling import apply_cross_sectional_scaling

        # The clip has to sit below what a 6-name cross-section can even
        # produce: with population std, one outlier against N-1 identical
        # values maxes out at sqrt(N-1) = 2.236, so a clip of 3.0 could never
        # bind here and the test would pass without testing anything.
        scaled = apply_cross_sectional_scaling(
            panel, feature_columns=["x"], clip=1.5
        )

        assert scaled["F"]["x"].iloc[0] == pytest.approx(1.5)

    def test_drops_dates_with_too_thin_a_cross_section(self):
        """A z-score across two names says almost nothing, and mixing those
        rows in gives the model two different feature definitions.

        Only the thin dates go — the well-populated ones on either side stay.
        """
        import pandas as pd

        dates = pd.bdate_range("2024-01-01", periods=4)
        # A and B run the whole window; the rest only join on the last two
        # dates, so dates 0 and 1 have a cross-section of two.
        panel = {
            "A": pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}, index=dates),
            "B": pd.DataFrame({"x": [2.0, 3.0, 4.0, 5.0]}, index=dates),
        }
        for i, ticker in enumerate(("C", "D", "E", "F")):
            panel[ticker] = pd.DataFrame(
                {"x": [3.0 + i, 4.0 + i]}, index=dates[2:]
            )

        from portfolio_agent.features.scaling import apply_cross_sectional_scaling

        scaled = apply_cross_sectional_scaling(
            panel, feature_columns=["x"], min_names=5
        )

        assert list(scaled["A"].index) == list(dates[2:])
        assert list(scaled["C"].index) == list(dates[2:])

    def test_a_universe_thin_on_every_date_is_left_unscaled(self):
        """Degrade, do not destroy.

        Two tickers have no cross-section on any date. Emptying the panel would
        throw the whole training set away over a property of the universe
        rather than of the data — and apply_cross_sectional_target falls back
        the same way, so the label and the inputs must agree about which rows
        still exist.
        """
        import pandas as pd

        dates = pd.bdate_range("2024-01-01", periods=4)
        panel = {
            "A": pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}, index=dates),
            "B": pd.DataFrame({"x": [2.0, 3.0, 4.0, 5.0]}, index=dates),
        }
        from portfolio_agent.features.scaling import apply_cross_sectional_scaling

        scaled = apply_cross_sectional_scaling(
            panel, feature_columns=["x"], min_names=5
        )

        for ticker in panel:
            np.testing.assert_array_equal(
                scaled[ticker]["x"].to_numpy(), panel[ticker]["x"].to_numpy()
            )

    def test_is_deterministic(self):
        panel = self._panel()
        first = self._scaled(panel)
        second = self._scaled(panel)

        for ticker in panel:
            np.testing.assert_array_equal(
                first[ticker].to_numpy(), second[ticker].to_numpy()
            )
