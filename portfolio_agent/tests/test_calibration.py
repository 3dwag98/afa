"""Tests for isotonic confidence calibration (src/calibration.py)."""

import numpy as np
import pytest

from portfolio_agent.src.calibration import (
    IsotonicCalibrator,
    calibration_error,
    pool_adjacent_violators,
)


class TestPoolAdjacentViolators:
    def test_an_already_monotone_series_is_left_alone(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([0.1, 0.2, 0.3, 0.4])

        knot_x, knot_y = pool_adjacent_violators(x, y)

        assert np.allclose(np.interp(x, knot_x, knot_y), y)

    def test_a_violation_is_pooled_into_its_average(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 0.5])

        knot_x, knot_y = pool_adjacent_violators(x, y)
        fitted = np.interp(x, knot_x, knot_y)

        assert fitted[0] == pytest.approx(0.0)
        assert fitted[1] == pytest.approx(0.75)
        assert fitted[2] == pytest.approx(0.75)

    def test_pooling_cascades_backwards(self):
        """Merging two blocks can push the merged mean below the block before
        it, so the check has to repeat rather than run once."""
        x = np.arange(4.0)
        y = np.array([0.0, 0.9, 0.8, 0.1])

        _, knot_y = pool_adjacent_violators(x, y)

        assert np.all(np.diff(knot_y) >= -1e-9)

    def test_unsorted_input_is_handled(self):
        x = np.array([3.0, 1.0, 2.0])
        y = np.array([0.9, 0.1, 0.5])

        knot_x, knot_y = pool_adjacent_violators(x, y)

        assert np.all(np.diff(knot_x) >= 0)
        assert np.all(np.diff(knot_y) >= -1e-9)


class TestIsotonicCalibrator:
    @staticmethod
    def _overconfident_sample(n=2000, seed=3):
        """Scores that rank well but whose scale badly overstates the win rate.

        True win probability is a compressed function of the score, so the raw
        score reads far more confident than the outcomes justify — the exact
        failure mode a network on noisy return data exhibits.
        """
        rng = np.random.default_rng(seed)
        scores = rng.uniform(-1.0, 1.0, size=n)
        true_probability = 0.45 + 0.1 * scores
        outcomes = (rng.uniform(size=n) < true_probability).astype(float)
        return scores, outcomes

    def test_calibration_reduces_expected_calibration_error(self):
        scores, outcomes = self._overconfident_sample()
        calibrator = IsotonicCalibrator.fit(scores, outcomes)

        raw = np.clip(0.5 + scores, 0.0, 1.0)
        calibrated = calibrator.predict(scores)

        assert calibration_error(calibrated, outcomes) < calibration_error(raw, outcomes)

    def test_the_fitted_map_is_monotone(self):
        """Monotonicity is what preserves the model's ranking — the part that
        was actually measured out of sample."""
        scores, outcomes = self._overconfident_sample()
        calibrator = IsotonicCalibrator.fit(scores, outcomes)

        grid = np.linspace(-1.0, 1.0, 200)
        probabilities = calibrator.predict(grid)

        assert np.all(np.diff(probabilities) >= -1e-9)

    def test_predictions_stay_inside_the_unit_interval(self):
        scores, outcomes = self._overconfident_sample()
        calibrator = IsotonicCalibrator.fit(scores, outcomes)

        probabilities = calibrator.predict(np.array([-1e6, 0.0, 1e6]))

        assert probabilities.min() >= 0.0
        assert probabilities.max() <= 1.0

    def test_out_of_range_scores_are_clamped_not_extrapolated(self):
        scores, outcomes = self._overconfident_sample()
        calibrator = IsotonicCalibrator.fit(scores, outcomes)

        assert calibrator.predict_one(1e6) == pytest.approx(max(calibrator.knot_y))
        assert calibrator.predict_one(-1e6) == pytest.approx(min(calibrator.knot_y))

    def test_too_few_samples_refuses_to_fit(self):
        """Below a few hundred observations the per-bin frequencies are noise,
        and the fit would encode that noise as a correction."""
        rng = np.random.default_rng(0)
        assert IsotonicCalibrator.fit(rng.uniform(size=20), rng.integers(0, 2, 20)) is None

    def test_degenerate_outcomes_refuse_to_fit(self):
        """All wins says nothing about where the threshold belongs."""
        scores = np.linspace(0, 1, 500)
        assert IsotonicCalibrator.fit(scores, np.ones(500)) is None

    def test_mismatched_lengths_return_none(self):
        assert IsotonicCalibrator.fit(np.zeros(10), np.zeros(5)) is None

    def test_round_trips_through_json(self, tmp_path):
        scores, outcomes = self._overconfident_sample()
        calibrator = IsotonicCalibrator.fit(scores, outcomes)

        path = tmp_path / "calibration.json"
        calibrator.save(path)
        restored = IsotonicCalibrator.load(path)

        assert np.allclose(restored.predict(scores), calibrator.predict(scores))
        assert restored.n_samples == calibrator.n_samples

    def test_loading_a_missing_or_corrupt_file_returns_none(self, tmp_path):
        assert IsotonicCalibrator.load(tmp_path / "absent.json") is None

        broken = tmp_path / "broken.json"
        broken.write_text("{not json")
        assert IsotonicCalibrator.load(broken) is None

    def test_from_dict_rejects_incomplete_payloads(self):
        assert IsotonicCalibrator.from_dict(None) is None
        assert IsotonicCalibrator.from_dict({}) is None
        assert IsotonicCalibrator.from_dict({"knot_x": [1.0], "knot_y": []}) is None


class TestCalibrationError:
    def test_a_perfectly_calibrated_predictor_scores_zero(self):
        probabilities = np.full(1000, 0.5)
        outcomes = np.tile([0.0, 1.0], 500)

        assert calibration_error(probabilities, outcomes) == pytest.approx(0.0, abs=1e-9)

    def test_a_confidently_wrong_predictor_scores_high(self):
        probabilities = np.full(100, 0.95)
        outcomes = np.zeros(100)

        assert calibration_error(probabilities, outcomes) == pytest.approx(0.95, abs=1e-9)

    def test_probability_one_lands_in_the_top_bin(self):
        """A digitize that spills p == 1.0 past the last edge indexes off the
        end of the bin table."""
        assert calibration_error(np.ones(50), np.ones(50)) == pytest.approx(0.0)

    def test_empty_input_scores_zero(self):
        assert calibration_error([], []) == 0.0
