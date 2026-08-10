"""Isotonic calibration: turning model scores into probabilities that hold up.

A network trained on pinball loss produces quantiles of a forward return. Turn
those into a "chance this trade profits" and you have a number that *looks*
like a probability and behaves like one only by accident. Neural networks on
noisy financial data are systematically overconfident: the raw score at which
the model says 80% is typically won far less than 80% of the time. That matters
here more than in most settings, because the number feeds Kelly sizing and the
trigger engine's expected-value hurdle, and both are more sensitive to an
optimistic p than to almost anything else.

Calibration fixes the mapping without touching the model. Split the
out-of-sample scores into bins, measure the realized win rate in each, and fit
a monotone function from score to realized frequency. Monotone is the whole
trick: it preserves the model's *ranking* — which is where the alpha is, and
which is measured out-of-sample — while discarding its *scale*, which was
never measured at all.

The fit is the Pool-Adjacent-Violators Algorithm (PAVA), implemented here
directly rather than pulled from scikit-learn:

    minimize  sum_i w_i (y_i - f(x_i))^2   subject to f non-decreasing

PAVA sweeps left to right, and whenever a block's mean falls below its
predecessor's it merges the two and re-averages. It is exact (not iterative),
runs in O(n) after the sort, and the whole thing is forty lines — a scikit-learn
dependency for it would be the larger commitment.

Calibrators serialize to plain JSON so a fitted mapping ships next to the model
checkpoint and can be inspected without loading torch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Below this many out-of-sample observations, the realized frequencies in each
# bin are noise and the fitted map would encode that noise as a correction.
MIN_CALIBRATION_SAMPLES = 100


def pool_adjacent_violators(
    x: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> tuple:
    """Fit a non-decreasing step function to (x, y) by least squares.

    Args:
        x: Predictor values. Need not be sorted.
        y: Observed outcomes (0/1 for win/loss, or any real values).
        weights: Optional per-observation weights; uniform when omitted.

    Returns:
        (knot_x, knot_y): the sorted x values and the fitted non-decreasing
        values at them, with consecutive duplicates collapsed.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)

    order = np.argsort(x, kind="mergesort")
    x, y, w = x[order], y[order], w[order]

    # Each block holds a weighted mean and the total weight behind it. A block
    # whose mean sits below the previous block's violates monotonicity, so the
    # two are merged and the merged mean re-checked against *its* predecessor —
    # which is why this is a while loop and not a single pass.
    block_means: List[float] = []
    block_weights: List[float] = []
    block_sizes: List[int] = []

    for value, weight in zip(y, w):
        block_means.append(value)
        block_weights.append(weight)
        block_sizes.append(1)
        while len(block_means) > 1 and block_means[-2] > block_means[-1]:
            merged_weight = block_weights[-2] + block_weights[-1]
            merged_mean = (
                (block_means[-2] * block_weights[-2] + block_means[-1] * block_weights[-1])
                / merged_weight
                if merged_weight > 0
                else block_means[-1]
            )
            block_means[-2:] = [merged_mean]
            block_weights[-2:] = [merged_weight]
            block_sizes[-2:] = [block_sizes[-2] + block_sizes[-1]]

    fitted = np.repeat(np.asarray(block_means), np.asarray(block_sizes))

    # Collapse runs of identical fitted values into knots. Every original x
    # still maps to exactly its fitted value (a dropped point sits between two
    # knots that share that value), and predict() interpolates linearly between
    # distinct levels rather than stepping — the standard isotonic predictor,
    # and the smoother choice for a probability map.
    keep = np.ones(len(x), dtype=bool)
    keep[1:-1] = ~(
        np.isclose(fitted[1:-1], fitted[:-2]) & np.isclose(fitted[1:-1], fitted[2:])
    )
    return x[keep], fitted[keep]


@dataclass
class IsotonicCalibrator:
    """A fitted monotone map from raw model score to realized probability.

    Attributes:
        knot_x: Sorted raw scores at which the fitted map changes value.
        knot_y: Calibrated probabilities at those scores, non-decreasing.
        n_samples: How many observations the fit was built from.
        base_rate: Unconditional win rate in the fitting sample, reported so a
            calibrated probability can be read against the trivial baseline.
    """

    knot_x: List[float]
    knot_y: List[float]
    n_samples: int = 0
    base_rate: float = 0.5

    @classmethod
    def fit(
        cls,
        scores: Sequence[float],
        outcomes: Sequence[float],
        min_samples: int = MIN_CALIBRATION_SAMPLES,
    ) -> Optional["IsotonicCalibrator"]:
        """Fit on out-of-sample scores and their realized binary outcomes.

        **Out-of-sample is not optional.** Fitting on the same predictions the
        model was trained against measures how well the model memorized its
        training set and produces a map that makes an overfitted model look
        perfectly calibrated. The trainer fits this on walk-forward test-fold
        predictions for exactly that reason.

        Args:
            scores: Raw model outputs (any monotone score; needs no scale).
            outcomes: 1.0 for a profitable outcome, 0.0 otherwise.
            min_samples: Refuse to fit below this many usable observations.

        Returns:
            A fitted calibrator, or None when there is too little data or the
            outcomes are degenerate (all wins or all losses, which carries no
            information about where the score threshold should sit).
        """
        scores = np.asarray(scores, dtype=float).ravel()
        outcomes = np.asarray(outcomes, dtype=float).ravel()
        if scores.size != outcomes.size or scores.size == 0:
            return None

        finite = np.isfinite(scores) & np.isfinite(outcomes)
        scores, outcomes = scores[finite], outcomes[finite]
        if scores.size < min_samples:
            return None

        binary = (outcomes > 0).astype(float)
        if binary.min() == binary.max():
            return None

        knot_x, knot_y = pool_adjacent_violators(scores, binary)
        return cls(
            knot_x=[float(v) for v in knot_x],
            knot_y=[float(min(1.0, max(0.0, v))) for v in knot_y],
            n_samples=int(scores.size),
            base_rate=float(binary.mean()),
        )

    def predict(self, scores) -> np.ndarray:
        """Map raw scores to calibrated probabilities in [0, 1].

        Scores outside the fitted range are clamped to the end knots rather
        than extrapolated: the fit says nothing about behaviour beyond the data
        it saw, and extrapolating a monotone map is how a model ends up
        reporting probabilities above 1 for its most extreme outputs.
        """
        values = np.asarray(scores, dtype=float)
        if not self.knot_x:
            return np.full(values.shape, self.base_rate, dtype=float)
        calibrated = np.interp(values, self.knot_x, self.knot_y)
        return np.clip(calibrated, 0.0, 1.0)

    def predict_one(self, score: float) -> float:
        """Calibrate a single score."""
        return float(self.predict(np.asarray([score]))[0])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "knot_x": self.knot_x,
            "knot_y": self.knot_y,
            "n_samples": self.n_samples,
            "base_rate": self.base_rate,
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> Optional["IsotonicCalibrator"]:
        """Rebuild from to_dict() output, or None for a missing/invalid payload."""
        if not payload:
            return None
        knot_x = payload.get("knot_x")
        knot_y = payload.get("knot_y")
        if not knot_x or not knot_y or len(knot_x) != len(knot_y):
            return None
        return cls(
            knot_x=[float(v) for v in knot_x],
            knot_y=[float(v) for v in knot_y],
            n_samples=int(payload.get("n_samples", 0)),
            base_rate=float(payload.get("base_rate", 0.5)),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Optional["IsotonicCalibrator"]:
        file = Path(path)
        if not file.exists():
            return None
        try:
            return cls.from_dict(json.loads(file.read_text()))
        except (json.JSONDecodeError, TypeError, ValueError):
            return None


def calibration_error(
    probabilities: Sequence[float],
    outcomes: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Expected calibration error: mean |predicted - realized| across bins.

    Weighted by bin population, so a bin holding three observations cannot
    dominate the score. This is the number that says whether calibration
    actually helped — a model whose 80% bucket wins 55% of the time has an ECE
    the loss curve will never reveal.

    Args:
        probabilities: Predicted probabilities in [0, 1].
        outcomes: Realized binary outcomes.
        n_bins: Number of equal-width bins over [0, 1].

    Returns:
        ECE in [0, 1]; 0.0 when there is nothing to score.
    """
    p = np.asarray(probabilities, dtype=float).ravel()
    y = (np.asarray(outcomes, dtype=float).ravel() > 0).astype(float)
    if p.size == 0 or p.size != y.size:
        return 0.0

    finite = np.isfinite(p) & np.isfinite(y)
    p, y = p[finite], y[finite]
    if p.size == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # `right=True` so the top bin includes p == 1.0 rather than spilling past
    # the last edge into an index that does not exist.
    bins = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)

    total_error = 0.0
    for b in range(n_bins):
        mask = bins == b
        count = int(mask.sum())
        if count == 0:
            continue
        total_error += count * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return total_error / p.size
