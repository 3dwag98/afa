"""Input standardization for the sequence forecasting models.

Why this exists: the registered features are on wildly different scales, and
several of them are *price levels*. `sma_20` for a ₹90 small-cap and for MRF at
₹1,50,000 differ by four orders of magnitude, while `return_1d` lives in
[-0.2, 0.2] and `rsi_14` in [0, 100]. Feeding that straight into an LSTM has
two failure modes, and the platform hit both:

* **fp16 overflow.** The largest representable half-precision value is 65504.
  A price-level feature above that becomes `inf` on the first autocast matmul,
  the loss becomes NaN, and every epoch after it prints `nan`.
* **Exploding activations in fp32.** Even without autocast, inputs of 1e5
  through a recurrent layer at lr=3e-3 diverge within a few hundred steps.

Rolling z-scoring inside the feature pipeline (`features.normalize`) is not a
substitute: those same feature frames are consumed by the rule-based strategies,
whose thresholds are stated in raw units ("RSI below 30"), so normalizing there
would change what every other strategy trades. Standardizing here instead keeps
the model's *inputs* well-conditioned while leaving the shared feature pipeline
untouched, and — because the fitted constants ship inside the checkpoint
metadata — guarantees inference applies exactly the transform training used.

The statistics are always fitted on training rows only, never on validation or
test rows, so this introduces no look-ahead.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

# Standard deviations below this are treated as "constant feature" and scaled
# by 1.0 instead: dividing by a near-zero spread turns a column that carries no
# information into one that dominates the input.
_MIN_STD = 1e-8

# Post-standardization clip, in standard deviations. Price-level features are
# heavily right-skewed across a 4000-name Indian universe, so the largest names
# sit tens of sigma above the mean even after centring. Clipping bounds the
# input the network can ever see without discarding the row.
DEFAULT_CLIP = 10.0


class FeatureScaler:
    """Per-feature mean/std standardization with a hard clip.

    Args:
        mean: Per-feature means, shape (n_features,).
        std: Per-feature standard deviations, shape (n_features,).
        clip: Absolute bound applied after standardizing.
    """

    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        clip: float = DEFAULT_CLIP,
    ):
        self.mean = np.asarray(mean, dtype=np.float64).ravel()
        self.std = np.asarray(std, dtype=np.float64).ravel()
        if self.mean.shape != self.std.shape:
            raise ValueError(
                f"mean and std must describe the same features, got "
                f"{self.mean.shape} and {self.std.shape}"
            )
        # Guard the divisor here rather than at fit time, so a scaler
        # reconstructed from metadata written by an older run is safe too.
        self.std = np.where(np.isfinite(self.std) & (self.std > _MIN_STD), self.std, 1.0)
        self.mean = np.where(np.isfinite(self.mean), self.mean, 0.0)
        self.clip = float(clip)

    @property
    def n_features(self) -> int:
        return int(self.mean.size)

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        clip: float = DEFAULT_CLIP,
    ) -> "FeatureScaler":
        """Fit on a (n_samples, n_features) block of *training* rows only.

        Non-finite entries are ignored rather than poisoning the statistics —
        a single `inf` left over from a division by a zero price would
        otherwise make the whole column's mean `inf` and every standardized
        value NaN.
        """
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(f"expected a 2-D feature block, got shape {matrix.shape}")
        if matrix.shape[0] == 0:
            raise ValueError("cannot fit a scaler on zero rows")

        clean = np.where(np.isfinite(matrix), matrix, np.nan)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(clean, axis=0)
            std = np.nanstd(clean, axis=0)
        return cls(mean=mean, std=std, clip=clip)

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Standardize a (…, n_features) array, returning float32.

        The output is guaranteed finite: anything non-finite on the way in (or
        produced by the division) is mapped to 0.0, which is the standardized
        value of "this feature at its training mean" — the most neutral thing
        the model can be shown.
        """
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.shape[-1] != self.n_features:
            raise ValueError(
                f"scaler was fitted on {self.n_features} features but got "
                f"{matrix.shape[-1]}"
            )
        with np.errstate(invalid="ignore", divide="ignore"):
            scaled = (matrix - self.mean) / self.std
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=self.clip, neginf=-self.clip)
        return np.clip(scaled, -self.clip, self.clip).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable form, for the checkpoint metadata."""
        return {
            "mean": [float(v) for v in self.mean],
            "std": [float(v) for v in self.std],
            "clip": self.clip,
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> Optional["FeatureScaler"]:
        """Rebuild from metadata, or None when a checkpoint carries none.

        Checkpoints trained before input standardization existed have no
        `feature_scaler` key; returning None there keeps them loading and
        scoring exactly as they did.
        """
        if not payload:
            return None
        mean, std = payload.get("mean"), payload.get("std")
        if not mean or not std:
            return None
        return cls(mean=mean, std=std, clip=float(payload.get("clip", DEFAULT_CLIP)))
