"""The one place training checkpoints are written and read.

Every trainer routes through `save_artifact`, which is the only reason the
on-disk contract can be relied on. The schema is the one the existing strategy
loaders already read, not a new one:

    {
        "model_state_dict": {...},
        "metadata": {
            "feature_names":  [...],        # column order the net was fitted on
            "feature_scaler": {"mean": [...], "std": [...], "clip": ...},
            "trainer": "sac",
            ...trainer-specific keys...
        },
    }

`strategies/india_sac.py::load` reads exactly `metadata["feature_names"]` and
`metadata["feature_scaler"]`, and rebuilds the standardizer with
`FeatureScaler.from_dict`, which expects `mean`/`std`/`clip`.

Two failure modes this module exists to make impossible:

1. **A checkpoint that silently ships no scaler.** `FeatureScaler` exposes
   `.mean`/`.std` and serializes through `.to_dict()`. Code written against
   scikit-learn's `.mean_`/`.scale_` spelling finds neither, and if it reaches
   for them behind a `hasattr` guard it writes `None` without complaint. The
   model then trains on standardized inputs and scores raw ones — a skew that
   produces plausible-looking numbers and no error anywhere. `save_artifact`
   takes the `FeatureScaler` object itself and calls `to_dict()` for you.

2. **A checkpoint that cannot be loaded back.** The strategy loaders call
   `torch.load(..., weights_only=True)`, which refuses arbitrary pickled
   objects. Metadata assembled from pandas or numpy carries `np.float32`,
   `np.int64` and `Timestamp` values that a caller never notices writing, so
   metadata is coerced to plain Python primitives on the way out.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

from portfolio_agent.features.scaling import FeatureScaler

from .base import TrainingArtifact

logger = logging.getLogger(__name__)


def _plain(value: Any) -> Any:
    """Coerce numpy/pandas scalars and containers to JSON- and torch-safe types.

    `weights_only=True` loading and `json.dump` both reject the numpy scalar
    types that leak out of any pandas computation, and neither failure is
    obvious at the call site that produced them.
    """
    # Import locally: this module is imported by config-resolution paths that
    # must work on installs without the scientific stack fully present.
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Anything else (a Timestamp, a pydantic model, a custom object) becomes a
    # string rather than breaking the save. Losing fidelity in a provenance
    # field is strictly better than losing the trained weights.
    return str(value)


def build_metadata(
    *,
    feature_names: List[str],
    scaler: Optional[FeatureScaler],
    trainer: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the metadata block, guaranteeing the fields inference needs.

    Args:
        feature_names: Column order the network was fitted against.
        scaler: The fitted standardizer, or None if the trainer fitted on raw
            features. None is recorded explicitly rather than omitted, so a
            loader can tell "trained on raw features" apart from "someone
            forgot to record the scaler".
        trainer: Registry name of the producing trainer, for provenance.
        extra: Trainer-specific keys (hidden_dim, sequence_length, ...).

    Returns:
        A metadata dict safe to embed in a checkpoint.
    """
    if not feature_names:
        raise ValueError(
            "refusing to write a checkpoint with no feature names: inference "
            "assembles its state vector from this list, and a checkpoint "
            "without it cannot be scored correctly by construction"
        )

    metadata: Dict[str, Any] = {
        "feature_names": list(feature_names),
        "feature_scaler": scaler.to_dict() if scaler is not None else None,
        "trainer": trainer,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        # Explicit keys win over caller-supplied ones only for the two fields
        # inference cannot do without; everything else the trainer knows best.
        for key, value in extra.items():
            if key not in ("feature_names", "feature_scaler"):
                metadata[key] = value
    return _plain(metadata)


def save_artifact(
    artifact: TrainingArtifact,
    path: Path | str,
    *,
    sidecar_path: Optional[Path | str] = None,
) -> Path:
    """Write a checkpoint, and optionally a JSON sidecar of its metadata.

    Args:
        artifact: Weights plus metadata. `metadata` must already carry
            `feature_names` — build it with `build_metadata`.
        path: Destination `.pt` file. Parent directories are created.
        sidecar_path: Optional JSON file receiving metadata and metrics.
            `MLStrategy` reads `models/metadata.json` rather than the
            checkpoint, so the supervised trainer writes one.

    Returns:
        The path written.
    """
    import torch

    if "feature_names" not in artifact.metadata:
        raise ValueError(
            "artifact.metadata is missing 'feature_names' — construct it with "
            "training.artifacts.build_metadata() so the fields inference "
            "depends on cannot be omitted"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": artifact.state_dict,
        "metadata": _plain(artifact.metadata),
        "metrics": _plain(artifact.metrics),
    }
    torch.save(payload, path)
    logger.info("Saved checkpoint to %s", path)

    if sidecar_path is not None:
        sidecar_path = Path(sidecar_path)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar = dict(_plain(artifact.metadata))
        sidecar["metrics"] = _plain(artifact.metrics)
        with open(sidecar_path, "w") as handle:
            json.dump(sidecar, handle, indent=2)
        logger.info("Saved metadata sidecar to %s", sidecar_path)

    return path


def load_artifact(path: Path | str) -> TrainingArtifact:
    """Read a checkpoint written by `save_artifact`.

    Tolerates the older flat layout (metadata keys at the top level) that
    `agents/trainer.py` writes, so a supervised checkpoint trained before this
    package existed still loads.

    Raises:
        FileNotFoundError: If the checkpoint is absent. Callers must treat this
            as a failure, never as "start from random weights" — an untrained
            network emits confident-looking nonsense.
    """
    import torch

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}")

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a checkpoint dict")

    state_dict = payload.get("model_state_dict", payload)
    metadata = payload.get("metadata")
    if metadata is None:
        # Flat layout: everything that is not the weights is provenance.
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in ("model_state_dict", "optimizer_state_dict", "metrics")
        }
    return TrainingArtifact(
        state_dict=state_dict,
        metadata=dict(metadata),
        metrics=dict(payload.get("metrics", {}) or {}),
    )
