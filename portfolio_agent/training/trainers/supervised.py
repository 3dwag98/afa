"""The existing supervised pipeline, exposed as a registered trainer.

This is a thin adapter, not a reimplementation. `agents/trainer.py::run_training`
already does the hard parts — walk-forward validation, the cross-sectional
target transform, quantile heads, confidence calibration, the checkpoint and
the `models/metadata.json` sidecar `MLStrategy` reads — and all of it is
load-bearing. Re-deriving any of that inside a new trainer would fork behaviour
that took several rounds of fixes to get right.

What the adapter adds is the two things the registry contract needs: a declared
hyperparameter schema (so `--set` validates against the supervised knobs, and
rejects one aimed at a different trainer), and a pinnable universe (so a
supervised model and an RL model can be compared on the same names).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

from pydantic import Field

from ..base import BaseTrainer, TrainerConfig, TrainingArtifact, TrainingData
from ..registry import register_trainer

logger = logging.getLogger(__name__)


class SupervisedTrainerConfig(TrainerConfig):
    """Knobs the supervised pipeline reads off `config.training`.

    These mirror `config/schema.py::TrainingConfig`; the field documentation
    there is the authority on what each one means and why its default is what
    it is. Listing them here is what lets a typo like `sequence_lenght=60` fail
    at startup with the correct spelling in the message.
    """

    model: str = Field(default="lstm", description="Registered architecture (models/registry.py).")
    target: str = Field(default="return_5d", description="Forward-return label.")
    sequence_length: int = Field(default=60, gt=0, description="Input window length.")
    feature_normalization: str = Field(default="cross_sectional")
    target_transform: str = Field(default="cross_sectional_rank")
    max_abs_target: float = Field(default=5.0, gt=0.0)
    use_synthetic_data: bool = Field(
        default=False,
        description="Train on generated OHLCV instead of the cache. For offline "
        "smoke tests only — a model fitted on it must never be scored.",
    )


@register_trainer("supervised")
class SupervisedTrainer(BaseTrainer):
    """Adapter over `agents/trainer.py::run_training`."""

    name = "supervised"
    strategy_name = None  # produces the checkpoint MLStrategy loads, not one strategy's

    @classmethod
    def config_model(cls) -> Type[TrainerConfig]:
        return SupervisedTrainerConfig

    def __init__(self) -> None:
        self._app_config: Any = None
        self._universe: Optional[List[str]] = None

    def prepare(
        self, app_config: Any, universe: List[str], cfg: TrainerConfig
    ) -> TrainingData:
        """Record what to train on; `run_training` owns the real data path.

        Returning a placeholder rather than building a second panel is
        deliberate. `run_training` constructs its panel differently on purpose
        (grouped by split then ticker, so a single index split lands on the
        chronological boundaries), and building a parallel one here would mean
        two data paths that must be kept in agreement forever.
        """
        self._app_config = app_config
        self._universe = list(universe)

        return TrainingData(
            features_by_ticker={},
            prices_by_ticker={},
            tickers=list(universe),
            feature_names=_supervised_feature_names(),
        )

    def fit(self, data: TrainingData, cfg: TrainerConfig) -> TrainingArtifact:
        assert isinstance(cfg, SupervisedTrainerConfig)
        from portfolio_agent.agents.trainer import run_training

        app_config = self._app_config
        if app_config is None:
            raise RuntimeError("prepare() must run before fit()")

        # Push the resolved hyperparameters onto the AppConfig `run_training`
        # reads, so a `--set epochs=5` reaches the loop rather than being
        # accepted and ignored.
        merged = app_config.model_copy(deep=True)
        for key, value in cfg.model_dump().items():
            if hasattr(merged.training, key):
                setattr(merged.training, key, value)

        metadata = run_training(merged, universe=self._universe)

        # run_training writes its own checkpoint and sidecar in the shape
        # MLStrategy expects, so the artifact returned here is for the bulk
        # report and provenance — save_artifact must not overwrite it.
        return TrainingArtifact(
            state_dict={},
            metadata={
                "feature_names": metadata.get("feature_names", data.feature_names),
                "feature_scaler": metadata.get("feature_scaler"),
                "trainer": self.name,
                "model_architecture": metadata.get("model_architecture", cfg.model),
                "sequence_length": metadata.get("sequence_length", cfg.sequence_length),
                "checkpoint_written_by": "agents.trainer.run_training",
            },
            metrics=_headline_metrics(metadata),
        )

    @property
    def writes_own_checkpoint(self) -> bool:
        """`run_training` already wrote `models/<model>_best.pt` and the sidecar."""
        return True


def _supervised_feature_names() -> List[str]:
    """The feature list the supervised pipeline is hardcoded against."""
    from portfolio_agent.agents.trainer import TRAINING_FEATURE_NAMES

    return list(TRAINING_FEATURE_NAMES)


def _headline_metrics(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the few numbers a bulk comparison table wants out of the metadata."""
    metrics: Dict[str, Any] = {
        "epochs_trained": metadata.get("epochs_trained"),
        "best_val_loss": (metadata.get("metrics") or {}).get("best_val_loss"),
    }
    walk_forward = metadata.get("walk_forward")
    if isinstance(walk_forward, dict):
        for key in ("mean_ic", "mean_rank_ic", "hit_rate"):
            if key in walk_forward:
                metrics[key] = walk_forward[key]
    test_metrics = metadata.get("test_metrics")
    if isinstance(test_metrics, dict):
        for key, value in test_metrics.items():
            if isinstance(value, (int, float)):
                metrics[f"test_{key}"] = value
    return {k: v for k, v in metrics.items() if v is not None}
