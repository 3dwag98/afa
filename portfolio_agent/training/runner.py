"""One training run, end to end.

Resolve the trainer and its config, pin the universe, prepare, fit, save. Every
entry point — the CLI, a bulk sweep, a notebook cell — comes through here, so
they cannot drift apart in what they mean by "train this strategy".
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .base import TrainerConfig, TrainingArtifact
from .config import resolve_training_config
from .universe import UniverseSnapshot, resolve_universe

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path("models")


@dataclass
class TrainingRun:
    """The record of one completed run.

    Carries enough to reproduce it — the resolved config and the universe
    fingerprint — because a metrics table without provenance invites comparing
    two runs that were never comparable.
    """

    strategy: Optional[str]
    trainer: str
    config: TrainerConfig
    artifact: TrainingArtifact
    universe: UniverseSnapshot
    checkpoint_path: Optional[Path]
    duration_seconds: float
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def summary(self) -> Dict[str, Any]:
        """A flat row for a comparison table."""
        row: Dict[str, Any] = {
            "strategy": self.strategy or "-",
            "trainer": self.trainer,
            "tickers": len(self.universe),
            "universe": self.universe.fingerprint,
            "seconds": round(self.duration_seconds, 1),
            "status": "ok" if self.ok else "failed",
        }
        if self.error:
            row["error"] = self.error
            return row
        primary = self.artifact.primary_metric()
        if primary is not None:
            row["metric"] = round(primary, 6)
        for key, value in self.artifact.metrics.items():
            if isinstance(value, (int, float)) and key != "history":
                row[key] = round(float(value), 6)
        return row


def checkpoint_path_for(
    strategy: Optional[str],
    trainer_name: str,
    *,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    model_name: Optional[str] = None,
    suffix: str = ".pt",
) -> Path:
    """Where a run's weights go.

    Defaults to `<models_dir>/<model_name>_best.pt`, matching what the strategy
    loaders already look for: `IndiaSACStrategy` builds
    `models/india_sac_best.pt` from its `model_name` param, and `MLStrategy`
    builds `models/<architecture>_best.pt`.

    `suffix` comes from the trainer, because not every payload is a torch file
    — the boosting baseline writes `.joblib`. It is the trainer's
    `checkpoint_suffix`, so the reported path and the written path cannot
    disagree.
    """
    name = model_name or strategy or trainer_name
    return Path(models_dir) / f"{name}_best{suffix}"


def run_training_job(
    app_config: Any,
    strategy: Optional[str] = None,
    *,
    trainer: Optional[str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    universe: Optional[Sequence[str]] = None,
    snapshot: Optional[Path | str] = None,
    universe_size: Optional[int] = None,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    model_name: Optional[str] = None,
    strategy_config_file: Optional[Path | str] = None,
    save: bool = True,
) -> TrainingRun:
    """Train one strategy and (by default) write its checkpoint.

    Args:
        app_config: Loaded AppConfig.
        strategy: Strategy to train. None runs the supervised default.
        trainer: Explicit trainer name, overriding every other source.
        overrides: Highest-precedence hyperparameters.
        universe: Explicit ticker list.
        snapshot: Path to a saved universe snapshot; ignored if `universe` is
            given.
        universe_size: Size for a fresh draw, when neither of the above is set.
        models_dir: Checkpoint destination directory.
        model_name: Checkpoint basename, defaulting to the strategy name.
        strategy_config_file: Explicit strategy YAML.
        save: Write the checkpoint. False is for notebook experiments that
            should not clobber a good model.

    Returns:
        A `TrainingRun`. Failures are captured on the record rather than raised,
        so one bad entry in a sweep does not discard the rest.
    """
    started = time.monotonic()

    trainer_name, trainer_class, cfg = resolve_training_config(
        app_config,
        strategy,
        trainer=trainer,
        overrides=overrides,
        strategy_config_file=strategy_config_file,
    )

    snap = resolve_universe(
        app_config,
        tickers=universe,
        snapshot=snapshot,
        size=universe_size,
        name=strategy or trainer_name,
    )

    logger.info(
        "Training strategy=%s trainer=%s on %d tickers (universe %s)",
        strategy or "-", trainer_name, len(snap), snap.fingerprint,
    )

    instance = trainer_class()
    try:
        data = instance.prepare(app_config, list(snap.tickers), cfg)
        artifact = instance.fit(data, cfg)
    except Exception as exc:
        logger.exception("Training failed for strategy=%s trainer=%s", strategy, trainer_name)
        return TrainingRun(
            strategy=strategy,
            trainer=trainer_name,
            config=cfg,
            artifact=TrainingArtifact(state_dict={}),
            universe=snap,
            checkpoint_path=None,
            duration_seconds=time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
        )

    # Record what the run actually saw, so a checkpoint can be traced back to
    # its sample without consulting a log.
    artifact.metadata.setdefault("universe_fingerprint", snap.fingerprint)
    artifact.metadata.setdefault("universe_name", snap.name)
    artifact.metadata.setdefault("n_tickers", len(snap))

    checkpoint: Optional[Path] = None
    if instance.writes_own_checkpoint:
        # `fit` already persisted; report where, do not write over it.
        checkpoint = checkpoint_path_for(
            strategy, trainer_name, models_dir=models_dir,
            model_name=model_name or artifact.metadata.get("model_architecture"),
            suffix=instance.checkpoint_suffix,
        )
    elif save:
        checkpoint = instance.write_checkpoint(
            artifact,
            checkpoint_path_for(
                strategy, trainer_name, models_dir=models_dir,
                model_name=model_name, suffix=instance.checkpoint_suffix,
            ),
        )

    return TrainingRun(
        strategy=strategy,
        trainer=trainer_name,
        config=cfg,
        artifact=artifact,
        universe=snap,
        checkpoint_path=checkpoint,
        duration_seconds=time.monotonic() - started,
    )
