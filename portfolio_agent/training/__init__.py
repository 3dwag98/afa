"""Pluggable training: registered procedures, per-strategy configs, bulk runs.

The platform already made features, model architectures and strategies
pluggable. This package is the fourth seam — *how* a strategy is trained —
plus the two things that seam needs to be usable: hyperparameters that are
validated per trainer rather than pooled in one global block, and universes
that can be pinned so two runs are comparable.

    from portfolio_agent.config.loader import load_config
    from portfolio_agent.training import run_training_job

    run = run_training_job(load_config(), "india_sac", overrides={"epochs": 50})
    print(run.summary())

Nothing here imports PyTorch at module scope: `list_trainers()` and config
resolution work on installs without the `gpu` extra, which is what keeps
rule-based backtests torch-free.
"""

from .base import BaseTrainer, TrainerConfig, TrainingArtifact, TrainingData
from .bulk import BulkJob, BulkReport, run_bulk, sweep
from .config import resolve_training_config
from .registry import (
    get_trainer,
    is_trainer_registered,
    list_trainers,
    register_trainer,
)
from .runner import TrainingRun, run_training_job
from .universe import UniverseSnapshot, resolve_universe

__all__ = [
    "BaseTrainer",
    "BulkJob",
    "BulkReport",
    "TrainerConfig",
    "TrainingArtifact",
    "TrainingData",
    "TrainingRun",
    "UniverseSnapshot",
    "get_trainer",
    "is_trainer_registered",
    "list_trainers",
    "register_trainer",
    "resolve_training_config",
    "resolve_universe",
    "run_bulk",
    "run_training_job",
    "sweep",
]
