"""Training several configurations in one go, on one universe.

Two shapes of bulk run, because they answer different questions:

- **A list of jobs.** Train everything that needs training — one strategy per
  entry, each with its own trainer and knobs.
- **A sweep.** One strategy, one trainer, the cross product of some
  hyperparameter grid.

Both pin the universe *once* and hand the same snapshot to every job. That is
the whole point: a comparison table whose rows were fitted on different draws
from the cache measures the draws as much as the models, and nothing in the
resulting numbers tells you which is which.

A failing job is recorded and the sweep continues. Losing nine good results
because the tenth had a bad hyperparameter is a worse outcome than a table with
one `failed` row in it.

    from portfolio_agent.training.bulk import BulkJob, run_bulk

    report = run_bulk(config, [
        BulkJob(strategy="india_sac", overrides={"epochs": 50}),
        BulkJob(strategy="india_sac", overrides={"epochs": 50, "gamma": 0.9}),
    ], universe_size=40)
    print(report.to_frame())
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .runner import DEFAULT_MODELS_DIR, TrainingRun, run_training_job
from .universe import UniverseSnapshot, resolve_universe

logger = logging.getLogger(__name__)


@dataclass
class BulkJob:
    """One entry in a bulk run.

    Attributes:
        strategy: Strategy to train. None runs the supervised default.
        trainer: Explicit trainer, overriding the strategy's declaration.
        overrides: Hyperparameters for this entry only.
        label: Name for the report row. Defaults to a readable rendering of
            the strategy, trainer and whatever overrides distinguish it.
        model_name: Checkpoint basename. Sweeps set this per entry, otherwise
            every entry overwrites the same file and only the last survives.
        save: Whether to write a checkpoint at all.
    """

    strategy: Optional[str] = None
    trainer: Optional[str] = None
    overrides: Dict[str, Any] = field(default_factory=dict)
    label: Optional[str] = None
    model_name: Optional[str] = None
    save: bool = True

    def resolved_label(self) -> str:
        if self.label:
            return self.label
        parts = [self.strategy or "supervised"]
        if self.trainer:
            parts.append(self.trainer)
        if self.overrides:
            parts.append(
                ",".join(f"{k}={v}" for k, v in sorted(self.overrides.items()))
            )
        return " | ".join(parts)


@dataclass
class BulkReport:
    """Every run in a bulk execution, plus the universe they shared."""

    runs: List[TrainingRun]
    labels: List[str]
    universe: UniverseSnapshot

    @property
    def failures(self) -> List[str]:
        return [label for label, run in zip(self.labels, self.runs) if not run.ok]

    def rows(self) -> List[Dict[str, Any]]:
        """Flat summary rows, one per job."""
        return [
            {"label": label, **run.summary()}
            for label, run in zip(self.labels, self.runs)
        ]

    def to_frame(self):
        """The report as a DataFrame, best metric first.

        Successful runs sort above failed ones regardless of metric, so a
        failure never lands at the top of a table someone skims. Note the sort
        key is an explicit boolean rather than the `status` column: "failed"
        precedes "ok" alphabetically, so sorting on the text puts the failures
        exactly where they must not be.
        """
        import pandas as pd

        frame = pd.DataFrame(self.rows())
        if "status" not in frame.columns:
            return frame

        frame = frame.assign(_failed=(frame["status"] != "ok"))
        sort_columns = ["_failed"]
        ascending = [True]
        if "metric" in frame.columns:
            sort_columns.append("metric")
            ascending.append(False)

        return (
            frame.sort_values(by=sort_columns, ascending=ascending, kind="stable")
            .drop(columns="_failed")
            .reset_index(drop=True)
        )

    def best(self) -> Optional[TrainingRun]:
        """The successful run with the highest primary metric, if any."""
        scored = [
            (run.artifact.primary_metric(), run)
            for run in self.runs
            if run.ok and run.artifact.primary_metric() is not None
        ]
        if not scored:
            return None
        return max(scored, key=lambda pair: pair[0])[1]


def run_bulk(
    app_config: Any,
    jobs: Sequence[BulkJob],
    *,
    universe: Optional[Sequence[str]] = None,
    snapshot: Optional[Path | str] = None,
    universe_size: Optional[int] = None,
    universe_name: str = "bulk",
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    save_snapshot_to: Optional[Path | str] = None,
) -> BulkReport:
    """Run every job against one pinned universe.

    Args:
        app_config: Loaded AppConfig.
        jobs: What to train.
        universe: Explicit ticker list, pinned for every job.
        snapshot: Saved snapshot to load instead.
        universe_size: Size for a fresh draw when neither is given.
        universe_name: Label for the drawn universe.
        models_dir: Checkpoint destination.
        save_snapshot_to: Write the resolved snapshot here, so a later run can
            reproduce the comparison exactly.

    Returns:
        A `BulkReport`.
    """
    if not jobs:
        raise ValueError("run_bulk needs at least one job")

    snap = resolve_universe(
        app_config,
        tickers=universe,
        snapshot=snapshot,
        size=universe_size,
        name=universe_name,
    )
    if save_snapshot_to:
        snap.save(save_snapshot_to)

    logger.info(
        "Bulk run: %d jobs on %d tickers (universe %s)",
        len(jobs), len(snap), snap.fingerprint,
    )

    runs: List[TrainingRun] = []
    labels: List[str] = []

    for index, job in enumerate(jobs, start=1):
        label = job.resolved_label()
        labels.append(label)
        logger.info("[%d/%d] %s", index, len(jobs), label)

        run = run_training_job(
            app_config,
            job.strategy,
            trainer=job.trainer,
            overrides=job.overrides,
            # The snapshot's tickers, not the snapshot path: every job must see
            # this exact list even if the file is rewritten mid-sweep.
            universe=snap.tickers,
            models_dir=models_dir,
            model_name=job.model_name,
            save=job.save,
        )
        runs.append(run)

        if not run.ok:
            logger.warning("[%d/%d] %s failed: %s", index, len(jobs), label, run.error)

    report = BulkReport(runs=runs, labels=labels, universe=snap)
    if report.failures:
        logger.warning(
            "%d/%d jobs failed: %s", len(report.failures), len(jobs), report.failures
        )
    return report


def sweep(
    strategy: Optional[str],
    grid: Mapping[str, Iterable[Any]],
    *,
    trainer: Optional[str] = None,
    base_overrides: Optional[Mapping[str, Any]] = None,
    save: bool = False,
) -> List[BulkJob]:
    """Expand a hyperparameter grid into jobs.

    Args:
        strategy: Strategy each job trains.
        grid: Field name -> values to try. The cross product is taken.
        trainer: Explicit trainer for every job.
        base_overrides: Settings held fixed across the sweep.
        save: Whether jobs write checkpoints. Defaults to False — a sweep is
            usually asking which settings to use, not producing the model, and
            writing every combination to disk is rarely what was wanted. Each
            job that does save gets a distinct `model_name` so they cannot
            overwrite one another.

    Returns:
        One `BulkJob` per grid point.
    """
    if not grid:
        raise ValueError("sweep needs at least one hyperparameter")

    keys = list(grid)
    jobs: List[BulkJob] = []

    for values in itertools.product(*(list(grid[key]) for key in keys)):
        overrides = dict(base_overrides or {})
        point = dict(zip(keys, values))
        overrides.update(point)

        suffix = "_".join(f"{k}{v}" for k, v in point.items())
        jobs.append(
            BulkJob(
                strategy=strategy,
                trainer=trainer,
                overrides=overrides,
                label=" | ".join(
                    [strategy or "supervised", ",".join(f"{k}={v}" for k, v in point.items())]
                ),
                model_name=f"{strategy or 'supervised'}_{suffix}" if save else None,
                save=save,
            )
        )
    return jobs
