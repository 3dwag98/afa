"""Append-only record of every backtest configuration that has been run.

The Deflated Sharpe Ratio needs two numbers this platform did not previously
have: N, the number of configurations tried, and V[SR], the variance of the
Sharpe estimates across them. Neither can be reconstructed after the fact —
runs that were tried and abandoned leave no trace, and those are exactly the
trials that inflate the maximum. So they have to be recorded as they happen.

This is a 100-line file with more scientific value than any model in the
repository: without it, a reported Sharpe is a number with no denominator.

Format is JSONL — one object per run, appended, never rewritten. That makes it
safe under concurrent runs, trivially greppable, and impossible to
accidentally "clean up" into a summary that has lost the failures.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TRIAL_LOG = "output/trials.jsonl"


@dataclass
class Trial:
    """One backtest run, keyed by the configuration that produced it."""

    run_id: str
    timestamp: str
    config_hash: str
    sharpe: float
    # The parameters that were varied for this run, so a reader can tell which
    # dimension of the search space a trial explored without diffing configs.
    parameters: Dict[str, Any] = field(default_factory=dict)
    cagr: Optional[float] = None
    max_drawdown: Optional[float] = None
    n_observations: Optional[int] = None
    n_trades: Optional[int] = None
    strategy: Optional[str] = None
    notes: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=str)


def config_hash(config: Any) -> str:
    """Stable short hash of a configuration.

    Two runs with the same hash are the same trial and should not both count
    toward N — re-running an identical backtest is not a new experiment. The
    hash is over the serialized config, sorted, so key ordering cannot make
    identical configurations look different.
    """
    if hasattr(config, "model_dump"):
        payload = config.model_dump(mode="json")
    elif isinstance(config, dict):
        payload = config
    else:
        payload = {"repr": repr(config)}
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def record_trial(trial: Trial, path: str = DEFAULT_TRIAL_LOG) -> bool:
    """Append one trial. Never raises — a failed write must not lose a backtest.

    Returns:
        True when the line was written.
    """
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Append mode with a single write() call: on POSIX, writes under
        # PIPE_BUF to a file opened O_APPEND are atomic, so parallel runs
        # interleave whole lines rather than corrupting each other's.
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(trial.to_json() + os.linesep)
        return True
    except OSError:
        logger.warning("Could not append to the trial log at %s", path, exc_info=True)
        return False


def read_trials(path: str = DEFAULT_TRIAL_LOG) -> List[Trial]:
    """Every trial recorded so far. Missing file reads as no trials."""
    return list(iter_trials(path))


def iter_trials(path: str = DEFAULT_TRIAL_LOG) -> Iterator[Trial]:
    """Stream trials, skipping any line that does not parse.

    A malformed line is a lost trial, not a lost log: a half-written record
    from a killed process must not make the whole history unreadable.
    """
    source = Path(path)
    if not source.exists():
        return

    known_fields = set(Trial.__dataclass_fields__)
    with open(source, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                yield Trial(**{k: v for k, v in payload.items() if k in known_fields})
            except (json.JSONDecodeError, TypeError):
                logger.debug("Skipping unparseable trial log line %d", line_number)


def trial_statistics(
    path: str = DEFAULT_TRIAL_LOG,
    deduplicate: bool = True,
) -> Dict[str, Any]:
    """The (N, V[SR]) pair the Deflated Sharpe Ratio needs.

    Args:
        path: Trial log to read.
        deduplicate: Count each distinct config_hash once. Re-running the same
            backtest is not an independent trial, and letting repeats inflate
            N would deflate the Sharpe for the wrong reason.

    Returns:
        Dict with n_trials, sharpe_variance, and the best/mean Sharpe seen.
        n_trials is 0 and sharpe_variance 0.0 on an empty or missing log,
        which callers should treat as "DSR is not computable yet" rather than
        as "there was one trial".
    """
    sharpes_by_hash: Dict[str, float] = {}
    all_sharpes: List[float] = []

    for trial in iter_trials(path):
        # A JSON null, a NaN or a non-numeric sharpe is an unusable record,
        # not a reason to abandon the whole log.
        try:
            sharpe = float(trial.sharpe)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(sharpe):
            continue
        all_sharpes.append(sharpe)
        # Keep the first result for a given configuration, so a re-run cannot
        # quietly replace a recorded number with a luckier one.
        sharpes_by_hash.setdefault(trial.config_hash, sharpe)

    sharpes = list(sharpes_by_hash.values()) if deduplicate else all_sharpes
    if not sharpes:
        return {
            "n_trials": 0,
            "sharpe_variance": 0.0,
            "best_sharpe": 0.0,
            "mean_sharpe": 0.0,
            "n_records": len(all_sharpes),
        }

    array = np.asarray(sharpes, dtype=float)
    return {
        "n_trials": int(array.size),
        # ddof=1: V[SR] is an estimate of the population dispersion of Sharpe
        # across the trials that *could* have been run, not a description of
        # the ones that were.
        "sharpe_variance": float(np.var(array, ddof=1)) if array.size > 1 else 0.0,
        "best_sharpe": float(array.max()),
        "mean_sharpe": float(array.mean()),
        "n_records": len(all_sharpes),
    }
