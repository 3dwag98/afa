"""What produced a result, recorded at the moment it was produced.

The gap this fills
------------------
Training writes a checkpoint. Evaluation finds it by filename convention.
Nothing anywhere records that *this* metric came from *that* model, fitted on
*that* universe, with *those* settings, against *that* data, from *that*
commit. Six weeks later the only way to find out is to reconstruct it from a
shell history that no longer exists, and the usual outcome is that a number
gets quoted with no way to check it.

The universe fingerprint added with the training layer was the first piece of
this, and it stopped at the checkpoint. A manifest carries it the rest of the
way.

Why this is small
-----------------
Under the forecasting premise a run is `(features, labels, model, split) ->
metrics` — no order book, no fills, no portfolio state. So "everything needed
to reproduce this" is a config hash, a universe fingerprint, a code revision, a
data fingerprint, the resolved settings and the metrics. That fits in a JSON
file, which is why this is a file format rather than a database.

The dirty flag is not decoration
--------------------------------
A result produced from uncommitted code cannot be reproduced by anyone,
including the person who produced it, and it is indistinguishable from a
reproducible one unless something says so. `git_dirty` is recorded on every
manifest, and the rendered note says it in the header rather than in a footnote
— because the entire value of a provenance record is that the bad case is
visible without looking for it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

#: Where manifests land. One JSON file per run, named by run id, so the
#: directory is browsable and a manifest can be moved or mailed on its own.
DEFAULT_RUNS_DIR = Path("runs")

#: Libraries whose version can change a number. Recorded because "the metric
#: moved and nothing changed" is almost always one of these.
TRACKED_LIBRARIES = ("numpy", "pandas", "scipy", "torch", "scikit-learn", "pyarrow")


def _git(*args: str) -> Optional[str]:
    """Run a git command, returning None when git or the repo is unavailable."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True, text=True, timeout=10,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - env dependent
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_state() -> Dict[str, Any]:
    """Commit, branch and whether the tree had uncommitted changes.

    `dirty` is the field that matters. Everything else is context; that one is
    the difference between a result someone else can reproduce and a result
    that exists only on one machine.
    """
    revision = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": revision,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # None, not False, when git could not be consulted: "we did not check"
        # and "we checked and it was clean" are different claims, and only one
        # of them supports reproducing anything.
        "dirty": None if status is None else bool(status),
        "dirty_files": (
            [line[3:] for line in status.splitlines()[:20]] if status else []
        ),
    }


def library_versions(names: Sequence[str] = TRACKED_LIBRARIES) -> Dict[str, str]:
    """Installed versions of the libraries that can move a metric."""
    from importlib.metadata import PackageNotFoundError, version

    found: Dict[str, str] = {}
    for name in names:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            continue
    return found


def stable_hash(payload: Any, length: int = 12) -> str:
    """A short, stable hash of any JSON-serializable structure.

    `sort_keys` is what makes it stable: two configs that differ only in the
    order pydantic happened to emit their fields must hash the same, or the
    fingerprint reports a change that did not happen.
    """
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def config_fingerprint(app_config: Any) -> str:
    """Hash of the resolved application config."""
    if hasattr(app_config, "model_dump"):
        payload = app_config.model_dump(mode="json")
    elif isinstance(app_config, Mapping):
        payload = dict(app_config)
    else:  # pragma: no cover - defensive
        payload = str(app_config)
    return stable_hash(payload)


def data_fingerprint(
    tickers: Sequence[str], cache_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Identify the bars a run actually read.

    Hashes each file's size and modification time rather than its contents. A
    content hash over 2,400 parquet files costs seconds per run and buys
    precision nobody needs — what this has to catch is "the cache was
    refreshed between these two runs", and mtime plus size catches that.

    Stated plainly so nobody assumes more: this detects a *changed* cache, not
    a cache whose contents were rewritten byte-identically at the same
    timestamp.
    """
    from portfolio_agent.src.data_store import DATA_DIR, _ticker_filename

    directory = Path(cache_dir) if cache_dir is not None else DATA_DIR
    entries: List[Any] = []
    missing = 0
    for ticker in sorted(tickers):
        path = directory / _ticker_filename(ticker)
        try:
            stat = path.stat()
        except OSError:
            missing += 1
            continue
        entries.append([ticker, stat.st_size, int(stat.st_mtime)])

    return {
        "cache_dir": str(directory),
        "n_symbols": len(entries),
        "n_missing": missing,
        "fingerprint": stable_hash(entries),
        "method": "size+mtime per file, not a content hash",
    }


@dataclass
class RunManifest:
    """Everything needed to explain, compare and reproduce one run.

    Deliberately a flat-ish dataclass rather than a nested schema: a manifest
    is read by a human six weeks later at least as often as by code, and a
    two-level JSON file is readable in a terminal.
    """

    run_id: str
    kind: str                       # "train" | "evaluate" | other
    created_at: str
    strategy: Optional[str] = None
    trainer: Optional[str] = None

    config_fingerprint: Optional[str] = None
    universe_fingerprint: Optional[str] = None
    universe_name: Optional[str] = None
    n_symbols: int = 0

    settings: Dict[str, Any] = field(default_factory=dict)
    split: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)

    git: Dict[str, Any] = field(default_factory=git_state)
    data: Dict[str, Any] = field(default_factory=dict)
    libraries: Dict[str, str] = field(default_factory=library_versions)
    environment: Dict[str, str] = field(default_factory=lambda: {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    })
    notes: List[str] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def reproducible(self) -> bool:
        """Whether this run could be reproduced from what is recorded.

        False for a dirty tree and for an unknown git state alike. The
        conservative reading is the right one: a manifest that cannot say the
        code was committed cannot promise the result can be recreated.
        """
        return self.git.get("dirty") is False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, runs_dir: Path | str = DEFAULT_RUNS_DIR) -> Path:
        """Write `<runs_dir>/<run_id>.json`."""
        directory = Path(runs_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.run_id}.json"
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2, default=str)
        logger.info("Wrote run manifest %s", path)
        return path

    def summary(self) -> Dict[str, Any]:
        """One row for a table of runs."""
        row: Dict[str, Any] = {
            "run_id": self.run_id,
            "kind": self.kind,
            "created_at": self.created_at,
            "strategy": self.strategy or "-",
            "trainer": self.trainer or "-",
            "symbols": self.n_symbols,
            "universe": self.universe_fingerprint or "-",
            "reproducible": self.reproducible,
        }
        for key, value in self.metrics.items():
            if isinstance(value, (int, float, bool)):
                row[key] = value
        return row

    def render(self) -> str:
        """A plain-text summary, for a terminal."""
        lines = [
            f"Run {self.run_id}  ({self.kind})",
            "=" * 62,
            f"  created     {self.created_at}",
            f"  strategy    {self.strategy or '-'}"
            + (f"   trainer {self.trainer}" if self.trainer else ""),
            f"  universe    {self.universe_fingerprint or '-'} "
            f"({self.n_symbols} symbols, {self.universe_name or 'unnamed'})",
            f"  config      {self.config_fingerprint or '-'}",
        ]
        commit = (self.git.get("commit") or "unknown")[:12]
        if self.git.get("dirty") is True:
            lines.append(
                f"  code        {commit} on {self.git.get('branch') or '?'}  "
                f"** DIRTY WORKING TREE — this result cannot be reproduced **"
            )
        elif self.git.get("dirty") is None:
            lines.append(f"  code        {commit}  (git state unknown)")
        else:
            lines.append(f"  code        {commit} on {self.git.get('branch') or '?'}")

        if self.data:
            missing = self.data.get("n_missing", 0)
            # The missing count is not a footnote: a fingerprint over zero
            # files is a perfectly valid hash of nothing, and without this it
            # looks exactly like a fingerprint over a cache that was read.
            suffix = f", {missing} not found in the cache" if missing else ""
            lines.append(
                f"  data        {self.data.get('fingerprint', '-')} "
                f"({self.data.get('n_symbols', 0)} files{suffix})"
            )
        if self.split:
            rendered = ", ".join(f"{k}={v}" for k, v in self.split.items())
            lines.append(f"  split       {rendered}")

        if self.metrics:
            lines += ["", "  Metrics"]
            for key, value in sorted(self.metrics.items()):
                if isinstance(value, float):
                    lines.append(f"    {key:<28}{value:>14.6f}")
                elif isinstance(value, (int, bool)):
                    lines.append(f"    {key:<28}{value:>14}")

        if self.timings:
            rendered = "  ".join(f"{k} {v:.1f}s" for k, v in self.timings.items())
            lines += ["", f"  Timings     {rendered}"]

        if self.artifacts:
            lines += ["", "  Artifacts"]
            for key, value in self.artifacts.items():
                lines.append(f"    {key:<14}{value}")

        for note in self.notes:
            lines += ["", f"  Note: {note}"]
        return "\n".join(lines)


def new_run_id(kind: str, strategy: Optional[str] = None) -> str:
    """A sortable, human-readable, collision-resistant run id.

    Timestamp first so `ls runs/` is chronological, then what it was, then four
    random hex characters — two runs of one configuration started in the same
    second are a normal thing to do when comparing seeds, and a timestamp alone
    would have one overwrite the other.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    parts = [stamp, kind]
    if strategy:
        parts.append("".join(c for c in strategy if c.isalnum() or c in "-_")[:24])
    parts.append(os.urandom(2).hex())
    return "-".join(parts)


def build_manifest(
    kind: str,
    *,
    app_config: Any = None,
    strategy: Optional[str] = None,
    trainer: Optional[str] = None,
    universe: Optional[Sequence[str]] = None,
    universe_fingerprint: Optional[str] = None,
    universe_name: Optional[str] = None,
    settings: Optional[Mapping[str, Any]] = None,
    split: Optional[Mapping[str, Any]] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    timings: Optional[Mapping[str, float]] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
    notes: Optional[Sequence[str]] = None,
    extras: Optional[Mapping[str, Any]] = None,
    cache_dir: Optional[Path] = None,
) -> RunManifest:
    """Assemble a manifest, filling in everything derivable from the environment.

    Args:
        kind: "train", "evaluate", or another short label.
        app_config: Loaded AppConfig, hashed into `config_fingerprint`.
        universe: Tickers the run used, fingerprinted against the cache.
        settings: Resolved hyperparameters or evaluation arguments.
        split: The CV scheme — horizon, embargo, folds — so a metric is never
            separated from the split that produced it.
        artifacts: Paths written, e.g. `{"checkpoint": "models/x_best.pt"}`.
    """
    manifest = RunManifest(
        run_id=new_run_id(kind, strategy),
        kind=kind,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        strategy=strategy,
        trainer=trainer,
        config_fingerprint=config_fingerprint(app_config) if app_config is not None else None,
        universe_fingerprint=universe_fingerprint,
        universe_name=universe_name,
        n_symbols=len(universe) if universe is not None else 0,
        settings=_plain(settings or {}),
        split=_plain(split or {}),
        metrics=_plain(metrics or {}),
        timings={k: float(v) for k, v in (timings or {}).items()},
        artifacts={k: str(v) for k, v in (artifacts or {}).items()},
        notes=list(notes or []),
        extras=_plain(extras or {}),
    )
    if universe:
        manifest.data = data_fingerprint(universe, cache_dir)
    if manifest.git.get("dirty"):
        manifest.notes.append(
            "The working tree had uncommitted changes when this ran, so the "
            "code that produced these numbers is not recoverable from the "
            "recorded commit."
        )
    return manifest


def _plain(value: Any) -> Any:
    """Convert to something `json.dump` handles without a custom encoder.

    Manifests are read by other tools, so a numpy float64 that serializes only
    under `default=str` and comes back as the string "0.042" is a landmine.
    """
    import numpy as np

    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(mode="json"))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def load_manifest(path: Path | str) -> RunManifest:
    """Read a manifest written by `RunManifest.save`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No manifest at {path}")
    with open(path, "r") as handle:
        payload = json.load(handle)
    known = {f for f in RunManifest.__dataclass_fields__}
    return RunManifest(**{k: v for k, v in payload.items() if k in known})


def find_manifest(
    run_id: str, runs_dir: Path | str = DEFAULT_RUNS_DIR
) -> RunManifest:
    """Look up a manifest by run id, or by a unique prefix of one.

    Prefix matching because run ids are long by design and nobody retypes a
    timestamp. An ambiguous prefix raises and lists the candidates rather than
    picking one — silently rendering the wrong run's note would defeat the
    purpose of having manifests at all.
    """
    directory = Path(runs_dir)
    exact = directory / f"{run_id}.json"
    if exact.exists():
        return load_manifest(exact)

    matches = sorted(directory.glob(f"{run_id}*.json")) if directory.exists() else []
    if not matches:
        raise FileNotFoundError(
            f"No run matching {run_id!r} in {directory}. "
            f"List what is there with `portfolio-agent report --list`."
        )
    if len(matches) > 1:
        names = [p.stem for p in matches[:10]]
        raise ValueError(
            f"{run_id!r} matches {len(matches)} runs: {names}. Use a longer prefix."
        )
    return load_manifest(matches[0])


def list_manifests(
    runs_dir: Path | str = DEFAULT_RUNS_DIR, limit: Optional[int] = None
) -> List[RunManifest]:
    """Every manifest in `runs_dir`, newest first."""
    directory = Path(runs_dir)
    if not directory.exists():
        return []
    paths = sorted(directory.glob("*.json"), reverse=True)
    if limit is not None:
        paths = paths[:limit]

    manifests: List[RunManifest] = []
    for path in paths:
        try:
            manifests.append(load_manifest(path))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable manifest %s: %s", path, exc)
    return manifests
