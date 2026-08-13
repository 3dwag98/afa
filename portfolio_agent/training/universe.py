"""Pinned ticker sets, so two runs are comparable.

Comparing two strategies is only meaningful when they saw the same names.
`resolve_backtest_universe` draws from whatever happens to be in the parquet
cache, and its `purpose` argument deliberately offsets the seed so training and
backtesting sample *differently*. Both behaviours are right for a single run
and wrong for a comparison: train model A on Monday and model B on Tuesday
after a data sync, and the two are fitted on different universes, so the gap
between their scores is partly just the gap between their samples.

A snapshot fixes the universe once and hands the same list to every later run —
each notebook cell, each entry in a bulk sweep. It carries a content hash, so a
run can *assert* it trained on the intended names rather than trusting that it
did.

    from portfolio_agent.training.universe import UniverseSnapshot

    snap = UniverseSnapshot.create(config, size=50, name="compare-2026q1")
    snap.save("universe/compare-2026q1.json")
    ...
    snap = UniverseSnapshot.load("universe/compare-2026q1.json")
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: Default home for snapshot files. Kept out of `models/` so a snapshot is not
#: mistaken for a trained artifact and deleted with one.
DEFAULT_SNAPSHOT_DIR = Path("universe")


def _fingerprint(tickers: Sequence[str]) -> str:
    """A short, stable hash of a ticker set.

    Order-insensitive by construction: two runs that resolved the same names in
    a different order are the same universe, and should compare equal.
    """
    joined = "\n".join(sorted(tickers))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


@dataclass
class UniverseSnapshot:
    """A frozen list of tickers plus how it was drawn.

    Attributes:
        tickers: The names, sorted. Sorting is what makes the snapshot's
            identity independent of the cache's directory order.
        name: Human label, used in bulk-run reports.
        created_at: ISO timestamp of when the draw happened.
        source: How it was produced ("cache", "explicit", "file").
        params: The draw parameters, recorded so the sample can be explained
            later even though it will not be redrawn.
    """

    tickers: List[str]
    name: str = "default"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    source: str = "cache"
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tickers:
            raise ValueError(
                "a universe snapshot needs at least one ticker — an empty draw "
                "usually means the parquet cache is empty, so run "
                "`portfolio-agent download-data` first"
            )
        # Deduplicate and sort here rather than trusting callers, so the
        # fingerprint is a property of the set and not of the call site.
        self.tickers = sorted(set(self.tickers))

    @property
    def fingerprint(self) -> str:
        """Short content hash — equal iff two snapshots hold the same names."""
        return _fingerprint(self.tickers)

    def __len__(self) -> int:
        return len(self.tickers)

    @classmethod
    def create(
        cls,
        app_config: Any,
        *,
        size: Optional[int] = None,
        name: str = "default",
        selection: Optional[str] = None,
        seed: Optional[int] = None,
        purpose: str = "train",
    ) -> "UniverseSnapshot":
        """Draw a universe from the local parquet cache and freeze it.

        Args:
            app_config: Loaded AppConfig, supplying the `data:` defaults.
            size: How many names to keep. None uses `data.universe_size`.
            name: Label for reports.
            selection: "alphabetical" or "random"; None uses the config value.
            seed: Base seed for a random draw; None uses the config value.
            purpose: Passed through to `resolve_backtest_universe`, which
                offsets the seed so training and backtesting draw different
                samples of the same cache.
        """
        from portfolio_agent.src.universe import resolve_backtest_universe

        data_cfg = getattr(app_config, "data", None)
        resolved_size = size if size is not None else getattr(data_cfg, "universe_size", None)
        resolved_selection = selection or getattr(data_cfg, "universe_selection", "alphabetical")
        resolved_seed = seed if seed is not None else getattr(data_cfg, "universe_seed", 42)

        tickers = resolve_backtest_universe(
            force_full_download=False,
            max_tickers=resolved_size,
            selection=resolved_selection,
            seed=resolved_seed,
            purpose=purpose,
        )
        return cls(
            tickers=list(tickers),
            name=name,
            source="cache",
            params={
                "size": resolved_size,
                "selection": resolved_selection,
                "seed": resolved_seed,
                "purpose": purpose,
            },
        )

    @classmethod
    def from_tickers(cls, tickers: Sequence[str], name: str = "explicit") -> "UniverseSnapshot":
        """Freeze an explicitly supplied list — the notebook path."""
        return cls(tickers=list(tickers), name=name, source="explicit")

    def save(self, path: Path | str) -> Path:
        """Write the snapshot to JSON, creating parent directories."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "created_at": self.created_at,
            "source": self.source,
            "params": self.params,
            "fingerprint": self.fingerprint,
            "tickers": self.tickers,
        }
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2)
        logger.info(
            "Saved universe snapshot %r (%d tickers, fingerprint=%s) to %s",
            self.name, len(self.tickers), self.fingerprint, path,
        )
        return path

    @classmethod
    def load(cls, path: Path | str) -> "UniverseSnapshot":
        """Read a snapshot back, verifying it has not been edited in place.

        A mismatched fingerprint is a warning rather than an error: hand-editing
        a snapshot to drop a delisted name is legitimate. Silence would not be —
        the file is the evidence that two runs were comparable.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No universe snapshot at {path}")

        with open(path, "r") as handle:
            payload = json.load(handle)

        snapshot = cls(
            tickers=list(payload.get("tickers", [])),
            name=payload.get("name", path.stem),
            created_at=payload.get("created_at", ""),
            source=payload.get("source", "file"),
            params=dict(payload.get("params", {})),
        )
        recorded = payload.get("fingerprint")
        if recorded and recorded != snapshot.fingerprint:
            logger.warning(
                "Universe snapshot %s was edited after it was written "
                "(recorded fingerprint %s, actual %s). Runs using it are still "
                "comparable with each other, but not with earlier runs.",
                path, recorded, snapshot.fingerprint,
            )
        return snapshot


def resolve_universe(
    app_config: Any,
    *,
    tickers: Optional[Sequence[str]] = None,
    snapshot: Optional[Path | str] = None,
    size: Optional[int] = None,
    name: str = "default",
    purpose: Optional[str] = None,
) -> UniverseSnapshot:
    """Get the universe for a run, from the most specific source available.

    Order: an explicit ticker list, then a saved snapshot file, then a fresh
    draw from the cache. Bulk runs and notebooks pass a snapshot so every
    entry in a comparison sees identical names.

    Args:
        purpose: Which draw to take when falling through to a fresh one —
            `TRAINING_PURPOSE` for anything that fits a model,
            `MEASUREMENT_PURPOSE` for anything that scores one. **Required in
            spirit, optional in signature only because every existing training
            caller means the former.**

            This defaulted silently to `"train"`, so `evaluate` — which never
            passed it — scored the training draw while `backtest` traded the
            measurement draw. At `universe_size=50` the two shared 6 names.
            A default that is right for one caller and wrong for another is
            how that happens, so the parameter is now explicit at every call
            site that is not fitting a model.
    """
    from portfolio_agent.src.universe import TRAINING_PURPOSE

    if tickers:
        return UniverseSnapshot.from_tickers(tickers, name=name)
    if snapshot:
        return UniverseSnapshot.load(snapshot)
    return UniverseSnapshot.create(
        app_config, size=size, name=name, purpose=purpose or TRAINING_PURPOSE
    )
