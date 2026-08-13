"""Run manifests: what produced a result, recorded when it was produced.

    portfolio-agent report --list             # every run, newest first
    portfolio-agent report --run 20260812     # a run's note, by id or prefix

A manifest is one JSON file per run carrying the config hash, universe
fingerprint, code revision and dirty flag, data fingerprint, resolved settings,
split, metrics and timings. `report.py` renders one as a standalone HTML page
with no external requests — including the decay chart, which is inline SVG for
the same reason.

Rendering re-reads the manifest and computes nothing, so a note and the run it
describes can never disagree.
"""

from .manifest import (
    DEFAULT_RUNS_DIR,
    RunManifest,
    build_manifest,
    config_fingerprint,
    data_fingerprint,
    find_manifest,
    git_state,
    library_versions,
    list_manifests,
    load_manifest,
    new_run_id,
    stable_hash,
)
from .report import render_index, render_note, write_note

__all__ = [
    "DEFAULT_RUNS_DIR",
    "RunManifest",
    "build_manifest",
    "config_fingerprint",
    "data_fingerprint",
    "find_manifest",
    "git_state",
    "library_versions",
    "list_manifests",
    "load_manifest",
    "new_run_id",
    "render_index",
    "render_note",
    "stable_hash",
    "write_note",
]
