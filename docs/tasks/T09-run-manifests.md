# T09 — Run manifests and rendered research notes

**Status:** done · **Effort:** ~2 days · **Depends on:** T04
**Plan reference:** `docs/forecasting_plan.html` Architecture register A9; Part 4

## Goal

Make every result traceable to the code, config, data and universe that
produced it — and readable a month later.

## Why

Training writes a checkpoint; evaluation reads it by filename convention.
Nothing records that *this* result came from *that* model trained on *that*
universe with *those* settings. The universe fingerprint added with the
training layer is the first piece of this and currently stops at the
checkpoint.

Under the forecasting premise this is much smaller than it would have been: a
run is `(features, labels, model, split) -> metrics`, not a full simulation.

## Manifest contents

Config hash, universe fingerprint and name, code revision and dirty flag, data
store revision, trainer and its resolved settings, CV scheme with horizon and
embargo, metrics, timings, and the library versions that matter.

## Rendered note

Each manifest renders to a standalone HTML page: settings, metrics table, decay
curve, IC by regime, and the caveats that apply. Makes results comparable
across weeks and stops the accumulation of undocumented experiments nobody can
reproduce.

## Acceptance criteria

- [x] Every `evaluate` and `train` run writes a manifest — including failed
      training runs, which are exactly the ones worth reconstructing.
- [x] Two runs of one manifest's configuration reproduce its metrics exactly.
- [x] `report --run ID` renders the note without re-running anything.
- [x] A dirty working tree is recorded as such.

## Files

| File | Change |
| --- | --- |
| `portfolio_agent/provenance/manifest.py` | New — `RunManifest`, fingerprints, git state, lookup |
| `portfolio_agent/provenance/report.py` | New — standalone HTML note, inline-SVG decay chart |
| `portfolio_agent/provenance/__init__.py` | New — package surface |
| `portfolio_agent/training/runner.py` | Writes a manifest per run; `run_id` on `TrainingRun` |
| `portfolio_agent/evaluation/harness.py` | Writes a manifest per evaluation |
| `portfolio_agent/cli.py` | `report` command |
| `portfolio_agent/tests/test_run_manifests.py` | New — 48 tests |

## Three decisions

**An unknown git state is not clean.** `dirty` is tri-state — `True`, `False`,
or `None` when git could not be consulted — and `reproducible` requires an
explicit `False`. "We did not check" and "we checked and it was clean" are
different claims and only one of them supports reproducing anything.

**The dirty warning leads the note.** It sits above the metrics table, not in a
footer, and a test asserts that ordering. A number from uncommitted code is
otherwise indistinguishable from one that is reproducible, so burying the
warning would make the note complicit in the confusion it exists to prevent.
`report --run` exits **2** for such a run, so a script cannot quote it without
noticing; `--allow-dirty` opts out.

**Provenance never breaks a run.** Manifest writing is wrapped and swallowed on
both paths. A forty-minute training job must not be lost to a full disk in the
bookkeeping, and an evaluation that is already computed must not be thrown away
to protect a record of it.

## Notes on the format

The data fingerprint hashes each file's size and mtime, not its contents — it
has to catch "the cache was refreshed between these two runs", and a content
hash over 2,400 parquet files costs seconds per run for precision nobody needs.
The manifest says so in a `method` field rather than leaving the reader to
assume more.

The HTML note makes no external requests at all, including for the decay chart,
which is inline SVG drawn from the numbers. A research note that silently fails
to render when a CDN is unreachable is worse than a plain table.
