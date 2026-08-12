# T09 — Run manifests and rendered research notes

**Status:** not started · **Effort:** ~2 days · **Depends on:** T04
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

- [ ] Every `evaluate` and `train` run writes a manifest.
- [ ] Two runs of one manifest's configuration reproduce its metrics exactly.
- [ ] `report --run ID` renders the note without re-running anything.
- [ ] A dirty working tree is recorded as such, so a result from uncommitted
      code cannot be mistaken for a reproducible one.
