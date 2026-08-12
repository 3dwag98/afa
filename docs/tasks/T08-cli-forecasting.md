# T08 — CLI surface for the forecasting workflow

**Status:** not started · **Effort:** ~3 days · **Depends on:** T02, T04, T07
**Plan reference:** `docs/forecasting_plan.html` Part 6 (CLI)

## Goal

Make the forecasting workflow the path of least resistance, and give the CLI
its first tests.

## Why

`cli.py` is 873 lines with **zero test mentions** — the only surface anyone
touches, and the one place with no test at all. Three of the bugs found while
building the training layer were in exactly this shape of code: plumbing that
looked obviously correct.

## Commands

| Command | Status | Purpose |
| --- | --- | --- |
| `evaluate` | new, primary | Forecast skill for one strategy |
| `compare` | new | Several strategies, one panel, one table |
| `data build` / `validate` / `status` | new | Replaces and extends `download-data` |
| `report` | new | Render a run manifest to standalone HTML |
| `list-features` | new | The one registry with no inspection command |
| `train`, `train-bulk` | unchanged | Already correct |
| `backtest` | demoted | Kept, no longer the default path |
| `run-agent` | removed | Live path, frozen |

## `evaluate`

```
portfolio-agent evaluate --strategy lstm
  --horizons 1,5,10,21          decay curve
  --cv purged|walkforward       default purged
  --embargo N
  --neutralize sector,beta,size
  --by-regime
  --baseline gbm
  --universe-snapshot PATH
  --set key=value
  --output DIR
```

Two design choices to preserve:

- **`--cv` defaults to purged.** The correct method should be what you get by
  not thinking about it; the leaky one should require asking for it.
- **`--baseline` is a flag, not a separate run.** "Better than gradient
  boosting on identical splits" is the claim that matters, and one flag means
  nobody skips it.

## Smaller additions

`--dry-run` (resolve config, print, exit), `--seed`, `--limit N`,
`--fail-on-warning` on validate, `--keep-raw` and `--years` on data build.

**Deliberately not added:** a `--quick` preset that loosens CV or shortens the
sample. Presets that trade correctness for speed get used by default, and the
resulting numbers are indistinguishable from real ones in every report they
appear in. `--limit` narrows the universe honestly and visibly.

## Acceptance criteria

- [ ] Every subcommand has a `--help` smoke test.
- [ ] Every subcommand has one end-to-end invocation against synthetic data.
- [ ] Unknown options fail with a message naming the valid ones.
- [ ] `--json` output parses for every command that produces results.
