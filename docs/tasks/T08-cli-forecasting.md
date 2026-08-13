# T08 — CLI surface for the forecasting workflow

**Status:** done · **Effort:** ~3 days · **Depends on:** T02, T04, T07
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

- [x] Every subcommand has a `--help` smoke test — parametrized off the parser
      itself, so a new command that ships without one fails the file.
- [x] Every subcommand has one end-to-end invocation against synthetic data.
      Two exceptions, named in the test docstrings: `download-data` needs the
      network and `run-agent` drives the frozen live path, so both are asserted
      to parse and reach their first decision with their flags intact.
- [x] Unknown options fail with a message naming the valid ones.
- [x] `--json` output parses for every command that produces results.

## Files

| File | Change |
| --- | --- |
| `portfolio_agent/cli_forecast.py` | New — `evaluate`, `compare`, `list-features`, `data build` |
| `portfolio_agent/cli.py` | Registers them; `--config` ordering fix |
| `portfolio_agent/tests/test_cli.py` | New — 71 tests, the CLI's first |

## The bug the tests caught immediately

`--config` was threaded to the forecasting commands *before* the line that
applies it, so those commands silently loaded `config.yaml` no matter what was
passed. A global flag the parser accepts and one branch never reads — precisely
the shape of bug this task exists for, found by the first test written for it.

## Measured

`evaluate --strategy momentum --limit 120 --stride 10 --cv purged --embargo 2
--horizons 1,5,21 --neutralize beta,size --baseline low_volatility`, on the
real cache, produces in one command: rank IC with a Newey–West t, decile
profile and monotonicity, five purged folds, the raw-vs-neutralized split, a
decay curve, and this:

```
  momentum               +0.0606    4.77    +0.1129%        +0.217
  low_volatility         +0.0639    5.23    -0.1728%        -0.117

  momentum does NOT beat low_volatility (-0.0033 IC).
  On identical features and splits, the simpler model wins.
```

Which is the `--baseline` flag doing exactly the job it was added for.

`compare` also surfaced something worth knowing: `rule_based` scores
`score_dispersion` of 0.016 and zero rankable dates — it emits one floor value
for 98% of the universe, so there is no cross-section to rank. Its IC is not
low because its claims are wrong; it barely makes any.
