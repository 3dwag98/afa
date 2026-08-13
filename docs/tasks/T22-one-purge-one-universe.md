# T22 — One purge, one measurement universe

**Status:** done · **Effort:** ~1 day · **Depends on:** nothing
**Review reference:** round-three plan, finding C

## Goal

Overlap measured in sessions everywhere, and one universe for everything that
scores a strategy rather than fits one.

## Why

Two unrelated defects with the same shape: an expression restated in a second
place, where the copy was wrong in a way nothing could see.

### The embargo was in calendar days

`agents/trainer.py` computed `test_end_date + pd.Timedelta(days=embargo)`.
`validation/purged.py` opens by ruling that out:

> A horizon of 5 means five *trading sessions*, not five calendar days. A
> weekend or an exchange holiday would make a calendar-day arithmetic silently
> wrong by a variable amount, so everything here is computed on positions
> within a sorted date index.

An embargo of 5 spanning a weekend excluded **three** sessions. Spanning
Diwali, fewer. The setting was a no-op in practice — an expanding window never
produces training rows after its test fold — but it was kept precisely so it
would mean what it says once the scheme changes, and in the wrong units it
would not have.

### `evaluate` drew the training universe

`purpose` offsets the RNG so a model is not scored on the names it was fitted
on. The offset is two-way; `resolve_universe` defaulted to `"train"`, and
`evaluate` never overrode it. Measured on a 400-name cache at
`universe_size=50`, with the shipped `universe_selection: random`:

    evaluate (fell through to "train")  vs  backtest        6 names shared / 50

So an IC and an equity curve for "the same strategy" described substantially
different markets, and the platform printed them side by side.

## Approach

**`MEASUREMENT_PURPOSE` and `TRAINING_PURPOSE`** are named constants in
`src/universe.py`. The bug was a default argument nobody passed, and a shared
constant is the thing a reader notices is missing. Every scoring call site —
`harness`, `decay`, `neutralize`, `cli_forecast`, and now the backtest — passes
it explicitly; `resolve_universe` still defaults to training, because every
caller that does not pass one is fitting a model.

**The backtest accepts `--universe-snapshot`.** This is the structural fix and
the seed alignment is the fallback: pinning is the only way to put a backtest
and an evaluation on identical names, and `evaluate` and `train` have accepted
a snapshot since T09 while the backtest could not — so the two sides of "does
the measured IC show up in the equity curve" could not be pinned together even
in principle.

**The embargo is positional.** `frame.index[frame.index > test_end_date]`, then
index into it — the same convention `validation/purged.py` uses, clamped so an
embargo longer than the remaining history does not run off the end.

**`gbm`'s purge routes at `label_window_overlaps`.** It was
`max(cut - purge, 0)`; the shared predicate marks exactly the positions with
`p + horizon >= cut`. Verified identical across 45 combinations of
(n_dates, split, horizon) — but they agreed by coincidence of two people
deriving the same expression, which is the state T12 removed for rank IC and
T14 for the market composite.

## Acceptance criteria

- [x] Evaluation and backtest draw the same names; training still differs.
- [x] The old divergence recorded as large, so it is not re-introduced as a
      harmless default.
- [x] Every scoring path passes the purpose explicitly, asserted by source
      inspection.
- [x] `--universe-snapshot` on the backtest, and a test showing both sides pin
      to one fingerprint.
- [x] The embargo is in sessions; a weekend no longer shrinks it to three.
- [x] It clamps rather than raising when longer than the remaining history.
- [x] One overlap predicate, with the equivalence to the old arithmetic checked
      rather than assumed.

## Files

| File | Change |
| --- | --- |
| `portfolio_agent/src/universe.py` | `MEASUREMENT_PURPOSE`, `TRAINING_PURPOSE` |
| `portfolio_agent/training/universe.py` | `resolve_universe(purpose=...)` |
| `portfolio_agent/evaluation/{harness,decay,neutralize}.py`, `cli_forecast.py` | Pass the measurement purpose |
| `portfolio_agent/agents/backtester.py` | Resolves through `resolve_universe`; accepts a snapshot |
| `portfolio_agent/cli.py` | `backtest --universe-snapshot` |
| `portfolio_agent/agents/trainer.py` | Embargo in sessions |
| `portfolio_agent/training/trainers/gbm.py` | `_purge_cutoff_position` via `label_window_overlaps` |
| `portfolio_agent/tests/test_one_purge_one_universe.py` | New — 23 tests |

## What this changes for a user

**Every `evaluate` and `compare` number moves**, because those commands now
draw the measurement universe instead of the training one. This is the
correction: an evaluation that scores the training draw is measuring a model on
its own fitting sample, and one that disagrees with the backtest cannot be
compared to it.

Pin a snapshot and neither the seed nor the purpose matters:

```bash
portfolio-agent evaluate  --strategy momentum --universe-snapshot universe/q3.json
portfolio-agent backtest  --strategy momentum --universe-snapshot universe/q3.json
```

## Still open

The remaining panel divergences — normalization hardwired off in the backtest,
two hardcoded feature lists, four minimum-history thresholds, the label
transform applied in training only — are T21's, which unifies the builders
themselves rather than their arguments.
