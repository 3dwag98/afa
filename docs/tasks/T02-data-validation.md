# T02 — Ingest invariants and the data inspection commands

**Status:** not started · **Effort:** ~2 days · **Depends on:** T01 (uses the wider schema)
**Plan reference:** `docs/forecasting_plan.html` Part 1 (data layer), Part 6 (CLI)

## Goal

Turn data quality from a hope into a test suite, and give the platform a way to
answer "what is actually in the store" without writing a script.

## Why

The five-year window went unnoticed until someone measured the parquet files by
hand. A `data status` command would have surfaced it on day one. More
generally: bad bars become bad labels, and a bad label is indistinguishable
from a hard-to-forecast day in every metric downstream.

## Scope

**In**
- Invariant checks that run on ingest and can be run standalone.
- `data validate` and `data status` commands.
- Non-zero exit on violation, so the suite works as a gate.

**Out**
- Cross-source reconciliation (needs a second source).
- Band and surveillance checks (need NSE-sourced fields).

## Invariants

| Check | Rationale |
| --- | --- |
| `high >= max(open, close)` and `low <= min(open, close)` | Structural; a violation is a parser or source bug |
| `volume >= 0`, `close > 0` | A zero or negative price poisons every ratio feature |
| Sessions reconcile against the trading calendar | Distinguishes a holiday from missing data |
| No duplicate dates per symbol | Re-exports carry corrected duplicates |
| Adjustment factor changes only on recorded corporate actions | Catches a silent re-adjustment upstream |
| `|return|` beyond a plausible bound is flagged, not dropped | A 20% circuit move is real; a 900% move is a split that escaped adjustment |

The last one matters most: the existing `max_abs_target` filter *drops* extreme
labels, which silently discards genuine corporate-action days rather than
recognising them.

## `data status` output

Per store and per symbol: date span, session count, coverage against the
calendar, gap count and longest gap, corporate actions found per year, first
and last bar, and how many symbols fall below a usable-history threshold.

## Acceptance criteria

- [ ] `data validate` exits non-zero on a seeded violation of each invariant.
- [ ] `data status` reports the span and coverage of the current cache.
- [ ] Invariants run automatically during ingest and refuse to write a store
      that fails a structural check.
- [ ] Tests seed each violation and assert it is caught.
