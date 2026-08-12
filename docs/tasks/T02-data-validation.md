# T02 — Ingest invariants and the data inspection commands

**Status:** done · **Effort:** ~2 days · **Depends on:** T01 (uses the wider schema)
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

- [x] `data validate` exits non-zero on a seeded violation of each invariant.
- [x] `data status` reports the span and coverage of the current cache.
- [x] Invariants run automatically during ingest and refuse to write a store
      that fails a structural check.
- [x] Tests seed each violation and assert it is caught.

## Files

| File | Change |
| --- | --- |
| `portfolio_agent/data_quality/invariants.py` | New — the checks, the severity split, the inferred calendar |
| `portfolio_agent/data_quality/status.py` | New — store inventory |
| `portfolio_agent/data_quality/__init__.py` | New — package surface |
| `portfolio_agent/src/hf_dataset.py` | Structural gate before every parquet write |
| `portfolio_agent/src/data_store.py` | `read_cached_bars` — reads any cache dir, without forward-filling |
| `portfolio_agent/cli.py` | `data status` and `data validate` |
| `portfolio_agent/tests/test_data_quality.py` | New — 53 tests |

## Two decisions worth recording

**The calendar is inferred, not imported.** "Sessions reconcile against the
trading calendar" needs a calendar, and NSE holidays move every year. Rather
than pin a holiday package that is wrong the year after it is pinned, the
calendar is derived from the cross-section: a date on which a quorum of
*covered* symbols has a bar is a session; a date almost nobody traded is a
holiday. "Covered" is load-bearing — a symbol that listed in 2023 has no
opinion about 2021, so it is excluded from the denominator outside its own
span, or every session before the newest listing would fall below quorum.

**Two severities, and the split is the design.** Structural violations are
impossible in correct data, so ingest refuses the write and `validate` exits
non-zero. Advisory findings are plausible-but-notable and never fail the gate
by default. This is a correction to how the platform behaved: `max_abs_target`
*drops* labels past a threshold, discarding genuine corporate-action days along
with the errors and leaving no record. A 20% move is an NSE band doing its job;
a 900% move is a split that escaped adjustment. Both survive here.

Reading the store also needed its own primitive. `DataStore.load_ticker_data`
forward-fills up to three missing days, and a gap detector reading through a
gap filler reports no gaps — hence `read_cached_bars`.

## What it found on the shipped cache

`data validate --limit 300` on the 2,397-file cache:

- **6 structural violations** across 6 symbols — bars whose low is above the
  open/close, or high below them. Genuinely impossible bars, shipped.
- **9 symbols with suspected unadjusted splits** (+63% to +90% in one session).
- **5 symbols with extreme-but-real moves**, correctly classified as advisory —
  including `ADANIENT.NS` at +28% on 2023-02-01/02.
- **117 symbols missing sessions**, the great majority a single shared date —
  one partial trading day, which the grouped report makes obvious and a flat
  list would have buried.

`data status` prints the motivating defect in one line: **span 2021-08-09 ..
2026-08-07 (5.0 years)**, and that all 2,397 cached symbols predate the wider
schema, so corporate actions are invisible until the cache is refreshed.
