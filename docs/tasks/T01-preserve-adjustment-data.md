# T01 — Preserve adjustment data at ingest, make the history window explicit

**Status:** done · **Effort:** ~1 day + a refetch · **Depends on:** none
**Plan reference:** `docs/forecasting_plan.html` Part 1 (data layer), Part 7 week 1

## Goal

Stop discarding the corporate-action columns the source already provides, store
raw prices alongside adjusted ones, and make the history window a visible
decision rather than a silent default.

## Why this is first

Two findings, both measured on the 2,397 committed parquet files:

**The cache carries `Date, open, high, low, close, volume` and nothing else.**
The HuggingFace source provides `adj_close`, `dividends` and `stock_splits`.
`normalize_frame` applies the adjustment and then drops `adj_close`; the other
two are never read. So the corporate-action data is *already in the source
being used* and is thrown away at ingest. Consequences:

- Raw prices cannot be recovered, so any circuit-band or price-level check
  compares against a back-adjusted number that is not what traded.
- There is no corporate-action event list, so a demerger or rights issue
  arrives as an unexplained large return and is silently dropped by the
  `max_abs_target` filter rather than recognised.

**Every file spans exactly 2021-08-09 to 2026-08-07 — five years to the day.**
That is `default_history_years: 5` flowing into
`start_date = end_date - years*365`. It is a configuration choice, not a limit
of the source. The sample therefore begins *after* the COVID crash: it contains
one bull run, one rate-hike correction, and no crisis. Every tail estimate,
regime model and drawdown forecast on this platform is fitted on data with no
crash in it.

## Scope

**In**
- Keep `adj_close`, `dividends`, `stock_splits` through `normalize_frame`.
- Store both raw and adjusted OHLC. Adjusted stays the default for returns.
- Derive an adjustment factor series and expose corporate actions as a table.
- Raise `default_history_years` and surface it as a CLI option.
- Tests covering the round trip and the new columns.

**Out**
- The bitemporal store (`docs/data_architecture.html`) — a later task.
- NSE-sourced fields (series, delivery, bands) — needs a new source.
- Point-in-time index membership — separate acquisition, see T-backlog.

## Files

| File | Change |
| --- | --- |
| `portfolio_agent/src/hf_dataset.py` | `normalize_frame` keeps the adjustment columns; emit raw legs as `open_raw` … `close_raw` |
| `portfolio_agent/src/data_store.py` | Persist the wider schema; readers tolerate the older narrow one |
| `portfolio_agent/config/schema.py` | `default_history_years` default raised; document why |
| `config.yaml` | Same, with the reasoning in a comment |
| `portfolio_agent/cli.py` | `--years` / `--from` / `--to` on the download command |
| `portfolio_agent/tests/test_hf_dataset.py` | Extend for the preserved columns |

## Approach

1. Change `normalize_frame` to return the adjustment columns rather than
   dropping them, and to emit the raw price legs under distinct names. Keep the
   adjusted legs where they are so nothing downstream moves.
2. Add `corporate_actions_from_frame()` deriving a tidy event table from
   `stock_splits` / `dividends`, plus the `adj_close/close` factor as a
   cross-check. Where the explicit columns and the derived factor disagree,
   record both — the disagreement is a data-quality signal, not something to
   resolve silently.
3. Widen the default history window. The correct value depends on what the
   source actually holds, which is the first thing the implementation should
   measure and report.
4. Backwards compatibility: a cache written before this change has the narrow
   schema. Readers must treat the new columns as optional, so an existing
   install keeps working until it refetches.

## Acceptance criteria

- [x] A freshly built cache carries raw and adjusted prices and the two
      corporate-action columns.
- [x] `corporate_actions_from_frame()` recovers known splits and bonuses on at
      least three symbols, verified against the derived factor.
- [x] A cache written under the old schema still loads, with the new columns
      absent rather than erroring.
- [x] The history window is reported at ingest, so a five-year window can never
      again be a silent default.
- [x] Existing tests pass unchanged.

## Risks

- **The source may not hold more than five years.** If so, the widening half of
  this task is a no-op and that finding changes the plan — which is exactly why
  it is measured on day one rather than assumed.
- The refetch cannot run in CI or the dev container: `huggingface.co` is
  blocked by the network policy. Code and tests land here; the refetch is a
  local step.

## Outcome

Done. The ingest was discarding `adj_close`, `adj_factor`, `dividends` and
`stock_splits` that the source already supplied, so a corporate action was
indistinguishable from a price move after the fact. Raw OHLC legs are now
captured before adjustment, `adj_factor` is recorded explicitly (1.0 when
nothing was adjusted), and `corporate_actions_from_frame` reports stated and
derived actions *unmerged* — a derived action that no stated one explains is
the signal that the source re-adjusted history.

`default_history_years` went from 5 to 20. Five years was a config default
nobody had measured, and it put the sample start *after* the COVID crash, so no
regime model on this platform had ever seen one.

14 new tests; suite 1236 passed. Two existing tests were rewritten — one of
them, `test_drops_the_non_ohlcv_columns`, was pinning the bug in place.
