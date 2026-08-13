# T12 — One rank IC

**Status:** done · **Effort:** ~1 day · **Depends on:** nothing
**Review reference:** `docs/architecture_review_2.html`, "The headline"

## Goal

One definition of the information coefficient, reachable from every entry
point that reports one.

## Why

The architecture review found three implementations and two of them disagreed
about the sign. Searching by shape rather than by name turned up a fourth.

On a panel where the signal orders every date perfectly but its level runs
against the market's:

| Implementation | Method | Result |
| --- | --- | --- |
| `evaluation/metrics.py` | per date, Spearman | **+1.0000** |
| `src/performance_stats.py` | per date, Spearman | **+1.0000** |
| `training/trainers/gbm.py` | per date, Spearman | **+1.0000** |
| `agents/trainer.py` | pooled, Pearson-on-ranks | **−0.9988** |

`agents/trainer.py` ranked every observation in one pool. Pooled, the
correlation is dominated by whether the model's level tracks the market's from
day to day — a market view, which a long-only ranking book cannot trade. Per
date, it asks whether the model ordered that day's names. The two are different
quantities, and on `relative_target` runs the pooled one was the headline
metric that model selection read.

A second, quieter disagreement: `src/performance_stats.py` annualized ICIR by
`sqrt(periods / horizon)` while `evaluation/metrics.py` reported the raw ratio
under the same key. Same signal, two ICIRs, a factor of 16 apart at a daily
horizon.

## Approach

`evaluation/metrics.py` is the definition. Everything else adapts to it.

- `rank_ic_from_arrays(scores, labels, dates)` — the panel function for callers
  holding three parallel arrays, which is what trainers have. This is what each
  copy had grown its own loop for.
- `gbm.rank_ic_by_date` and `performance_stats.rank_information_coefficient`
  keep their names — their callers and tests read well with them — and delegate.
- `evaluate_predictions` takes `dates` and computes IC per date. **Without
  dates it reports NaN**, not a pooled number: there is no cross-section to
  correlate within, and "not measured" is the honest answer.
- ICIR is the raw mean/sd everywhere, because that is what the literature
  quotes. The annualized figure keeps its own key, `icir_annualized`.

### Threading dates to the trainer

The neural trainer discarded them twice over, so both paths were fixed:

- **Walk-forward.** `_stack_blocks` returns arrays and drops the index.
  `_stacked_dates` re-derives the dates from the same block list, offset by
  `sequence_length` because a `TimeSeriesDataset` spends its first
  `sequence_length` rows as history for the first prediction.
- **Single split.** `feature_df` was built with `ignore_index=True`. The index
  is kept now, and `data/dataset.py` exposes `chronological_split_bounds` so the
  date slice and the array slice cannot drift apart — two copies of
  `int(n * 0.70)` is exactly how they would.

## What changed in the numbers

**The walk-forward fixture had two tickers.** Two names on a date is not a
cross-section, and every date fell below `MIN_CROSS_SECTION_NAMES`. The pooled
implementation returned a confident correlation anyway and the summary printed
it as the model's skill. The fixture now uses six, and a second test asserts
that the two-ticker case reports NaN.

**`gbm.py` carried its own `min_names: int = 5`.** Equal to
`MIN_CROSS_SECTION_NAMES` by coincidence rather than by construction. Now
shared, with a test on the default.

## Acceptance criteria

- [x] One module computes a per-date rank correlation; a test searches by shape
      for any other, the way T10 does for RSI.
- [x] All four entry points return the same series on the same input.
- [x] The adversarial panel is a test, so the +1.00/−0.99 gap stays measurable.
- [x] `evaluate_predictions` reports NaN without dates rather than a pooled
      number, and raises on a misaligned date array.
- [x] `icir` means the same thing in both modules; `icir_annualized` preserves
      the old figure.
- [x] Dates are aligned to predictions, asserted against the real
      `TimeSeriesDataset` rather than against a reimplementation of its offset.

## Files

| File | Change |
| --- | --- |
| `portfolio_agent/evaluation/metrics.py` | New — `rank_ic_from_arrays` |
| `portfolio_agent/src/performance_stats.py` | Delegates; `icir` raw, `icir_annualized` added |
| `portfolio_agent/training/trainers/gbm.py` | `rank_ic_by_date` delegates; shared threshold |
| `portfolio_agent/agents/trainer.py` | `dates` parameter; `_stacked_dates`; NaN-safe fold summary |
| `portfolio_agent/data/dataset.py` | New — `chronological_split_bounds`, `test_split_dates` |
| `portfolio_agent/tests/test_one_rank_ic.py` | New — 18 tests |
| `portfolio_agent/tests/test_trainer.py` | Fixture widened to 6 tickers; 2 tests added |
| `portfolio_agent/tests/test_performance_stats.py` | ICIR convention; 1 test added |

## Found on the way, not fixed here

**Sequences straddle ticker boundaries.** `_stack_blocks` concatenates
per-ticker frames and `TimeSeriesDataset` slides a window across the join, so a
window near the end of one ticker's block reads the next ticker's history as
its own. With ~250 rows per ticker and a 30-row sequence, roughly 12% of
training sequences mix two stocks. Out of scope for a task about IC, and large
enough to deserve its own — filed as T18.
