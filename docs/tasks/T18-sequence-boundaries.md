# T18 — Stop sequences straddling ticker boundaries

**Status:** done · **Effort:** ~1 day · **Depends on:** T12 (which found it, and
whose date alignment has to move in lockstep)
**Found by:** T12, recorded in `T12-one-rank-ic.md` under "Found on the way"

## Goal

A training sequence must come from one instrument.

## Why

`_stack_blocks` concatenates per-ticker frames into one matrix and
`TimeSeriesDataset` slid a fixed window across it. A window starting near the
end of ticker A's block and ending inside ticker B's fed the model one stock's
price history and asked it to predict a different stock's move. Nothing raised;
the run trained, converged, and reported metrics.

`load_data`'s own docstring acknowledged it and dismissed it:

> Sequence windows that straddle two concatenated tickers' boundaries mix data
> from different instruments; this is a bounded, documented limitation of
> pooling multiple series through a single-series windowing dataset ... it
> affects at most `sequence_length * (n_tickers - 1)` windows out of the full
> panel.

The bound is arithmetically correct. The conclusion drawn from it is wrong,
because *the full panel is not the panel that matters.*

## What it actually cost, measured

At the shipped `training.sequence_length = 60`:

| tickers | rows each | windows | clean | straddling | share |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 300 | 2,340 | 1,920 | 420 | **17.9%** |
| 50 | 300 | 14,940 | 12,000 | 2,940 | **19.7%** |
| 200 | 300 | 59,940 | 48,000 | 11,940 | **19.9%** |
| 50 | 1,000 | 49,940 | 47,000 | 2,940 | 5.9% |
| 500 | 1,250 | 624,940 | 595,000 | 29,940 | 4.8% |

Roughly a fifth of a typical training panel, not the ~12% first estimated. The
loss scales with the *ticker count*, not the row count, so a wide shallow
universe is the bad case — and a wide shallow universe is what
`--universe-size 500` produces.

### The part that makes it a defect rather than a rounding error

`create_dataloaders` splits 70/15/15. Each ticker's 15% share of the shipped
`data.min_history_days = 250` is about 37 rows, against a 60-row window.

**Nothing fits.** Every validation and every test sample the old code produced
spanned at least one join. The held-out metrics those loaders reported were
computed on sequences no single stock ever experienced.

| split | rows/ticker | old samples | clean | straddling |
| --- | ---: | ---: | ---: | ---: |
| train | 210 | 10,440 | 7,500 | 28.2% |
| validation | 45 | 2,190 | 0 | **100%** |
| test | 45 | 2,190 | 0 | **100%** |

## Approach

`sequence_target_positions(group_lengths, sequence_length)` is the single
definition of *which rows get a prediction*. Three callers have to agree on it
— the dataset that indexes the rows, `test_split_dates`, and the trainer's
`_stacked_dates` — and when they disagree nothing raises: rank IC just
correlates each prediction against a different day's cross-section and reports
a confident number about the wrong thing. Same discipline as T12.

**Filtering positions, not concatenating per-ticker datasets.** Both drop
exactly the same windows. The filter keeps which-rows-are-predicted in one
array that the date-alignment code reads directly, instead of re-deriving it
from a list of sub-dataset lengths; and a `ConcatDataset` would copy each
ticker's rows into its own tensor when the panel is already one contiguous
matrix.

**`group_lengths` travels with the arrays.** `_stack_blocks` returns a
`StackedPanel` carrying them, because they cannot be recovered from the frame:
dates repeat across tickers and the blocks cover overlapping calendar periods.

**A stacked panel passed without boundaries is refused.** A single instrument's
dates only ever increase, so a backwards step proves the frame is several
tickers stacked. `create_dataloaders` raises on that rather than windowing
blind — refusing beats producing a model and no error at all.

## The behaviour change users will hit

Under the shipped default, `create_dataloaders` now **raises** instead of
building a validation set, because no window fits inside one ticker's 37-row
slice. That is the correction, not a new restriction: it never could build a
valid one, it just did not say so. The message names both ways out — lower
`training.sequence_length`, or raise `data.min_history_days`.

## Acceptance criteria

- [x] No sequence spans two tickers; asserted by checking every sample's
      feature column is internally contiguous.
- [x] The old contamination stays reproducible, so the fix is measurable rather
      than described.
- [x] One definition of which rows are predicted, shared by all three callers.
- [x] T12's date alignment still holds, strengthened to the multi-ticker case
      and matched on exact target values rather than plausibility.
- [x] Boundaries that do not sum to the row count are refused.
- [x] Panels where no window fits are refused with an actionable message.
- [x] A single-instrument frame behaves exactly as before, error text included.

## Files

| File | Change |
| --- | --- |
| `portfolio_agent/data/dataset.py` | `sequence_target_positions`, `group_lengths_in_slice`, boundary-aware `TimeSeriesDataset`, `create_dataloaders` guard |
| `portfolio_agent/agents/trainer.py` | `StackedPanel`, `split_ordered_panel`, `_stack_blocks`/`_stacked_dates`/`_make_loader` carry boundaries |
| `portfolio_agent/tests/test_sequence_boundaries.py` | New — 20 tests |
| `portfolio_agent/tests/test_one_rank_ic.py` | Alignment strengthened; 2 tests added |

## Not measured

**What this does to walk-forward rank IC on real data.** The contaminated
sequences were being fed a different stock's history, which is noise rather
than leakage, so the expected direction is that IC improves — but that is a
prediction, not a measurement, and it needs the market cache this environment
does not have.
