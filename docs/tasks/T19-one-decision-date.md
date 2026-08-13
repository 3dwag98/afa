# T19 — One decision-date contract

**Status:** done · **Effort:** ~1 day · **Depends on:** nothing
**Review reference:** round-three plan, finding B

## Goal

`evaluate`, `backtest` and `train` must read the same numbers for the same
strategy on the same date.

## Why

They did not. On date D:

| path | slice | `features.iloc[-1]` described |
| --- | --- | --- |
| `backtest_engine.py:559` | `df.index < D` — **exclusive** | the close of **D-1** |
| `harness.py:560` | `frame.loc[:D]` — **inclusive** | the close of **D** |
| trainers | row `t` labelled from `t` | the close of **D** |

Every feature in `features/technical.py` already shifts its own inputs by a
bar, so the backtest was reading one *further* session stale than the
evaluator, on identical inputs. Add the T+1 open fill and its
decision-to-entry gap was ~2 sessions against the evaluator's 0.

**The rank IC `evaluate` reported was not the IC the backtest traded**, and
nothing anywhere said so.

## Why inclusive is the right side to converge on

Not a coin flip — the engine's own loop settles it. `run_backtest` runs seven
steps per session, and signal generation is step **E**, *after* step D has
marked the book to market at D's close:

    D. Mark to market at T's close  ->  that day's equity point
    E. Score the universe
    F. Queue orders for T+1's open

The engine is standing at the end of session T. The close it just valued the
portfolio with is knowable. Deciding on T-1 data there and then waiting until
T+1 to act was an off-by-one against its own position in the loop, not a
safety margin — and it is the sequence a live desk would never run, since the
orchestrator computes after the close and trades the next morning.

What actually prevents look-ahead is unchanged: features shift internally, and
`close` — the one deliberate exception — is the decision date's reference
price, knowable at that session's close by construction.

## The test that was pinning it

`TestLookAheadBiasPrevention` asserted `hist_data.index.max() < test_date`.
That conflates two different things: *not seeing the future*, which is the
invariant, and *where the decision point sits*, which is a convention. It has
been replaced by the two assertions that mean something:

- signals see D and nothing after it;
- **rewriting every bar after D leaves D's signals bit-identical** — stronger
  than an index bound, because it would catch a feature that reached forward
  regardless of how the frame was sliced.

## Found on the way: the trainers were discarding labelled rows

`build_forward_return` was `close.shift(-h).pct_change(h)`. That lands on the
same value as the direct expression — shifting then differencing over the same
span gives `close[t+h]/close[t] - 1` either way — but it *also* NaNs the
**first** `h` rows, because `pct_change` has nothing to difference against
there. Those rows have a perfectly well-defined forward return.

Verified: values identical wherever both are defined; coverage 55 rows vs 50 on
a 60-row series at `h=5`. On a ticker at the 252-session minimum with a 21-day
label that is **8% of its training rows**, dropped at the start of the sample
where the long-lookback features have only just warmed up. The evaluation
harness never had it, so training and evaluation disagreed about how much of
each ticker was labelled at all.

Now stated as `close.shift(-periods) / close - 1.0`, with a parametrized test
asserting only the final `h` rows are NaN.

## Acceptance criteria

- [x] Backtest, evaluation and training agree on the decision date.
- [x] Feature vectors bit-identical between the slice-then-build and
      build-then-slice orders, across five dates in the sample.
- [x] Asserted end to end through the real `BacktestEngine`, not only through
      `build_features`.
- [x] Look-ahead re-tested as "a future bar cannot change today's signal".
- [x] The divergence recorded as material, so the exclusive slice is not
      re-introduced on the grounds that it cannot matter much.
- [x] Benchmark closes and OHLC bars truncate to the same length — a close
      series one bar longer than its own range would have the regime read a
      trend from one session and an ADX from the one before it.
- [x] The label keeps the first rows of the sample.

## Files

| File | Change |
| --- | --- |
| `portfolio_agent/src/backtest_engine.py` | `_get_historical_data_up_to` → `_history_through`, inclusive; both benchmark slices; five stale comments |
| `portfolio_agent/features/labels.py` | `build_forward_return` stops discarding the first `h` rows |
| `portfolio_agent/tests/test_paths_agree.py` | New — 17 tests |
| `portfolio_agent/tests/test_backtest_engine.py` | Look-ahead and benchmark tests re-stated on the real invariant |

## What this changes for a user

**Every backtest number moves.** The engine now decides on one session fresher
data. That is the correction — the old numbers described a strategy deciding on
stale inputs and then waiting a day to act, which is neither what `evaluate`
measured nor what a live desk would do. Nothing about the correction is a
loosening: the future-bar test is stricter than the index bound it replaces.
