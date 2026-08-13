# T03 — Purged and embargoed cross-validation

**Status:** done · **Effort:** ~3 days · **Depends on:** none
**Plan reference:** `docs/forecasting_plan.html` Part 4 (additions, first entry)

## Goal

Cross-validation that accounts for overlapping labels, so a reported forecast
skill is not partly a measurement of leakage.

## Why this is the highest-priority correctness item

Predicting a 5-day forward return from daily observations means consecutive
training samples share four days of outcome. Standard cross-validation trains
on samples whose labels overlap the test fold and reports skill that does not
exist out of sample.

**Every other number this platform produces is measured through this.** Until
it is in place, an improvement in rank IC cannot be distinguished from an
increase in leakage.

The existing walk-forward respects chronology, which handles the coarse
problem, but does not purge the overlap at the fold boundary or embargo the
serial correlation that bridges it.

## Approach

- **Purge**: drop training samples whose label window intersects the test
  fold's window. With horizon `h`, that is the `h` samples either side.
- **Embargo**: additionally drop `e` samples after the test fold, so serial
  correlation in features cannot carry information across the boundary. `e` is
  usually a small multiple of `h`.
- Applies per symbol on a date index, not per row of a stacked panel — the
  panel groups by split then ticker, so a row-index purge is meaningless.

## Files

| File | Change |
| --- | --- |
| `portfolio_agent/validation/purged.py` | New: `PurgedTimeSeriesSplit`, `purge_and_embargo` |
| `portfolio_agent/agents/trainer.py` | Walk-forward gains a purged mode |
| `portfolio_agent/tests/test_purged_cv.py` | New |

## Acceptance criteria

- [x] With horizon `h` and embargo `e`, no training sample's label window
      intersects its test fold, asserted directly on index sets.
- [x] A deliberately leaky synthetic signal — the label itself as a feature —
      scores near-perfectly under naive CV and near-zero under purged CV. This
      is the test that proves the mechanism works.
- [x] Existing walk-forward behaviour is reachable unchanged, so the difference
      between the two can be quantified rather than assumed.

## The measurement this unlocks

Rank IC of the existing LSTM under purged CV, against the same model under the
current walk-forward. The gap between those two numbers is leakage that has
been reported as skill. Expect it to be uncomfortable.

## Outcome

Done, and the task's own premise needed correcting first. I claimed
walk-forward "does not purge the overlap at the fold boundary". It **does** —
`history.iloc[:-horizon_days]`, under a comment mislabelling it an embargo.
That was the right operation with the wrong name.

What was genuinely missing: a real embargo (no setting existed), a unit that
could be tested directly, and an assertion of the property purging exists to
guarantee. `validation/purged.py` supplies all three, computed on **positions in
a sorted date index** rather than calendar days — a horizon of 5 means five
trading sessions, and calendar arithmetic is silently wrong by a variable
amount across a weekend or an exchange holiday.

One test expectation of mine was wrong and exposed a redundancy: purging
requires `p <= test_end_pos` and the embargo requires `p > test_end_pos`, so
the two are disjoint by construction and the reconciliation between them was
dead code. Removed, and the disjointness is now the test.

29 new tests; suite 1250 passed.
