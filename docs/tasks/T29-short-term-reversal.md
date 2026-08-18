# T29 — Short-term reversal

**Status:** done · **Effort:** ~half a day · **Depends on:** T25, T28
**Review reference:** round-three plan, Phase 3

## Goal

Measure the effect momentum's skip-month exists to avoid, instead of assuming
it.

## Why

`mom_9m_skip1m` is:

```python
close.shift(21) / close.shift(21 + 189) - 1
```

Those 21 skipped sessions are not an implementation detail. They are there
because the most recent month **reverses** rather than continues (Jegadeesh
1990, Lehmann 1990), and including it would drag the momentum signal against
itself.

The platform has applied that correction on the strength of the literature and
has **never measured the effect it corrects for on this data.** That is a
standing assumption sitting underneath the flagship strategy, and it is cheap
to check: one feature and a sign flip.

```bash
portfolio-agent compare --strategies momentum,reversal
```

- A flat or negative reversal spread means the skip is buying nothing here, and
  those 21 sessions could be folded back into the formation return.
- A positive one means the skip is earning its keep — which is worth more than
  the assumption it currently rests on.

## The window has to be exact

`return_21d` is `close.shift(1).pct_change(21)`, which is *precisely* the
window `mom_9m_skip1m` drops. A test asserts the two series are **equal**, not
similar:

```python
skipped = close.shift(1) / close.shift(1 + 21) - 1.0
pd.testing.assert_series_equal(build_features(ohlcv, ["return_21d"]), skipped)
```

If the windows were merely close, `compare momentum,reversal` would be
comparing two different questions and the answer would be uninterpretable.

## Costs decide this one, not the spread

Reversal is the most turnover-intensive effect in the book: a one-month
formation window implies replacing most of the decile every month.

Computed from the shipped Indian schedule rather than quoted from memory:

| | |
| --- | --- |
| buy | **40.5 bps** |
| sell | **39.0 bps** |
| round trip | **79.4 bps** |
| 12 full rebalances/yr | **9.53% of capital** |
| 52 full rebalances/yr | **41.3% of capital** |

9.53% a year in friction is **larger than most published gross reversal
spreads.** So the number that settles this strategy is not its spread but
T13's `breakeven_round_trip_cost` — the round trip at which the edge reaches
exactly zero — together with `--slippage-bps`. A gross reversal spread quoted
without them is not a result.

The tests pin the arithmetic at both ends: a panel reshuffled every date (the
limiting case a monthly book approaches) and one whose ordering never changes,
which pays nothing.

## Microstructure is the second reason to read it carefully

A reversal sort concentrates in exactly the names T14's tradability screen
exists for. A stock that fell hard on no volume, or one pinned at its lower
circuit, **prints the return this ranks highest while offering nothing to
buy.** The screen is inherited and on by default, and turning it off makes this
strategy look far better than it is.

## The seam, again

Third Phase 3 strategy, and the first that needed **no new extraction**:

| task | needed |
| --- | --- |
| T27 residual momentum | `_formation_metric` hook |
| T28 betting-against-beta | `higher_metric_is_better` property |
| T29 reversal | *nothing* — overrides both, adds none |

`ShortTermReversalStrategy` is a `MomentumStrategy` subclass that overrides the
metric and the sort direction. Tradability screen, regime filter, volatility
targeting, decile selection all inherited. That is what the seam was for.

It is also the only Phase 3 strategy whose metric is a **per-ticker** feature —
it needs a cross-section to rank in, but not to compute. So it exercises the
path where `required_cross_sectional_features()` is legitimately empty.

## What changed

- `features/technical.py` — `return_21d`.
- `strategies/cross_sectional.py` — `ShortTermReversalStrategy`,
  `REVERSAL_FEATURE`; and the two constants T28 left sitting between import
  statements moved below the import block.

## What this does not do

**No residual reversal.** Da, Liu & Schaumburg (2014) find the residual version
substantially stronger than the raw one, and T27 built `residual_returns`, so
it would be a short addition. It is deliberately left out: the plan scoped this
task as "one new feature and a sign flip", and the question being answered —
*is the skip-month earning its keep* — is about the raw effect the skip
removes. A residual variant answers a different and also interesting question,
and should be its own task with its own comparison.

## Verification

```bash
python -m pytest portfolio_agent/tests/test_short_term_reversal.py -q
portfolio-agent compare --strategies momentum,reversal --slippage-bps 25
```
