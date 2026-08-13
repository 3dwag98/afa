# T13 — Net the evaluation layer of costs

**Status:** done · **Effort:** ~1 day · **Depends on:** T04 (harness), T05 (the
raw-vs-adjusted reporting pattern)
**Review reference:** `docs/architecture_review_2.html`, "Architecture"

## Goal

Report what a signal earns after paying to harvest it, using the cost model the
repository already has.

## Why

`src/execution_sim.py` prices the Indian schedule accurately — STT, exchange
charges, SEBI fees, GST, stamp duty, slippage — and even exposes
`cost_fraction_per_side` and `round_trip_cost_pct`, quantity-free helpers built
for exactly this. The live path used them. The evaluation layer did not, so
every decile spread published so far was gross, and a signal earning 40 bps a
month looked identical to one earning 40 bps a month *after* paying 80 bps to
get it.

The round trip is **0.79%** at the shipped 25 bps/side slippage assumption.
Momentum's raw decile spread has to clear that on every rebalance.

## The thing this deliberately does not do

**There is no "net IC" column.** Rank IC is a Spearman correlation, and
subtracting the same cost from every name's forward return is a monotone
transform of the labels — the ranks do not move, so the correlation is
identical to the last decimal. A net IC column would be the gross number with a
different label, which is the most expensive kind of metric because it looks
like corroboration.

A test asserts the invariance bit-for-bit, and a second test records the case
that *would* justify one: slippage scales with illiquidity, so a per-name cost
is not a constant and can move the ranks. Nothing reports that today, and the
distinction is in a test rather than a comment so it stays true.

What costs actually change is whether the spread survives being harvested.

## Approach

Turnover is the missing term, and it is measurable rather than assumable — the
panel carries `(date, symbol, score)`, which is enough to see how much of the
top bucket is replaced between rebalances. A slow signal that reshuffles 8% of
its book a month is a completely different proposition from a fast one that
replaces 60%, at identical gross spread.

    cost_per_rebalance = one_way_turnover × round_trip_cost
    net_long_short     = gross_spread − 2 × cost_per_rebalance
    net_long_only      = top_decile_return − benchmark − cost_per_rebalance

Two round trips on the long-short leg because both books turn over: the long
side sells what leaves the top decile, the short side covers what leaves the
bottom. One on the long-only leg — **which is the binding case**, because this
platform never shorts, and T05 already found low volatility ranks the
cross-section well while having a *negative* decile spread.

Also reported: **breakeven round-trip cost**, the level at which the edge
reaches exactly zero. "Would this work at 40 bps instead of 80" is the question
that follows every negative net spread, and answering it should not need a
re-run.

## Acceptance criteria

- [x] Costs read from `execution_sim`, not restated. A test doubles the STT
      rate and checks the evaluation layer follows.
- [x] STT charged on **both** legs — 0.1% each way on delivery is the largest
      statutory component, and charging it once would understate a round trip
      by 10 bps.
- [x] Turnover measured from the panel, with the number of rebalances it
      averaged over reported alongside it.
- [x] Gross and net side by side, never net alone.
- [x] Rank IC provably unchanged, asserted end to end, and the report says why
      rather than leaving it implied.
- [x] A signal with no turnover has an infinite breakeven and a net spread
      equal to its gross one.
- [x] `--gross` and `--slippage-bps` on the CLI; net is the default, because a
      gross spread is the number that looks best and means least.

## Files

| File | Change |
| --- | --- |
| `portfolio_agent/evaluation/costs.py` | New — `CostModel`, turnover, `NetSpread`, `evaluate_net`, `cost_notes` |
| `portfolio_agent/evaluation/harness.py` | `costs` field, `charge_costs`/`slippage_per_side`, render section |
| `portfolio_agent/evaluation/__init__.py` | Exports |
| `portfolio_agent/cli_forecast.py` | `--gross`, `--slippage-bps` |
| `portfolio_agent/tests/test_costs.py` | New — 39 tests |
| `portfolio_agent/tests/test_forecast_harness.py` | Determinism check made NaN-aware |

## Found on the way

**The determinism test compared dicts with `==`.** On a noise panel the gross
spread is negative, which makes "what share of it did costs eat" a ratio with
no readable sign — reported as NaN. Since `nan != nan`, two identical runs
compared unequal and the test called it a reproducibility failure. Fixed in the
test, NaN-aware: two runs producing the same NaN *is* agreement. Worth noting
because the failure mode is general — any future metric that is legitimately
undefined would have tripped it the same way.

**The HTML report needed no change.** It already renders every scalar in
`manifest.metrics`, so the cost keys appear automatically. That is the payoff
from `to_dict()` being flat.
