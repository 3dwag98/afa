# T26 — From a ranking to a book

**Status:** done · **Effort:** ~1 day · **Depends on:** T13, T25
**Review reference:** round-three plan, finding E

## Goal

A path from a forecast to an allocated, cost-charged book — the step that turns
a ranking into a strategy.

## Why

Two halves of one thing, with nothing between them.

### The evaluation layer stopped at the spread

`evaluate` reported a decile spread. A decile spread **is** a book: the return
of an equal-weighted basket of the top bucket. Nothing said so, because the
choice was implicit in `bucket_analysis` taking a mean — so every number the
platform has published about a strategy silently assumed the one allocation
rule that needs no covariance and no view.

That is a defensible default and an indefensible silence. Equal weighting is
genuinely hard to beat, but "we chose equal weighting and here is what the
alternatives did" and "we took a mean" are different claims, and only the first
is a result.

### The portfolio library had no callers

`src/portfolio.py` is 821 lines: Ledoit-Wolf and shrunk-EWMA covariance, risk
contributions, a diversification ratio, an exact capped-simplex projection,
`optimize_long_only`, and Hierarchical Risk Parity. `src/portfolio_optimizer.py`
adds mean-variance with sector constraints.

| function | non-test callers |
| --- | --- |
| `ledoit_wolf_covariance`, `summarize_book_risk` | `backtest_engine.py` |
| `optimize_long_only` | `portfolio_optimizer.py` — which nothing imports |
| `hierarchical_risk_parity`, `shrunk_ewma_covariance` | **none** |

## What shipped

`evaluation/allocation.py`: `build_book` → weights, `evaluate_book` → the
cost-charged equity curve, `compare_schemes` → one row per rule.
`evaluate_panel(weighting=..., returns=...)` reports it alongside the spread.

Four schemes, in increasing order of what they assume:

| scheme | needs | what it is for |
| --- | --- | --- |
| `equal` | scores | the spread's own rule, made explicit. The baseline everything else must beat. |
| `inverse_vol` | + returns | size down what moves. Diagonal only, so no correlation estimate to get wrong. |
| `hrp` | + returns | the correlation structure without inverting it and without expected returns — the right default when the view is a *ranking*. |
| `mean_variance` | + returns | the textbook answer, and the only scheme that treats the score as a return. |

Inverse **volatility**, not inverse variance: variance-weighting is a far more
aggressive tilt toward the quietest names, and on Indian small caps the
quietest names are frequently the least liquid — which is the failure T14's
tradability screen exists for, and there is no reason to walk back into it
through the weighting rule.

## Two decisions worth naming

### The cap defaults to None, and that default is load-bearing

A cap of `c` on an `n`-name book is unsatisfiable when `c * n < 1`: `n` names
cannot each hold less than `1/n` of a fully invested book. **A decile book is
small.** A 10-bucket split of a 60-name universe holds six names, so any cap
below 0.167 cannot bind and equal weights are the only feasible answer.

A 0.10 default — which is what this task was first written with — made all four
schemes return *identical* weights on a realistic universe. `compare_schemes`
printed four indistinguishable rows, and the obvious reading of that table
("the weighting rule doesn't matter") would have been entirely an artifact of
the cap.

Uncapped is the honest default for a book whose concentration is already set by
the selection. A caller who passes a cap that cannot bind is told so in the
notes, and `cap_is_binding` is public so the check is available before a
comparison is trusted.

### The score is a rank percentile, not a return

Every strategy here emits a 0-100 *goodness* score. Only `mean_variance` needs
it to be a return, and it is not one. So the cross-section is centered and
scaled to a declared `expected_return_spread` — best-ranked name at
`+spread/2`, worst at `-spread/2` — which puts the assumption in one visible
place rather than smuggling it in as "the score, divided by 100".

Centering matters as much as scaling: an all-positive mu makes a long-only
optimizer want the whole budget invested no matter how weak the ranking is.

## What the schemes actually do

Measured on a synthetic 60-name factor panel with deliberately unequal
volatility, top decile, daily rebalance:

| scheme | mean names | mean max weight | annualized vol |
| --- | --- | --- | --- |
| `equal` | 6.0 | 0.167 | 0.196 |
| `inverse_vol` | 6.0 | 0.246 | 0.178 |
| `hrp` | 6.0 | 0.340 | 0.167 |
| `mean_variance` | **1.6** | **0.908** | **0.334** |

HRP does what it is for — the lowest realized volatility. `mean_variance`
concentrates the book into one or two names and nearly doubles the volatility,
which is precisely the behaviour `optimize_long_only`'s and
`hierarchical_risk_parity`'s own docstrings warn about ("an optimizer handed a
noisy mu will happily concentrate the entire book in whichever name got the
luckiest sample"). Having it in the table is worth more than being told.

Breadth therefore travels into the report with the return. Without
`book_mean_names_held` and `book_mean_max_weight`, the difference between the
first row and the last is invisible.

`mean_variance` also rebalances **from the book it holds**, not from cash,
charging the round-trip cost as `optimize_long_only`'s turnover penalty — its
own docstring calls that penalty load-bearing — and warm-starting the
subgradient ascent, which is what makes a per-date optimization affordable at
all (300 iterations against the function's own default of 2000).

## One convention recorded rather than derived

Establishing the book from cash reports 0.5 turnover, which looks like an
understatement of a 100% buy. But the caller charges `turnover x round_trip`,
and establishing pays only the buy leg — so half a round trip *is* one leg,
which is the cost actually incurred. That is a coincidence of two conventions
rather than a derivation, and it is stated in `weight_turnover`'s docstring and
pinned by a test, so the day either convention moves it fails rather than
quietly mis-charging.

## What this does not do

The book is built from the top bucket only, so it inherits the decile
selection wholesale — no score-proportional sizing across the whole
cross-section, and no short leg (this platform does not short). Sector and
turnover constraints exist in `src/portfolio_optimizer.py` and are not wired;
that needs the sector map T31 supplies.

## Verification

```bash
python -m pytest portfolio_agent/tests/test_allocation.py -q
```

```python
from portfolio_agent.evaluation import compare_schemes
compare_schemes(panel, returns=trailing_returns)
```
