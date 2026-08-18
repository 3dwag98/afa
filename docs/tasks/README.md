# The forecasting pivot: the task log

Each task's own file carries its spec, its acceptance criteria, and what
actually shipped — including the places where the spec turned out to be wrong.

**Round one (T01–T11)** built the evaluation layer: measure forecast skill
directly, without simulating a book. **Round two (T12–T18)** came out of
[`docs/architecture_review_2.html`](../architecture_review_2.html), which read
the merged tree against current literature and found that the platform
contradicted itself about its own headline number. **Round three (T19–)** is
the platform review: the three paths that build a feature panel — `evaluate`,
`backtest`, `train` — disagreed on ten things, and two of those disagreements
produced plausible numbers rather than errors.

The premise behind all of them: **no trades will ever be executed, so tracking
error is acceptable and forecast skill is the thing worth measuring.** That
single decision is why the evaluation layer routes around `BacktestEngine`
rather than decomposing it, why the live path is frozen rather than improved,
and why the data work is about provenance rather than latency.

| Task | What it did | Tests added |
| --- | --- | --- |
| [T01](T01-preserve-adjustment-data.md) | Keep the adjustment data the ingest was discarding; 5→20 years | 14 |
| [T02](T02-data-validation.md) | Ingest invariants, `data status` / `data validate` | 53 |
| [T03](T03-purged-cv.md) | Purged CV as a testable unit, plus the embargo that was missing | 29 |
| [T04](T04-forecast-harness.md) | Forecast evaluation without simulating a book | 47 |
| [T05](T05-neutralized-ic-and-decay.md) | Neutralized IC and decay curves | 39 |
| [T06](T06-gbm-baseline.md) | Gradient-boosting baseline, the model to beat | 44 |
| [T07](T07-installable-package.md) | Make the package installable; `--config` | 82 |
| [T08](T08-cli-forecasting.md) | `evaluate`, `compare`, `list-features`, `data build` | 70 |
| [T09](T09-run-manifests.md) | Run manifests and rendered research notes | 48 |
| [T10](T10-remove-dead-code.md) | Delete what is actively misleading | 26 |
| [T11](T11-freeze-execution.md) | Freeze the live-trading namespace | 67 |

## Round two: what the review produced

| Task | What it did | Tests added |
| --- | --- | --- |
| [T12](T12-one-rank-ic.md) | One rank IC instead of four, two of which disagreed about the sign | 21 |
| [T13](T13-net-of-costs.md) | Charge the Indian cost schedule against a forecast | 39 |
| [T14](T14-idiosyncratic-volatility.md) | Sort low volatility on the CAPM residual, not the total | 26 |
| [T15](T15-point-in-time-membership.md) | Rank each date against who was actually in the index | 39 |
| [T16](T16-rank-ic-objective.md) | Train on the metric the model is judged on | 31 |
| [T17](T17-src-restructure.md) | Resolve where `src/` answers one question twice | 23 |
| [T18](T18-sequence-boundaries.md) | Stop training sequences straddling ticker boundaries | 22 |

## Round three: making the three paths agree

| Task | What it did | Tests added |
| --- | --- | --- |
| [T19](T19-one-decision-date.md) | One decision-date convention; the backtest was a session staler | 11 |
| [T20](T20-strategy-context-contract.md) | One `StrategyContext` contract; `rule_based` was scoring 0.0 for every name | 14 |
| [T22](T22-one-purge-one-universe.md) | Overlap in sessions, not calendar days; `evaluate` drew the training universe | 20 |
| [T21](T21-one-feature-set.md) | One feature set, derived from the strategy rather than hardcoded twice | 21 |
| [T23](T23-one-panel-policy.md) | Warm-up derived from the features, and actually loaded | 25 |

## Phase two: making it pluggable

| Task | What it did | Tests added |
| --- | --- | --- |
| [T24](T24-cross-sectional-features.md) | A feature registry that can see the universe, not one ticker | 36 |
| [T25](T25-strategy-ergonomics.md) | Declare the strategy contract callers were already relying on | 32 |
| [T26](T26-signal-to-portfolio.md) | From a ranking to a cost-charged book; four weighting schemes | 38 |

## Phase three: strategies the seams had to support

| Task | What it did | Tests added |
| --- | --- | --- |
| [T27](T27-residual-momentum.md) | Momentum on the CAPM residual; corrects `QUANT_RESEARCH.md` §12(c) | 25 |
| [T28](T28-betting-against-beta.md) | Long the low-beta decile, and an IC split by market state | 33 |

## What round two found

**The platform reported the information coefficient four different ways, and
the one driving model selection had the wrong sign.** `agents/trainer.py`
pooled every observation into a single rank correlation, which measures whether
a score tracks the market's level rather than whether it ordered any day's
cross-section. On a signal that orders every date perfectly while its level
runs against the market's, the pooled figure is −0.99 and the per-date figure
is +1.00.

**Every decile spread published so far was gross of costs** — while an accurate
NSE model sat unused in `execution_sim.py`. The round trip is 0.79% at 25
bps/side slippage, and the signal's own measured turnover is what decides
whether an edge survives it.

**The default configuration could not build one clean validation sample.** Each
ticker's 15% split of `min_history_days=250` is ~37 rows against a 60-row
sequence window, so *every* validation and test sequence spanned two stocks.
The held-out metrics were computed on histories no single stock experienced.

**Two families of risk arithmetic disagreed by 2.5x on position size**, and the
dead one understated portfolio volatility 4.1x by assuming zero correlation —
next to a module that already exposed `correlation_risk_multiple` to measure
exactly that error.

**The biggest number in the pipeline is still unmeasured.** Survivorship bias
in Indian indices runs to ~4.94pp of annual return overstatement, larger than
either strategy's neutralized alpha. T15 built the mechanism; the data itself
needs NSE circulars or a paid vendor, and every run without it now says so.

## What round one found

These are the results, not the mechanics. Each is in the relevant task file with
its numbers.

**Neither strategy survives neutralization intact.** Momentum's rank IC is
+0.061 raw and +0.026 once beta and size are removed; low volatility's is
+0.061 and +0.018. So 58% and 71% of what looked like alpha is factor loading.
For a low-volatility screen that is close to tautological — it *is* a beta bet —
and the number now says so instead of appearing as selection skill.

**Low volatility ranks the cross-section better than momentum and still has a
negative decile spread.** It orders names well and cannot be traded long-only.
Separating those two facts is the entire reason the evaluation layer exists; an
equity curve reports their product and never their difference.

**Momentum does not beat gradient boosting** on identical features and splits.

**Momentum's IC rises to day 3 and is flat to day 21** — a slow signal, so a
monthly rebalance keeps essentially all of it. The single-horizon number could
not support that conclusion either way.

**~~`rule_based` makes almost no claims.~~ Retracted — that was a harness bug,
not a property of the strategy.** The original finding read: *"score dispersion
is 0.016: one floor value for 98% of the universe, and no cross-section left to
rank."* [T20](T20-strategy-context-contract.md) found the cause. `rule_based`
took its component weights from `StrategyContext.weights`, and the evaluation
harness never set that field, so the weighted sum ran over an empty mapping and
returned **0.0 for every name**. The floor was the harness. Reading its own
configured weights, the same strategy scores the same cross-section at
dispersion **1.0** — every name distinctly ranked.

**The shipped data had six impossible bars** — low above the open/close — plus
nine symbols with one-session moves between +63% and +90%, which no NSE price
band permits and which are splits that escaped adjustment.

## Deliberately left undone

Each of these is stated on the output it affects, not only here — a caveat that
lives in a document is one people stop reading.

**The ~80 MB of market data already in git history is not reclaimed.** T10 stops
the growth; reclaiming the history needs a rewrite that invalidates every clone
and open branch. Noted in [`docs/OBTAINING_DATA.md`](../OBTAINING_DATA.md), and
a test asserts it stays noted rather than quietly presented as fixed.

**No sector map ships with the repository (finding A8).** Sector neutralization
is implemented and tested — pass `sector_map=` and it works — but every result
that runs without one says in its printed notes that it is *not* sector-neutral.
Indian momentum concentrates hard by sector, so this is exactly where an
apparent alpha most often turns out to be a sector bet.

**No point-in-time membership file ships either (T15).** It is the one input
here that cannot be reconstructed from prices: NSE circulars are authoritative
but are days of manual assembly, and the alternative is a paid vendor. Every
evaluation without one prints the bias and its magnitude.

**Nothing in round two has been run against real market data.** The numbers
above are measured — the four-way IC comparison, the 0.79% round trip, the
100%-contaminated validation split, the 2.5x and 4.1x risk disagreements — but
they are measured on the code and on synthetic panels built to have the
relevant shape. Whether Indian equity data has that shape is the next thing to
find out, and it needs the cache.

## Getting started

```bash
uv sync --extra hf --extra gbm
portfolio-agent data build --years 20      # download, then check what arrived
portfolio-agent evaluate --strategy momentum --baseline gbm --neutralize beta,size
portfolio-agent report --run <id>
```

The comparisons round two made possible, each one command:

```bash
# Does the residual sort survive where the total-volatility sort did not?
portfolio-agent compare --strategies low_volatility,low_volatility_idio \
    --neutralize beta,size

# Does the spread survive its own turnover?
portfolio-agent evaluate --strategy momentum --slippage-bps 40

# Does optimizing IC beat optimizing squared error?
portfolio-agent train --trainer gbm
portfolio-agent train --trainer rank_ic
```
