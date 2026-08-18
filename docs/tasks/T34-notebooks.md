# T34 — Notebooks

**Status:** done · **Effort:** ~1 day · **Depends on:** T33
**Review reference:** round-three plan, Phase 5 (closes the round)

## Goal

Stop the shipped notebooks from contradicting the package, and give the
evaluation layer the notebook it never had.

## `afa_lab.py` had drifted, and the drift was the package's own old bugs

`notebooks/standalone/README.md` claims the 2,000-line standalone file
reproduces the package's behaviour without importing it. **Nothing checked
that**, and it had stopped being true in two specific ways — both of them
defects the package had already fixed.

### A pooled rank IC

Exactly the defect T12 removed, still driving `val_rank_ic` in the notebook's
own training loop.

A pooled rank correlation measures whether the score tracks the market's
*level* over time; the model is used to order **one day's** cross-section.
Those are different questions with different answers. A test demonstrates the
gap on a constructed case: each date ordered perfectly, with the levels running
opposite ways across dates, gives **+1.00 per date and below −0.5 pooled**.

**The dates were available the whole time.** The panel already carried
`keys_val` as `(date, symbol)` pairs; `_rank_ic` was simply never handed them.
Without dates it now returns NaN rather than pooling — a pooled number is not a
worse estimate of the same quantity, it is an estimate of a different one.

### A total-volatility low-volatility sort

T14 moved the package to the CAPM residual on the finding that the residual
sort survives out of sample where the total sort does not. The standalone file
kept the total sort **under the same name**.

It now defaults to the residual, computed through the same closed form
(`var(r_i) − β²·var(r_m)`, no regression loop). A test asserts the two
implementations agree to **1e-9** rather than merely correlating — which is the
strongest form of the claim the README makes.

`sort_on="total"` is kept deliberately, so the two can be compared. That
comparison *is* the finding.

### And the generated file can go stale

`afa_lab.py` is assembled from `build/blocks/*.py`. A block edited without
regenerating leaves the shipped artifact behind, which is how a 2,000-line
generated file drifts unnoticed. A test checks the shipped file still contains
what the blocks define.

## `Lab` could not reach the evaluation layer

`Lab` had `train`, `backtest`, `compare` and `sweep`. It had no `evaluate`.

So the notebooks could train and backtest a strategy and **could not ask the
question the platform exists to answer.** Added:

| Method | Question |
| --- | --- |
| `Lab.evaluate` | Does the ranking carry information? |
| `Lab.neutralized` | How much survives removing beta and size? |
| `Lab.compare_forecasts` | Several strategies, one universe, one table |

None of them trains. Evaluation is about the signal a strategy *already* emits,
which is why it works on the rule-based strategies that have no checkpoint at
all — and a test asserts that by parsing the method's AST rather than grepping
its source.

## The new notebook

`03_forecast_lab.ipynb`. It opens on the distinction rather than on an API:

> A backtest reports the *product* of two things — does the signal order the
> cross-section, and does that ordering survive being turned into a book — and
> never their difference. Round one found a strategy that ranks the
> cross-section well and has a **negative** decile spread. One equity curve
> cannot say that; it just looks bad.

Eight sections: pin a universe, evaluate, read what the run *did not have*,
neutralize, decay, a cross-strategy table, the weighting-scheme comparison
(T26), the market-state split (T28), and the run manifest.

## How the notebook tests work

They do **not** execute a kernel — that needs cached prices and the optional
extras, and a test that requires a data download is one that gets skipped.

They check the two things that break first when the package moves underneath:

1. every code cell still **parses** (IPython magics stripped the way the kernel
   strips them);
2. every name a cell **imports still exists**.

The second caught a real one while this notebook was being written: it imported
`evaluate_decay` from a module that exports `decay_curve`. That reads perfectly
on the page and dies on the first cell — precisely the failure a documentation
pass cannot catch by reading.

## What changed

- `notebooks/standalone/build/blocks/models.py` — per-date `_rank_ic`,
  `_dates_of`, `MIN_CROSS_SECTION_NAMES`.
- `notebooks/standalone/build/blocks/strategies.py` —
  `idiosyncratic_volatility`, `low_volatility_scores(sort_on=...)`.
- `notebooks/standalone/build/make_notebooks.py` — call sites pass `close`.
- `notebooks/standalone/afa_lab.py` — regenerated (1,983 → 2,119 lines).
- `portfolio_agent/lab.py` — `evaluate`, `neutralized`, `compare_forecasts`.
- `notebooks/03_forecast_lab.ipynb` (new), `notebooks/README.md`.
- `portfolio_agent/tests/test_standalone_agrees.py` (new) — 22 tests.

## What this does not do

**The standalone notebooks are not regenerated from `make_notebooks.py`.** Only
`afa_lab.py` is, and the two `.ipynb` call sites it feeds were updated by hand
in the generator. Regenerating the notebooks themselves is a larger change than
this task's scope and would produce a diff nobody can review.

**No notebook is executed in CI.** Parsing and import-resolution are the checks
that pay for themselves without a data download; running them needs the cache,
which this environment does not have.

## Verification

```bash
python -m pytest portfolio_agent/tests/test_standalone_agrees.py -q
```
