# T16 — Train on the metric the model is judged on

**Status:** done · **Effort:** ~1 day · **Depends on:** T06 (the GBM baseline
and its panel), T12 (one definition of rank IC)
**Review reference:** `docs/architecture_review_2.html`, "Architecture" —
objective mismatch

## Goal

Close the gap between what the model minimizes and what decides whether it was
any good.

## Why

Every trainer fits a pointwise loss — squared error, pinball — and is then
ranked by rank IC. Those are different objectives. Squared error on a
cross-sectional rank label is minimized by predicting each name's *conditional
mean rank*, which is an excellent way to be nearly constant and score an IC of
approximately zero.

The `gbm` baseline went further: `_fit_with_early_stopping` chose **which
iteration ships** by validation MSE, while `_score` reported `val_rank_ic` and
`primary_metric()` picked that up. So the selection step never looked at the
number in the summary table.

LambdaRankIC (arXiv 2605.00501, May 2026) makes the case for closing this
directly, in the LambdaRank tradition: you cannot differentiate a rank metric —
ranks are piecewise constant — so you derive per-item gradients from the metric
and hand those to the booster.

## Two changes, separable on purpose

**1. The baseline selects on IC.** `selection_metric` is `"rank_ic"` by
default, `"mse"` still available so the difference stays measurable rather than
becoming a claim in a commit message. One sign convention in the loop: the
score is always maximized and MSE enters negated, because two loops that must
stay in step is how a comparison silently inverts.

**2. A trainer whose gradient is the objective's.** Registered as `rank_ic`.

## What the objective actually is

The differentiable surrogate, not the paper's lambda construction. Per date,
the loss is the negative Pearson correlation between raw scores and the
cross-sectionally ranked label:

    L = −(1/T) Σ_t corr(s_t, y_t)

    s̃ = s − mean(s)    ỹ = y − mean(y)
    a = <s̃,ỹ>   b = ‖s̃‖   c = ‖ỹ‖   corr = a/(b·c)

    ∂corr/∂s_i = [ ỹ_i − corr·(c/b)·s̃_i ] / (b·c)

A surrogate for Spearman rather than Spearman itself — it *becomes* Spearman
when the scores are replaced by their own ranks. Two reasons to start here: the
label is already a cross-sectional rank under the default transform, so one
side of the correlation is the rank side; and the gradient is closed-form and
exact, so it can be **checked against a numerical derivative** rather than
trusted. It matches finite differences to 1.5e-11.

**Naming.** `rank_ic`, not `lambdarank_ic`. It optimizes an IC surrogate, which
is the paper's premise, but it is not the paper's lambda formulation and should
not borrow the name.

### The degenerate case that would have broken it silently

At iteration one every score is identical, so ‖s̃‖ = 0 and the correlation is
undefined. A naive implementation divides by zero on the first step and emits
NaN for the rest of the fit. The limit is well defined — with no score
dispersion the direction that most increases correlation is the centred label —
and that is what `date_ic_gradient` returns.

## Why a hand-rolled boosting loop

`HistGradientBoostingRegressor` takes no custom objective; scikit-learn exposes
a fixed set of losses. But gradient boosting with a custom objective is exactly
"fit a tree to the negative gradient, take a shrunk step", so the loop is
short. It uses `DecisionTreeRegressor` and **shares `build_gbm_panel` with the
baseline**, so both trainers see an identical panel, split, purge, label and
scaler — which is what makes a comparison between them a comparison of
objectives and nothing else.

Subsampling is by **date**, not by row. The gradient is defined against each
date's own mean, so half a date's names give the other half the wrong centring:
a row subsample would corrupt the objective rather than regularize it.

## What it buys, measured

On a panel whose label variance is 96% market move — the shape of an equity
panel — trained on identical data:

| trees | IC objective | squared error |
| --- | --- | --- |
| 5 | **+0.5249** | +0.0552 |
| 10 | +0.5661 | +0.2684 |
| 20 | +0.6789 | +0.6086 |
| 40 | +0.6823 | +0.6708 |
| 80 | +0.6825 | +0.6786 |

**The finding is about rate, not ceiling.** Squared error has to explain the
market before it can reach the cross-section, so its early capacity buys no
ordering skill at all — at five trees it is 10× behind. Given enough trees it
catches up, and a test asserts that too, because the honest claim is narrower
than "the new objective is better."

It matters because early stopping cuts the budget. A model stopped at 10 trees
under the old setup shipped an IC of 0.27 where 0.57 was available on the same
data.

The IC-objective model is correspondingly **much worse at squared error** — 10×
worse — and a test pins that. Its output is an ordering, not a return forecast,
and it should not be mistaken for one.

## Acceptance criteria

- [x] Gradient checked against a numerical derivative, not another formula.
- [x] Sums to zero within a date (correlation is shift-invariant) and scales as
      1/scale (it is scale-invariant).
- [x] Both degenerate cases handled: constant scores give the centred label
      direction, constant labels give no direction.
- [x] Thin dates excluded at `MIN_CROSS_SECTION_NAMES`, so the objective is not
      optimized on cross-sections the metric will refuse to score.
- [x] Averaged over scored dates, so step size does not depend on panel length.
- [x] Early stopping maximizes validation rank IC; truncation is exact.
- [x] Identical panel to the baseline, so the comparison isolates the objective.
- [x] Registered, torch-free, joblib checkpoint.
- [x] `selection_metric` defaults to `rank_ic`; `mse` kept and refused if
      misspelled.

## Files

| File | Change |
| --- | --- |
| `portfolio_agent/training/trainers/rank_ic.py` | New — gradient, objective, `RankICTrainer`, `AdditiveEnsemble` |
| `portfolio_agent/training/trainers/gbm.py` | `selection_metric`; early stopping maximizes a score |
| `portfolio_agent/training/trainers/__init__.py` | Registers `rank_ic` |
| `portfolio_agent/tests/test_rank_ic_objective.py` | New — 31 tests |
| `portfolio_agent/tests/test_gbm_trainer.py` | Torch-less trainer list now includes `rank_ic` |

## Found on the way

**`rank_ic` needs no torch.** It shares gbm's panel builder and scikit-learn is
its only requirement, so the torch-less install test now lists
`gbm,rank_ic,supervised`. That is the intended outcome, not a side effect: the
objective comparison is available on an install without the `gpu` extra.

## Not done here

**Neither trainer has been run on real data.** The comparison this enables —
`train --trainer gbm` against `train --trainer rank_ic` on the same universe —
needs the market cache, which this environment does not have. The synthetic
result above establishes that the objective does what it claims on a panel with
the right shape; it does not establish that Indian equity data has that shape.
