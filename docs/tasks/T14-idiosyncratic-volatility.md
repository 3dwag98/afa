# T14 — Re-sort low volatility on idiosyncratic volatility

**Status:** done · **Effort:** ~1 day · **Depends on:** T05 (the neutralization
result that motivates it)
**Review reference:** `docs/architecture_review_2.html`, "The strategies,
against current work"

## Goal

Sort the low-volatility anomaly on the CAPM residual, and make the two sorts
comparable without editing a config.

## Why

Total volatility decomposes:

    var(total) = beta² × var(market) + var(residual)

So a total-volatility sort ranks a high-beta index proxy and a wildly
idiosyncratic small-cap identically. T05 already measured what that does here:
low volatility's rank IC was **+0.061 raw and +0.018 once beta and size were
removed** — 71% of the apparent alpha was factor loading. For a volatility
screen that is close to tautological. It *is* a beta bet, and the neutralized
number said so.

The 2025 low-risk-anomaly literature (JFE's beta-anomaly volatility puzzle, the
Nordic-evidence paper) finds the two sorts behave very differently out of
sample: idiosyncratic-volatility sorts survive where beta sorts largely do not.
That makes this the smallest literature-indicated change available, and the
rolling-beta machinery was already written for T05.

## Approach

**The residual variance has a closed form.** Fitting a rolling CAPM per date
per symbol with an explicit regression would be a Python loop over
(dates × symbols). Under OLS,

    var(residual) = var(r_i) − beta² × var(r_m),   beta = cov(r_i, r_m) / var(r_m)

which is the same identity as `var(r_i) × (1 − rho²)`, and every term is a
`rolling` operation. Causality comes free: a `rolling` window ends at the row
it labels. A test checks the closed form against an explicit least-squares fit
on the same window, because the whole point is that it avoids the regression —
so the regression is the independent check.

**Two registered names, not a config flag.** `low_volatility` keeps the total
sort; `low_volatility_idio` defaults to the residual. A comparison that
requires editing a file between runs is a comparison that does not get made,
and `compare --strategies low_volatility,low_volatility_idio` now just works.
An explicit `sort_on` param still wins, so the subclass is a default, not a
lock.

**No partial fallback.** A symbol whose residual could not be estimated is not
scored on total volatility instead. Mixing two measures into one ranking is
exactly the failure T12 removed for rank IC, and a partially-idiosyncratic sort
would be *harder* to notice than an empty one, because the numbers would still
look reasonable.

**One definition of "the market".** `market_composite` is now the single
equal-weighted composite, used by this module and by
`evaluation/neutralize.rolling_beta`, which previously spelled out
`returns.mean(axis=1)` itself. A test asserts it stays shared.

**Lag-safe, matching its neighbour.** `realized_vol_60` is
`close.shift(1).pct_change()`, so `idiosyncratic_vol_from_closes` is too. If
the two disagreed, the total-vs-idiosyncratic comparison would be confounded by
a one-day alignment difference rather than measuring the decomposition. Window
defaults to 60 for the same reason.

## What the tests establish

On a synthetic panel with one pure-beta name, one pure-residual name and one of
each:

| | total vol | idiosyncratic vol |
| --- | --- | --- |
| 2× market, no residual | 0.2938 | **0.0000** |
| zero beta, all residual | 0.3469 | 0.3169 |

The two sorts pick different names — which is the point, and why this is a
different strategy rather than a tweak.

## The cost of the 60-day window, stated

A name with a *true* beta of zero still fits a non-zero beta on any finite
window, and the regression removes whatever variance that chance fit explains.
On 60 sessions of the test's seed the estimated beta comes out at **0.96** and
the fit takes **17%** of the variance with it. An explicit regression agrees
with the closed form to the last decimal, so this is a property of 60-day
estimation, not of the implementation. A test shows the shortfall falls below
5% at a 500-day window, and `vol_window` is configurable for that reason. The
default stays at 60 so the comparison is about the decomposition rather than
about the window.

## Acceptance criteria

- [x] Residual volatility computed without a per-date regression loop, checked
      against an explicit least-squares fit.
- [x] Never negative, and a flat market leaves the return as its own residual
      rather than dividing by zero.
- [x] Lag-safe on the same convention as `realized_vol_60`.
- [x] Both sorts registered and directly comparable from the CLI.
- [x] The existing `low_volatility` selection is unchanged — asserted against
      the raw feature column.
- [x] No silent fallback to total volatility for an unestimable symbol.
- [x] `rolling_beta` and this module agree on what the market is.

## Files

| File | Change |
| --- | --- |
| `portfolio_agent/features/market_relative.py` | New — `market_composite`, `rolling_idiosyncratic_vol`, `idiosyncratic_vol_from_closes` |
| `portfolio_agent/strategies/cross_sectional.py` | `sort_on` param, `_idiosyncratic_vol`, `IdiosyncraticLowVolatilityStrategy` |
| `portfolio_agent/strategies/registry.py` | Registers `low_volatility_idio` |
| `portfolio_agent/evaluation/neutralize.py` | `rolling_beta` routed at `market_composite` |
| `portfolio_agent/tests/test_idiosyncratic_volatility.py` | New — 26 tests |

## Not done here

**The two sorts have not been run against real data.** The comparison this
enables — `compare --strategies low_volatility,low_volatility_idio
--neutralize beta,size` — needs the market cache, which this environment does
not have. The number that matters is whether the idiosyncratic sort's
neutralized IC holds up better than +0.018, and it is one command away.
