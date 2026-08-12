# T05 — Neutralized IC and signal decay curves

**Status:** done · **Effort:** ~3 days · **Depends on:** T04 (harness)
**Plan reference:** `docs/forecasting_plan.html` Part 4 (additions)

## Goal

Separate stock selection from factor tilt, and measure how long an edge
persists.

## Why

**Neutralization.** A cross-sectional signal can score well simply by loading
on a sector or on beta. Indian momentum concentrates hard by sector — IT
through 2020-21, PSU banks through 2022-23 — so this is precisely where an
apparent alpha most often turns out to be a sector bet wearing a signal's
clothes. The gap between raw and neutralized IC is the share that is genuinely
stock selection.

**Decay.** IC as a function of horizon determines rebalancing frequency, and
therefore whether an edge survives costs at all. It also distinguishes a
genuine slow signal from a fast one that is mostly microstructure noise.

## Approach

- Neutralize by cross-sectional regression of the signal on the chosen
  exposures each date, keeping the residual. Report raw and residual IC side by
  side rather than replacing one with the other.
- Exposures: sector (once a map exists), market beta (rolling), size
  (log market cap — needs shares outstanding, so **free float is a known gap**;
  until then use log traded value as a proxy and label it as such).
- Decay: evaluate the same signal against forward returns at 1, 2, 3, 5, 10 and
  21 sessions and plot IC against horizon.

## Acceptance criteria

- [x] Raw and neutralized IC reported together for every strategy.
- [x] A synthetic pure-sector signal neutralizes to **exactly** zero IC.
- [x] Decay curve produced for every registered strategy.
- [x] The size-proxy substitution is stated in the output, not just the code.

## Files

| File | Change |
| --- | --- |
| `portfolio_agent/evaluation/neutralize.py` | New — residualization, exposures, `evaluate_neutralized` |
| `portfolio_agent/evaluation/decay.py` | New — `DecayCurve`, `decay_curve`, shape reading |
| `portfolio_agent/evaluation/harness.py` | `extra_horizons` and `keep_prices` on the panel builder |
| `portfolio_agent/tests/test_neutralized_ic_and_decay.py` | New — 39 tests |

## Two bugs the acceptance tests found

**A pure sector signal neutralized to IC 0.18, not 0.** When the exposures span
the signal exactly, the residual is float noise — and that noise is *not*
random. Every name in a sector runs identical arithmetic, so all of them get
the same 1e-17 value and the residual is still a perfect sector ordering three
hundred orders of magnitude below anything meaningful. Spearman does not care
about scale. `residualize` now snaps a residual whose dispersion is negligible
against the original's to exactly zero, and the report says the exposures
explained the signal entirely.

**A 252-session beta window on a strided panel was a five-year window.** The
window is stated in sessions and the panel is indexed in rows, so at
`stride=5` the rolling window spanned 1,260 sessions — and then never filled,
because the panel did not have 252 strided rows. 126 of 200 dates were being
skipped. Windows are now converted, and the conversion is reported.

## Measured

**Neutralization**, 150 NSE names, 5-day horizon, beta + size proxy:

| | raw IC | neutralized IC | retained |
| --- | --- | --- | --- |
| momentum | +0.061 (t 4.14) | +0.026 (t 2.09) | 42% |
| low_volatility | +0.061 (t 6.31) | +0.018 (t 2.16) | 29% |

Both keep a significant residual IC, but **most of what looked like alpha is
factor loading** — 58% and 71% respectively. For `low_volatility` that is
close to tautological: a low-volatility screen is a beta bet by construction,
and the number says so.

**Decay**, momentum, same universe:

| horizon | 1d | 2d | 3d | 5d | 10d | 21d |
| --- | --- | --- | --- | --- | --- | --- |
| mean IC | +0.024 | +0.030 | +0.043 | +0.042 | +0.041 | +0.043 |

The IC *rises* to day 3 and is flat out to a month. That is the signature of a
slow signal rather than microstructure, and it means a monthly rebalance keeps
essentially all of it — a conclusion the single-horizon number could not
support either way.

## Dependency worth naming

Sector neutralization needs a sector map, and none ships with the repository
(`A8`). Until one exists this task delivers beta and size neutralization only,
which is still worth having.
