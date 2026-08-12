# T05 — Neutralized IC and signal decay curves

**Status:** not started · **Effort:** ~3 days · **Depends on:** T04 (harness)
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

- [ ] Raw and neutralized IC reported together for every strategy.
- [ ] A synthetic pure-sector signal neutralizes to approximately zero IC —
      the test that proves the neutralization does what it claims.
- [ ] Decay curve produced for every registered strategy.
- [ ] The size-proxy substitution is stated in the output, not just the code.

## Dependency worth naming

Sector neutralization needs a sector map, and none ships with the repository
(`A8`). Until one exists this task delivers beta and size neutralization only,
which is still worth having.
