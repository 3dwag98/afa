# T11 — Freeze the execution and live-trading namespace

**Status:** not started · **Effort:** ~1 day · **Depends on:** none
**Plan reference:** `docs/forecasting_plan.html` Part 3 (frozen)

## Goal

Move the machinery for a book that will never trade behind a clear boundary,
without losing what it knows.

## Why

Under the forecasting premise, order creation, fills, tax lots, the circuit
breaker, sector caps and the live orchestrator are all machinery for placing
trades. Leaving them in the main namespace means they keep accruing maintenance
and keep appearing in every search result, while nobody exercises them.

This also resolves the "built but never switched on" half of architecture
finding `A5` by decision rather than continued deferral.

## Freeze, do not delete

The value there is accumulated correctness — gap-aware stop fills, holiday
handling, delisting logic, the tax-lot treatment. None of it is cheap to
rebuild, and all of it is right. Deleting would be a false economy.

## Scope

| Component | Disposition |
| --- | --- |
| Order creation, fills, tax lots | Move to `execution/` |
| Live orchestrator, storage, reporting, outcomes | Move to `execution/` |
| Execution simulation, liquidity tradability gates | Move to `execution/`; keep the liquidity *features* |
| Kelly sizing, circuit breaker, sector cap | Stay, used by the demoted `backtest` |
| `run-agent` command | Removed from the CLI |

## Approach

1. Create `portfolio_agent/execution/` with a README stating plainly that it is
   not maintained under the current premise, what it was for, and what would
   need checking before reviving it.
2. Move the modules, updating imports.
3. Exclude the namespace from required CI checks but keep its tests runnable,
   so it can be revived without archaeology.
4. Remove `run-agent`.

## Acceptance criteria

- [ ] The forecasting path imports nothing from `execution/`.
- [ ] `backtest` still runs, since it remains a secondary check.
- [ ] Frozen tests still pass when run explicitly.
- [ ] The README says why, not just what.

## Note

Several of the modules with broken flat imports (`A1`) live here — orchestrator,
storage, reporting, outcomes. Freezing them shrinks the surface T07 has to fix,
so doing this first makes that task smaller.
