# T11 — Freeze the execution and live-trading namespace

**Status:** done · **Effort:** ~1 day · **Depends on:** none
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

- [x] The forecasting path imports nothing from `execution/`.
- [x] `backtest` still runs, since it remains a secondary check.
- [x] Frozen tests still pass when run explicitly.
- [x] The README says why, not just what.

## Note

Several of the modules with broken flat imports (`A1`) live here — orchestrator,
storage, reporting, outcomes. Freezing them shrinks the surface T07 has to fix,
so doing this first makes that task smaller.

## Outcome

Done. `orchestrator.py`, `storage.py`, `reporting.py` and `outcomes.py` moved
into `portfolio_agent/execution/` with their five test files, and `run-agent`
was removed from the CLI. The namespace is the boundary: research code that
imports from `execution/` is doing something the freeze forbids, and 67 tests
assert no research module does.

Two modules deliberately did **not** move. `src/execution_sim.py` is used by
`BacktestEngine`, which is research, and `src/models.py` holds dataclasses
shared by both sides. Moving either would have made the freeze a lie about
where the boundary really is.

A dead `try/except ImportError` in the orchestrator was collapsed on the way
through — it existed to let the module run as a loose script, which is the
same ambiguity T07 removed everywhere else.

67 new tests; suite 1261 passed.
