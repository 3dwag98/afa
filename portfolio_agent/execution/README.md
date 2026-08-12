# execution/ — frozen

**This code is not maintained.** It works, its tests pass, and nothing in the
research path depends on it. It is here rather than deleted because what it
knows is expensive to rebuild and cheap to keep.

## Why it was frozen

The platform's purpose changed: it forecasts on historical data and places no
trades. Order creation, fills, tax lots and the live daily run are machinery
for a book that does not exist. Left in the main namespace they would keep
accruing maintenance and keep appearing in every search result while nobody
exercised them — the "built but never switched on" state that
`docs/architecture_review.html` flags as `A5`, resolved here by decision rather
than continued deferral.

## What is here

| Module | What it does |
| --- | --- |
| `orchestrator.py` | The daily live run: score, size, comply, report, persist |
| `storage.py` | SQLite persistence for recommendations, outcomes and the agent brain |
| `reporting.py` | The Excel recommendation report |
| `outcomes.py` | Marking past recommendations to market, feeding the learning loop |

Their tests moved with them and still pass.

## What deliberately did *not* move

- **`src/execution_sim.py`** — the transaction-cost model. `BacktestEngine`
  uses it, and the backtest is kept as a secondary check.
- **`src/models.py`** — the shared dataclasses. Read by `indicators.py`,
  `learning.py` and the backtest engine.
- **Kelly sizing, the circuit breaker, the sector cap** — all still reachable
  from the backtest.
- **The liquidity *features*.** Only the tradability *gate* was a live concern;
  the measurements are useful inputs to a forecast.

## What this code knows that is worth keeping

Rebuilding any of this would mean rediscovering the same edge cases:

- **Gap-aware stop fills.** A stop is filled at `min(open, stop)` for longs and
  `max(open, target)` for take-profits, with the realized distance recorded.
  An overnight gap is where a large share of an Indian equity's move happens,
  and filling at the stop price pretends the gap did not occur.
- **Holiday and delisting handling.** Orders landing on a non-session are
  filled at the next one; delisted tickers are closed rather than held forever
  at a stale price.
- **Tax lots.** Entry prices tracked per lot rather than averaged, so realized
  gains are computed against what was actually paid.

## Reviving it

If the platform ever places trades again, check these before trusting anything
here:

1. **Settlement.** India moved to T+1 in January 2023 and runs an optional T+0
   for a growing list of scrips. Any assumption in this code predates that.
2. **The cost model.** `execution_sim` uses a flat basis-point figure. It was a
   placeholder then and it is a placeholder now.
3. **Circuit bands and surveillance.** `ASM`/`GSM`/`ESM` state and daily price
   bands are not modelled, so this code will happily "trade" a session that was
   never tradeable. See `docs/data_architecture.html`.
4. **Series.** `BE` and `BZ` are trade-to-trade — compulsory delivery, no
   intraday exit. Not modelled anywhere.

## CI

Excluded from the required checks. The tests still run on request:

```bash
pytest portfolio_agent/execution/
```

Keep them passing if you touch the modules they cover. If that becomes a
burden, that is the signal to delete rather than to let them rot.
