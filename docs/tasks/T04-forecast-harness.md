# T04 — Forecast evaluation harness

**Status:** not started · **Effort:** ~1 week · **Depends on:** T03 (purged CV)
**Plan reference:** `docs/forecasting_plan.html` Part 1 (evaluation layer), Part 4

## Goal

Make forecast skill the primary measured artifact, replacing the equity curve,
and take `BacktestEngine` off the critical path for research.

## Why

The platform can currently only evaluate a signal by simulating a book. That
requires the 2,412-line engine with its order router, tax lots, circuit breaker
and Kelly sizing — machinery for a book that will never trade. It also means
every evaluation is filtered through portfolio-construction choices that have
nothing to do with whether the forecast is any good.

This resolves architecture finding `A3` by routing around the engine rather
than decomposing it: same benefit, far less work, engine left intact for the
day a portfolio question is actually asked.

## Metrics

| Metric | What it answers |
| --- | --- |
| Rank IC (Spearman), per date | Does the predicted ordering match the realized one |
| ICIR | Is that agreement stable, or one good month |
| Newey–West t-statistic on the IC series | Is it distinguishable from zero, given autocorrelation |
| Decile spread and monotonicity | Is the signal broad, or driven only by its extreme tail |
| Hit rate | Directional accuracy, per horizon |
| Forecast error distribution | Where the model is wrong, not just how often |

Monotonicity across deciles is the check that catches a signal driven entirely
by its tail — common, and expensive when mistaken for breadth.

Newey–West already exists in `src/risk_analytics.py`; the IC series is exactly
the autocorrelated series it was built for.

## Files

| File | Change |
| --- | --- |
| `portfolio_agent/evaluation/harness.py` | New: `evaluate_forecast()` |
| `portfolio_agent/evaluation/metrics.py` | New: IC, ICIR, decile, decay |
| `portfolio_agent/tests/test_forecast_harness.py` | New |

## Acceptance criteria

- [ ] Produces every metric above for any registered strategy without
      constructing a `BacktestEngine`.
- [ ] A signal built from future returns scores near-perfect IC; a random
      signal scores near zero with a t-statistic that does not reject.
- [ ] Deterministic: two runs of one configuration agree exactly.
- [ ] Output is a structured result object, renderable to a table or a report.
