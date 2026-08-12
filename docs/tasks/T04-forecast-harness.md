# T04 — Forecast evaluation harness

**Status:** done · **Effort:** ~1 week · **Depends on:** T03 (purged CV)
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
| `portfolio_agent/evaluation/metrics.py` | New — IC, ICIR, Newey–West, buckets, rank error, dispersion, decay |
| `portfolio_agent/evaluation/harness.py` | New — `evaluate_forecast`, `build_forecast_panel`, `evaluate_panel`, `ForecastEvaluation` |
| `portfolio_agent/evaluation/__init__.py` | New — package surface |
| `portfolio_agent/tests/test_forecast_harness.py` | New — 47 tests |

## Acceptance criteria

- [x] Produces every metric above for any registered strategy without
      constructing a `BacktestEngine`. *(Asserted literally: the engine's
      `__init__` is patched to raise, and a full panel is built through it.)*
- [x] A signal built from future returns scores near-perfect IC; a random
      signal scores near zero with a t-statistic that does not reject.
- [x] Deterministic: two runs of one configuration agree exactly.
- [x] Output is a structured result object, renderable to a table or a report.

## Three things worth knowing

**The null test needed to be a calibration test.** A single random panel that
fails to reject is also consistent with a statistic that never rejects
anything, and one that *does* reject is consistent with a correctly-sized test
having an ordinary bad day — the first seed tried produced p=0.004 on pure
noise. The real claim is the rejection *rate*, so the test runs 60 independent
null panels and checks it lands near the nominal 5%. It measures 3.3%.

**A perfect signal read as "not significant" until it was fixed.** An oracle's
IC is exactly 1.0 on every date, so the series has zero dispersion and the
Newey–West standard error is zero. Returning `t=0, p=1` there labels the
strongest evidence available insignificant. The limit is infinite, and
`_ratio` now distinguishes zero-over-zero from non-zero-over-zero.

**The Newey–West lag has to follow the sampling stride, not just the horizon.**
Sampled daily a 5-day label overlaps its next four neighbours; sampled every
fifth day it overlaps none of them. Using `horizon - 1` regardless corrects for
an overlap that is not there — conservative, but wrong, and wrong precisely on
the fast runs someone strided to make cheap. `overlap_lags(horizon, stride)`
computes it.

## Measured

150 NSE names from the parquet cache, 5-day horizon, 150 evaluation dates:

| | momentum | low_volatility |
| --- | --- | --- |
| mean rank IC | +0.0425 | +0.0509 |
| Newey–West t | +4.08 | +4.94 |
| decile spread | +0.19% | **-0.01%** |
| monotonicity | +0.37 | **-0.18** |

Both have a statistically significant IC. Only one of them has a decile profile
that a long-only book could act on — `low_volatility` ranks the cross-section
better than momentum does and still has a *negative* top-minus-bottom spread.
Separating those two facts is the reason this task exists; an equity curve
reports their product and never their difference.
