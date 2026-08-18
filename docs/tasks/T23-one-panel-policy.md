# T23 — One panel policy

**Status:** done · **Effort:** ~1 day · **Depends on:** T19, T21
**Review reference:** round-three plan, finding C

## Goal

The three settings that decide what a feature *is* — how much history it gets,
whether it is normalized, and which labels survive — hold the same value on
every path that builds a panel.

## Why

T19 made the three paths agree about what day it is. T21 made them agree about
which features to build. Three settings were left, and each of them changes the
number a feature reports rather than which feature is reported. All three
differed silently.

### Minimum history was a guess in four places

| path | threshold |
| --- | --- |
| `src/backtest_engine.py` | 20 rows |
| `evaluation/harness.py` | 252 rows |
| `training/data.py` | 252 rows |
| `agents/trainer.py` | `data.min_history_days` |

Four numbers, four modules, no agreement — and none of them derived from
anything. All four were reaching for the same quantity: the longest lookback
among the features actually requested. That is a property of the request, not a
constant, so `features/pipeline.warmup_rows` now computes it. Each feature is
built once on a synthetic probe series and the first row where it resolves is
its warm-up; the answer for a set is the largest.

Measured:

| requested by | rows |
| --- | --- |
| `momentum` | 211 |
| `india_sac` | 211 |
| `rule_based` | 201 |
| `low_volatility_idio` | 62 |
| `default` feature set | 51 |
| `tradability` feature set | 21 |

Against the backtest's 20-row bar, six of the ten features `momentum` requires
are still NaN on the last row:

    mom_9m_skip1m, realized_vol_60, traded_value_60,
    zero_return_fraction_60, circuit_lock_fraction_60,
    operator_trap_fraction_60

`mom_9m_skip1m` is the value the strategy ranks on. Nothing here was a
borderline call about how much history is *enough* — the ranking key did not
exist.

A probe rather than a declared constant matters for what comes next. A feature
added with a three-year lookback raises the threshold on the day it is
registered; a constant would keep its old value and start scoring NaN.

### And the warm-up was never loaded

Raising the bar surfaced the reason it had been low. `_load_all_data` requested
each ticker's data **from `start_date`**:

```python
df = load_ticker_data(ticker,
                      start_date=self.start_date.strftime('%Y-%m-%d'),
                      end_date=self.end_date.strftime('%Y-%m-%d'))
```

A backtest beginning 2023-01-02 therefore had no bar before 2023-01-02. `sma_200`
was undefined for the first 200 scored sessions no matter how much history the
cache actually held, and the 20-row eligibility bar admitted those tickers
anyway. The opening months of every backtest on record ranked the universe on
undefined numbers.

The engine now loads from `start_date` minus the warm-up, converted to calendar
days at 1.6x (7/5 for weekends, the remainder for exchange holidays).
Over-reading costs one wider slice of a parquet file; under-reading costs the
warm-up. The scored window is unchanged — the extra bars are warm-up only, and
a test asserts `master_date_index` still begins on `start_date`.

This is also why `test_parallel_determinism`'s fixture had to move. Its
docstring claimed "enough history for the rule-based strategy" while the series
began on the backtest's own start date. Those tests were exercising the
parallel machinery on undefined features, and the fixture now starts two years
earlier.

### Normalization was hardwired off in one path

`evaluation/harness.py` and both trainers read `features.normalize` from the
config. `backtest_engine.py` called `build_features` with no `normalize`
argument at all, taking the pipeline's own default of `False`. The two agree
only because the shipped `config.yaml` sets `normalize: false` against a schema
default of `True` — the day anyone relies on the schema default, the evaluation
path z-scores and the backtest does not.

The consequence is worse than a rescale. `_normalize_features` applies its own
`.shift(1)`:

```python
normalized = (series - rolling_mean) / rolling_std
return normalized.shift(1)
```

So a normalize divergence is not a difference in units, it is **a second
session of lag** — precisely the disagreement T19 removed, re-created through a
config value. `agents/backtester.py` now maps both `features.normalize` and
`features.normalize_window` into the engine, and the two parallel-worker
globals carry them so `--parallel` cannot change a signal either.

### The outlier filter sat on one side of a fork

`agents/trainer.prepare_features` has dropped labels beyond ±5.0 since a run was
poisoned by a single bad bar — a split that escapes adjustment produces an
eleven-million-percent "return", and one such row dominates a squared-error
objective completely.

`build_gbm_panel` assembles its label itself rather than going through
`prepare_features`, so it never had the filter. Both now call
`features/labels.drop_absurd_labels`.

**Dropped, not clipped.** A clip piles a spike of samples at the bound and
teaches the model that the bound is a common outcome, trading one distortion
for a subtler one. The ±5.0 default admits any genuinely reachable Indian move
— five consecutive 20% upper circuits compound to +149% — and rejects only
arithmetic that cannot be a price.

## What changed

- `features/pipeline.py` — `warmup_rows(feature_names)` and the cached
  per-feature probe behind it.
- `features/labels.py` — `drop_absurd_labels`, `DEFAULT_MAX_ABS_LABEL`.
- `src/backtest_engine.py` — `_required_history_rows`; loads from
  `start_date` minus the warm-up; eligibility uses the derived threshold rather
  than 20; `feature_normalize` / `feature_normalize_window` constructor
  parameters threaded through both `build_features` call sites and the worker
  globals.
- `agents/backtester.py` — maps both normalization fields from `AppConfig`.
- `agents/trainer.py` — the inline outlier block calls the shared filter.
- `training/trainers/gbm.py` — applies the shared filter after labelling.
- `tests/test_paths_agree.py` — the warm-up class.
- `tests/test_parallel_determinism.py` — fixture extended to 780 sessions
  beginning 2021-01-04.

## What this does not do

`_required_history_rows` is the *feature* warm-up. It is not a statement about
how much history a strategy needs to be meaningful — a 60-session volatility is
defined at 62 rows and still noisy at 62 rows. Sample-adequacy is a separate
question and remains the caller's.

Two thresholds also survive outside the engine: the harness and
`training/data.py` still carry their own 252. They are now redundant rather than
wrong (252 exceeds every measured warm-up), and folding them in is left to the
task that gives those two a shared panel builder.

## Verification

```bash
python -m pytest portfolio_agent/tests/test_paths_agree.py -q
python -m pytest portfolio_agent/tests/test_parallel_determinism.py -q
```

The warm-up numbers in the table above are reproducible:

```python
from portfolio_agent.features.pipeline import warmup_rows
from portfolio_agent.strategies.registry import load_strategy
from portfolio_agent.config.schema import StrategyConfig

s = load_strategy(StrategyConfig(type="momentum", params={}))
warmup_rows(s.required_features())   # 211
```
