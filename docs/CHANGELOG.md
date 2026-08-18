# Changelog

Breaking changes and corrections, newest first. Each entry links the task file
that carries the full argument and the numbers.

The rule for what appears here: anything that **changes a number the platform
previously reported**, anything that **removes or renames something a caller
used**, and anything that **retracts a published finding**. Additive features
are in the task log rather than here.

---

## Round three — the platform review (T19–T32)

### Numbers that changed

**The backtest was reading inputs one session staler than the evaluator.**
`BacktestEngine` sliced `df.index < decision_date` while the evaluation harness
sliced `frame.loc[:date]`. Same strategy, same date, feature vectors one full
session apart — so the IC `evaluate` reported was never the IC the backtest
traded. Both are now inclusive. **Every backtest result predating this differs
from the current one.** ([T19](tasks/T19-one-decision-date.md))

**`rule_based` was scoring 0.0 for every name under `evaluate`.** The harness
never populated `StrategyContext.weights`, so the weighted sum ran over an
empty mapping. Any `rule_based` IC, dispersion or decile spread on record was
measuring the harness, not the strategy. ([T20](tasks/T20-strategy-context-contract.md))

**`evaluate` and `backtest` were drawing different universes.** The universe
RNG is offset by `crc32(purpose)`; `backtest` passed `"backtest"` and
`evaluate` fell through to `"train"`. Measured on a 400-name cache at
`universe_size=50`: **6 names shared out of 50.**
([T22](tasks/T22-one-purge-one-universe.md))

**The backtest never loaded its warm-up.** `_load_all_data` requested history
*from* `start_date`, so no bar existed before the first scored session.
`sma_200` was undefined for the first 200 sessions of every run, and the 20-row
eligibility bar admitted those tickers anyway — at that bar, **six of the ten
features `momentum` requires are NaN, including the one it ranks on.**
([T23](tasks/T23-one-panel-policy.md))

**The embargo was computed in calendar days.** `pd.Timedelta(days=embargo)` on
a trading-session quantity: an embargo of 5 spanning a weekend excluded three
sessions. A no-op under an expanding walk-forward, and wrong the moment the
scheme changes. ([T22](tasks/T22-one-purge-one-universe.md))

### Retracted

**"`rule_based` makes almost no claims."** The published finding read *"score
dispersion is 0.016: one floor value for 98% of the universe."* That was the
harness zeroing every score, not a property of the strategy. Re-measured after
the fix: **1.0.** ([T20](tasks/T20-strategy-context-contract.md))

**`QUANT_RESEARCH.md` §12(c): "idiosyncratic momentum needs §8's data."**
False. A CAPM residual needs a market return and a rolling beta, both of which
T14 built. Now shipped as `residual_momentum`.
([T27](tasks/T27-residual-momentum.md))

**`QUANT_RESEARCH.md` §7: pairs trading blocked by `BaseStrategy`'s per-ticker
interface.** Half right. The obstruction was one layer away — the *feature*
layer could not express a relationship between two tickers, and T24's
cross-sectional registry removed it. `BaseStrategy` never had to change.
([T30](tasks/T30-pairs-cointegration.md))

### Removed or renamed

| Was | Now | Why |
|---|---|---|
| `register_strategy(name, cls)` | `@register_strategy(name)` | Matches the other three registries. `rule_based` was registered twice, from two files. ([T25](tasks/T25-strategy-ergonomics.md)) |
| `_rank_and_select_decile` | `rank_and_select` | Everything a new cross-sectional strategy needs was module-private. ([T25](tasks/T25-strategy-ergonomics.md)) |
| `hasattr(strategy, "load")` | `BaseStrategy.load()` | The probe was the shape of a contract that existed in practice and not in the type. ([T25](tasks/T25-strategy-ergonomics.md)) |
| `_get_historical_data_up_to` | `_history_through` | The old name did not say whether "up to" included the date. ([T19](tasks/T19-one-decision-date.md)) |
| `neutralize.rolling_beta` (impl.) | `features/market_relative.rolling_beta` | Beta is a characteristic of a stock; the old home imported `market_composite` from the feature layer to compute it. The name still works and delegates. ([T24](tasks/T24-cross-sectional-features.md)) |
| two `position_scale` parsers | `StrategySignal.scaled_quantity` | Byte-identical copies in the engine and the orchestrator, kept in step by a comment. ([T25](tasks/T25-strategy-ergonomics.md)) |
| `TRAINING_FEATURE_NAMES`, `DEFAULT_GBM_FEATURES` | `features/sets.py` | Two hardcoded copies of eight names, neither derived from any strategy. ([T21](tasks/T21-one-feature-set.md)) |

### Behaviour preserved on purpose

**An unparseable `position_scale` still sizes a position *fully*.** Both sizing
paths have always done this. It is arguably backwards — a corrupt multiplier is
a bug, and sizing fully on a bug is the more expensive failure — but tightening
it is a risk decision rather than an ergonomics one, and burying a behaviour
change in a refactor is how one escapes review. Recorded in the docstring and
pinned by a test that says it is preserved rather than chosen.
([T25](tasks/T25-strategy-ergonomics.md))

### New dependency

`statsmodels>=0.14.0` is now declared. It was already present transitively via
`arch`; the Engle-Granger test needs MacKinnon's critical values, and relying
on another package's dependency for something load-bearing is how a working
install breaks on an unrelated upgrade. ([T30](tasks/T30-pairs-cointegration.md))

---

## Round two — the measurement review (T12–T18)

**The information coefficient was reported four different ways, and the one
driving model selection had the wrong sign.** `agents/trainer.py` pooled every
observation into a single rank correlation, which measures whether a score
tracks the market's *level* rather than whether it ordered any day's
cross-section. On a signal that orders every date perfectly while its level
runs against the market, the pooled figure is **−0.99** and the per-date figure
is **+1.00**. ([T12](tasks/T12-one-rank-ic.md))

**Every decile spread published before T13 was gross of costs**, while an
accurate NSE cost model sat unused in `execution_sim.py`. The round trip is
**0.79%** at 25 bps a side. Costs are charged by default now; `--gross` opts
out. ([T13](tasks/T13-net-of-costs.md))

**The default configuration could not build one clean validation sample.** Each
ticker's 15% split of `min_history_days=250` is ~37 rows against a 60-row
sequence window, so **every** validation and test sequence spanned two stocks.
The held-out metrics were computed on histories no single stock experienced.
([T18](tasks/T18-sequence-boundaries.md))

**Low volatility now sorts on the CAPM residual, not total volatility.**
Different sort, different names, different results — total volatility mixes
beta and idiosyncratic volatility, and the 2025 literature finds only the
second survives out of sample. The old sort is still available as
`low_volatility`. ([T14](tasks/T14-idiosyncratic-volatility.md))

**Two families of risk arithmetic disagreed by 2.5x on position size**, and the
dead one understated portfolio volatility 4.1x by assuming zero correlation.
The second family was deleted. ([T17](tasks/T17-src-restructure.md))

---

## Round one — the forecasting pivot (T01–T11)

**`run-agent` was removed** and `portfolio_agent/execution/` frozen. Nothing in
this repository places an order; the live path is kept for reference and not
improved. Documentation still describing `run-agent` was corrected in T33.
([T11](tasks/T11-freeze-execution.md))

**Ingest keeps the adjustment data it was discarding**, and history went from 5
years to 20. Any cached parquet written before T01 lacks the adjustment columns
and should be rebuilt: `portfolio-agent data build --years 20`.
([T01](tasks/T01-preserve-adjustment-data.md))

**The package became installable** (`portfolio-agent`, `--config`), which moved
every entry point. ([T07](tasks/T07-installable-package.md))

---

## Standing caveats

These are not changes; they are properties of every number the platform
produces, and each is printed in the run's own notes.

| Missing input | What the result then is |
|---|---|
| point-in-time membership | ranked against the names that survived to be downloaded — **~4.94pp of annual return overstatement** on Indian indices, larger than either strategy's neutralized alpha |
| a sector map | not sector-neutral, and Indian momentum concentrates hard by sector |
| fundamentals | controls for no accounting characteristic |
| free float | size is a traded-value proxy, not market cap |
| FII/DII flows | no flow-regime conditioning |

**Nothing since round two has been run against real market data.** The measured
numbers throughout the task log are measured on the code and on synthetic
panels built to have the relevant shape. Whether Indian equity data has that
shape is the next thing to find out, and it needs the cache.
