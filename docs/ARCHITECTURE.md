# AFA Architecture

How the platform is put together, what runs where, and — in the most detail —
how the **model/strategy layer** works, since that is the part you extend.

Companion documents:

- **[STRATEGIES.md](STRATEGIES.md)** — the plug-and-play guide to creating,
  updating and deleting strategies.
- **[QUANT_RESEARCH.md](QUANT_RESEARCH.md)** — the research basis for each
  strategy and risk model.

## Contents

- [The one-paragraph version](#the-one-paragraph-version)
- [System overview](#system-overview)
- [The strategy layer](#the-strategy-layer)
  - [The interface](#the-interface)
  - [The three dispatch paths](#the-three-dispatch-paths)
  - [Registry and configuration flow](#registry-and-configuration-flow)
  - [Ensembles (UMAs)](#ensembles-umas)
  - [The ML strategy and the training loop](#the-ml-strategy-and-the-training-loop)
  - [Self-learning weights](#self-learning-weights)
- [The backtest engine](#the-backtest-engine)
- [The live agent](#the-live-agent)
- [Concurrency map](#concurrency-map)
- [Device selection](#device-selection)
- [Report generation and data lineage](#report-generation-and-data-lineage)
- [Determinism guarantees](#determinism-guarantees)
- [Module map](#module-map)

## The one-paragraph version

Cached OHLCV parquet files are turned into feature matrices by a **feature
registry**, scored by a **strategy** looked up from a **strategy registry**,
sized and gated by **risk and compliance** rules, and written out as Excel
reports. The same strategy object is used by the live agent and the backtest
engine, so a strategy cannot behave one way in a backtest and another way
live. Everything expensive — data loading, per-ticker Monte Carlo, model
training, model inference — is parallelized, but always in a way that leaves
results identical to the serial path.

## System overview

```mermaid
flowchart TB
    subgraph ingest["Data layer"]
        YF["yfinance"] -->|"batch_download_and_cache()<br/>thread pool"| CACHE[("data/market_data/<br/>*.parquet")]
        CACHE --> LOAD["load_ticker_data()"]
    end

    subgraph feat["Feature layer"]
        LOAD --> BUILD["build_features(df, names)<br/>features/pipeline.py"]
        FREG["feature registry<br/>features/registry.py"] -.->|"lag-safe indicators"| BUILD
    end

    subgraph strat["Strategy layer"]
        BUILD --> SCORE["strategy.score() /<br/>score_batch()"]
        SREG["strategy registry<br/>strategies/registry.py"] -.->|"load_strategy(config)"| SCORE
        MC["Monte Carlo<br/>src/monte_carlo.py"] -.->|"StrategyContext.mc_result"| SCORE
        BRAIN["learned weights<br/>data/agent_brain.json"] -.->|"StrategyContext.weights"| SCORE
    end

    SCORE --> SIG["StrategySignal per ticker"]

    subgraph consume["Consumers"]
        SIG --> LIVE["live agent<br/>src/orchestrator.py"]
        SIG --> BT["backtest engine<br/>src/backtest_engine.py"]
    end

    subgraph out["Outputs"]
        LIVE --> RISK1["risk sizing + compliance"] --> XL1["Agent_Orchestrator_Output.xlsx"]
        LIVE --> DB[("SQLite<br/>recommendations,<br/>outcomes")]
        BT --> EXEC["execution simulator<br/>costs, slippage, tax"] --> ANA["RiskAnalyzer"] --> XL2["Backtest_Report.xlsx"]
    end
```

The important structural property is the **single scoring path**: both
consumers call the same `BaseStrategy` object built by the same registry from
the same config. There is no separate "backtest scoring" implementation to
drift out of sync with the live one.

## The strategy layer

### The interface

Every strategy — built-in or yours — implements one small interface.

```mermaid
classDiagram
    class BaseStrategy {
        <<abstract>>
        +name
        +supports_gpu_batch
        +requires_full_batch
        +required_features()
        +score(symbol, features, context)
        +score_batch(features_by_symbol, context)
        +entry_rules()
        +exit_rules()
    }

    class RuleBasedStrategy {
        trend, breakout, volume, Monte Carlo
        YAML-configured weights
    }
    class MomentumStrategy {
        cross-sectional ranking
        requires_full_batch is True
    }
    class LowVolatilityStrategy {
        cross-sectional ranking
        requires_full_batch is True
    }
    class MLStrategy {
        supports_gpu_batch is True
        +load()
    }
    class EnsembleStrategy {
        members with weights
        weighted_blend or vote
    }

    BaseStrategy <|-- RuleBasedStrategy
    BaseStrategy <|-- MomentumStrategy
    BaseStrategy <|-- LowVolatilityStrategy
    BaseStrategy <|-- MLStrategy
    BaseStrategy <|-- EnsembleStrategy
    EnsembleStrategy o-- BaseStrategy : members
```

Only three members are mandatory: `name`, `required_features()`, and
`score()`. The two boolean properties are how a strategy tells its callers how
it wants to be invoked.

Inputs and outputs are fixed shapes (`strategies/types.py`):

```mermaid
flowchart LR
    subgraph in["Inputs to score()"]
        F["features: DataFrame<br/>one row per bar,<br/>one column per required feature"]
        C["StrategyContext<br/>• risk: RiskParams<br/>• weights: learned component weights<br/>• mc_result: MonteCarloResult | None<br/>• run_id"]
    end
    subgraph out["Output"]
        S["StrategySignal<br/>• signal: BUY/SELL/HOLD/WATCH/AVOID<br/>• score: 0-100<br/>• trigger, entry/stop/target<br/>• reward_risk, probability_profit<br/>• component_scores, rationale, extra"]
    end
    F --> SC["score()"] --> S
    C --> SC
```

`rationale` is not decoration — it is what lands in the Excel report's
rationale column and explains to a human why a ticker did or did not qualify.

### The three dispatch paths

A caller picks **one** of three ways to score a universe, based on the two
boolean properties. This is the single most important diagram for
understanding strategy performance:

```mermaid
flowchart TD
    START["universe of eligible tickers"] --> Q1{"requires_full_batch<br/>or supports_gpu_batch?"}

    Q1 -->|yes| BATCH["score_batch(all tickers, context)<br/>ONE call"]
    Q1 -->|no| Q2{"parallel enabled<br/>and >1 ticker?"}

    Q2 -->|yes| POOL["process pool<br/>one task per ticker<br/>results reassembled in universe order"]
    Q2 -->|no| LOOP["plain loop over score()"]

    BATCH --> B1["MLStrategy: one stacked tensor,<br/>one GPU forward pass"]
    BATCH --> B2["Momentum / LowVolatility:<br/>ranking needs the whole universe"]

    POOL --> R["dict[ticker, StrategySignal]"]
    LOOP --> R
    B1 --> R
    B2 --> R
```

The two flags mean different things and should not be confused:

| Property | Means | Set by | Consequence of getting it wrong |
|---|---|---|---|
| `supports_gpu_batch` | "`score_batch()` is a genuine batched forward pass" | `MLStrategy` | Performance only — you lose GPU batching |
| `requires_full_batch` | "my signal depends on ranking against the whole universe" | `MomentumStrategy`, `LowVolatilityStrategy` | **Correctness** — ranking against a universe of one is meaningless |

A cross-sectional strategy scored one ticker at a time is not slow, it is
wrong: the top decile of a single stock is always that stock.

### Registry and configuration flow

```mermaid
sequenceDiagram
    participant CLI as cli.py
    participant CFG as config/loader.py
    participant REG as strategies/registry.py
    participant S as YourStrategy
    participant YAML as config/strategies/*.yaml

    CLI->>CFG: load_config()
    CFG-->>CLI: AppConfig (config.yaml + AFA_* env overrides)
    CLI->>CLI: strategy_config = config.strategy.copy()<br/>type = --strategy, config_path = --strategy-config
    CLI->>REG: load_strategy(strategy_config)
    REG->>REG: look up STRATEGY_REGISTRY[type]
    REG->>S: YourStrategy(strategy_config)
    S->>YAML: read config_path (optional)
    YAML-->>S: rules / weights / members
    REG-->>CLI: BaseStrategy instance
    opt strategy defines load()
        CLI->>S: load()
        Note over S: e.g. read models/lstm_best.pt
    end
    CLI->>S: score() / score_batch()
```

The registry is explicit — `register_strategy("name", Class)` — rather than
import-scanning, so the set of available strategies is knowable statically and
`portfolio-agent list-strategies` can never lie.

### Ensembles (UMAs)

A UMA is itself just a strategy, so it can be backtested, run live, or nested
inside another UMA.

```mermaid
flowchart TD
    UMA["EnsembleStrategy<br/>(type: ensemble)"] --> Y["example_uma.yaml<br/>method: weighted_blend | vote"]
    UMA --> M1["member: rule_based (0.6)"]
    UMA --> M2["member: lstm (0.4)"]

    M1 --> S1["StrategySignal"]
    M2 --> S2["StrategySignal"]

    S1 --> COMB{"method"}
    S2 --> COMB

    COMB -->|weighted_blend| WB["map signal to strength<br/>BUY=1, WATCH=0.3, HOLD=0,<br/>AVOID=-0.3, SELL=-1<br/>weighted mean of strength,<br/>score, prices, probability"]
    COMB -->|vote| VT["bucket to BUY/SELL/HOLD<br/>majority: >50% agree<br/>unanimous: all agree"]

    WB --> OUT["combined StrategySignal"]
    VT --> OUT
```

Two consequences worth knowing:

- A UMA is **not** GPU-batched even when a member is. Members are scored
  per-ticker so that a rule-based member receives its genuine per-ticker Monte
  Carlo result. For maximum ML throughput, run that strategy directly.
- Cross-sectional strategies (`momentum`, `low_volatility`) **cannot** be UMA
  members, because a UMA scores members per-ticker and their ranking would
  degenerate.

### The ML strategy and the training loop

Training and inference are two separate programs joined by a checkpoint and a
metadata file:

```mermaid
flowchart LR
    subgraph train["portfolio-agent train"]
        direction TB
        T1["resolve universe"] --> T2["per-ticker load + featurize<br/>process pool"]
        T2 --> T3["70/15/15 split per ticker,<br/>then concatenate"]
        T3 --> T4["TimeSeriesDataset<br/>sliding windows"]
        T4 --> T5["DataLoader<br/>pin_memory on CUDA"]
        T5 --> T6["train loop<br/>AMP on CUDA, early stopping"]
    end

    T6 --> CKPT[("models/lstm_best.pt")]
    T6 --> META[("models/metadata.json<br/>feature_names, target,<br/>sequence_length")]

    subgraph infer["backtest / run-agent with --strategy lstm"]
        direction TB
        L["ModelLoader.load_model()"] --> RF["required_features()<br/>= metadata feature_names"]
        RF --> BF["build_features per ticker"]
        BF --> STACK["stack last sequence_length rows<br/>for every eligible ticker"]
        STACK --> FWD["ONE forward pass<br/>(n_tickers, seq_len, n_features)"]
        FWD --> P["prediction -> probability -> signal"]
    end

    CKPT --> L
    META --> L
```

`metadata.json` is the contract between the two halves: the feature list the
model was trained on becomes the feature list the strategy requests at
inference time, so the two can never silently disagree.

Panel construction detail worth knowing: each ticker is split 70/15/15
*individually* and then all train parts are concatenated, followed by all
validation parts, then all test parts. That ordering makes the dataloader's
single top-level 70/15/15 index split land exactly on those boundaries, so
validation and test represent every ticker rather than only the last few.

### Self-learning weights

The rule-based scoring components carry weights that adapt to realized
outcomes. The same pure function drives both the live agent and the backtest,
so learning cannot drift between them.

```mermaid
flowchart LR
    TH["realized trade history"] --> EL["evaluate_and_learn()<br/>strategies/weighting.py"]
    W0["current weights"] --> EL
    EL --> W1["adjusted weights<br/>(normalized to 100)"]
    W1 --> BRAIN[("agent_brain.json<br/>live")]
    W1 --> SNAP["brain_evolution snapshots<br/>(backtest, every 20 trading days)"]
    SNAP --> SHEET["Brain_Evolution sheet"]
```

## The backtest engine

Each trading day is replayed in real market order. The ordering is not
cosmetic: it is what keeps the equity curve honest and prevents look-ahead.

```mermaid
sequenceDiagram
    autonumber
    participant D as Day T
    participant E as BacktestEngine
    participant X as ExecutionSimulator
    participant S as Strategy

    D->>E: A. fill orders queued on T-1, at T's OPEN
    E->>X: costs, slippage, market impact, capital gains tax
    X-->>E: adjusted price + friction
    D->>E: B. check stops/targets against T's HIGH/LOW
    D->>E: C. liquidate anything that stopped trading
    D->>E: D. mark to market at T's CLOSE
    Note over E: this is the day's equity point —<br/>it includes everything above
    D->>S: E. score universe using data strictly BEFORE T
    S-->>E: signals
    E->>E: F. queue T+1 orders (SELLs first, then BUYs by score)
    E->>E: G. every 20 days, evaluate_and_learn()
```

Look-ahead prevention is structural, not incidental:

- Signals for day T are computed from `df[df.index < T]` — strictly earlier bars.
- Orders decided on day T execute at **T+1's open**, never at a price known
  when the decision was made.
- Position sizing uses T's end-of-day equity, which is known before T+1's open.

Position accounting is tracked explicitly in `open_positions` (cost basis,
first entry date, quantity), which is what makes realized P&L, holding period
and STCG/LTCG classification correct in the trade log.

## The live agent

```mermaid
flowchart TD
    A["run-agent"] --> B["init SQLite, load brain"]
    B --> C["load trade outcomes -> brain.trade_history"]
    C --> D["evaluate_and_learn() -> updated weights"]
    D --> E["load_strategy(config.strategy)"]
    E --> F["load_or_fetch_data()"]
    F --> G["per-ticker prep: indicators + Monte Carlo + features<br/>PROCESS POOL (order-stable)"]
    G --> H{"requires_full_batch<br/>or supports_gpu_batch?"}
    H -->|yes| I["score_batch(all tickers)"]
    H -->|no| J["score() per ticker with its own mc_result"]
    I --> K["position sizing (fixed-fractional or fractional Kelly)"]
    J --> K
    K --> L["compliance checks"]
    L --> M["rank by score, persist to SQLite"]
    M --> N["Excel report + run log"]
```

## Concurrency map

Each hot path uses the executor that matches the work, and every one of them
is result-identical to its serial equivalent.

```mermaid
flowchart TB
    subgraph threads["Threads — network-bound"]
        DL["chunk downloads<br/>data/download_workers (4)<br/>src/data_store.py"]
    end
    subgraph procs["Processes — CPU-bound"]
        TP["training panel build<br/>training/data_load_workers<br/>agents/trainer.py"]
        BS["backtest signal scoring<br/>--parallel --workers N<br/>src/backtest_engine.py"]
        LP["live per-ticker prep<br/>data/ticker_prep_workers<br/>src/orchestrator.py"]
    end
    subgraph gpu["GPU — batched tensor math"]
        TR["training loop (AMP, cuDNN benchmark)"]
        INF["ML inference: one stacked forward pass per day"]
    end
    subgraph dl["DataLoader workers"]
        DW["training/num_workers (2)<br/>pin_memory on CUDA"]
    end
```

| Path | Executor | Why | Knob |
|---|---|---|---|
| Market data download | `ThreadPoolExecutor` | Blocked on sockets; the GIL is released during I/O, and processes would only add pickling cost | `data.download_workers`, `download-data --workers` |
| Training panel build | `ProcessPoolExecutor` | Parquet decode + indicator math is CPU-bound Python | `training.parallel_data_loading`, `training.data_load_workers` |
| Backtest signal scoring | `ProcessPoolExecutor` | Monte Carlo + indicator math per ticker per day | `backtest --parallel --workers N` |
| Live per-ticker prep | `ProcessPoolExecutor` | Same work as above, once per run | `data.parallel_ticker_prep`, `data.ticker_prep_workers` |
| Batch tensor training | GPU | Matrix math | `training.device`, `use_mixed_precision`, `use_torch_compile` |
| ML inference | GPU | All eligible tickers in one forward pass | `backtest --device cuda` |
| Batch feeding | DataLoader workers | Overlaps host-side batch prep with device compute | `training.num_workers` |

Two design rules keep this safe:

1. **The pool is per run, not per unit of work.** The backtest engine creates
   one worker pool for the whole run and installs the run-constant inputs
   (strategy, feature list, risk params, Monte Carlo settings) through a pool
   initializer. Only the per-day varying arguments travel with each task.
2. **Results are reassembled in input order**, never in completion order — see
   [Determinism guarantees](#determinism-guarantees).

## Device selection

One resolver decides the device, and it never hands back an accelerator that
PyTorch cannot actually use.

```mermaid
flowchart TD
    REQ["requested device<br/>(--device or config.training.device)"] --> AUTO{"auto?"}
    AUTO -->|yes| C1{"cuda available?"}
    C1 -->|yes| CUDA["cuda"]
    C1 -->|no| M1{"mps available?"}
    M1 -->|yes| MPS["mps"]
    M1 -->|no| CPU["cpu"]

    AUTO -->|"no: cuda"| C2{"cuda available?"}
    C2 -->|yes| CUDA
    C2 -->|no| WARN["ONE warning + reason + fix"] --> CPU

    AUTO -->|"no: mps"| M2{"mps available?"}
    M2 -->|yes| MPS
    M2 -->|no| WARN

    AUTO -->|"no: cpu"| CPU

    CUDA --> BACK["written back to config.training.device<br/>so dataloaders, AMP and checkpoint<br/>metadata all agree"]
    MPS --> BACK
    CPU --> BACK
```

When CUDA is requested but unavailable, the diagnostic distinguishes the cases
that need different fixes:

| Symptom | Meaning | Fix |
|---|---|---|
| `torch.version.cuda is None` | CPU-only PyTorch wheel — what PyPI ships for Windows, so `uv sync --extra gpu` alone is not enough | Install a CUDA build from the PyTorch index |
| Built with CUDA, `CUDA_VISIBLE_DEVICES` empty | GPUs hidden from this process | Unset the variable |
| Built with CUDA, no device visible | Missing/older driver, or no NVIDIA GPU | Update the driver, or install a build for an older CUDA |

Run `portfolio-agent gpu-check` to see which case applies.

## Report generation and data lineage

```mermaid
flowchart LR
    ENG["BacktestEngine"] --> EQ["daily_equity_curve"]
    ENG --> TL["trade_log"]
    ENG --> BE["brain_evolution"]
    ENG --> DAL["daily_activity_log"]

    EQ --> RA["RiskAnalyzer<br/>(seeded bootstrap MC)"]
    TL --> RA
    RA --> AN["analytics report"]

    AN --> XL["export_backtest_excel()"]
    EQ --> XL
    TL --> XL
    BE --> XL
    DAL --> XL

    XL --> S1["Executive_Summary"]
    XL --> S2["Equity_Curve + charts"]
    XL --> S3["Trade_Log (16 cols)"]
    XL --> S4["Daily_Trade_Log (11 cols)"]
    XL --> S5["Monthly_Heatmap"]
    XL --> S6["Brain_Evolution"]
    XL --> S7["Monte_Carlo_Simulations"]
```

Two contracts govern the numbers that reach Excel:

- **Units.** Every percentage metric crosses the boundary in *percent* units
  (`18.5` means 18.5%), declared per metric in
  `backtest_reporting.py::SUMMARY_METRICS`. The exporter divides by 100 once,
  because Excel percent formats multiply by 100 on display. Ratios (Sharpe,
  Sortino, profit factor) are written as plain numbers.
- **Columns.** `EXPECTED_COLUMNS` (16) and `EXPECTED_DAILY_COLUMNS` (11) are
  the only trade-log schemas. Values are written at looked-up column indices,
  never hardcoded positions.

Realized P&L reads `net_pnl` and counts only **closed round trips** — an open
BUY leg carries `exit_date: None` and a negative P&L (its transaction cost),
and counting it would report every open position as a losing trade.

## Determinism guarantees

The same inputs produce the same report, whether or not you pass `--parallel`
and no matter how many workers you use. Three mechanisms provide that:

```mermaid
flowchart TD
    W1["worker 1 finishes ticker C"] --> COL["collector"]
    W2["worker 2 finishes ticker A"] --> COL
    W3["worker 3 finishes ticker B"] --> COL
    COL --> ORD["reassembled in UNIVERSE order:<br/>A, B, C"]
    ORD --> RANK["orders queued:<br/>SELLs first, then BUYs by descending score<br/>(ticker breaks ties)"]
    RANK --> FILL["fills against finite cash"]
    FILL --> REP["identical trades, equity curve, report"]
```

1. **Order-stable collection.** Parallel results are keyed by ticker and
   re-read in input order, so completion order never leaks into results. This
   matters because cash is finite: whichever BUY is considered first is the one
   that gets filled.
2. **Ranked order queueing.** BUY candidates are queued by descending signal
   score with the ticker as tie-breaker — reproducible, and the behaviour a
   capital-constrained portfolio wants anyway.
3. **Seeded simulations.** Both the per-symbol forward Monte Carlo and the
   portfolio-level bootstrap Monte Carlo are seeded (`simulation.random_seed`),
   and the bootstrap draws from a local generator so it neither disturbs nor
   depends on global NumPy state.

This is enforced by tests, not just convention: `tests/test_parallel_determinism.py`
runs the same backtest serially and in parallel and compares the exported
workbook sheet by sheet.

## Module map

```
portfolio_agent/
├── cli.py                  entry point: download-data, train, backtest,
│                           run-agent, list-strategies, gpu-check
├── config/
│   ├── schema.py           pydantic AppConfig (the full settings surface)
│   ├── loader.py           config.yaml + AFA_* env overrides
│   └── strategies/*.yaml   per-strategy rule files, incl. example_uma.yaml
├── features/
│   ├── registry.py         @register_feature name -> function
│   ├── technical.py        the lag-safe indicators themselves
│   └── pipeline.py         build_features(df, names) -> DataFrame
├── strategies/
│   ├── base.py             BaseStrategy — the interface you implement
│   ├── types.py            RiskParams, StrategyContext, StrategySignal
│   ├── registry.py         register_strategy / load_strategy
│   ├── rule_based.py       trend + breakout + volume + Monte Carlo
│   ├── cross_sectional.py  momentum, low_volatility (requires_full_batch)
│   ├── ml_strategy.py      trained model, GPU-batched (supports_gpu_batch)
│   ├── ensemble.py         UMAs: weighted_blend / vote
│   └── weighting.py        pure weight-adaptation used live and in backtests
├── models/
│   ├── registry.py         @register_model name -> nn.Module class
│   └── pytorch_models.py   LSTM forecaster
├── agents/
│   ├── trainer.py          training loop, panel construction, checkpointing
│   └── backtester.py       wires strategy -> engine -> analytics -> Excel
├── data/dataset.py         TimeSeriesDataset + DataLoader construction
├── utils/device.py         device resolution and GPU diagnostics
└── src/
    ├── orchestrator.py     the live daily loop
    ├── backtest_engine.py  event-driven backtest
    ├── execution_sim.py    Indian market costs, slippage, STCG/LTCG
    ├── risk.py             position sizing incl. fractional Kelly
    ├── risk_analytics.py   CAGR/Sharpe/Sortino/drawdown, bootstrap MC
    ├── monte_carlo.py      per-symbol forward simulation (scoring input)
    ├── volatility_models.py GJR-GARCH(1,1)
    ├── compliance.py       eligibility gates
    ├── data_store.py       parquet cache + downloads
    ├── universe.py         ticker universe resolution
    ├── reporting.py        live-agent Excel report
    └── backtest_reporting.py backtest Excel report
```

Note the two distinct Monte Carlo modules — they answer different questions:

| | `src/monte_carlo.py` | `src/risk_analytics.py` |
|---|---|---|
| Scope | one symbol | whole portfolio |
| Direction | forward from today | resampling of realized trades |
| Role | **input** to strategy scoring | **output** metric in the report |
| Method | lognormal shocks (optionally GARCH-driven) | bootstrap resampling of trade returns |
