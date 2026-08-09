# AFA — Autonomous Financial Advisor

A lightweight, CLI-first platform for training and backtesting trading strategies on Indian equities (NSE/BSE), with GPU acceleration for model training and ML-strategy inference.

- **Decision support only**: this system does not place real trades. It runs in paper trading / decision support mode only.
- **No broker integration**: there is no execution path to any real broker.
- **Educational purpose**: for research and education. Past performance does not guarantee future results.

## Documentation

| Document | What's in it |
|---|---|
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | How the platform works, with diagrams — the strategy/model layer in detail, the backtest day loop, the concurrency map, device selection, and report data lineage |
| **[docs/STRATEGIES.md](docs/STRATEGIES.md)** | Plug-and-play guide to **creating, updating and deleting** strategies, with worked examples for each kind, plus adding features and model architectures |
| **[docs/QUANT_RESEARCH.md](docs/QUANT_RESEARCH.md)** | The academic basis for every strategy and risk model |

## Table of Contents

- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [GPU / CUDA setup](#gpu--cuda-setup)
- [Strategies (plug-and-play)](#strategies-plug-and-play)
- [UMAs — combining strategies](#umas--combining-strategies)
- [Quant research basis](#quant-research-basis)
- [Training](#training)
- [Parallelism](#parallelism)
- [Configuration](#configuration)
- [Scheduling (cron / Task Scheduler)](#scheduling-cron--task-scheduler)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Optional: Docker](#optional-docker)
- [Safety & Guardrails](#safety--guardrails)

## Quick Start

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies (add --extra gpu for torch/CUDA support)
uv sync

# Download market data for the configured universe
uv run portfolio-agent download-data --universe-size 50

# Run the live paper-trading agent (writes an Excel recommendation report)
uv run portfolio-agent run-agent

# Backtest the rule-based strategy over 1 year
uv run portfolio-agent backtest --strategy rule_based --years 1

# Train the LSTM forecaster on cached data, then backtest it
uv sync --extra gpu
uv run portfolio-agent train --device auto
uv run portfolio-agent backtest --strategy lstm --years 1
```

## CLI Reference

All commands are subcommands of `portfolio-agent` (equivalently `uv run python -m portfolio_agent.cli`).

```
portfolio-agent download-data [--force] [--universe-size N] [--workers N]
    Download and cache OHLCV data for the resolved ticker universe.
    Chunks are fetched concurrently (default 4); use --workers 1 if the
    data provider rate-limits you.

portfolio-agent train [--device auto|cuda|mps|cpu]
    Train the configured model (default: LSTM) on real cached market data.
    Set training.use_synthetic_data: true in config.yaml to train on
    synthetic data instead (offline/CI testing only).

portfolio-agent backtest
    [--strategy rule_based|lstm|ensemble|...]   Strategy to backtest (default: config.strategy.type)
    [--strategy-config PATH]       Strategy YAML override (e.g. a UMA ensemble file)
    [--parallel] [--workers N]     Parallelize rule-based signal generation across CPU workers
    [--use-trained-model]          Shorthand for --strategy lstm
    [--years N | --start-date/--end-date]
    [--device auto|cuda|mps|cpu]   Device for ML-strategy inference
    [--output PATH]                Excel report path

portfolio-agent run-agent [--force-refresh] [--simulate-outcome] [--update-outcomes]
    Run the live daily paper-trading loop: fetch data, score every ticker with
    the configured strategy, save recommendations, export an Excel report.

portfolio-agent list-strategies [--name NAME] [--strategy-config PATH]
    List registered strategies. With --name, show that strategy's entry/exit
    rules and required features (pass --strategy-config to inspect a UMA file).

portfolio-agent gpu-check
    Report which compute devices this install can actually use, and — when
    CUDA is unavailable — why, plus the exact command to fix it.
```

`--strategy` and `--strategy-config` are also accepted by `backtest` (see below) to select any registered strategy — `rule_based`, `lstm`, `ensemble` (a UMA), or a custom one you register yourself.

## GPU / CUDA setup

`uv sync --extra gpu` installs `torch` from PyPI. **On Windows that wheel is CPU-only**, so `--device cuda` will correctly fall back to CPU no matter how good your GPU is. Check what you actually have:

```bash
portfolio-agent gpu-check
```

If it reports a CPU-only build, install a CUDA build from the PyTorch index (pick the URL matching your driver at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)):

```bash
uv pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu126 torch
portfolio-agent gpu-check     # should now report CUDA available: True
```

Device selection resolves once, up front, and never returns an accelerator PyTorch cannot use. Requesting an unavailable device prints one warning explaining the cause and the fix, then runs on CPU — the resolved device is written back into the config so dataloaders, mixed precision and the saved checkpoint metadata all agree with what was printed.

| `--device` | Behaviour |
|---|---|
| `auto` (default) | CUDA if usable, else MPS, else CPU |
| `cuda` | CUDA if usable, else CPU with a diagnostic |
| `mps` | Apple Metal if usable, else CPU with a diagnostic |
| `cpu` | CPU |

CUDA is used for two things: **training** (with automatic mixed precision and cuDNN benchmarking) and **ML-strategy inference**, where all eligible tickers on a date are scored in a single batched forward pass. Rule-based strategies are CPU work; parallelize those with `--parallel` instead.

## Strategies (plug-and-play)

Every strategy — built-in or your own — implements one interface (`portfolio_agent/strategies/base.py::BaseStrategy`) and is looked up by name from `portfolio_agent/strategies/registry.py`. Because the live agent and the backtest engine both go through this same registry, they always make identical decisions from identical inputs — there's no separate "live" vs "backtest" scoring logic to keep in sync.

Built-in strategies:

- **`rule_based`** (default) — "Trend + Breakout + Volume + Monte Carlo probability" scoring, configured via `config/strategies/trend_breakout.yaml`. Component weights self-adjust over time based on realized win rate (`strategies/weighting.py`). Cheap to evaluate; parallelizes across CPU workers for large universes (`--parallel`).
- **`momentum`** — cross-sectional momentum: long the top decile of the eligible universe by 9-month (skip 1-month) formation return (Jegadeesh-Titman convention). Params: `top_percentile` (default 0.1), `min_universe` (default 5, below which every ticker is `AVOID` since ranking isn't reliable).
- **`low_volatility`** — the low-volatility anomaly: long the bottom decile by trailing 60-day realized volatility. Same params as `momentum`.
- **`lstm`** — a trained sequence-forecasting model (`portfolio_agent/models/pytorch_models.py`). During backtesting, all eligible tickers on a given date are batched into a single GPU forward pass (`strategies/ml_strategy.py::score_batch`) rather than scored one at a time.
- **`ensemble`** — combines multiple strategies into one; see [UMAs](#umas--combining-strategies) below.

`momentum` and `low_volatility` are **cross-sectional**: a ticker's signal depends on where it ranks against the *entire* eligible universe that round, not on its own history alone (`BaseStrategy.requires_full_batch`). Both the backtest engine and the live orchestrator detect this and call `score_batch()` with every eligible ticker at once rather than looping per-ticker. For the same reason they cannot be used as UMA members today (a UMA scores members per-ticker) — use them directly instead. See [Quant research basis](#quant-research-basis) for the math.

**Adding your own strategy** is three steps:

1. Subclass `BaseStrategy` (see `strategies/rule_based.py` for a rule-based example, `strategies/ml_strategy.py` for an ML example, `strategies/cross_sectional.py` for a ranking example) and implement `name`, `required_features()`, and `score(symbol, features, context) -> StrategySignal`.
2. Register it: `register_strategy("my_strategy", MyStrategy)` in `strategies/registry.py` (or call `register_strategy` yourself before running the CLI, e.g. from a small bootstrap script).
3. Use it: `portfolio-agent backtest --strategy my_strategy`.

No other code needs to change — `run-agent`, `backtest`, and any UMA that references your strategy by name all pick it up automatically through the registry.

> **Full guide: [docs/STRATEGIES.md](docs/STRATEGIES.md)** — complete worked examples for all four kinds of strategy, how to update one safely, how to remove one cleanly, plus adding indicators and model architectures. For how the layer works internally, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## UMAs — combining strategies

A **UMA** (Unified Multi-strategy Agent) combines two or more registered strategies into a single strategy, so you can, for example, blend the rule-based strategy with a trained LSTM rather than choosing one or the other. A UMA is itself just a strategy (type `ensemble`) — it can be backtested, run live, or even nested inside another UMA.

Define one in a YAML file (see `config/strategies/example_uma.yaml`):

```yaml
name: "Trend+ML Blend"
method: weighted_blend      # or "vote"
vote:
  mode: majority            # "majority" or "unanimous" — only used by method: vote

members:
  - type: rule_based
    weight: 0.6
    config_path: config/strategies/trend_breakout.yaml
  - type: lstm
    weight: 0.4
    params:
      model_name: lstm
```

Run it:

```bash
portfolio-agent backtest --strategy ensemble --strategy-config config/strategies/example_uma.yaml
portfolio-agent list-strategies --name ensemble --strategy-config config/strategies/example_uma.yaml
```

Two combination methods, selectable per UMA:

- **`weighted_blend`** (default) — each member's signal is mapped to a strength (BUY=1, WATCH=0.3, HOLD=0, AVOID=-0.3, SELL=-1) and averaged by weight; score, entry/stop/target, and probability-of-profit are likewise weighted averages. Good default for mixing strategies of different character (e.g. a fast rule-based signal with a slower ML one).
- **`vote`** — each member casts a BUY/SELL/HOLD-bucketed vote; `vote.mode: majority` requires >50% agreement, `vote.mode: unanimous` requires all members to agree. More conservative — fewer but higher-conviction signals.

Notes:
- Member weights only matter for `weighted_blend`; they're ignored by `vote`.
- A UMA is not GPU-batched even if one of its members is (correctness — a rule-based member needs a genuine per-ticker Monte Carlo result, which the batched path skips). If you want maximum ML-inference throughput, run that strategy directly (`--strategy lstm`) rather than wrapping it in a UMA.
- `list-strategies --name ensemble --strategy-config <file>` shows you the resolved member list and weights for a given UMA file.

## Quant research basis

**[docs/QUANT_RESEARCH.md](docs/QUANT_RESEARCH.md)** is the mathematical/research foundation behind the platform's strategies and risk models — academic evidence (with an emphasis on India-specific studies), exact formulations, and an honest list of what's implementable with OHLCV-only data versus what needs a new data source (fundamentals, institutional flows). Covers:

- Cross-sectional momentum and the low-volatility anomaly (`strategies/cross_sectional.py`)
- GJR-GARCH(1,1) conditional volatility with Student-t innovations, used as an optional drop-in replacement for the Monte Carlo simulation's flat historical-volatility assumption (`src/volatility_models.py`; enable via `simulation.use_garch_volatility: true`)
- Fractional-Kelly position sizing, estimated from realized trade history (`src/risk.py::calculate_kelly_quantity`; enable via `risk.use_kelly_sizing: true`)
- The original trend/breakout/volume/Monte-Carlo rule-based strategy
- Researched-but-not-implemented strategy families (cointegration pairs trading, Fama-French factors, quality/QMJ, FII/DII flows, calendar anomalies) and exactly why each is scoped out (architectural gap vs. data gap vs. weak evidence)

## Training

```bash
uv sync --extra gpu
uv run portfolio-agent train --device auto
```

`train` builds its training panel from the **full cached ticker universe** (whatever's in `data/market_data/*.parquet` — no separate download needed if you've already run `download-data` or otherwise populated that cache) rather than a synthetic stand-in. Each ticker is loaded, featurized, and chronologically split (70/15/15) individually, then concatenated so validation/test proportionally represent every ticker, not just the last one processed.

Optimizations already applied, controlled via `config.yaml`'s `training` section:

| Setting | Default | What it does |
|---|---|---|
| `parallel_data_loading` | `true` | Loads and featurizes tickers across a CPU process pool (`data_load_workers`, default: CPU count) instead of one at a time — this is what makes training on ~2,400 tickers practical. |
| `use_mixed_precision` | `true` | Automatic mixed precision (`torch.amp`) on CUDA. |
| `use_torch_compile` | `false` | Wraps the model with `torch.compile()` for faster training (PyTorch 2.0+, biggest win on CUDA). Off by default — enable it once you're doing longer training runs. |
| `batch_size` | `128` | Sized for GPU throughput; lower it on CPU-only or memory-constrained machines. |
| `num_workers` | `2` | PyTorch `DataLoader` workers (separate from `data_load_workers`, which is for building the panel, not iterating it). |

Plus, already in place from the underlying `DataLoader`/device setup: `pin_memory` on CUDA, `persistent_workers`/`prefetch_factor` for the training loop, and `cudnn.benchmark` enabled on fixed-size CUDA inputs.

Set `training.use_synthetic_data: true` to fall back to generated random-walk data instead (offline/CI testing only — real training should leave this `false`).

## Parallelism

Every hot path uses the executor that matches the work, and **all of them are speed-only**: parallel and serial runs produce identical results.

| Path | Executor | Enable / tune |
|---|---|---|
| Market data download | Thread pool (network-bound) | `download-data --workers N`, `data.download_workers` (default 4) |
| Training panel build | Process pool (CPU-bound) | `training.parallel_data_loading` (default on), `training.data_load_workers` |
| Backtest signal scoring | Process pool (CPU-bound) | `backtest --parallel --workers N` |
| Live per-ticker prep | Process pool (CPU-bound) | `data.parallel_ticker_prep` (default on), `data.ticker_prep_workers` |
| Model training | GPU | `training.device`, `use_mixed_precision`, `use_torch_compile` |
| ML-strategy inference | GPU | `backtest --device cuda` (one batched forward pass per date) |
| Batch feeding | DataLoader workers | `training.num_workers` |

```bash
# 2-year rule-based backtest over 120 tickers: ~2.3x faster on 4 cores,
# byte-identical Excel report
portfolio-agent backtest --strategy rule_based --years 2 --parallel --workers 4
```

Determinism is guaranteed by construction, not by luck: parallel results are reassembled in universe order (never completion order), orders are queued SELL-first then BUYs by descending score so that finite cash is allocated reproducibly, and both Monte Carlo simulations are seeded from `simulation.random_seed`. `tests/test_parallel_determinism.py` enforces this by running the same backtest both ways and comparing the exported workbook sheet by sheet. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#determinism-guarantees).

`--parallel` is not free: for small universes, process startup can cost more than it saves. It pays off from a few dozen tickers upward.

## Configuration

Edit `config.yaml` at the repo root. Every field can also be overridden via environment variables using the `AFA_` prefix with double-underscore nesting (see `.env.example`):

```bash
AFA_RISK__PORTFOLIO_VALUE_INR=500000 uv run portfolio-agent run-agent
```

Key sections: `data` (universe/tickers), `strategy`, `training`, `backtest`, `risk`, `learning`, `simulation` (Monte Carlo), `compliance`, `paths`.

```yaml
data:
  download_workers: 4          # concurrent chunk downloads; set to 1 if rate-limited
  parallel_ticker_prep: true   # prepare tickers across a CPU pool during run-agent
  ticker_prep_workers: null    # null = CPU count
compliance:
  paper_trading_mode: true   # must remain true
risk:
  portfolio_value_inr: 308733
  risk_per_trade_pct: 0.01
  use_kelly_sizing: false    # true = fractional-Kelly sizing once enough realized trades exist
  kelly_fraction: 0.5        # kappa in [0, 1]; 0.5 = half-Kelly
  kelly_min_trades: 20       # minimum realized trades before Kelly is trusted (else fixed-fractional)
simulation:
  use_garch_volatility: false   # true = GJR-GARCH(1,1) volatility forecast instead of flat historical std
```

## Scheduling (cron / Task Scheduler)

There's no scheduler baked into the app — run the CLI on your own schedule.

**Linux/macOS (cron):**
```
# Daily run at 15:45 IST on weekdays
45 15 * * 1-5 cd /path/to/afa && uv run portfolio-agent run-agent

# Update outcomes at 16:00 IST on weekdays
0 16 * * 1-5 cd /path/to/afa && uv run portfolio-agent run-agent --update-outcomes
```

**Windows (Task Scheduler):** create a task that runs `uv run portfolio-agent run-agent` from the project directory on your desired schedule.

## Testing

```bash
uv run pytest portfolio_agent/tests/ -q
```

## Project Structure

```
afa/
├── config.yaml                 # single nested config (see Configuration)
├── portfolio_agent/            # the package
│   ├── cli.py                  # single CLI entry point
│   ├── config/                 # schema.py, loader.py, strategies/*.yaml (incl. example_uma.yaml)
│   ├── strategies/             # base.py, types.py, rule_based.py, cross_sectional.py, ml_strategy.py, ensemble.py, weighting.py, registry.py
│   ├── features/               # lag-safe technical indicators + pipeline
│   ├── models/                 # PyTorch model definitions
│   ├── agents/                 # trainer.py, backtester.py
│   ├── src/                    # orchestrator, backtest engine, data store, risk.py (incl. Kelly sizing),
│   │                           # volatility_models.py (GJR-GARCH), monte_carlo.py, compliance, ...
│   └── tests/
├── docs/
│   ├── ARCHITECTURE.md         # how it all works, with diagrams
│   ├── STRATEGIES.md           # create / update / delete a strategy
│   └── QUANT_RESEARCH.md       # research basis for every strategy/risk model
├── data/                       # gitignored: market_data/*.parquet cache, agent_brain.json, sqlite db
├── output/                     # gitignored: Excel reports
├── models/                     # gitignored: trained model checkpoints
└── logs/                       # gitignored
```

## Optional: Docker

The primary workflow is local `uv`. A minimal optional `Dockerfile` is provided for reproducibility on a fresh machine:

```bash
docker build -t afa .                              # CPU only
docker build --build-arg INSTALL_GPU=true -t afa:gpu .   # with torch/CUDA
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output afa
```

## Safety & Guardrails

- **Paper trading only** — no live trading is enabled.
- **No leverage, no short selling** — long positions only, fully funded.
- **Position size cap** — configurable via `risk.max_single_position_pct`.
- **Risk per trade cap** — configurable via `risk.risk_per_trade_pct`.
- **Penny stock filter** — `compliance.min_price_inr`.

MIT License — for educational purposes only. This system is not investment advice.
