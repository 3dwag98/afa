# AFA — Autonomous Financial Advisor

A lightweight, CLI-first platform for training and backtesting trading strategies on Indian equities (NSE/BSE), with GPU acceleration for model training and ML-strategy inference.

- **Decision support only**: this system does not place real trades. It runs in paper trading / decision support mode only.
- **No broker integration**: there is no execution path to any real broker.
- **Educational purpose**: for research and education. Past performance does not guarantee future results.

## Table of Contents

- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Strategies](#strategies)
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
portfolio-agent download-data [--force] [--universe-size N]
    Download and cache OHLCV data for the resolved ticker universe.

portfolio-agent train [--device auto|cuda|mps|cpu]
    Train the configured model (default: LSTM) on real cached market data.
    Set training.use_synthetic_data: true in config.yaml to train on
    synthetic data instead (offline/CI testing only).

portfolio-agent backtest
    [--strategy rule_based|lstm]   Strategy to backtest (default: config.strategy.type)
    [--parallel] [--workers N]     Parallelize rule-based signal generation across CPU workers
    [--use-trained-model]          Shorthand for --strategy lstm
    [--years N | --start-date/--end-date]
    [--device auto|cuda|mps|cpu]   Device for ML-strategy inference
    [--output PATH]                Excel report path

portfolio-agent run-agent [--force-refresh] [--simulate-outcome] [--update-outcomes]
    Run the live daily paper-trading loop: fetch data, score every ticker with
    the configured strategy, save recommendations, export an Excel report.
```

## Strategies

The platform trains and backtests strategies through a single canonical interface (`portfolio_agent/strategies/base.py::BaseStrategy`), so the live agent and the backtest engine always make identical decisions from identical inputs:

- **`rule_based`** (default) — "Trend + Breakout + Volume + Monte Carlo probability" scoring, configured via `config/strategies/trend_breakout.yaml`. Component weights self-adjust over time based on realized win rate (`strategies/weighting.py`). Cheap to evaluate; can be parallelized across CPU workers for large universes (`--parallel`).
- **`lstm`** — a trained sequence-forecasting model (`portfolio_agent/models/pytorch_models.py`). During backtesting, all eligible tickers on a given date are batched into a single GPU forward pass (`strategies/ml_strategy.py::score_batch`) rather than scored one at a time.

Add a new strategy by implementing `BaseStrategy` and registering it in `portfolio_agent/strategies/registry.py`.

## Configuration

Edit `config.yaml` at the repo root. Every field can also be overridden via environment variables using the `AFA_` prefix with double-underscore nesting (see `.env.example`):

```bash
AFA_RISK__PORTFOLIO_VALUE_INR=500000 uv run portfolio-agent run-agent
```

Key sections: `data` (universe/tickers), `strategy`, `training`, `backtest`, `risk`, `learning`, `simulation` (Monte Carlo), `compliance`, `paths`.

```yaml
compliance:
  paper_trading_mode: true   # must remain true
risk:
  portfolio_value_inr: 308733
  risk_per_trade_pct: 0.01
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
│   ├── config/                 # schema.py, loader.py, strategies/*.yaml
│   ├── strategies/             # base.py, rule_based.py, ml_strategy.py, weighting.py, registry.py
│   ├── features/               # lag-safe technical indicators + pipeline
│   ├── models/                 # PyTorch model definitions
│   ├── agents/                 # trainer.py, backtester.py
│   ├── src/                    # orchestrator, backtest engine, data store, risk, compliance, ...
│   └── tests/
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
