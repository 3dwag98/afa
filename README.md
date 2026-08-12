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
| **[docs/REVIEW_STATUS.md](docs/REVIEW_STATUS.md)** | Item-by-item status against the quantitative review: what is done, what is not, and what each gap is blocked on. Every "done" names the code and the test that pins it |

## Table of Contents

- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [GPU / CUDA setup](#gpu--cuda-setup)
- [Strategies (plug-and-play)](#strategies-plug-and-play)
  - [Scoring modes](#scoring-modes)
- [UMAs — combining strategies](#umas--combining-strategies)
- [Quant research basis](#quant-research-basis)
- [Training](#training)
- [Parallelism](#parallelism)
- [Configuration](#configuration)
- [Scheduling (cron / Task Scheduler)](#scheduling-cron--task-scheduler)
- [Backtest findings and known limitations](#backtest-findings-and-known-limitations)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Optional: Docker](#optional-docker)
- [Safety & Guardrails](#safety--guardrails)

## Quick Start

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies (--extra hf for the HuggingFace data source,
# --extra gpu for torch/CUDA support)
uv sync --extra hf

# Download 5 years of Indian-market OHLCV from the HuggingFace dataset
uv run portfolio-agent download-data

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
portfolio-agent download-data
    [--source huggingface|yfinance]  Data source (default: config.data.source)
    [--years N]                      Years of history to keep (default: 5)
    [--hf-dataset ID]                Hub dataset repo id
    [--hf-revision REV]              Pin to a branch/tag/commit for reproducibility
    [--universe-size N]              Cap the number of tickers ingested
    [--force] [--workers N]          yfinance path only
    Download and cache OHLCV into data/market_data/*.parquet.
    huggingface (default) reads vishnun0027/indian-market-historical-ohlcv,
    one parquet per symbol, and also caches the benchmark index (^NSEI).
    yfinance fetches per ticker in concurrent chunks; use --workers 1 if
    the provider rate-limits you.

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

`uv sync --extra gpu` installs `torch` from PyPI. **On Windows that wheel is CPU-only**, so `--device cuda` will correctly fall back to CPU no matter how good your GPU is.

Use the CUDA extra instead, which pulls from PyTorch's own wheel index:

```bash
# CUDA 12.6 (also: --extra cu121)
uv sync --extra hf --extra cu126
portfolio-agent gpu-check     # should report CUDA available: True
```

Note `--extra` takes one value at a time, so each extra needs its own flag — `--extra hf --extra cu126`, not `--extra hf cu126`.

This replaces the manual two-step of installing the CPU build and force-reinstalling over it. That still works if you need a CUDA version the extras do not cover — pick the URL matching your driver at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/):

```bash
uv pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu126 torch
```

`gpu`, `cu126` and `cu121` all provide `torch` and are declared mutually exclusive, so uv will tell you if you ask for more than one rather than silently resolving to whichever it saw last.

Check what you actually ended up with at any point:

```bash
portfolio-agent gpu-check
```

Device selection resolves once, up front, and never returns an accelerator PyTorch cannot use. Requesting an unavailable device prints one warning explaining the cause and the fix, then runs on CPU — the resolved device is written back into the config so dataloaders, mixed precision and the saved checkpoint metadata all agree with what was printed.

| `--device` | Behaviour |
|---|---|
| `auto` (default) | CUDA if usable, else MPS, else CPU |
| `cuda` | CUDA if usable, else CPU with a diagnostic |
| `mps` | Apple Metal if usable, else CPU with a diagnostic |
| `cpu` | CPU |

CUDA is used for two things: **training** (with automatic mixed precision and cuDNN benchmarking) and **ML-strategy inference**, where all eligible tickers on a date are scored in a single batched forward pass. Rule-based strategies are CPU work; parallelize those with `--parallel` instead.

### Mixed precision is refused on cards that cannot do it safely

`use_mixed_precision` is a request, not a command. Before enabling `torch.amp`, training checks the actual card and turns AMP off — printing the reason — when fp16 would be unsafe or pointless:

- **GeForce GTX 16-series (1650/1660/1660 Ti).** These Turing dies report compute capability 7.5, the same as an RTX 2060, but ship **without tensor cores**. AMP buys no speedup on them, and their fp16 path is the one widely reported to return NaN. Left enabled, it produced a training run where every epoch printed `nan`.
- **Anything older than compute capability 7.0** (pre-Volta), where fp16 is emulated rather than accelerated.

`portfolio-agent gpu-check` reports the verdict for your card under `Mixed precision:`. Nothing needs changing in `config.yaml` — the check is automatic, and the run continues in fp32.

## Strategies (plug-and-play)

Every strategy — built-in or your own — implements one interface (`portfolio_agent/strategies/base.py::BaseStrategy`) and is looked up by name from `portfolio_agent/strategies/registry.py`. Because the live agent and the backtest engine both go through this same registry, they always make identical decisions from identical inputs — there's no separate "live" vs "backtest" scoring logic to keep in sync.

Built-in strategies:

- **`rule_based`** (default) — "Trend + Breakout + Volume + Monte Carlo probability" scoring, configured via `config/strategies/trend_breakout.yaml`. Component weights self-adjust over time based on realized win rate (`strategies/weighting.py`). Cheap to evaluate; parallelizes across CPU workers for large universes (`--parallel`). Three combination rules are available — see [Scoring modes](#scoring-modes) below.
- **`momentum`** — cross-sectional momentum: long the top decile of the eligible universe by 9-month (skip 1-month) formation return (Jegadeesh-Titman convention). Params: `top_percentile` (default 0.1), `min_universe` (default 5, below which every ticker is `AVOID` since ranking isn't reliable).
- **`low_volatility`** — the low-volatility anomaly: long the bottom decile by trailing 60-day realized volatility. Same params as `momentum`.
- **`lstm`** — a trained sequence-forecasting model (`portfolio_agent/models/pytorch_models.py`). During backtesting, all eligible tickers on a given date are batched into a single GPU forward pass (`strategies/ml_strategy.py::score_batch`) rather than scored one at a time.
- **`ensemble`** — combines multiple strategies into one; see [UMAs](#umas--combining-strategies) below.

`momentum` and `low_volatility` are **cross-sectional**: a ticker's signal depends on where it ranks against the *entire* eligible universe that round, not on its own history alone (`BaseStrategy.requires_full_batch`). Both the backtest engine and the live orchestrator detect this and call `score_batch()` with every eligible ticker at once rather than looping per-ticker. For the same reason they cannot be used as UMA members today (a UMA scores members per-ticker) — use them directly instead. See [Quant research basis](#quant-research-basis) for the math.

### Scoring modes

`rule_based` combines its four components under one of three rules, set by
`scoring.method` in the strategy YAML or `scoring_mode` in its params. They
differ in what the score *means*, not in which components feed it.

| Mode | Combination | Score scale | Use when |
|---|---|---|---|
| `weighted_sum` (default) | Weighted sum of the raw component values | 0–100, absolute | You want a fixed quality bar: a name clears 60 on its own merits, and on a bad day nothing clears it |
| `rank_composite` | Weighted sum of cross-sectional percentile ranks | 0–100, relative | Components are on incommensurable scales and you want the combination invariant to each one's units |
| `probit_composite` | Ranks → Φ⁻¹ → weighted sum → standardized per date | 0–100 via Φ, with a mean-zero unit-variance z alongside | The score is consumed as a *magnitude* — e.g. as the expected-return input to the portfolio optimizer — rather than only as an ordering |

Two consequences worth stating plainly before switching:

- **Both cross-sectional modes convert the entry threshold from an absolute bar
  into a percentile.** Under them a roughly fixed share of the universe clears
  60 every day, whatever the market is doing. That is the intended behaviour for
  a ranking system and the wrong behaviour for a "only trade genuinely good
  setups" mandate. Both are off by default for this reason.
- **Neither fixes a component that discriminates nothing.** Ranking ties hand
  every name the same percentile, so a near-constant component contributes a
  flat number under all three rules — a different flat number, but still flat,
  and still spending its full share of the weight budget. Making influence track
  discrimination is a change to the weight learner, not to the combination rule.

`probit_composite` exists because a percentile is a uniform variate: a weighted
sum of uniforms has a spread that depends on how many components were
measurable and how correlated they were that day, so the same 0.72 means
different things on different dates. Pushing ranks through the inverse normal
CDF and standardizing the result makes the composite mean-zero and
variance-one on *every* date, which is the contract an optimizer needs from an
alpha input. The reported `score` still maps back to 0–100 through Φ so the
existing `>= 60` / `>= 45` gates keep working — Φ is monotone, so the ordering
is unchanged and `>= 60` acquires a cleaner reading: "top 40% of today's
cross-section". The raw z is exposed as `signal.extra["composite_z"]`.

Both cross-sectional modes report `requires_full_batch = True`, so the
orchestrator and the backtest engine score the whole eligible universe in one
call. A UMA that reaches a cross-sectional member through per-ticker scoring is
rejected at load time rather than silently ranking each name against itself.

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

Three combination methods, selectable per UMA:

- **`trigger`** — members are converted to `ModelVerdict`s and arbitrated by [`src/trigger_engine.py`](portfolio_agent/src/trigger_engine.py). **Use this for anything that trades real money.** The other two average, and averaging is the wrong operation for votes on a decision: a strong BUY blended with a strong SELL comes out as a weak BUY — a trade neither member would take, entered exactly when the models disagree most. The trigger engine instead discounts buy-side conviction by the strongest opposing conviction (`c_eff = c_buy × (1 − max c_opposing)`), applies hard vetoes for tradability, regime and expected value, and emits a **position-size multiplier** so a trade that barely clears its bar is taken at half size.
- **`weighted_blend`** (default, kept so existing UMA files behave as they did) — each member's signal is mapped to a strength (BUY=1, WATCH=0.3, HOLD=0, AVOID=-0.3, SELL=-1) and averaged by weight; score, entry/stop/target, and probability-of-profit are likewise weighted averages. Cheap and smooth, and wrong in the way described above whenever members conflict.
- **`vote`** — each member casts a BUY/SELL/HOLD-bucketed vote; `vote.mode: majority` requires >50% agreement, `vote.mode: unanimous` requires all members to agree. It cannot manufacture a signal out of disagreement, but it discards conviction magnitude, position sizing and the expected-value hurdle.

### The multi-regime meta-orchestrator

[`config/strategies/uma_meta_orchestrator.yaml`](portfolio_agent/config/strategies/uma_meta_orchestrator.yaml) is the production configuration: four sleeves, arbitrated by the trigger engine and gated on the Nifty 50 regime.

```bash
portfolio-agent backtest --strategy ensemble \
  --strategy-config config/strategies/uma_meta_orchestrator.yaml
```

A `regimes:` block maps each market state to the members allowed to generate BUY signals in it:

| Regime | Definition (Nifty 50) | Sleeves permitted |
|---|---|---|
| `BULL_RISK_ON` | Above 200-day SMA, realized vol < target | Quality momentum, trend rules, neural |
| `BEAR_CRASH_RISK` | Below 200-day SMA, **or** vol > 1.5× target | Defensive low-volatility only |
| `SIDEWAYS_CHOP` | Within 2% of the 200-day SMA and ADX < 20 | Trend rules, neural, defensive |
| `NEUTRAL` | Above the SMA but vol between target and 1.5× | Everything except momentum |

The definitions overlap by design, so the classifier checks them in a specific order — a volatility spike is unambiguous panic wherever price sits and wins first; chop is the most specific condition and is checked before the bear branch's "below the SMA" clause, because an index 1% under its average in a calm market is a range, not a bear.

Members outside a regime's list are **muted, not vetoed**: one sleeve being out of season must not stop the sleeve that is in it. A regime the map does not mention — including `UNKNOWN`, when no benchmark is cached — permits every member, since not knowing the regime is not evidence that every model is wrong.

Notes:
- Member weights only matter for `weighted_blend`; they're ignored by `vote` and `trigger`.
- Cross-sectional members (`momentum`, `low_volatility`) require `method: trigger`, which scores every member across the whole eligible universe before arbitrating. The averaging methods combine through per-ticker `score()`, where decile ranking degenerates to a universe of one, so they reject such members at construction.
- A UMA takes each member's name from the UMA file (`name:` on the member, or `params.name`), which is what `regimes:` keys off. Member names must be unique — the trigger engine treats each verdict as an independent voice.
- Inside a batched UMA, a `rule_based` member receives no per-ticker Monte Carlo result and scores its `MC_Prob` component at zero. Mixing an MC-dependent member with a cross-sectional one is a real trade-off, not a free composition.
- `list-strategies --name ensemble --strategy-config <file>` shows you the resolved member list, weights, trigger thresholds and regime map for a given UMA file.

## Quant research basis

**[docs/QUANT_RESEARCH.md](docs/QUANT_RESEARCH.md)** is the mathematical/research foundation behind the platform's strategies and risk models — academic evidence (with an emphasis on India-specific studies), exact formulations, and an honest list of what's implementable with OHLCV-only data versus what needs a new data source (fundamentals, institutional flows). Covers:

- Cross-sectional momentum and the low-volatility anomaly (`strategies/cross_sectional.py`)
- GJR-GARCH(1,1) conditional volatility with Student-t innovations, used as an optional drop-in replacement for the Monte Carlo simulation's flat historical-volatility assumption (`src/volatility_models.py`; enable via `simulation.use_garch_volatility: true`)
- Fractional-Kelly position sizing in *allocation* units — the fraction of wealth a stop-loss trade justifies, which is the binary-bet Kelly fraction divided by the loss-given-stop (`src/risk.py::kelly_allocation_fraction`; enable via `risk.use_kelly_sizing: true`)
- Portfolio covariance estimation and constrained long-only allocation: Ledoit-Wolf shrinkage, a turnover-penalized mean-variance optimizer, and hierarchical risk parity for when expected returns are not trustworthy (`src/portfolio.py`)
- Selection-bias-aware performance statistics — probabilistic and deflated Sharpe, probability of backtest overfitting, cross-sectional rank IC, and the Newey-West correction for overlapping labels (`src/performance_stats.py`)
- Bayesian shrinkage of the simulated drift, whose standard error over five years of daily data is roughly 14% a year (`src/monte_carlo.py::shrink_drift`)
- A Markov-switching regime model — a K-state Gaussian HMM by Baum-Welch with K chosen by BIC, emitting filtered state probabilities rather than a hard label (`src/markov_regime.py`)
- Reinforcement learning for the *exposure* decision, scoped to what a single market trajectory can actually support — see §27 for why the obvious deep-RL-over-prices version does not (`src/rl.py`)
- The original trend/breakout/volume/Monte-Carlo rule-based strategy
- Researched-but-not-implemented strategy families (cointegration pairs trading, Fama-French factors, quality/QMJ, FII/DII flows, calendar anomalies) and exactly why each is scoped out (architectural gap vs. data gap vs. weak evidence)

These were added in response to a quantitative review; §21–26 of the research doc set out what each was measuring wrongly and what the corrected version measures instead. **[docs/REVIEW_STATUS.md](docs/REVIEW_STATUS.md)** tracks that review item by item — what is done, what is partly done and why, and what is not started — including the finding that invalidates every cross-sectional backtest number until the universe becomes point-in-time.

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
| `use_mixed_precision` | `true` | Automatic mixed precision (`torch.amp`), on CUDA cards that have tensor cores. Silently — and deliberately — ignored elsewhere; see [above](#mixed-precision-is-refused-on-cards-that-cannot-do-it-safely). |
| `use_torch_compile` | `false` | Wraps the model with `torch.compile()` for faster training (PyTorch 2.0+, biggest win on CUDA). Off by default — enable it once you're doing longer training runs. |
| `batch_size` | `128` | Sized for GPU throughput; lower it on CPU-only or memory-constrained machines. |
| `num_workers` | `2` | PyTorch `DataLoader` workers (separate from `data_load_workers`, which is for building the panel, not iterating it). |
| `model` | `lstm` | Architecture from `models/registry.py`. **`patchtst` is the recommended one** — see below. |
| `loss` | `quantile` | Training objective. See below. |
| `quantiles` | `[0.1, 0.5, 0.9]` | Percentiles of the forward return the model predicts. |
| `calibrate_confidence` | `true` | Fits an isotonic score→probability map on the walk-forward folds and ships it with the checkpoint. |

### Inputs are standardized before they reach the network

Half the model's features are price *levels* (`sma_20`, `sma_50`, `macd`, `atr_14`). Across a 4,000-name Indian universe those span roughly ₹5 to ₹1,50,000, sitting next to `return_1d` at ±0.02 and `rsi_14` in [0, 100]. Fed in raw, that breaks training two ways, and both present identically as a NaN loss on every epoch:

- fp16 tops out at **65,504**, so any feature above it becomes `inf` on the first autocast matmul;
- even in fp32, inputs of 1e5 through a recurrent layer at `lr=3e-3` diverge within a few hundred steps.

`portfolio_agent/features/scaling.py` therefore standardizes each feature (mean 0, std 1, clipped to ±10σ) before it reaches the model. The statistics are fitted on **training rows only** — per fold during walk-forward validation, so no fold sees its own test period's distribution — and the fitted constants are written into `models/metadata.json` and the checkpoint, so `MLStrategy` applies the identical transform at inference. Checkpoints trained before this existed carry no scaler and keep being scored on raw features unchanged.

This is deliberately *not* `features.normalize`: that flag rewrites the shared feature pipeline, whose output the rule-based strategies read in raw units ("RSI below 30"), so turning it on to fix training would change what every other strategy trades.

**Two normalizations run, and they answer different questions.** The scaler
above is a *conditioning* fix — it keeps the numbers in a range the network can
train on. `training.feature_normalization: cross_sectional` (the default) adds a
statistical one in front of it: each feature is z-scored across the universe
**on each date** rather than against a pooled five-year mean.

The difference matters because a pooled z-score answers "is this RSI high for
this stock over the sample", while a model choosing *between* stocks needs "is
this RSI high relative to what else I could buy today". The pooled form also
quietly puts the market factor back into every column that
`training.target_transform` just removed from the label: on a day the whole
market gapped down, every name's return feature reads extreme against a
five-year mean, and the network is handed a market state a long-only book
cannot act on.

The cross-sectional transform **cannot leak, by construction** — it fits no
state and carries nothing across dates, so the transform for date *t* reads only
rows dated *t*. That is a stronger guarantee than "the statistics were fitted on
the training split", which is a property of the calling code rather than of the
transform, and it is testable directly: the suite rewrites every later row and
asserts the earlier dates come back bit-for-bit identical. Dates with too thin a
cross-section are dropped, and a universe with no cross-section on any date is
left unscaled rather than emptied — the same fallback `target_transform` uses,
so the label and the inputs never disagree about which rows exist.

Set `feature_normalization: global` to restore the pooled behaviour. The global
scaler runs either way: after the cross-sectional pass it is close to a no-op,
but it is the transform that ships in the checkpoint metadata and guarantees
inference reproduces training.

Rows that are non-finite — not just NaN — are dropped when the panel is built. Several features are ratios (`return_1d` divides by the previous close, `bollinger_pct_b` by the band width), and a cached bar with a zero price makes them infinite; `dropna()` alone leaves those rows in, and each one produces a NaN loss.

### What the model predicts, and why it isn't a single number

Squared error is minimized by the conditional mean, and the conditional mean of a 5-day equity return is very close to a constant. A network trained on `MSELoss` therefore converges to a near-constant output that scores excellently on the loss curve and forecasts nothing — the mean-reversion trap. It also hands downstream code a bare point estimate, which the trigger engine cannot turn into an expected value without inventing a distribution around it.

The default is **pinball (quantile) loss** over the 10th, 50th and 90th percentiles. A constant answer cannot satisfy three asymmetric penalties at once, and the outer pair is a confidence interval that comes out of the fit rather than being bolted on: `MLStrategy` derives its stop and target from the predicted 10th/90th percentiles instead of fixed 2%/3% cuts, so a name the model reads as wide gets a wide stop. Crossed quantiles are repaired by sorting at inference — exact and free — rather than penalized during training, where the penalty would distort the quantiles it was protecting.

Set `loss: mse` to restore the single-output point forecast.

### Architectures

| `training.model` | What it is |
|---|---|
| `lstm` | The original vanilla LSTM. Compresses a 60-day multi-feature window into one hidden vector and predicts from the final timestep — everything the sequence held has to survive that bottleneck. |
| `patchtst` | **Recommended.** Cuts the window into 5-day patches and attends over them: a single day's return is nearly pure noise, a week of them has shape. Channel-independent encoding (every feature runs through the same weights, separately) keeps attention from fitting spurious cross-feature relationships, and per-window instance normalization lets one set of weights serve a ₹30 small-cap and a ₹3,000 large-cap. Attention costs 12×12 instead of 60×60. |

```bash
uv run portfolio-agent train --device auto   # set training.model: patchtst in config.yaml
```

Existing single-output checkpoints keep loading unchanged: head width and quantile levels come from `models/metadata.json` and default to the old scalar shape when absent.

### Confidence calibration

Networks on noisy financial data are systematically overconfident — the score at which the model says 80% is typically won far less than 80% of the time. That matters more here than in most settings, because the number feeds Kelly sizing and the trigger engine's expected-value hurdle, and both are far more sensitive to an optimistic `p` than a pessimistic one.

`calibrate_confidence` fits an isotonic (monotone) map from raw score to realized win rate on the **walk-forward test folds** — the only genuinely out-of-sample scores a run produces. Fitting on training predictions would measure memorization and hand back a map that makes an overfitted model look perfectly calibrated. Monotonicity is the point: it preserves the model's ranking, which walk-forward actually measured, and discards its scale, which nothing did. Training prints the expected calibration error before and after, so the correction is auditable rather than a black box.

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

Determinism is guaranteed by construction, not by luck: parallel results are reassembled in universe order (never completion order), orders are queued SELL-first then BUYs by descending score so that finite cash is allocated reproducibly, and both Monte Carlo simulations are seeded from `simulation.random_seed`. `portfolio_agent/tests/test_parallel_determinism.py` enforces this by running the same backtest both ways and comparing the exported workbook sheet by sheet. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#determinism-guarantees).

`--parallel` is not free: for small universes, process startup can cost more than it saves. It pays off from a few dozen tickers upward.

### Watching a backtest run

`portfolio-agent backtest` draws a progress bar for each of the two phases that take real time — reading the universe off disk, then replaying the trading days:

```
Loading ticker data: 100%|██████████| 3847/3847 [02:11<00:00, 29.3ticker/s]
Loaded 3612/3847 tickers with usable history
Replaying 2020-08-11 to 2025-08-10 with strategy 'Trend Breakout Volume MC' over 1237 trading days
Backtesting:  34%|███▍      | 421/1237 [08:52<17:12, 1.27s/day, date=2022-04-19, equity=1,143,208, open=7, trades=96]
```

The day bar carries the simulated date, current equity, open positions and cumulative trade count, so a run that has quietly stopped trading is visible immediately rather than at the end. Progress goes to the terminal directly and suppresses itself when output is redirected to a file; `--no-progress` turns it off explicitly.

(Previously the engine reported progress through `logger.info`, but the CLI configures no logging handlers, so those records were discarded and a multi-hour run printed nothing between "Running backtest..." and its final report.)

## Configuration

Edit `config.yaml` at the repo root. Every field can also be overridden via environment variables using the `AFA_` prefix with double-underscore nesting (see `.env.example`):

```bash
AFA_RISK__PORTFOLIO_VALUE_INR=500000 uv run portfolio-agent run-agent
```

Key sections: `data` (universe/tickers), `strategy`, `training`, `backtest`, `risk`, `learning`, `simulation` (Monte Carlo), `compliance`, `paths`.

```yaml
data:
  source: huggingface          # huggingface | yfinance
  hf_dataset_id: vishnun0027/indian-market-historical-ohlcv
  hf_revision: null            # pin a branch/tag/commit for a reproducible backtest
  hf_adjust_prices: true       # back-adjust OHLC by adj_close/close (splits/dividends)
  benchmark_symbol: "^NSEI"    # index driving the momentum crash filter
  default_history_years: 5     # history kept, both sources
  download_workers: 4          # concurrent chunk downloads (yfinance); 1 if rate-limited
  parallel_ticker_prep: true   # prepare tickers across a CPU pool during run-agent
risk:
  portfolio_value_inr: 308733
  risk_per_trade_pct: 0.01
  risk_free_rate: 0.065         # annualized Sharpe/Sortino hurdle; overridden by
                                # paths.risk_free_rate_csv when that file exists
  use_kelly_sizing: false       # true = fractional-Kelly once enough realized trades exist
  kelly_fraction: 0.25          # kappa; hard-capped at 0.25 (quarter-Kelly) in src/risk.py
  kelly_min_trades: 50          # realized trades before Kelly is trusted (else fixed-fractional)
  kelly_shrinkage_strength: 20  # Beta prior pulling the win rate toward 0.5
  max_sector_pct: 0.25          # max share of portfolio in any one sector
  max_unknown_sector_pct: 0.30  # aggregate budget for tickers missing from the map
  max_portfolio_drawdown_pct: 0.15  # halt new buys past this drawdown
  drawdown_reentry_pct: 0.10        # resume buying once recovered to here
  slippage_pct_per_side: 0.0025     # assumed slippage when costing a signal
compliance:
  paper_trading_mode: true      # must remain true
  min_reward_risk: 1.2          # applied to reward:risk NET of round-trip costs
  min_price_inr: 20.0           # penny-stock floor
  target_prob_profit: 0.55      # Monte Carlo probability gate
simulation:
  method: block_bootstrap       # gaussian | block_bootstrap | jump_diffusion
  use_garch_volatility: false   # true = GJR-GARCH(1,1) instead of flat historical std
  separate_overnight_gaps: true # fit GARCH to sessions, add gap risk separately
  prior_annual_drift_std: 0.10  # fixed fallback prior on the spread of true drifts
  use_empirical_drift_prior: true   # measure that prior from the cross-section instead
training:
  walk_forward_splits: 5        # expanding-window validation folds; 0 to skip
  target_transform: cross_sectional_rank  # absolute | cross_sectional_demean | ..._rank
  feature_normalization: cross_sectional  # global | cross_sectional
paths:
  trial_log: output/trials.jsonl        # append-only; supplies N for the deflated Sharpe
  risk_free_rate_csv: data/risk_free_rate.csv  # optional date,annualized_yield series
```

> The block above is grouped by the section each key actually belongs to.
> `min_reward_risk`, `min_price_inr` and `target_prob_profit` live under
> `compliance`, while the sector, drawdown and slippage controls live under
> `risk` — a distinction worth checking against `config.yaml` before copying a
> key from one section into another.

### Risk controls

| Control | Config | What it does |
|---|---|---|
| Momentum crash filter | `momentum` strategy params | Stands momentum down when the market is below its 200-day average *and* volatility is above 1.5× target — the state momentum crashes in ([§12](docs/QUANT_RESEARCH.md)) |
| Volatility targeting | `volatility_target` (0.20) | Scales each position by target vol / realized vol; never levers up |
| Cost-aware signals | `risk.slippage_pct_per_side` | Reward:risk is reported net of round-trip friction (~0.8%), so the `min_reward_risk` gate compares money actually kept |
| Tradability screen | `liquidity_filter` params | Drops circuit-locked and zombie stocks from the ranking ([§15](docs/QUANT_RESEARCH.md)) |
| Sector cap | `risk.max_sector_pct`, `risk.max_unknown_sector_pct` | Trims orders so no sector exceeds 25%, with unmapped tickers sharing a wider 30% budget so an incomplete map isn't a bypass. **Requires a `ticker,sector` CSV at `paths.sector_map_csv`** — with no map at all the cap is inactive (and logs a warning), since capping the unmapped pool would limit total invested capital rather than sector concentration |
| Drawdown breaker | `risk.max_portfolio_drawdown_pct` | Halts new entries past 15% drawdown, re-arms at 10% |
| Drawdown liquidation | `risk.liquidate_on_drawdown_halt` | **Off by default.** Also sells the whole book when the breaker trips. Left off because open positions already carry stops and force-liquidating at a drawdown trough turns a bad quarter into a permanent loss; turn it on for mandates with a hard equity floor |
| Lower-circuit exit | `risk.exit_on_lower_circuit_lock` | Exits a holding that closed pinned at its lower circuit at the next session, instead of waiting for a modelled stop no bid will fill |
| Signal arbitration | UMA `trigger:` block | Blocks the trade when a strong model opposes, when expected value misses the hurdle, or when the regime mutes every buyer — and sizes the rest by conviction |
| Kelly guards | `risk.kelly_*` | 50-trade floor, Beta-shrunk win rate, kappa hard-capped at quarter-Kelly, and **both inputs measured net of friction** on the live path as well as in backtests |

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

## Backtest findings and known limitations

Running the end-to-end backtest surfaced three defects that no unit test would
have caught, because each one presents as a *plausible number* rather than an
error. They are fixed; the findings are recorded here because two of them
change how the platform should be configured.

**The engine discarded every strategy's exit plan.** Filled positions were
given a hardcoded 5% stop and 10% target regardless of what the strategy
computed. That made `atr_stop_multiplier` dead config, and — worse — meant
`min_reward_risk` screened signals on a net-of-cost reward:risk derived from
ATR levels the engine then ignored, gating on one exit plan and trading
another. The quantile model's distribution-derived stop and target went the
same way, and Kelly's payoff ratio was estimated from trades exited under a
rule the signals were never screened against. Fills now carry the signal's own
levels, as distances re-applied to the price actually paid.

**Exit horizon has to match signal horizon, and by default it did not.** Once
the exit plan actually reached the fill, the same momentum signal over the same
universe and window behaved completely differently depending on the stop width:

| `atr_stop_multiplier` | Round trips | Median hold | Stop-loss exits | Gross return on deployed | Net P&L |
|---|---|---|---|---|---|
| 1.5 (default) | 590 | 5 days | 71% | **−2.02%** | −₹481,861 |
| 6.0 | 113 | **94 days** | 31% | **+4.44%** | +₹55,807 |

Cross-sectional momentum forms on a 9-month window. At 1.5× ATR it was exiting
in a working week, which turns a multi-month factor into day-trading and pays
~0.8% of round-trip friction for the privilege. The signal has edge; the stop
was cutting the thesis short rather than protecting it. The default stays 1.5
because it suits the per-ticker trend/breakout strategy the platform started
with — **set `risk.atr_stop_multiplier` to match your signal's horizon.**

**The drawdown breaker could deadlock.** It halts new buys at 15% drawdown and
re-arms on recovery to 10%. But halting buys does not freeze the book: open
positions keep exiting through their stops until only cash is left, and cash
cannot appreciate back toward a peak it is measured against. A 5-year run
tripped in month seven and sat in cash for four years, reporting a 15.04% max
drawdown that actually meant "stopped trading". `risk.drawdown_halt_max_days`
(60) now re-arms on a cooldown and resets the peak.

### End-to-end results after the fixes

120 tickers, 2021-01 to 2025-12 (~4.3 years of cached data), ₹10L initial
capital, fractional-Kelly sizing, full friction stack (STT 0.1% both legs,
stamp duty, GST, exchange and SEBI charges, ATR- and volume-scaled slippage,
STCG/LTCG), `atr_stop_multiplier: 6.0`:

| | Total return | CAGR | Max DD | Win rate | Profit factor | Sharpe |
|---|---|---|---|---|---|---|
| `momentum` alone | **+15.2%** | 3.27% | **12.9%** | 53.4% | 1.60 | −0.40 |
| Meta-orchestrator UMA | +1.5% | 0.34% | 27.8% | 44.8% | 1.03 | −0.44 |

Read the Sharpe carefully: it is negative *despite* a positive return, because
`RiskAnalyzer` measures excess return over a 6.5% Indian risk-free rate. A
3.3% CAGR loses to government bonds. The strategy survives the friction stack —
profit factor 1.60 net of every cost the simulator models — but it does not
yet clear the opportunity cost of not trading at all.

The meta-orchestrator underperforming its own momentum sleeve is worth naming
rather than hiding: on this universe the additional sleeves diluted the one
that had edge, and the regime map was running off the composite fallback rather
than a real index. That is a result about this configuration and this data, not
a verdict on the architecture — but it is the result.

### What is not verified here

- **The performance target is not met.** The roadmap's goal of a net Sharpe
  above 1.2 with max drawdown under 15% is not reached on the data available in
  this repository. Max drawdown clears comfortably for the momentum sleeve
  (12.9%); Sharpe does not, and nothing here should be read as claiming it does.
- **The cache is ~4.5 years, not 10.** `data/market_data/` starts in 2021, so
  the 2018 IL&FS crisis and the 2020 COVID crash are outside it. The regime
  classifier is exercised on the 2022 small-cap drawdown and the 2023 mid-cap
  rally only.
- **The universe is an alphabetical slice, not the Nifty 500.**
  `resolve_backtest_universe(max_tickers=N)` takes the first N cached tickers,
  which is heavily micro-cap. Decile ranking over that is not the cross-section
  the research describes.
- **`^NSEI` is not cached**, so the regime filter runs off the equal-weighted
  composite of the traded universe rather than the real index. That fallback is
  what `src/regime.py` was designed for and it produces a sensible regime series
  (bear through 2022, bull through 2023, neutral from mid-2024), but an index
  feed is the better gauge.

## Testing

```bash
uv run pytest portfolio_agent/tests/ -q
```

**Optional extras change what collects.** `uv sync --frozen` installs neither
`torch` nor `cvxpy`, and the two behave differently when absent:

| Extra | Without it | Install |
|---|---|---|
| `torch` | Six test files **fail collection**, so the whole run aborts rather than skipping | `uv sync --extra gpu` (or `--extra cu126` / `--extra cu121` for CUDA) |
| `cvxpy` | `test_portfolio_optimizer.py` skips cleanly via `importorskip` | `uv sync --extra optimize` |

The torch case is worth knowing before assuming a red suite means a real
failure: the six files import `torch` at module scope, so a missing optional
dependency reads as `6 errors during collection` rather than as skips. To run
everything the CI runs:

```bash
uv sync --extra gpu --extra optimize --extra hf
uv run pytest portfolio_agent/tests/ -q
```

## Project Structure

```
afa/
├── config.yaml                 # single nested config (see Configuration)
├── portfolio_agent/            # the package
│   ├── cli.py                  # single CLI entry point
│   ├── config/                 # schema.py, loader.py, strategies/*.yaml (incl.
│   │                           # uma_meta_orchestrator.yaml, the multi-regime UMA)
│   ├── strategies/             # base.py, types.py (incl. ModelVerdict), rule_based.py,
│   │                           # cross_sectional.py, ml_strategy.py, ensemble.py,
│   │                           # weighting.py, registry.py
│   ├── features/               # lag-safe technical indicators, pipeline, and
│   │                           # scaling.py (global + per-date cross-sectional)
│   ├── models/                 # LSTM + PatchTST, pinball loss, model registry
│   ├── agents/                 # trainer.py, backtester.py
│   ├── src/                    # the engine room — see docs/ARCHITECTURE.md for the
│   │                           # full annotated inventory. Broadly:
│   │                           #   run loops    orchestrator.py, backtest_engine.py
│   │                           #   sizing       risk.py (Kelly in allocation units),
│   │                           #                portfolio.py (covariance + HRP),
│   │                           #                portfolio_optimizer.py (QP w/ sector caps)
│   │                           #   measurement  risk_analytics.py, performance_stats.py
│   │                           #                (PSR/DSR, trial log), outcomes.py
│   │                           #   simulation   monte_carlo.py, volatility_models.py,
│   │                           #                execution_sim.py
│   │                           #   gating       compliance.py, liquidity.py, sectors.py,
│   │                           #                trigger_engine.py, regime.py,
│   │                           #                markov_regime.py, rl.py, calibration.py
│   │                           #   data / io    data_store.py, hf_dataset.py, universe.py,
│   │                           #                storage.py, reporting.py,
│   │                           #                backtest_reporting.py, indicators.py
│   └── tests/                  # 1,111 tests (with all optional extras installed)
├── docs/
│   ├── ARCHITECTURE.md         # how it all works, with diagrams
│   ├── STRATEGIES.md           # create / update / delete a strategy
│   ├── QUANT_RESEARCH.md       # research basis for every strategy/risk model
│   └── REVIEW_STATUS.md        # item-by-item status against the quant review
├── data/                       # gitignored: market_data/*.parquet cache, agent_brain.json, sqlite db
│                               # optional: sector_map.csv (ticker,sector) for concentration caps,
│                               # risk_free_rate.csv (date,annualized_yield) for the Sharpe hurdle
├── output/                     # gitignored: Excel reports; trials.jsonl (the DSR trial log)
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
- **Sector concentration cap** — `risk.max_sector_pct` (25% by default).
- **Drawdown circuit breaker** — new entries stop past `risk.max_portfolio_drawdown_pct`,
  resuming on recovery or after `risk.drawdown_halt_max_days`.
- **Momentum crash filter** — exposure is cut in the market state where momentum crashes.
- **Regime gating** — a UMA's `regimes:` map decides which sleeves may buy at all.
- **Signal arbitration** — conflicting models block the trade instead of averaging
  into one neither would take.
- **Penny stock filter** — `compliance.min_price_inr`.
- **Tradability screen** — circuit-locked (1/2/5/10/20% bands), operator-trapped and
  effectively-untraded stocks are excluded.
- **Lower-circuit exit** — a holding locked down is exited at the next session rather
  than waiting for a stop no bid will fill.
- **Costs charged to the decision, not just the fill** — signals are gated on
  reward:risk net of the full Indian friction stack, and Kelly's inputs are
  measured net of it on both the live and backtested paths.

MIT License — for educational purposes only. This system is not investment advice.
