# Notebooks

Two notebooks for working on strategies interactively. Both are thin wrappers
over `portfolio_agent.lab.Lab` — the logic lives in the package, so anything
you do here has a CLI equivalent and is reproducible outside a kernel.

| Notebook | Question it answers |
| --- | --- |
| `01_strategy_lab.ipynb` | How does *this* strategy train and backtest? |
| `02_compare_and_sweep.ipynb` | Which strategy, or which settings, is better? |
| `03_forecast_lab.ipynb` | Does the strategy's **ranking** carry information at all? |

`03` is usually the one to open first, and the reason is worth stating: a
backtest reports the *product* of two things — does the signal order the
cross-section, and does that ordering survive becoming a book — and never their
difference. Round one found a strategy that ranks the cross-section well and has
a **negative** decile spread. One equity curve cannot say that; it just looks
bad.

## Running them

The notebooks need the training extras and a Jupyter kernel:

```bash
uv sync --extra gpu
uv pip install jupyterlab
uv run jupyter lab
```

They also need cached price data — `portfolio-agent download-data` — because
every cell reads the same parquet cache the CLI does.

## The one idea worth knowing

Both notebooks open with a `Lab`, and a `Lab` **pins its universe once**:

```python
lab = Lab(universe_size=40)
lab.tickers        # the pinned names
lab.fingerprint    # short hash identifying this exact set
```

Every method on it — `train`, `backtest`, `compare`, `sweep` — uses those same
tickers.

That matters more than it looks. `resolve_backtest_universe` draws from
whatever happens to be in the cache, and it deliberately offsets its seed by
purpose so that training and backtesting sample *differently* (which is the
right default: scoring a model on the very names it was fitted on is not
out-of-sample in the cross-section). But it means a notebook that trains in one
cell and backtests in the next has silently used two different samples, and a
table comparing two models built that way is partly measuring the samples.

So: pin once, then compare. And write the universe down when a result is worth
keeping —

```python
lab.save_universe("universe/experiment.json")
```

— so a later session, or someone else's machine, can reproduce the comparison:

```python
lab = Lab(snapshot="universe/experiment.json")
```

```bash
portfolio-agent train-bulk --strategies india_sac \
    --universe-snapshot universe/experiment.json
```

## Committing notebooks

Clear outputs before committing. Executed notebooks carry megabytes of base64
images and, more importantly, absolute paths and ticker lists from whoever ran
them last, which makes every diff a merge conflict:

```bash
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```
