# Standalone strategy notebooks

Seven notebooks covering every strategy the framework implements, from raw
HuggingFace data through cleaning, training, backtesting and analysis.

**These do not import `portfolio_agent`.** They are a self-contained
re-implementation, so you can copy this folder anywhere — a laptop, Colab, a
fresh container — and run it without installing the platform. The only file they
need is `afa_lab.py`, which sits next to them.

## The notebooks

| Notebook | Strategy | Needs torch |
| --- | --- | --- |
| `00_data_ingestion.ipynb` | HuggingFace ingestion, cleaning, data-quality analysis | no |
| `01_rule_based.ipynb` | Trend + breakout + volume + Monte Carlo composite | no |
| `02_momentum.ipynb` | Cross-sectional 9m-skip-1m momentum, with a crash filter | no |
| `03_low_volatility.ipynb` | Inverse-volatility weighted low-vol | no |
| `04_lstm.ipynb` | Supervised LSTM on cross-sectional forward-return rank | yes |
| `05_sac_rl.ipynb` | Soft Actor-Critic continuous-allocation policy | yes |
| `06_ensemble_comparison.ipynb` | All of the above, blended and compared | yes |

**Run `00` first.** It populates `data_cache/`, which the others reuse, so the
download happens once rather than seven times.

## Setup

```bash
pip install pandas numpy pyarrow matplotlib huggingface_hub torch jupyterlab
jupyter lab
```

Torch is only needed for `04`, `05` and `06`; the other four run without it.

No HuggingFace token is required — the dataset
(`vishnun0027/indian-market-historical-ohlcv`, 2,421 NSE/BSE equities) is public.

### On Colab

The first cell of each notebook downloads `afa_lab.py` from the repo if it is
not already beside the notebook, so uploading a single `.ipynb` is enough.

## Data

One small parquet per symbol is fetched from the Hub, so a 30-name universe
downloads 30 small files rather than the dataset's full 283 MB.

Cleaning, in order:

1. **Back-adjust OHLC by `adj_close / close`.** On a 1:10 split the raw close
   drops 90% in one print — momentum reads that as a crash and ATR-derived stops
   blow out. Scaling all four price legs by the same per-row factor removes the
   discontinuity while leaving intraday relationships intact, so a locked session
   (high == low) stays locked.
2. **Coerce numerics.** One object-dtype column silently poisons every rolling
   statistic computed from it.
3. **Drop unparseable dates and missing closes** rather than forward-filling, so
   a gap stays visible as a gap.
4. **Drop duplicate sessions**, keeping the last.

### If HuggingFace is unreachable

`load_panel` falls back to a **synthetic** panel and says so loudly. The
synthetic generator carries a common market factor, per-name idiosyncratic
drift, volatility clustering and occasional gaps, so every downstream cell still
exercises properly — but any number produced from it describes the generator,
not the market. Never read a result off a synthetic run.

## What keeps the backtests honest

- **`execution_lag=1`.** A signal computed from day *t*'s close is traded into
  day *t+1*'s return. The engine refuses `execution_lag=0`, which is the single
  most common way a backtest manufactures a return that does not exist.
- **Chronological splits.** Train/validation/test are split by date across the
  whole panel — every symbol's training window ends before any symbol's
  validation window begins. Splitting by row after stacking symbols would put
  one name's future beside another's past.
- **Standardizers fitted on training rows only.** Fitting on the full history
  leaks the test period's moments into the transform. The leak is small and
  entirely invisible in the resulting metrics, which is what makes it worth
  being strict about.
- **Turnover measured against drifted weights.** A name that rose is a larger
  share of the book by itself; measuring turnover against the previous *target*
  would overstate trading and therefore costs.
- **Equal-weight buy-and-hold as the benchmark.** The honest comparison for a
  long-only stock picker is the same names held passively, not cash.

## What these notebooks do not establish

Stated plainly, because the charts are persuasive and the sample is not:

- **Survivorship bias.** The universe is today's large caps applied to history.
  Names that were large caps in 2018 and are not now are missing, and they are
  missing precisely because they did badly. Every long-only result is biased
  upward by an amount the notebooks cannot measure.
- **One universe, one period.** Thirty names over a few years is a single draw.
  The gap between two strategies here is well within what the draw alone could
  produce.
- **Flat 25 bps costs**, fills assumed at the close, and no circuit-limit
  modelling — on Indian equities a circuit-locked session is untradeable, and the
  simulation will happily trade it.
- **Parameters are chosen, not fitted.** Deliberately: tuning on this sample and
  reporting the result would be reporting the tuning.

The deliverable is a legible, modifiable mechanism — not evidence that any of
these strategies makes money.

## Editing

`afa_lab.py` is **generated**. Edit the sources under `build/blocks/` and
regenerate:

```bash
python build/assemble.py        # blocks/*.py  -> afa_lab.py
python build/make_notebooks.py  # cell builders -> *.ipynb
```

The blocks are separate modules so they can be imported and tested directly; the
notebooks import one flat file so a reader has one thing to open. The notebooks
themselves are generated for the same reason — the setup, ingestion and
evaluation sections are identical across all seven, and hand-copying meant they
drifted the moment one was fixed.

Clear outputs before committing:

```bash
jupyter nbconvert --clear-output --inplace *.ipynb
```

## Relationship to the framework

These re-implement the same strategies the platform ships
(`portfolio_agent/strategies/`), and the design choices are the same ones for
the same reasons. They are not a wrapper: the platform's engine models circuit
limits, liquidity screens, Kelly sizing, regime gating, walk-forward validation
and compliance rules that are deliberately out of scope here. Use these to
understand and modify a strategy; use the framework to run one.
