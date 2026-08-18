"""Generate the standalone strategy notebooks.

One generator rather than seven hand-maintained notebooks: the setup, ingestion
and evaluation sections are identical across them, and hand-copying meant they
drifted the moment one was fixed. Edit the cell builders here and re-run.

    python build/make_notebooks.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent

UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI",
    "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "NESTLEIND", "BAJFINANCE",
    "TATAMOTORS", "TATASTEEL", "POWERGRID", "NTPC", "ONGC", "HCLTECH",
    "JSWSTEEL", "GRASIM", "CIPLA", "COALINDIA",
]


_CELL_COUNTER = itertools.count(1)


def _cell_id() -> str:
    """Stable, deterministic cell ids so regenerating produces a clean diff."""
    return f"c{next(_CELL_COUNTER):04d}"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "id": _cell_id(), "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "id": _cell_id(), "execution_count": None,
            "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells: list) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------------------
# Cells shared by every notebook
# ---------------------------------------------------------------------------

SETUP = code('''
# Dependencies. Torch is only needed by the two learned strategies (04, 05).
# !pip install -q pandas numpy pyarrow matplotlib huggingface_hub torch

import sys, pathlib

# afa_lab.py sits next to this notebook. On Colab (or anywhere the file is
# missing) fetch it from the repo — that is the only network call that touches
# GitHub, and nothing else here imports the portfolio_agent package.
if not pathlib.Path("afa_lab.py").exists():
    import urllib.request
    URL = ("https://raw.githubusercontent.com/3dwag98/afa/main/"
           "notebooks/standalone/afa_lab.py")
    urllib.request.urlretrieve(URL, "afa_lab.py")
    print("fetched afa_lab.py")

sys.path.insert(0, ".")
import afa_lab as L

import numpy as np
import pandas as pd

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)
print("toolkit loaded | torch available:", L.TORCH_AVAILABLE)
''')

UNIVERSE_CELL = code(f'''
# NSE large caps. Any symbol absent from the dataset is skipped rather than
# failing the run, so this list does not have to be exactly right.
UNIVERSE = {json.dumps(UNIVERSE, indent=4)}

START_DATE = "2018-01-01"
END_DATE   = None          # None = up to the dataset's last session
CACHE      = "data_cache"  # downloaded parquet files land here and are reused

print(len(UNIVERSE), "symbols requested")
''')

INGEST = code('''
# Ingestion. One small parquet per symbol is pulled from the Hub dataset
# `vishnun0027/indian-market-historical-ohlcv` (2,421 NSE/BSE equities) and
# cleaned. Downloading per symbol rather than snapshotting the repo means a
# 30-name universe fetches 30 small files instead of 283 MB.
#
# Cleaning, in order: back-adjust OHLC by adj_close/close so a split is not read
# as a 90% crash, coerce numerics, drop unparseable dates and missing closes
# (rather than forward-filling, so a gap stays visible), and drop duplicate
# sessions keeping the last.
#
# If the Hub is unreachable the toolkit falls back to a synthetic panel and says
# so loudly. Synthetic results describe the generator, not the market.

panel = L.load_panel(UNIVERSE, start_date=START_DATE, end_date=END_DATE,
                     cache_dir=CACHE)

close = L.align_close_matrix(panel)
print(f"{len(panel)} symbols | {close.index.min().date()} -> {close.index.max().date()}"
      f" | {len(close)} sessions")
''')

QUALITY = code('''
quality = L.panel_quality(panel)
display(quality.sort_values("coverage").head(10))

L.plot_data_quality(quality)
L.plot_prices(panel, n=8)
''')

FEATURES = code('''
# Features are computed per symbol and left NaN until each window has filled.
# They are never back-filled: a back-filled indicator is a look-ahead, and it is
# invisible in every metric downstream.
feature_panel = L.build_feature_panel(panel)

sample = feature_panel[sorted(feature_panel)[0]]
print(f"{len(sample.columns)} features:", list(sample.columns))
display(sample.dropna().tail(3))
''')

CONFIG = code('''
# Simulation settings, shared by every notebook so the strategies are comparable.
#
# execution_lag=1 is the property that keeps this honest: a signal computed from
# day t's close is traded into day t+1's return. The engine refuses lag=0.
config = L.BacktestConfig(
    initial_capital=1_000_000.0,
    cost_bps=25.0,        # all-in round trip for Indian cash equities
    max_weight=0.10,
    rebalance_days=5,     # weekly; the main control over turnover
    max_gross=1.0,        # long-only, unlevered
    execution_lag=1,
)

benchmark = L.equal_weight_benchmark(close, config)
print("equal-weight buy & hold:",
      {k: round(v, 4) for k, v in benchmark.stats.items()
       if k in ("cagr", "sharpe", "max_drawdown")})
''')


def evaluate_cells(name: str, label: str) -> list:
    return [
        md(f"## Backtest\n\nAgainst equal-weight buy-and-hold of the same names — "
           f"the honest comparison for a long-only stock picker. Beating cash is "
           f"not the question."),
        code(f'''
result = L.run_backtest({name}_scores, close, config)

comparison = L.compare_stats({{"{label}": result, "equal weight": benchmark}})
display(comparison)
'''),
        md("## Analysis"),
        code(f'''
L.plot_equity({{"{label}": result}}, title="{label} vs equal weight",
              benchmark=benchmark.returns)
'''),
        code(f'L.plot_return_profile(result, "{label}")'),
        code(f'L.plot_exposure(result, "{label}")'),
        code(f'L.plot_weight_heatmap(result, title="{label}: allocation over time")'),
    ]


CAVEATS = md('''
---

## What this does and does not show

Read before quoting any number above.

- **Survivorship.** The universe is today's large caps, applied to history. Names
  that were large caps in 2018 and are not now are absent, and they are absent
  precisely because they did badly. Every long-only result here is biased upward
  by an amount this notebook cannot measure. A point-in-time constituent list is
  the only fix, and this dataset does not carry one.
- **One universe, one period.** Thirty names over a few years is a single draw.
  The difference between two strategies here is well within what the draw alone
  could produce.
- **Costs are a flat 25 bps.** Real cost scales with size and with how illiquid
  the name is, and the fill is assumed at the close. A strategy whose edge is
  this side of costs is not distinguishable from one that has no edge.
- **No point-in-time fundamentals, no corporate actions beyond the price
  adjustment**, and no circuit-limit modelling. On Indian equities a
  circuit-locked session is untradeable, and the simulation will happily trade it.
- **Parameters were chosen, not fitted.** Nothing here is tuned on a held-out
  period. That is deliberate — tuning on this sample and reporting the result
  would be reporting the tuning.

The purpose of these notebooks is to make the mechanism legible and modifiable,
not to establish that any of these strategies makes money.
''')


# ---------------------------------------------------------------------------
# Notebooks
# ---------------------------------------------------------------------------


def nb_data() -> dict:
    return notebook([
        md('''
# 00 — Data ingestion & cleaning

Pull Indian equity OHLCV from the HuggingFace dataset
`vishnun0027/indian-market-historical-ohlcv`, clean it, and look hard at what
arrived before building anything on it.

Standalone: this notebook does not import the `portfolio_agent` package. It needs
only `afa_lab.py`, which sits next to it.

**Run this first.** It populates `data_cache/`, which every other notebook reuses,
so the download happens once.
'''),
        md("## Setup"), SETUP,
        md("## Universe"), UNIVERSE_CELL,
        md('''
## Ingest

The dataset carries `date, open, high, low, close, adj_close, volume,
dividends, stock_splits, symbol` per file, one file per symbol under `stocks/`.
Indices such as `^NSEI` live under `indices/`.
'''),
        INGEST,
        md('''
## What cleaning actually did

The single most consequential step is the price adjustment. On a 1:10 split the
raw close drops 90% in one print — cross-sectional momentum reads that as a
crash, and every ATR-derived stop built on it blows out. Back-adjusting all four
price legs by the same `adj_close / close` factor removes the discontinuity
while leaving intraday relationships intact: a locked session (high == low)
stays locked.
'''),
        code('''
symbol = sorted(panel)[0]
frame = panel[symbol]

print(f"{symbol}: {len(frame)} sessions, {frame.index.min().date()} -> {frame.index.max().date()}")
display(frame.head(3))
display(frame.describe().T[["mean", "std", "min", "max"]])

returns = frame["close"].pct_change()
print("\\nlargest single-day moves (a split that slipped through the adjustment "
      "would show up here as a ~-90% day):")
display(returns.abs().nlargest(5))
'''),
        md('''
## Quality

Coverage below ~95% of business days means the symbol was suspended, newly
listed, or partially missing from the dataset — and a cross-sectional strategy
that ranks it against fully-covered names is comparing different things.
'''),
        QUALITY,
        md("## Features"), FEATURES,
        code('''
# How much history each feature needs before it produces anything. Long-lookback
# features are why a full trading year of warm-up is the default.
warmup = {c: int(sample[c].isna().sum()) for c in sample.columns}
display(pd.Series(warmup, name="leading NaNs").sort_values(ascending=False).head(10))
'''),
        code('''
# Cross-sectional dispersion: how much the ranked features actually separate
# names on a given date. A feature with no dispersion cannot drive a ranking.
import matplotlib.pyplot as plt

interesting = ["mom_9m_skip1m", "realized_vol_60", "rsi_14", "breakout_20"]
fig, axes = plt.subplots(1, len(interesting), figsize=(4 * len(interesting), 3.2))
for ax, name in zip(axes, interesting):
    matrix = L.cross_section(feature_panel, name)
    spread = matrix.std(axis=1)
    ax.plot(spread.index, spread, color=L.PALETTE[0], linewidth=1.1)
    ax.set_title(f"{name}\\ncross-sectional dispersion", fontsize=9)
    ax.grid(**L.GRID)
plt.tight_layout(); plt.show()
'''),
        md('''
## Next

`data_cache/` now holds the cleaned panel. The strategy notebooks reuse it:

| Notebook | Strategy |
| --- | --- |
| `01_rule_based.ipynb` | trend + breakout + volume + Monte Carlo composite |
| `02_momentum.ipynb` | cross-sectional momentum with a crash filter |
| `03_low_volatility.ipynb` | inverse-volatility weighted low-vol |
| `04_lstm.ipynb` | supervised LSTM on cross-sectional forward-return rank |
| `05_sac_rl.ipynb` | SAC reinforcement-learning allocation policy |
| `06_ensemble_comparison.ipynb` | all of them, blended and compared |
'''),
        CAVEATS,
    ])


def nb_rule_based() -> dict:
    return notebook([
        md('''
# 01 — Rule-based: trend + breakout + volume + Monte Carlo

A composite of four components, each mapped onto [0, 1] before weighting.

Mapping to a common scale first is what makes the weights mean what they say.
Combining raw components — an RSI in [0, 100] with a breakout in [-0.2, 0.2] —
lets whichever has the widest natural range dominate regardless of its weight.

Standalone: no `portfolio_agent` import. Run `00_data_ingestion.ipynb` first so
the panel is cached.
'''),
        md("## Setup"), SETUP, UNIVERSE_CELL, INGEST, FEATURES, CONFIG,
        md('''
## The four components

| Component | Reads | Mapped from |
| --- | --- | --- |
| Trend | `close / sma_200 - 1` | ±20%, saturating |
| Breakout | `close / high_20 - 1` | -10% to +5% |
| Volume | `volume / avg_volume_20` | 0.5x to 2.5x |
| Monte Carlo | P(higher in 21 sessions) | 0.30 to 0.70 |

The trailing high excludes today, so the breakout test is not self-referential —
comparing today's close against a window that contains it would make every new
high a breakout by construction.

The Monte Carlo term is a **block bootstrap** over trailing log returns, not a
Gaussian. Indian equity returns are fat-tailed and serially dependent enough
that a normal approximation understates both tails, and the left one is the
expensive side.
'''),
        code('''
params = L.RuleBasedParams(
    trend_weight=0.30,
    breakout_weight=0.25,
    volume_weight=0.20,
    mc_weight=0.25,
    min_score=0.55,            # composite needed to hold a name
    min_win_probability=0.50,
    require_trend=True,        # close > sma_200 is a hard gate, not a score
)

# The bootstrap is the slowest thing here. Set use_monte_carlo=False for a fast
# pass while you are changing weights; the gate then admits more names.
rule_based_scores = L.rule_based_scores(feature_panel, panel, params,
                                        use_monte_carlo=True)

held = (rule_based_scores > 0).sum(axis=1)
print(f"names passing the gate: mean {held.mean():.1f}, max {held.max()}, "
      f"zero on {(held == 0).mean():.1%} of sessions")
'''),
        code('''
# Where the score comes from, for one name. Each line is already on [0, 1], so
# the composite is directly readable as a weighted average.
symbol = sorted(feature_panel)[0]
f = feature_panel[symbol]

parts = pd.DataFrame({
    "trend": ((f["close"] / f["sma_200"] - 1).clip(-0.2, 0.2) + 0.2) / 0.4,
    "breakout": (f["breakout_20"].clip(-0.1, 0.05) + 0.1) / 0.15,
    "volume": (f["volume_ratio_20"].clip(0.5, 2.5) - 0.5) / 2.0,
}).dropna()

ax = parts.tail(250).plot(figsize=(11, 3.6), linewidth=1.1,
                          title=f"{symbol}: score components")
ax.axhline(params.min_score, color=L.PALETTE[1], linestyle="--", linewidth=1,
           label="min_score")
ax.grid(**L.GRID); ax.legend(ncol=4, fontsize=8)
import matplotlib.pyplot as plt; plt.tight_layout(); plt.show()
'''),
        *evaluate_cells("rule_based", "rule based"),
        md('''
## Sensitivity

The gate threshold is the parameter that matters most: it decides how often the
strategy is in the market at all. Sweeping it shows whether the result rests on
the signal or on one lucky cut point.
'''),
        code('''
sweep = {}
for threshold in (0.45, 0.50, 0.55, 0.60, 0.65):
    variant = L.RuleBasedParams(min_score=threshold)
    scores = L.rule_based_scores(feature_panel, panel, variant, use_monte_carlo=False)
    sweep[f"min_score={threshold}"] = L.run_backtest(scores, close, config)

table = L.compare_stats(sweep)
display(table[["sharpe", "cagr", "max_drawdown", "avg_positions", "ann_turnover"]])
L.plot_stats_table(table, title="Gate threshold sensitivity")
'''),
        CAVEATS,
    ])


def nb_momentum() -> dict:
    return notebook([
        md('''
# 02 — Cross-sectional momentum

Rank the universe on 9-month return skipping the most recent month, hold the top
quartile, and stand down in the state momentum crashes in.

**Why skip a month.** One-month returns reverse on average, so a momentum signal
that includes the latest month is partly buying the very thing that is about to
mean-revert. Jegadeesh & Titman skip it for exactly this reason.

**Why the crash filter.** Momentum's characteristic failure is not gradual
underperformance but a crash: it loses most in the rebound *after* a drawdown,
when the beaten-down names it is not long rally hardest. The state that predicts
it is observable in real time — market below its own 200-day average *and*
realized volatility elevated — so the strategy stands down there.

Standalone: no `portfolio_agent` import.
'''),
        md("## Setup"), SETUP, UNIVERSE_CELL, INGEST, FEATURES, CONFIG,
        md("## Signal"),
        code('''
momentum_scores = L.momentum_scores(
    feature_panel,
    top_fraction=0.25,   # hold the top quartile
    crash_filter=True,
    vol_target=0.25,
)

held = (momentum_scores > 0).sum(axis=1)
print(f"names held: mean {held.mean():.1f} | flat on {(held == 0).mean():.1%} of sessions")
'''),
        code('''
# What the crash filter costs and what it saves. The unfiltered version is the
# same signal with the stand-down removed.
unfiltered = L.momentum_scores(feature_panel, top_fraction=0.25, crash_filter=False)

filtered_result = L.run_backtest(momentum_scores, close, config)
unfiltered_result = L.run_backtest(unfiltered, close, config)

display(L.compare_stats({
    "with crash filter": filtered_result,
    "without": unfiltered_result,
    "equal weight": benchmark,
}))

L.plot_equity({"with crash filter": filtered_result, "without": unfiltered_result},
              title="Momentum: effect of the crash filter")
'''),
        *evaluate_cells("momentum", "momentum"),
        md('''
## Sensitivity

How concentrated should the book be? A narrower slice holds stronger names and
more idiosyncratic risk; a wider one converges on the index.
'''),
        code('''
sweep = {}
for fraction in (0.15, 0.25, 0.35, 0.50):
    scores = L.momentum_scores(feature_panel, top_fraction=fraction)
    sweep[f"top {fraction:.0%}"] = L.run_backtest(scores, close, config)

display(L.compare_stats(sweep)[["sharpe", "cagr", "max_drawdown", "avg_positions"]])
L.plot_stats_table(L.compare_stats(sweep), title="Concentration sensitivity")
'''),
        code('''
# Rebalance frequency trades signal freshness against cost. A signal this slow
# should not need daily trading, and daily trading of it mostly buys costs.
sweep = {}
for days in (1, 5, 21, 63):
    variant = L.BacktestConfig(cost_bps=config.cost_bps, max_weight=config.max_weight,
                               rebalance_days=days)
    sweep[f"every {days}d"] = L.run_backtest(momentum_scores, close, variant)

display(L.compare_stats(sweep)[["sharpe", "cagr", "ann_turnover"]])
'''),
        CAVEATS,
    ])


def nb_low_vol() -> dict:
    return notebook([
        md('''
# 03 — Low volatility

Hold the least volatile quartile, weighted by inverse volatility.

The low-volatility anomaly — that low-risk stocks have historically earned
*higher* risk-adjusted returns than high-risk ones, contradicting a plain CAPM
reading — is among the most replicated results in equities, and one of the few
that survives transaction costs at this turnover.

Weighting by inverse volatility rather than equally pushes the portfolio further
along the same axis the selection is made on, which is the point of the strategy.

Standalone: no `portfolio_agent` import.
'''),
        md("## Setup"), SETUP, UNIVERSE_CELL, INGEST, FEATURES, CONFIG,
        md("## Signal"),
        code('''
low_volatility_scores = L.low_volatility_scores(feature_panel, close, top_fraction=0.25)

held = (low_volatility_scores > 0).sum(axis=1)
print(f"names held: mean {held.mean():.1f}")

vol = L.cross_section(feature_panel, "realized_vol_60")
print("\\nrealized vol across the universe (annualized):")
display(vol.mean().sort_values().to_frame("mean vol").head(8))
'''),
        code('''
# The selection is a persistent property, not a fast-moving signal: volatility
# ranks are sticky, which is why this strategy turns over far less than momentum.
import matplotlib.pyplot as plt

ranks = vol.rank(axis=1, pct=True)
fig, ax = plt.subplots(figsize=(11, 4))
for i, symbol in enumerate(sorted(vol.columns)[:8]):
    ax.plot(ranks.index, ranks[symbol], linewidth=1.0,
            color=L.PALETTE[i % len(L.PALETTE)], label=symbol)
ax.axhline(0.25, color="#444444", linestyle="--", linewidth=1, label="selection cut")
ax.set_title("Volatility rank through time (low = selected)")
ax.set_ylabel("cross-sectional percentile")
ax.grid(**L.GRID); ax.legend(ncol=5, fontsize=8)
plt.tight_layout(); plt.show()
'''),
        *evaluate_cells("low_volatility", "low volatility"),
        md('''
## Does the anomaly show up here?

The claim is that realized volatility is *inversely* related to risk-adjusted
return. Testing it directly on this universe is more informative than the
backtest, because it does not depend on the portfolio construction.
'''),
        code('''
returns = close.pct_change()
by_symbol = pd.DataFrame({
    "ann_vol": returns.std() * np.sqrt(252),
    "ann_return": (1 + returns).prod() ** (252 / len(returns)) - 1,
})
by_symbol["sharpe"] = by_symbol["ann_return"] / by_symbol["ann_vol"]

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, column in zip(axes, ("ann_return", "sharpe")):
    ax.scatter(by_symbol["ann_vol"], by_symbol[column], color=L.PALETTE[0], s=30)
    fit = np.polyfit(by_symbol["ann_vol"], by_symbol[column], 1)
    xs = np.linspace(by_symbol["ann_vol"].min(), by_symbol["ann_vol"].max(), 50)
    ax.plot(xs, np.polyval(fit, xs), color=L.PALETTE[1], linestyle="--",
            label=f"slope {fit[0]:+.2f}")
    ax.set_xlabel("annualized volatility"); ax.set_ylabel(column)
    ax.legend(fontsize=8); ax.grid(**L.GRID)
fig.suptitle("Volatility versus outcome, per name", y=1.03)
plt.tight_layout(); plt.show()

print("A negative slope on the right-hand panel is the anomaly. On 30 names over")
print("a few years this is a very weak test — the standard error on that slope is")
print("large enough to admit either sign.")
'''),
        CAVEATS,
    ])


def nb_lstm() -> dict:
    return notebook([
        md('''
# 04 — Supervised LSTM

Train a two-layer LSTM on a window of features to predict each name's
**cross-sectional rank** of forward 5-day return, then hold the top quartile.

**Why rank, not return.** Most of the variance of an absolute forward return is
the market factor — nearly unforecastable, and for a long-only book with no
index hedge, unusable even when forecast correctly. The network would spend its
capacity on the one component it cannot act on. Ranking each name against the
rest of the universe on the same date leaves the idiosyncratic part, which is
what choosing between stocks can actually monetize.

**Why the split is by date.** Every symbol's training window ends before any
symbol's validation window begins. Splitting by row after stacking symbols would
put one name's future alongside another's past.

Standalone: no `portfolio_agent` import. Needs `torch`.
'''),
        md("## Setup"), SETUP, UNIVERSE_CELL, INGEST, FEATURES, CONFIG,
        md('''
## Build the supervised panel

The standardizer is fitted on **training rows only**. Fitting it on the whole
history leaks the test period's moments into the transform — a small leak, and
entirely invisible in the resulting metrics, which is what makes it worth being
strict about.
'''),
        code('''
supervised = L.build_supervised_panel(
    feature_panel, close,
    sequence_length=30,     # sessions of history per sample
    horizon=5,              # forward return being ranked
    train_fraction=0.70,
    val_fraction=0.15,
)

for split in ("train", "val", "test"):
    X = supervised[f"X_{split}"]
    print(f"{split:5s} {X.shape[0]:6d} windows  shape {X.shape[1:]}")
print(f"\\ntrain ends {supervised['train_end'].date()} | "
      f"validation ends {supervised['val_end'].date()}")
print("features:", supervised["features"])
'''),
        md('''
## Train

Early stopping is on validation loss and the returned weights are the **best**
epoch's, not the last. At this signal-to-noise ratio a model reliably keeps
improving in-sample long after it has stopped generalizing.

Rank IC (Spearman correlation between predicted and realized ordering) is the
metric that matches how the output is used — the predictions become a ranking,
so monotone agreement is what matters, not squared error. Values around 0.02–0.05
are normal and useful in equity forecasting; anything above ~0.15 on daily data
usually means a leak.
'''),
        code('''
trained = L.train_lstm(
    supervised,
    epochs=40,
    batch_size=256,
    learning_rate=1e-3,
    hidden_size=64,
    patience=6,
    device="auto",
    seed=42,
)

print(f"\\nbest epoch {trained['best_epoch']} | val loss {trained['best_val_loss']:.5f}")
L.plot_training_curve(trained["history"],
                      ["train_loss", "val_loss", "val_rank_ic"],
                      title="LSTM training")
'''),
        code('''
# Held-out rank IC — the number that decides whether anything below is meaningful.
import torch

model, device = trained["model"], trained["device"]
X_test = torch.tensor(supervised["X_test"]).to(device)
y_test = torch.tensor(supervised["y_test"]).to(device)

with torch.no_grad():
    predictions = model(X_test)

print(f"test rank IC: {L._rank_ic(predictions, y_test):+.4f}")
print(f"test MSE    : {float(torch.nn.functional.mse_loss(predictions, y_test)):.5f}")
print("\\nA rank IC near zero means the ranking below is noise, and the equity")
print("curve is measuring the sample rather than the model.")
'''),
        md('''
## Signal & backtest

Predictions are produced for every date, including the training period, so the
equity curve can be split. **Only the segment after the validation cut is
evidence of anything** — the part before it is the model recalling its own
training data.
'''),
        code('''
lstm_scores = L.lstm_scores(trained, feature_panel, close, top_fraction=0.25)
'''),
        *evaluate_cells("lstm", "lstm"),
        code('''
# In-sample versus out-of-sample, split at the validation cut. A large gap here
# is the normal outcome and the reason the full-period curve is not the result.
cut = supervised["val_end"]

oos_scores = lstm_scores[lstm_scores.index > cut]
oos_close = close[close.index > cut]
oos = L.run_backtest(oos_scores, oos_close, config)
oos_benchmark = L.equal_weight_benchmark(oos_close, config)

is_scores = lstm_scores[lstm_scores.index <= cut]
is_close = close[close.index <= cut]
in_sample = L.run_backtest(is_scores, is_close, config)

display(L.compare_stats({
    "in-sample (not evidence)": in_sample,
    "out-of-sample": oos,
    "equal weight (oos)": oos_benchmark,
}))

L.plot_equity({"out-of-sample": oos}, benchmark=oos_benchmark.returns,
              title=f"LSTM out-of-sample (from {cut.date()})", log_scale=False)
'''),
        CAVEATS,
    ])


def nb_sac() -> dict:
    return notebook([
        md('''
# 05 — Reinforcement learning: SAC allocation policy

Train a Soft Actor-Critic policy that maps a feature vector to an **allocation
weight in [0, 1]** per name — a continuous action, not a discrete buy/sell.

Three design points worth understanding before running it:

**The reward is a differential Sortino, net of turnover.**
`R_t = a_t · ret_{t+1} − cost·|a_t − a_{t−1}|` feeds an online Sortino whose
per-step increment is the reward. Summing those increments approximates the
Sortino ratio of the whole path, so a policy maximizing per-step reward is
maximizing a downside-aware ratio rather than raw return — the distinction that
stops it learning "hold maximum size always". Note the friction is a function of
the *action*: a cost charged as a constant every step shifts every action's
reward identically and therefore cannot penalize turnover at all.

**`gamma` defaults to 0.** Discounting assumes the action influences the next
state. A price-taking book does not move the market, so the observed state
sequence is exogenous, and bootstrapping a value function over it adds estimator
variance without adding signal. With `gamma=0` the critic learns
`Q(s,a) = E[r|s,a]` and the actor maximizes `Q − α·log π` — soft actor-critic
applied to what this decision actually is, a contextual bandit.

**Inference is deterministic.** SAC optimizes a stochastic policy — a squashed
Gaussian whose log-std head supplies the entropy term the algorithm is named
for. At scoring time the mean action is used instead: sampling would make two
runs of one backtest disagree.

Standalone: no `portfolio_agent` import. Needs `torch`.
'''),
        md("## Setup"), SETUP, UNIVERSE_CELL, INGEST, FEATURES, CONFIG,
        md('''
## Train

Experience is **re-collected every epoch** from the current policy. Training
against a buffer collected once means the actor only ever sees a randomly
initialized policy's decisions and never the consequences of its own.

The policy is trained on the first 70% of history only; everything after is held
out.
'''),
        code('''
sac = L.train_sac(
    feature_panel, close,
    epochs=40,
    batch_size=256,
    learning_rate=3e-4,
    hidden_dim=128,
    gamma=0.0,              # contextual bandit; see above
    tau=0.005,
    friction_cost=0.008,
    train_fraction=0.70,
    gradient_steps=150,
    device="auto",
    seed=42,
)

print(f"\\ntrained on data up to {sac['split_date'].date()}")
L.plot_training_curve(sac["history"], ["critic_loss", "actor_loss", "alpha"],
                      title="SAC training")
'''),
        md('''
## The learned policy

`threshold` is what turns a continuous allocation into a portfolio. Below it the
policy is not asking for a position, and keeping those weights would put a token
holding in every name in the universe.
'''),
        code('''
sac_scores = L.sac_scores(sac, feature_panel, close, threshold=0.60)

held = (sac_scores > 0).sum(axis=1)
print(f"names held: mean {held.mean():.1f} | flat on {(held == 0).mean():.1%} of sessions")
'''),
        code('''
# What the policy actually learned to output. A distribution piled against 0 or 1
# means the entropy term collapsed; a flat one near 0.5 means it never learned
# to discriminate.
raw = L.sac_scores(sac, feature_panel, close, threshold=0.0)
values = raw.to_numpy().ravel()
values = values[values > 0]

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
axes[0].hist(values, bins=60, color=L.PALETTE[0])
axes[0].axvline(0.60, color=L.PALETTE[1], linestyle="--", label="threshold")
axes[0].set_title("Distribution of allocation weights"); axes[0].legend(fontsize=8)
axes[0].grid(**L.GRID)

axes[1].plot(raw.index, raw.mean(axis=1), color=L.PALETTE[2], linewidth=1.1)
axes[1].axvline(sac["split_date"], color="#444444", linestyle=":", label="train/test cut")
axes[1].set_title("Mean allocation through time"); axes[1].legend(fontsize=8)
axes[1].grid(**L.GRID)
plt.tight_layout(); plt.show()
'''),
        *evaluate_cells("sac", "sac"),
        code('''
# Out-of-sample only — the policy never saw this period.
cut = sac["split_date"]
oos_close = close[close.index > cut]
oos = L.run_backtest(sac_scores[sac_scores.index > cut], oos_close, config)
oos_benchmark = L.equal_weight_benchmark(oos_close, config)

display(L.compare_stats({"sac (out-of-sample)": oos, "equal weight": oos_benchmark}))
L.plot_equity({"sac (out-of-sample)": oos}, benchmark=oos_benchmark.returns,
              title=f"SAC out-of-sample (from {cut.date()})", log_scale=False)
'''),
        md('''
## A caveat this design carries openly

The reward is net of turnover, which depends on the previous allocation — but
the state does not carry that previous allocation. The policy is therefore
mildly **partially observed**: it is penalized for turnover it cannot see. That
is a deliberate trade to keep the state vector purely feature-based; adding
previous allocation as a twelfth input would fix it at the cost of a state
representation that inference has to track.

Reinforcement learning on a few thousand transitions of one universe is also
close to the smallest problem this method is worth using on. Treat the mechanism
as the deliverable, not the equity curve.
'''),
        CAVEATS,
    ])


def nb_ensemble() -> dict:
    return notebook([
        md('''
# 06 — Ensemble & full comparison

Every strategy, on one universe, one period, one engine — then blended.

Comparing them is only meaningful because they share all three. The rankings
below still say more about this particular sample than about the strategies.

Standalone: no `portfolio_agent` import. Needs `torch` for the two learned members.
'''),
        md("## Setup"), SETUP, UNIVERSE_CELL, INGEST, FEATURES, CONFIG,
        md('''
## Build every member

The two learned members are trained here on the same panel. Keep the epoch counts
modest — this cell does two full training runs.
'''),
        code('''
members = {}

members["rule_based"] = L.rule_based_scores(feature_panel, panel, use_monte_carlo=False)
members["momentum"] = L.momentum_scores(feature_panel, top_fraction=0.25)
members["low_volatility"] = L.low_volatility_scores(feature_panel, close, top_fraction=0.25)

if L.TORCH_AVAILABLE:
    supervised = L.build_supervised_panel(feature_panel, close,
                                          sequence_length=30, horizon=5)
    trained = L.train_lstm(supervised, epochs=25, patience=5, verbose=False)
    members["lstm"] = L.lstm_scores(trained, feature_panel, close, top_fraction=0.25)
    print(f"lstm: best epoch {trained['best_epoch']}, "
          f"val loss {trained['best_val_loss']:.5f}")

    sac = L.train_sac(feature_panel, close, epochs=25, gradient_steps=100, verbose=False)
    members["sac"] = L.sac_scores(sac, feature_panel, close, threshold=0.60)
    print("sac: trained")

print("\\nmembers:", list(members))
'''),
        md('''
## Blend

Members are put on a common scale per date before weighting. Without that, a
member whose scores happen to be large (inverse volatility runs to tens) drowns
one whose scores are bounded in [0, 1], and the configured weights describe
something other than what the blend actually does.
'''),
        code('''
weights = {name: 1.0 / len(members) for name in members}   # start equal
ensemble_scores = L.ensemble_scores(members, weights)

results = {name: L.run_backtest(scores, close, config)
           for name, scores in members.items()}
results["ensemble"] = L.run_backtest(ensemble_scores, close, config)
results["equal weight"] = benchmark

table = L.compare_stats(results)
display(table[["sharpe", "sortino", "cagr", "max_drawdown", "calmar",
               "avg_positions", "ann_turnover"]])
'''),
        md("## Comparison"),
        code('L.plot_equity(results, title="All strategies", benchmark=benchmark.returns)'),
        code('L.plot_stats_table(table, title="Risk-adjusted comparison")'),
        code('''
# Correlation of daily returns. An ensemble only diversifies to the extent its
# members disagree — highly correlated members give the blend of one strategy
# with more turnover.
import matplotlib.pyplot as plt

daily = pd.DataFrame({name: result.returns for name, result in results.items()})
correlation = daily.drop(columns=["equal weight"], errors="ignore").corr()

fig, ax = plt.subplots(figsize=(6.5, 5.2))
image = ax.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(correlation))); ax.set_xticklabels(correlation.columns,
                                                           rotation=45, ha="right")
ax.set_yticks(range(len(correlation))); ax.set_yticklabels(correlation.index)
for i in range(len(correlation)):
    for j in range(len(correlation)):
        ax.text(j, i, f"{correlation.iloc[i, j]:.2f}", ha="center", va="center",
                fontsize=8, color="black")
ax.set_title("Strategy return correlation"); ax.grid(False)
fig.colorbar(image, ax=ax, shrink=0.8)
plt.tight_layout(); plt.show()
'''),
        code('''
# Rolling 6-month Sharpe: which member was carrying the blend, and when. A
# strategy that is only ever good in one regime shows up here and nowhere else.
window = 126
rolling = daily.rolling(window).mean() / daily.rolling(window).std() * np.sqrt(252)

ax = rolling.plot(figsize=(12, 4.2), linewidth=1.1)
ax.axhline(0, color="#444444", linewidth=1)
ax.set_title(f"Rolling {window}-day Sharpe by strategy")
ax.grid(**L.GRID); ax.legend(ncol=4, fontsize=8)
plt.tight_layout(); plt.show()
'''),
        md('''
## Weighting the blend

Equal weights are a defensible default and a poor optimum. Anything fitted on
this sample is fitted on the sample — the sweep below is here to show the
*spread* of outcomes across weightings, not to pick the best one.
'''),
        code('''
schemes = {"equal": {name: 1 / len(members) for name in members}}
if "momentum" in members and "low_volatility" in members:
    schemes["momentum tilt"] = {
        name: (0.5 if name == "momentum" else 0.5 / (len(members) - 1))
        for name in members
    }
    schemes["defensive tilt"] = {
        name: (0.5 if name == "low_volatility" else 0.5 / (len(members) - 1))
        for name in members
    }

sweep = {
    label: L.run_backtest(L.ensemble_scores(members, scheme), close, config)
    for label, scheme in schemes.items()
}
sweep["best single member"] = max(
    (results[name] for name in members),
    key=lambda r: r.stats.get("sharpe", -99),
)
display(L.compare_stats(sweep)[["sharpe", "cagr", "max_drawdown", "ann_turnover"]])
'''),
        CAVEATS,
    ])


NOTEBOOKS = {
    "00_data_ingestion.ipynb": nb_data,
    "01_rule_based.ipynb": nb_rule_based,
    "02_momentum.ipynb": nb_momentum,
    "03_low_volatility.ipynb": nb_low_vol,
    "04_lstm.ipynb": nb_lstm,
    "05_sac_rl.ipynb": nb_sac,
    "06_ensemble_comparison.ipynb": nb_ensemble,
}


def main() -> None:
    global _CELL_COUNTER
    for filename, builder in NOTEBOOKS.items():
        _CELL_COUNTER = itertools.count(1)
        path = OUT / filename
        path.write_text(json.dumps(builder(), indent=1) + "\n")
        print(f"wrote {filename}")


if __name__ == "__main__":
    main()
