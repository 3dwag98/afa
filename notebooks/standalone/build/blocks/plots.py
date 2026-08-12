# --- Visualization ------------------------------------------------------------

from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# A small, colour-blind-safe categorical palette (Okabe-Ito). Chosen so a
# multi-series chart stays readable in greyscale and for the ~8% of readers
# with a colour vision deficiency, which the default matplotlib cycle does not.
PALETTE = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
]
GRID = {"color": "#CCCCCC", "linewidth": 0.6, "alpha": 0.7}

plt.rcParams.update({
    "figure.figsize": (11, 4.5),
    "figure.dpi": 110,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": plt.cycler(color=PALETTE),
    "font.size": 10,
    "legend.frameon": False,
})


def _pct_axis(ax) -> None:
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))


def plot_prices(panel: Dict[str, pd.DataFrame], n: int = 8, title: str = "Price history") -> None:
    """Normalized price paths, so names of different absolute price compare."""
    fig, ax = plt.subplots()
    for symbol in list(sorted(panel))[:n]:
        close = panel[symbol]["close"].dropna()
        if close.empty:
            continue
        ax.plot(close.index, close / close.iloc[0], label=symbol, linewidth=1.2)
    ax.set_title(f"{title} (rebased to 1.0)")
    ax.set_ylabel("growth of 1")
    ax.grid(**GRID)
    ax.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_data_quality(quality: pd.DataFrame) -> None:
    """Coverage and volatility side by side — the two things that invalidate a run."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    order = quality.sort_values("coverage")
    axes[0].barh(order["symbol"], order["coverage"], color=PALETTE[0])
    axes[0].axvline(0.95, color=PALETTE[1], linestyle="--", linewidth=1,
                    label="95% of business days")
    axes[0].set_title("Session coverage")
    axes[0].set_xlabel("observed / expected business days")
    axes[0].legend(fontsize=8)
    axes[0].grid(**GRID)

    axes[1].scatter(quality["ann_vol"], quality["max_1d_move"],
                    color=PALETTE[2], s=28, alpha=0.85)
    for _, row in quality.iterrows():
        axes[1].annotate(row["symbol"], (row["ann_vol"], row["max_1d_move"]),
                         fontsize=6, alpha=0.7,
                         xytext=(3, 3), textcoords="offset points")
    axes[1].set_title("Volatility vs largest single-day move")
    axes[1].set_xlabel("annualized vol")
    axes[1].set_ylabel("max |1-day return|")
    axes[1].grid(**GRID)

    plt.tight_layout()
    plt.show()


def plot_equity(
    results: Dict[str, "BacktestResult"],
    benchmark: Optional[pd.Series] = None,
    title: str = "Equity curve",
    log_scale: bool = True,
) -> None:
    """Equity curves over drawdown — the two panels that matter, aligned in time."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})

    for i, (name, result) in enumerate(results.items()):
        equity = result.equity / result.equity.iloc[0]
        axes[0].plot(equity.index, equity, label=name,
                     color=PALETTE[i % len(PALETTE)], linewidth=1.4)
        axes[1].fill_between(result.drawdown.index, result.drawdown, 0,
                             color=PALETTE[i % len(PALETTE)], alpha=0.28)

    if benchmark is not None and len(benchmark):
        bench = (1.0 + benchmark.fillna(0.0)).cumprod()
        axes[0].plot(bench.index, bench / bench.iloc[0], label="benchmark",
                     color="#666666", linewidth=1.2, linestyle="--")

    axes[0].set_title(title)
    axes[0].set_ylabel("growth of 1")
    if log_scale:
        axes[0].set_yscale("log")
        axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}x"))
    axes[0].grid(**GRID)
    axes[0].legend(ncol=3, fontsize=9)

    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("peak-to-trough")
    _pct_axis(axes[1])
    axes[1].grid(**GRID)

    plt.tight_layout()
    plt.show()


def plot_return_profile(result: "BacktestResult", name: str = "strategy") -> None:
    """Distribution, rolling risk-adjusted return, and monthly seasonality."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

    returns = result.returns.dropna()

    axes[0].hist(returns, bins=60, color=PALETTE[0], alpha=0.85)
    axes[0].axvline(0, color="#444444", linewidth=1)
    axes[0].axvline(returns.mean(), color=PALETTE[1], linestyle="--", linewidth=1.2,
                    label=f"mean {returns.mean():.4f}")
    axes[0].set_title("Daily return distribution")
    axes[0].legend(fontsize=8)
    axes[0].grid(**GRID)

    window = min(126, max(21, len(returns) // 4))
    rolling = (returns.rolling(window).mean() / returns.rolling(window).std()) * np.sqrt(252)
    axes[1].plot(rolling.index, rolling, color=PALETTE[2], linewidth=1.2)
    axes[1].axhline(0, color="#444444", linewidth=1)
    axes[1].set_title(f"Rolling Sharpe ({window}d)")
    axes[1].grid(**GRID)

    monthly = returns.resample("ME").apply(lambda r: (1 + r).prod() - 1)
    colors = [PALETTE[2] if v >= 0 else PALETTE[1] for v in monthly]
    axes[2].bar(monthly.index, monthly, width=20, color=colors)
    axes[2].axhline(0, color="#444444", linewidth=1)
    axes[2].set_title("Monthly returns")
    _pct_axis(axes[2])
    axes[2].grid(**GRID)

    fig.suptitle(name, y=1.04, fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_exposure(result: "BacktestResult", name: str = "strategy") -> None:
    """What the book actually held: gross exposure, breadth, and turnover."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))

    gross = result.weights.sum(axis=1)
    axes[0].plot(gross.index, gross, color=PALETTE[0], linewidth=1.1)
    axes[0].set_title("Gross exposure")
    _pct_axis(axes[0])
    axes[0].grid(**GRID)

    breadth = (result.weights > 1e-6).sum(axis=1)
    axes[1].plot(breadth.index, breadth, color=PALETTE[3], linewidth=1.1)
    axes[1].set_title("Positions held")
    axes[1].grid(**GRID)

    cumulative_cost = result.costs.cumsum()
    axes[2].plot(cumulative_cost.index, cumulative_cost, color=PALETTE[1], linewidth=1.2)
    axes[2].set_title("Cumulative trading cost")
    _pct_axis(axes[2])
    axes[2].grid(**GRID)

    fig.suptitle(name, y=1.04, fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_weight_heatmap(result: "BacktestResult", max_symbols: int = 25,
                        title: str = "Allocation over time") -> None:
    """Which names the book was in, and when."""
    weights = result.weights
    if weights.empty:
        print("no weights to plot")
        return

    top = weights.mean().sort_values(ascending=False).head(max_symbols).index
    matrix = weights[top].T

    fig, ax = plt.subplots(figsize=(12, max(3.2, 0.26 * len(top))))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis",
                      interpolation="nearest")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top, fontsize=7)

    ticks = np.linspace(0, matrix.shape[1] - 1, min(10, matrix.shape[1])).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([matrix.columns[i].strftime("%Y-%m") for i in ticks],
                       rotation=45, ha="right", fontsize=8)
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="weight", shrink=0.8)
    plt.tight_layout()
    plt.show()


def plot_training_curve(history: pd.DataFrame, metrics: Sequence[str],
                        title: str = "Training") -> None:
    """One panel per tracked metric, sharing the epoch axis."""
    metrics = [m for m in metrics if m in history.columns]
    if not metrics:
        print("no metrics to plot")
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 3.4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric, colour in zip(axes, metrics, PALETTE):
        ax.plot(history["epoch"], history[metric], color=colour, linewidth=1.4)
        ax.set_title(metric)
        ax.set_xlabel("epoch")
        ax.grid(**GRID)

    fig.suptitle(title, y=1.05, fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_stats_table(stats: pd.DataFrame, title: str = "Performance") -> None:
    """Bar comparison of the statistics that are comparable across strategies."""
    columns = [c for c in ("sharpe", "sortino", "cagr", "max_drawdown") if c in stats.columns]
    if not columns:
        print("nothing comparable to plot")
        return

    fig, axes = plt.subplots(1, len(columns), figsize=(3.4 * len(columns), 3.4))
    if len(columns) == 1:
        axes = [axes]

    for ax, column in zip(axes, columns):
        values = stats[column].sort_values()
        colors = [PALETTE[1] if v < 0 else PALETTE[0] for v in values]
        ax.barh(values.index, values, color=colors)
        ax.set_title(column)
        ax.axvline(0, color="#444444", linewidth=1)
        ax.tick_params(labelsize=8)
        ax.grid(**GRID)

    fig.suptitle(title, y=1.04, fontsize=11)
    plt.tight_layout()
    plt.show()
