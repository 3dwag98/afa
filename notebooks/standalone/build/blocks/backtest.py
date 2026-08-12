# --- Backtest: target-weight portfolio simulation + performance statistics ----

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class BacktestConfig:
    """Simulation settings.

    Attributes:
        initial_capital: Starting equity, in INR.
        cost_bps: One-way cost in basis points, charged on traded notional.
            25 bps round-trip is a realistic all-in figure for Indian cash
            equities once brokerage, STT, exchange charges, GST, stamp duty and
            a modest slippage allowance are added up.
        max_weight: Cap on any single name's target weight.
        rebalance_days: Rebalance every N sessions. Daily rebalancing of a
            noisy signal spends most of its edge on costs; this is the main
            control over turnover.
        max_gross: Cap on total invested weight. 1.0 is long-only, unlevered.
        execution_lag: Sessions between a signal being observed and traded.
            Must be >= 1. Zero would let the simulation trade on a close it
            only knew after the fact — the single most common way a backtest
            manufactures a return that does not exist.
    """

    initial_capital: float = 1_000_000.0
    cost_bps: float = 25.0
    max_weight: float = 0.10
    rebalance_days: int = 5
    max_gross: float = 1.0
    execution_lag: int = 1

    def __post_init__(self) -> None:
        if self.execution_lag < 1:
            raise ValueError(
                "execution_lag must be at least 1 session: a signal computed from "
                "day t's close cannot be traded at day t's close"
            )
        if not 0 < self.max_weight <= 1.0:
            raise ValueError("max_weight must be in (0, 1]")


@dataclass
class BacktestResult:
    """Everything a run produced, ready to chart."""

    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    stats: Dict[str, float] = field(default_factory=dict)
    benchmark: Optional[pd.Series] = None

    @property
    def drawdown(self) -> pd.Series:
        peak = self.equity.cummax()
        return self.equity / peak - 1.0


def normalize_weights(
    raw: pd.DataFrame, config: BacktestConfig
) -> pd.DataFrame:
    """Turn arbitrary non-negative scores into tradeable target weights.

    Caps each name, renormalizes to `max_gross`, and leaves the remainder in
    cash rather than forcing full investment — a strategy that finds nothing
    worth holding should hold nothing, not the least-bad name available.
    """
    weights = raw.clip(lower=0.0).fillna(0.0)

    total = weights.sum(axis=1)
    scaled = weights.div(total.replace(0.0, np.nan), axis=0).fillna(0.0)
    scaled = scaled * config.max_gross

    # Capping frees up weight; one redistribution pass recovers most of it
    # without iterating to a fixed point, which is not worth the complexity
    # when the cap binds on a handful of names.
    #
    # Headroom is masked to names the strategy actually selected. Without that
    # mask the freed weight spreads across the *whole universe*, quietly turning
    # a concentrated signal into a near-index portfolio — a top-5 signal out of
    # 25 names came back holding 21 of them, with the extra 16 positions
    # contributed entirely by this line.
    selected = scaled > 0.0
    capped = scaled.clip(upper=config.max_weight)
    slack = (scaled.sum(axis=1) - capped.sum(axis=1)).clip(lower=0.0)
    headroom = (config.max_weight - capped).clip(lower=0.0).where(selected, 0.0)
    headroom_total = headroom.sum(axis=1).replace(0.0, np.nan)
    redistributed = capped + headroom.mul(slack / headroom_total, axis=0).fillna(0.0)

    return redistributed.clip(upper=config.max_weight).fillna(0.0)


def apply_rebalance_schedule(
    weights: pd.DataFrame, rebalance_days: int
) -> pd.DataFrame:
    """Hold targets between rebalance dates instead of re-trading every session."""
    if rebalance_days <= 1:
        return weights
    mask = pd.Series(False, index=weights.index)
    mask.iloc[::rebalance_days] = True
    held = weights.where(mask, other=np.nan)
    return held.ffill().fillna(0.0)


def run_backtest(
    target_weights: pd.DataFrame,
    close: pd.DataFrame,
    config: Optional[BacktestConfig] = None,
    benchmark: Optional[pd.Series] = None,
) -> BacktestResult:
    """Simulate a long-only target-weight portfolio.

    Args:
        target_weights: Desired weight per name per date, as computed from that
            date's close. Shifted forward by `execution_lag` internally, so the
            caller passes signals in signal-time and never has to remember to
            lag them.
        close: Close prices, same columns and index.
        config: Simulation settings.
        benchmark: Optional comparison series of returns.

    Returns:
        A `BacktestResult`.

    The accounting: weights drift with returns between rebalances (a name that
    doubled is a bigger share of the book), and turnover is measured against
    that *drifted* weight rather than the previous target. Measuring against the
    target overstates trading, because part of the move to the new target
    happened by itself.
    """
    config = config or BacktestConfig()

    close = close.sort_index()
    target_weights = target_weights.reindex(index=close.index, columns=close.columns).fillna(0.0)

    weights = normalize_weights(target_weights, config)
    weights = apply_rebalance_schedule(weights, config.rebalance_days)
    # The lag is applied after scheduling so a rebalance date means "the target
    # decided on this date", which is what a reader expects it to mean.
    weights = weights.shift(config.execution_lag).fillna(0.0)

    asset_returns = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    dates = close.index
    n_dates = len(dates)
    cost_rate = config.cost_bps / 10_000.0

    held = np.zeros(len(close.columns))
    equity = np.empty(n_dates)
    portfolio_returns = np.zeros(n_dates)
    turnover_series = np.zeros(n_dates)
    cost_series = np.zeros(n_dates)

    capital = config.initial_capital
    target_matrix = weights.to_numpy()
    return_matrix = asset_returns.to_numpy()

    for t in range(n_dates):
        target = target_matrix[t]

        traded = np.abs(target - held).sum()
        cost = traded * cost_rate
        held = target

        gross_return = float((held * return_matrix[t]).sum())
        net_return = gross_return - cost

        # Drift: a name that rose is now a larger share of the book. This is
        # what the next rebalance trades against.
        grown = held * (1.0 + return_matrix[t])
        total = grown.sum()
        if total > 0:
            held = grown * (held.sum() / total) if held.sum() > 0 else grown

        capital *= (1.0 + net_return)
        equity[t] = capital
        portfolio_returns[t] = net_return
        turnover_series[t] = traded
        cost_series[t] = cost

    equity_series = pd.Series(equity, index=dates, name="equity")
    returns_series = pd.Series(portfolio_returns, index=dates, name="return")

    result = BacktestResult(
        equity=equity_series,
        returns=returns_series,
        weights=weights,
        turnover=pd.Series(turnover_series, index=dates, name="turnover"),
        costs=pd.Series(cost_series, index=dates, name="cost"),
        benchmark=benchmark,
    )
    result.stats = performance_stats(returns_series, weights=weights,
                                     turnover=result.turnover)
    return result


def performance_stats(
    returns: pd.Series,
    weights: Optional[pd.DataFrame] = None,
    turnover: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """Headline performance statistics for a daily return series.

    Sortino uses downside deviation about zero, and returns 0.0 rather than an
    infinity when a series has no losing day — an infinity wins every
    comparison it enters, which is exactly the wrong behaviour in a ranking.
    """
    returns = returns.dropna()
    if returns.empty:
        return {}

    n = len(returns)
    years = n / TRADING_DAYS
    total_return = float((1.0 + returns).prod() - 1.0)
    cagr = float((1.0 + total_return) ** (1 / years) - 1.0) if years > 0 and total_return > -1 else np.nan

    vol = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS)) if n > 1 else 0.0
    sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(TRADING_DAYS)) if n > 1 and returns.std(ddof=1) > 0 else 0.0

    downside = np.minimum(returns.to_numpy(), 0.0)
    downside_dev = float(np.sqrt(np.mean(downside ** 2)))
    sortino = float(returns.mean() / downside_dev * np.sqrt(TRADING_DAYS)) if downside_dev > 1e-12 else 0.0

    equity = (1.0 + returns).cumprod()
    max_drawdown = float((equity / equity.cummax() - 1.0).min())

    stats = {
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": float(cagr / abs(max_drawdown)) if max_drawdown < -1e-9 and np.isfinite(cagr) else 0.0,
        "hit_rate": float((returns > 0).mean()),
        "best_day": float(returns.max()),
        "worst_day": float(returns.min()),
        "days": int(n),
    }

    if turnover is not None and len(turnover):
        stats["ann_turnover"] = float(turnover.sum() / max(years, 1e-9))
    if weights is not None and not weights.empty:
        gross = weights.sum(axis=1)
        stats["avg_exposure"] = float(gross.mean())
        stats["avg_positions"] = float((weights > 1e-6).sum(axis=1).mean())

    return stats


def equal_weight_benchmark(
    close: pd.DataFrame, config: Optional[BacktestConfig] = None
) -> BacktestResult:
    """Equal-weight buy-and-hold of the universe.

    The honest comparison for a long-only stock picker: beating cash is not the
    question, beating the same names held passively is.
    """
    config = config or BacktestConfig()
    available = close.notna().astype(float)
    weights = available.div(available.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    # Buy and hold: one rebalance schedule so wide it never re-trades.
    passive = BacktestConfig(
        initial_capital=config.initial_capital,
        cost_bps=config.cost_bps,
        max_weight=1.0,
        rebalance_days=max(len(close), 1),
        max_gross=1.0,
        execution_lag=config.execution_lag,
    )
    return run_backtest(weights, close, passive)


def compare_stats(results: Dict[str, BacktestResult]) -> pd.DataFrame:
    """Side-by-side statistics table, best Sharpe first."""
    frame = pd.DataFrame({name: result.stats for name, result in results.items()}).T
    if "sharpe" in frame.columns:
        frame = frame.sort_values("sharpe", ascending=False)
    return frame.round(4)
