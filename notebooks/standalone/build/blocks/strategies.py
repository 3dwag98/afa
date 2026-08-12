# --- Strategies ---------------------------------------------------------------
# Each returns a dates x symbols matrix of non-negative target scores. The
# backtester normalizes, caps and lags them; no strategy here does its own
# position sizing, so they are all comparable on the same engine.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def monte_carlo_win_probability(
    close: pd.Series,
    horizon: int = 21,
    n_paths: int = 400,
    lookback: int = 252,
    seed: int = 42,
) -> pd.Series:
    """Rolling probability that price is higher in `horizon` sessions.

    A block bootstrap over trailing log returns, not a Gaussian: Indian equity
    returns are fat-tailed and serially dependent enough that a normal
    approximation understates both tails, and the left one is the expensive
    side. Blocks of 5 preserve short-run autocorrelation.

    Computed on a coarse grid and forward-filled — running 400 paths on every
    session is the slowest thing in these notebooks and the estimate barely
    moves day to day.
    """
    rng = np.random.default_rng(seed)
    log_returns = np.log(close / close.shift(1)).dropna()
    probability = pd.Series(np.nan, index=close.index)

    block = 5
    n_blocks = max(1, horizon // block)
    step = 5

    positions = range(lookback, len(close), step)
    for i in positions:
        window = log_returns.iloc[max(0, i - lookback):i].to_numpy()
        if len(window) < block * 2:
            continue
        starts = rng.integers(0, len(window) - block, size=(n_paths, n_blocks))
        paths = np.stack([
            np.concatenate([window[s:s + block] for s in row]) for row in starts
        ])
        probability.iloc[i] = float((paths.sum(axis=1) > 0).mean())

    return probability.ffill()


# -- 1. Rule-based: trend + breakout + volume + Monte Carlo --------------------


@dataclass
class RuleBasedParams:
    """Weights and gates for the composite rule-based score."""

    trend_weight: float = 0.30
    breakout_weight: float = 0.25
    volume_weight: float = 0.20
    mc_weight: float = 0.25
    min_score: float = 0.55          # composite required to hold a name
    min_win_probability: float = 0.50
    require_trend: bool = True       # close > sma_200 is a hard gate, not a score


def rule_based_scores(
    feature_panel: Dict[str, pd.DataFrame],
    panel: Dict[str, pd.DataFrame],
    params: Optional[RuleBasedParams] = None,
    use_monte_carlo: bool = True,
) -> pd.DataFrame:
    """Composite of four components, each mapped to [0, 1] before weighting.

    Mapping every component onto a common scale before combining is what makes
    the weights mean what they say. Combining raw components — an RSI in
    [0,100] with a breakout in [-0.2, 0.2] — lets whichever has the widest
    natural range dominate regardless of its weight.
    """
    params = params or RuleBasedParams()
    scores: Dict[str, pd.Series] = {}

    for symbol, features in feature_panel.items():
        close = features["close"]

        # Trend: distance above the 200-day average, saturating at +20%.
        trend_raw = (close / features["sma_200"] - 1.0).clip(-0.2, 0.2)
        trend = (trend_raw + 0.2) / 0.4

        # Breakout: position against the trailing 20-day high (excluding today).
        breakout = (features["breakout_20"].clip(-0.1, 0.05) + 0.1) / 0.15

        # Volume confirmation: today's volume against its 20-day average.
        volume = (features["volume_ratio_20"].clip(0.5, 2.5) - 0.5) / 2.0

        if use_monte_carlo:
            win_probability = monte_carlo_win_probability(close)
        else:
            win_probability = pd.Series(0.5, index=close.index)
        mc = win_probability.clip(0.3, 0.7).sub(0.3).div(0.4)

        composite = (
            params.trend_weight * trend
            + params.breakout_weight * breakout
            + params.volume_weight * volume
            + params.mc_weight * mc
        )

        gate = composite >= params.min_score
        if params.require_trend:
            gate &= close > features["sma_200"]
        if use_monte_carlo:
            gate &= win_probability >= params.min_win_probability

        scores[symbol] = composite.where(gate, 0.0).fillna(0.0)

    return pd.DataFrame(scores).sort_index()


# -- 2. Cross-sectional momentum ----------------------------------------------


def momentum_scores(
    feature_panel: Dict[str, pd.DataFrame],
    top_fraction: float = 0.25,
    crash_filter: bool = True,
    vol_target: float = 0.25,
    min_traded_value: float = 0.0,
) -> pd.DataFrame:
    """Rank on 9-month momentum skipping the last month; hold the top slice.

    The crash filter is the part that earns its keep. Momentum's characteristic
    failure is not gradual underperformance but a crash: it loses most in the
    rebound *after* a drawdown, when the beaten-down names it is short (or
    simply not long) rally hardest. The state that predicts it is observable —
    market below its own 200-day average *and* realized volatility elevated —
    so the strategy stands down there rather than sizing through it.
    """
    from_column = lambda name: pd.DataFrame(
        {s: f[name] for s, f in feature_panel.items()}
    ).sort_index()

    momentum = from_column("mom_9m_skip1m")
    close = from_column("close")

    ranked = momentum.rank(axis=1, pct=True)
    selected = (ranked >= 1.0 - top_fraction).astype(float)

    if min_traded_value > 0:
        liquid = from_column("traded_value_60") >= min_traded_value
        selected = selected.where(liquid, 0.0)

    if crash_filter:
        # Equal-weight universe as the market proxy, so no index file is needed.
        market = close.mean(axis=1)
        market_trend = market > market.rolling(200, min_periods=200).mean()
        market_vol = np.log(market / market.shift(1)).rolling(60, min_periods=60).std() * np.sqrt(TRADING_DAYS)
        risk_on = market_trend & (market_vol <= 1.5 * vol_target)
        selected = selected.where(risk_on.reindex(selected.index).fillna(False), 0.0)

    # Score by rank within the selection, so the strongest names get more weight
    # than the marginal ones that just cleared the cut.
    return (selected * ranked).fillna(0.0)


# -- 3. Cross-sectional low volatility ----------------------------------------


def low_volatility_scores(
    feature_panel: Dict[str, pd.DataFrame],
    top_fraction: float = 0.25,
    min_history_vol: float = 1e-6,
) -> pd.DataFrame:
    """Hold the least volatile quartile, weighted by inverse volatility.

    The low-volatility anomaly is one of the most replicated results in equities
    and among the few that survive transaction costs at this turnover. Weighting
    by inverse vol rather than equally pushes the portfolio further along the
    same axis the selection is made on.
    """
    vol = pd.DataFrame(
        {s: f["realized_vol_60"] for s, f in feature_panel.items()}
    ).sort_index()

    ranked = vol.rank(axis=1, pct=True)
    selected = (ranked <= top_fraction).astype(float)

    inverse_vol = 1.0 / vol.clip(lower=min_history_vol)
    return (selected * inverse_vol).replace([np.inf, -np.inf], 0.0).fillna(0.0)


# -- 6. Ensemble ---------------------------------------------------------------


def ensemble_scores(
    members: Dict[str, pd.DataFrame],
    weights: Optional[Dict[str, float]] = None,
    normalize_members: bool = True,
) -> pd.DataFrame:
    """Weighted blend of member score matrices.

    Members are put on a common scale first, per date. Without that, a member
    whose scores happen to be large (inverse volatility runs to tens) drowns one
    whose scores are bounded in [0, 1], and the configured weights describe
    something other than what the blend actually does.
    """
    if not members:
        raise ValueError("ensemble needs at least one member")

    weights = weights or {name: 1.0 / len(members) for name in members}
    missing = set(members) - set(weights)
    if missing:
        raise ValueError(f"no weight given for member(s): {sorted(missing)}")

    index = sorted(set().union(*[m.index for m in members.values()]))
    columns = sorted(set().union(*[m.columns for m in members.values()]))

    total = pd.DataFrame(0.0, index=pd.Index(index), columns=columns)
    for name, matrix in members.items():
        aligned = matrix.reindex(index=index, columns=columns).fillna(0.0)
        if normalize_members:
            row_max = aligned.max(axis=1).replace(0.0, np.nan)
            aligned = aligned.div(row_max, axis=0).fillna(0.0)
        total += weights[name] * aligned

    return total


def regime_gate(
    scores: pd.DataFrame,
    close: pd.DataFrame,
    trend_window: int = 200,
) -> pd.DataFrame:
    """Mute every score when the market proxy is below its long-run average.

    A blunt instrument, and deliberately so: it is the one regime signal that
    is observable in real time without estimating anything.
    """
    market = close.mean(axis=1)
    risk_on = market > market.rolling(trend_window, min_periods=trend_window).mean()
    return scores.where(risk_on.reindex(scores.index).fillna(False), 0.0)
