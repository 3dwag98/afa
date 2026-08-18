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


TRADING_DAYS_PER_YEAR = 252


def idiosyncratic_volatility(
    close: pd.DataFrame, window: int = 60, lag: int = 1
) -> pd.DataFrame:
    """Volatility of the CAPM residual, per symbol, per date.

    **The residual, not the total**, and the distinction is the whole point of
    the sort. Total volatility mixes two things a cross-sectional book should
    keep apart: how much a stock moves *with* the market, and how much it moves
    on its own. Ranking on the sum puts high-beta names and
    idiosyncratically-wild names in the same bucket, and the 2025 work on the
    low-risk anomaly finds those two sorts behave very differently out of
    sample — idiosyncratic-volatility sorts survive where beta sorts largely do
    not.

    No regression loop is needed. For an OLS fit on a window,

        var(residual) = var(r_i) - beta^2 * var(r_m),  beta = cov(r_i, r_m) / var(r_m)

    and every term on the right is a `rolling` operation. Causality comes from
    `rolling` rather than being argued about: a window ends at the row it
    labels.

    The market is the equal-weighted composite of the cross-section itself,
    which is what this file has available and what the package falls back to.
    """
    returns = (close.shift(lag) if lag else close).pct_change()
    market = returns.mean(axis=1)

    floor = max(2, window // 2)
    market_var = market.rolling(window, min_periods=floor).var()
    safe_var = market_var.replace(0.0, np.nan)

    residual = {}
    for symbol in returns.columns:
        series = returns[symbol]
        own_var = series.rolling(window, min_periods=floor).var()
        covariance = series.rolling(window, min_periods=floor).cov(market)
        beta = covariance / safe_var
        variance = own_var - beta.pow(2) * market_var
        # Where the market was flat it explains nothing, so the residual is the
        # whole return rather than an undefined quantity.
        variance = variance.where(market_var.notna() & (market_var > 0), own_var)
        residual[symbol] = np.sqrt(variance.clip(lower=0.0))

    return pd.DataFrame(residual, index=returns.index) * np.sqrt(TRADING_DAYS_PER_YEAR)


def low_volatility_scores(
    feature_panel: Dict[str, pd.DataFrame],
    close: Optional[pd.DataFrame] = None,
    top_fraction: float = 0.25,
    min_history_vol: float = 1e-6,
    sort_on: str = "idiosyncratic",
) -> pd.DataFrame:
    """Hold the least volatile quartile, weighted by inverse volatility.

    The low-volatility anomaly is one of the most replicated results in equities
    and among the few that survive transaction costs at this turnover. Weighting
    by inverse vol rather than equally pushes the portfolio further along the
    same axis the selection is made on.

    `sort_on` defaults to `"idiosyncratic"` — the CAPM residual — matching the
    package since T14. `"total"` reproduces the original realized-volatility
    sort, which is worth keeping precisely so the two can be compared: they
    select different names, and the comparison is the finding.

    Args:
        feature_panel: Per-symbol feature frames. `realized_vol_60` is read for
            the total sort.
        close: Wide `(date x symbol)` closes. Required for the idiosyncratic
            sort, which needs the cross-section to residualize against.
        top_fraction: Quartile by default.
        min_history_vol: Floor before inverting, so a near-zero vol does not
            produce an unbounded weight.
        sort_on: `"idiosyncratic"` or `"total"`.

    Raises:
        ValueError: On an unknown `sort_on`, or the idiosyncratic sort without
            `close`. Falling back to the total sort would report one measure
            under the other's name.
    """
    if sort_on not in ("idiosyncratic", "total"):
        raise ValueError(
            f"sort_on must be 'idiosyncratic' or 'total', got {sort_on!r}"
        )

    if sort_on == "idiosyncratic":
        if close is None:
            raise ValueError(
                "the idiosyncratic sort needs `close` — a residual is defined "
                "against a cross-section. Falling back to the total sort would "
                "report one measure under the other's name."
            )
        vol = idiosyncratic_volatility(close).sort_index()
    else:
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
