"""Tradability screening: circuit locks, illiquidity and "zombie" stocks.

Motivation (docs/QUANT_RESEARCH.md section 15). Every ranking formula in this
platform assumes continuous price discovery and that a printed close is a
price you could have transacted at. On the NSE/BSE mid- and small-cap
segments, neither holds:

- **Circuit limits.** Stocks lock at 5%/10%/20% bands. An operator-driven pump
  locks a stock in the upper circuit for consecutive sessions: the momentum
  formula registers a huge P(t)/P(t-J) and screams BUY, while in reality there
  is no offer to lift. On the way back down the stock locks in the lower
  circuit and the position cannot be exited at all, so the realized loss
  blows through the modelled stop — which in turn inflates the payoff ratio
  b that Kelly sizes off.
- **The illiquidity illusion.** Low realized volatility is supposed to proxy
  for a stable business. In India it frequently proxies for *nothing trading*:
  a stock that prints the same close for days records r = 0 repeatedly, which
  mechanically suppresses its variance and pushes it straight into the
  low-volatility strategy's buy decile. The anomaly the strategy is trying to
  harvest is not the one it ends up holding.

Both are detectable from OHLCV alone:

- A circuit-locked session has **no intraday range** (high == low) together
  with a move from the prior close at or near a statutory band. A stock that
  genuinely traded flat all day has a range; one pinned at its limit does not.
- A zombie prints an unchanged close on a large share of sessions, and/or
  turns over trivial rupee value.

The screen is deliberately expressed as *fractions of a window* rather than
single-day flags, so one quiet session never disqualifies a stock and a
sustained pattern always does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

# Statutory NSE/BSE price bands. The smallest is what a "locked" move has to
# clear before a zero-range session is read as a circuit lock rather than a
# thinly-traded flat day.
CIRCUIT_BANDS = (0.05, 0.10, 0.20)
MIN_CIRCUIT_MOVE = 0.045  # just under the 5% band, allowing for tick rounding

# Screening defaults. Deliberately permissive: they are meant to exclude names
# that are structurally untradeable, not to second-guess a strategy's ranking.
DEFAULT_LIQUIDITY_WINDOW = 60
DEFAULT_MIN_TRADED_VALUE_INR = 5_000_000.0  # ₹50 lakh median daily turnover
DEFAULT_MAX_ZERO_RETURN_FRACTION = 0.30
DEFAULT_MAX_CIRCUIT_LOCK_FRACTION = 0.10


@dataclass
class TradabilityReport:
    """Whether a ticker can actually be traded, and why not if it cannot."""

    tradable: bool
    median_traded_value: float
    zero_return_fraction: float
    circuit_lock_fraction: float
    locked_today: bool
    reasons: List[str] = field(default_factory=list)


def circuit_locked_days(df: pd.DataFrame, min_move: float = MIN_CIRCUIT_MOVE) -> pd.Series:
    """Boolean series marking sessions that locked at a circuit limit.

    A locked session is identified by a zero intraday range (high == low)
    combined with a move from the previous close at or beyond the smallest
    statutory band. Both conditions are needed: a zero range alone is just an
    untraded day, and a large move alone is an ordinary volatile session.

    Args:
        df: OHLCV frame with 'high', 'low' and 'close' columns.
        min_move: Minimum absolute move from the prior close to count.

    Returns:
        Boolean Series aligned to df's index (False where undeterminable).
    """
    required = {"high", "low", "close"}
    if not required.issubset(df.columns) or len(df) < 2:
        return pd.Series(False, index=df.index)

    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")

    prev_close = close.shift(1)
    move = (close / prev_close - 1.0).abs()
    no_range = np.isclose(high, low, rtol=0.0, atol=1e-9)

    return pd.Series(no_range & (move >= min_move), index=df.index).fillna(False)


def zero_return_days(df: pd.DataFrame) -> pd.Series:
    """Boolean series marking sessions whose close did not move at all.

    This is the zombie signature that suppresses realized variance: an
    illiquid ticker that did not trade carries yesterday's close forward, so
    r = 0 rather than r = "small".
    """
    if "close" not in df.columns or len(df) < 2:
        return pd.Series(False, index=df.index)

    close = pd.to_numeric(df["close"], errors="coerce")
    changed = close.diff()
    return pd.Series(np.isclose(changed, 0.0, rtol=0.0, atol=1e-12), index=df.index).fillna(False)


def median_traded_value(df: pd.DataFrame, window: int = DEFAULT_LIQUIDITY_WINDOW) -> float:
    """Median daily turnover in rupees over the trailing window.

    Median rather than mean: a single delivery-heavy day (or one operator
    print) should not make an otherwise dead ticker look liquid.
    """
    if not {"close", "volume"}.issubset(df.columns) or df.empty:
        return 0.0

    close = pd.to_numeric(df["close"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    traded_value = (close * volume).dropna()
    if traded_value.empty:
        return 0.0
    return float(traded_value.iloc[-window:].median())


def assess_tradability(
    df: pd.DataFrame,
    window: int = DEFAULT_LIQUIDITY_WINDOW,
    min_traded_value_inr: float = DEFAULT_MIN_TRADED_VALUE_INR,
    max_zero_return_fraction: float = DEFAULT_MAX_ZERO_RETURN_FRACTION,
    max_circuit_lock_fraction: float = DEFAULT_MAX_CIRCUIT_LOCK_FRACTION,
) -> TradabilityReport:
    """Decide whether a ticker is realistically tradable, and say why not.

    Args:
        df: OHLCV frame for one ticker, oldest row first.
        window: Trailing sessions to screen over.
        min_traded_value_inr: Minimum median daily turnover.
        max_zero_return_fraction: Maximum share of unchanged closes before the
            ticker is treated as a zombie.
        max_circuit_lock_fraction: Maximum share of circuit-locked sessions
            before the ticker is treated as operator-driven / unexecutable.

    Returns:
        A TradabilityReport. An empty or too-short frame is reported as
        untradable — there is no evidence it *can* be traded, and this gate
        exists to keep uninvestable names out of a ranked portfolio.
    """
    if df is None or df.empty:
        return TradabilityReport(
            tradable=False, median_traded_value=0.0, zero_return_fraction=1.0,
            circuit_lock_fraction=0.0, locked_today=False,
            reasons=["no price history"],
        )

    recent = df.iloc[-window:] if len(df) > window else df

    locks = circuit_locked_days(df).iloc[-len(recent):]
    zeros = zero_return_days(df).iloc[-len(recent):]

    # The first row of any window has no prior close, so it can never be
    # classified; exclude it from both denominators rather than counting it
    # as a clean session.
    measurable = max(1, len(recent) - 1)
    circuit_lock_fraction = float(locks.sum()) / measurable
    zero_return_fraction = float(zeros.sum()) / measurable
    traded_value = median_traded_value(df, window)
    locked_today = bool(locks.iloc[-1]) if len(locks) else False

    reasons: List[str] = []
    if traded_value < min_traded_value_inr:
        reasons.append(
            f"illiquid: median turnover {traded_value:,.0f} < "
            f"{min_traded_value_inr:,.0f} INR/day"
        )
    if zero_return_fraction > max_zero_return_fraction:
        reasons.append(
            f"zombie: {zero_return_fraction:.0%} of sessions closed unchanged "
            f"(> {max_zero_return_fraction:.0%}): variance is suppressed by illiquidity, "
            f"not by stability"
        )
    if circuit_lock_fraction > max_circuit_lock_fraction:
        reasons.append(
            f"circuit-driven: {circuit_lock_fraction:.0%} of sessions locked at a "
            f"circuit limit (> {max_circuit_lock_fraction:.0%})"
        )
    if locked_today:
        reasons.append("locked at a circuit limit on the decision date; no fill available")

    return TradabilityReport(
        tradable=not reasons,
        median_traded_value=traded_value,
        zero_return_fraction=zero_return_fraction,
        circuit_lock_fraction=circuit_lock_fraction,
        locked_today=locked_today,
        reasons=reasons,
    )


def split_intraday_and_overnight(df: pd.DataFrame) -> Optional[tuple]:
    """Decompose close-to-close returns into intraday and overnight legs.

    A close-to-close return bundles two different processes: the overnight gap
    (open_t / close_{t-1} - 1), which reprices global cues, FII decisions and
    policy news in a single instantaneous jump, and the intraday session
    (close_t / open_t - 1), which is the continuous trading the GARCH recursion
    is actually a model of. See src/volatility_models.py for why keeping them
    apart matters.

    Args:
        df: OHLCV frame with 'open' and 'close' columns.

    Returns:
        (intraday_returns, overnight_returns) as numpy arrays, or None when the
        frame lacks the columns or the history to compute them.
    """
    if not {"open", "close"}.issubset(df.columns) or len(df) < 3:
        return None

    open_px = pd.to_numeric(df["open"], errors="coerce")
    close_px = pd.to_numeric(df["close"], errors="coerce")

    intraday = (close_px / open_px - 1.0)
    overnight = (open_px / close_px.shift(1) - 1.0)

    combined = pd.concat([intraday, overnight], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(combined) < 3:
        return None

    return combined.iloc[:, 0].to_numpy(), combined.iloc[:, 1].to_numpy()
