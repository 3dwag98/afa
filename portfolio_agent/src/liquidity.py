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

These are primitives, not the screen itself. They are wrapped as lag-safe
features (`features/technical.py`), and the thresholds live in
`strategies/cross_sectional.py::TradabilityFilter`, which reads those features
during ranking. Keeping the rules in exactly one place is deliberate: a second
copy operating on raw frames would drift from the feature path the moment
either window or threshold changed.
"""

from __future__ import annotations

from typing import Optional

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
