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
  with a move from the prior close landing on a statutory band. A stock that
  genuinely traded flat all day has a range; one pinned at its limit does not.
  The ladder includes the **1% and 2% bands** the exchanges impose ad hoc
  through the ASM/GSM surveillance frameworks, not just the 5/10/20% defaults
  — a small-cap pinned at a 1% upper circuit prints +1% on zero volume, and a
  detector keyed to "at least 5%" waves it straight through to the momentum
  ranking.
- An **operator trap** is the weaker but more common footprint: the stock
  trades through a real range and only locks late in the session, so it closes
  at its high on a band-sized move (high == close). The zero-range detector
  misses it entirely, yet by the close there is no offer left to lift.
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

from typing import Optional, Sequence

import numpy as np
import pandas as pd

# Statutory NSE/BSE price bands, including the 1% and 2% bands the exchanges
# impose ad hoc through the Additional Surveillance Measure (ASM) and Graded
# Surveillance Measure (GSM) frameworks on volatile, news-driven or
# operator-suspected scrips. A fixed "must move at least 4.5%" floor missed
# those entirely, and a ladder starting at 2% still misses the tightest ASM/GSM
# band: a surveillance-flagged small-cap pinned at a 1% upper circuit prints
# +1% on zero range, which momentum reads as strength for a stock nobody can
# buy. The 1% band is precisely the one applied to the names most likely to be
# operator-driven, so it is the one that matters most.
CIRCUIT_BANDS = (0.01, 0.02, 0.05, 0.10, 0.20)

# How far a move may sit from a band and still count as that band. Circuit
# prices are computed off the previous close and rounded to the tick, so an
# exact match is not something to rely on.
CIRCUIT_BAND_TOLERANCE = 0.004

# Cap on the tolerance as a share of the band itself. A flat 40 bp window is
# proportionate around a 5% band and absurd around a 1% one, where it would
# accept anything from 0.6% to 1.4% — half the ordinary quiet-day moves on the
# tape. Tick rounding on a circuit price is worth a few basis points, not tens,
# so tight bands get a proportionally tight window. Bands at 2% and above are
# unaffected (0.25 * 0.02 = 0.005 > 0.004), so this only narrows the new 1%
# band rather than re-tuning the ladder that was already in use.
CIRCUIT_BAND_RELATIVE_TOLERANCE = 0.25

# Screening defaults. Deliberately permissive: they are meant to exclude names
# that are structurally untradeable, not to second-guess a strategy's ranking.
DEFAULT_LIQUIDITY_WINDOW = 60
DEFAULT_MIN_TRADED_VALUE_INR = 5_000_000.0  # ₹50 lakh median daily turnover
DEFAULT_MAX_ZERO_RETURN_FRACTION = 0.30
DEFAULT_MAX_CIRCUIT_LOCK_FRACTION = 0.10
# Upper-circuit closes are rarer than full zero-range locks (a stock can run up
# and only lock late in the session), so the sustained-pattern threshold is
# tighter than the general circuit-lock one.
DEFAULT_MAX_OPERATOR_TRAP_FRACTION = 0.05


def band_tolerance(
    band: float,
    tolerance: float = CIRCUIT_BAND_TOLERANCE,
    relative_tolerance: float = CIRCUIT_BAND_RELATIVE_TOLERANCE,
) -> float:
    """Matching window around one band: the tighter of absolute and relative.

    See CIRCUIT_BAND_RELATIVE_TOLERANCE for why the window has to shrink with
    the band rather than stay flat across a ladder spanning 1% to 20%.
    """
    return min(tolerance, abs(band) * relative_tolerance)


def matches_circuit_band(
    move: np.ndarray,
    bands: Sequence[float] = CIRCUIT_BANDS,
    tolerance: float = CIRCUIT_BAND_TOLERANCE,
) -> np.ndarray:
    """Whether each absolute move sits at (or beyond) a statutory price band.

    Band *matching* rather than a single floor. A floor has to be set at the
    smallest band to catch 1% ASM/GSM locks, but then every zero-range day with
    a >=1% move counts as a lock, including ordinary thin trading. Matching each
    move against the actual band ladder (1/2/5/10/20%, within a tick-rounding
    tolerance that narrows with the band) catches the tight surveillance locks
    without also sweeping up everything above them. Moves beyond the widest band
    still count — nothing legitimate moves 20% with no intraday range.

    Args:
        move: Absolute returns from the prior close.
        bands: Statutory band ladder as fractions.
        tolerance: Absolute tolerance around each band, narrowed per band by
            CIRCUIT_BAND_RELATIVE_TOLERANCE.

    Returns:
        Boolean array of the same shape.
    """
    move = np.asarray(move, dtype=float)
    matched = np.zeros(move.shape, dtype=bool)
    for band in bands:
        matched |= np.abs(move - band) <= band_tolerance(band, tolerance)
    widest = max(bands)
    return matched | (move >= widest - band_tolerance(widest, tolerance))


def circuit_locked_days(
    df: pd.DataFrame,
    bands: Sequence[float] = CIRCUIT_BANDS,
    tolerance: float = CIRCUIT_BAND_TOLERANCE,
) -> pd.Series:
    """Boolean series marking sessions that locked at a circuit limit.

    A locked session is identified by a zero intraday range (high == low)
    combined with a move from the previous close that lands on a statutory
    band. Both conditions are needed: a zero range alone is just an untraded
    day, and a band-sized move alone is an ordinary volatile session that
    traded through a range.

    Args:
        df: OHLCV frame with 'high', 'low' and 'close' columns.
        bands: Statutory band ladder to match against.
        tolerance: Absolute tolerance around each band.

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
    at_band = matches_circuit_band(move.to_numpy(), bands, tolerance)

    return pd.Series(no_range & at_band, index=df.index).fillna(False)


def _signed_move(df: pd.DataFrame) -> Optional[pd.Series]:
    """Signed return from the previous close, or None if uncomputable."""
    if "close" not in df.columns or len(df) < 2:
        return None
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1)
    return close / prev_close - 1.0


def operator_trap_days(
    df: pd.DataFrame,
    bands: Sequence[float] = CIRCUIT_BANDS,
    tolerance: float = CIRCUIT_BAND_TOLERANCE,
) -> pd.Series:
    """Boolean series marking sessions that *closed pinned at the upper limit*.

    The signature is ``high == close`` together with an **upward** move landing
    on a statutory band. This is deliberately weaker than
    :func:`circuit_locked_days`, which additionally demands ``high == low``:

    - ``high == low`` describes a stock that gapped straight to its limit and
      never traded off it. That is the cleanest lock, and the one the 60-day
      structural screen is built on.
    - ``high == close`` also catches the session that traded through a real
      range and *then* locked — the classic operator footprint, where the buy
      side is walked up through the day until the circuit is hit and the stock
      shuts with orders stacked on the bid. It prints a genuine intraday range,
      so the stricter detector waves it straight through, and yet by the close
      there is no offer left to lift.

    Both flavours are untradeable at the close, which is the price every signal
    in this platform is generated from. Momentum reads the printed move as
    strength; the order queued against it cannot fill.

    The condition is asymmetric on purpose: a *lower*-circuit close (``low ==
    close`` on a band-sized fall) is a different problem — an exit that cannot
    be taken rather than an entry that cannot be filled — and is reported
    separately by :func:`lower_circuit_locked_days`.

    Args:
        df: OHLCV frame with 'high' and 'close' columns.
        bands: Statutory band ladder to match against.
        tolerance: Absolute tolerance around each band.

    Returns:
        Boolean Series aligned to df's index (False where undeterminable).
    """
    if not {"high", "close"}.issubset(df.columns) or len(df) < 2:
        return pd.Series(False, index=df.index)

    high = pd.to_numeric(df["high"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    move = _signed_move(df)
    if move is None:
        return pd.Series(False, index=df.index)

    closed_at_high = np.isclose(high, close, rtol=0.0, atol=1e-9)
    up_at_band = (move > 0).to_numpy() & matches_circuit_band(
        move.abs().to_numpy(), bands, tolerance
    )
    return pd.Series(closed_at_high & up_at_band, index=df.index).fillna(False)


def lower_circuit_locked_days(
    df: pd.DataFrame,
    bands: Sequence[float] = CIRCUIT_BANDS,
    tolerance: float = CIRCUIT_BAND_TOLERANCE,
) -> pd.Series:
    """Boolean series marking sessions that closed pinned at the *lower* limit.

    The mirror of :func:`operator_trap_days`: ``low == close`` on a band-sized
    fall. This is the exit-side hazard rather than the entry-side one — a
    holding locked down cannot be sold at the modelled stop, and every further
    session it stays locked realizes a loss the risk model never priced. The
    backtest engine reads this to fire an exit trigger the moment a lock is
    detected, instead of waiting for a stop that will never be touched at a
    fillable price.

    Args:
        df: OHLCV frame with 'low' and 'close' columns.
        bands: Statutory band ladder to match against.
        tolerance: Absolute tolerance around each band.

    Returns:
        Boolean Series aligned to df's index (False where undeterminable).
    """
    if not {"low", "close"}.issubset(df.columns) or len(df) < 2:
        return pd.Series(False, index=df.index)

    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    move = _signed_move(df)
    if move is None:
        return pd.Series(False, index=df.index)

    closed_at_low = np.isclose(low, close, rtol=0.0, atol=1e-9)
    down_at_band = (move < 0).to_numpy() & matches_circuit_band(
        move.abs().to_numpy(), bands, tolerance
    )
    return pd.Series(closed_at_low & down_at_band, index=df.index).fillna(False)


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
