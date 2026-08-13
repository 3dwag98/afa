"""What is actually in the store, answerable without writing a script.

The motivating failure: `default_history_years: 5` meant every cached series
began after the COVID crash, so an entire regime was absent from every backtest
the platform had ever run. Nothing surfaced it. It was found by counting rows
in parquet files by hand, months later. The point of this module is that the
same question — *what span does my data actually cover* — takes one command.

The numbers here are chosen to answer the questions that go wrong silently:

* **Span and sessions**, per symbol and across the store, because a config
  default is not a claim about what is on disk.
* **Coverage against the inferred calendar**, because a symbol with 900 bars
  over four years and a symbol with 900 bars over eight are different objects
  and the row count alone cannot tell them apart.
* **Longest gap**, because a hundred scattered missing days is a download
  problem and a hundred consecutive ones is a suspension.
* **Corporate actions per year**, because zero of them across 2,400 Indian
  names over five years means the adjustment data is not there — which was true
  of this cache until T01, and was invisible.
* **Schema coverage**, because the wider schema only applies to symbols
  ingested after it landed. A store where 2,397 files predate it and 3 do not
  will pass every check on the 3 and tell you nothing about the rest unless it
  says so out loud.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .invariants import (
    DEFAULT_CALENDAR_QUORUM,
    DEFAULT_MIN_SESSIONS,
    infer_trading_calendar,
)

logger = logging.getLogger(__name__)

#: Columns T01 added. Their absence is not an error — it means the symbol was
#: cached before the wider schema — but it does mean the adjustment-provenance
#: check cannot run, which is worth reporting rather than passing silently.
ADJUSTMENT_COLUMNS = ("adj_close", "adj_factor", "dividends", "stock_splits")
RAW_PRICE_COLUMNS = ("open_raw", "high_raw", "low_raw", "close_raw")


@dataclass(frozen=True)
class SymbolStatus:
    """One symbol's inventory line."""

    symbol: str
    sessions: int
    first_bar: Optional[pd.Timestamp]
    last_bar: Optional[pd.Timestamp]
    expected_sessions: int
    missing_sessions: int
    longest_gap: int
    has_adjustment_columns: bool
    has_raw_columns: bool
    corporate_actions: int
    years: float

    @property
    def coverage(self) -> float:
        """Share of the calendar's sessions this symbol actually has.

        1.0 when no calendar could be inferred, since "coverage against
        nothing" is not a number and reporting 0.0 would look like a failure.
        """
        if self.expected_sessions <= 0:
            return 1.0
        return 1.0 - self.missing_sessions / self.expected_sessions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "sessions": self.sessions,
            "first_bar": self.first_bar.strftime("%Y-%m-%d") if self.first_bar is not None else None,
            "last_bar": self.last_bar.strftime("%Y-%m-%d") if self.last_bar is not None else None,
            "years": round(self.years, 2),
            "coverage": round(self.coverage, 4),
            "missing_sessions": self.missing_sessions,
            "longest_gap": self.longest_gap,
            "corporate_actions": self.corporate_actions,
            "has_adjustment_columns": self.has_adjustment_columns,
            "has_raw_columns": self.has_raw_columns,
        }


@dataclass
class StoreStatus:
    """The whole store, summarized."""

    symbols: List[SymbolStatus] = field(default_factory=list)
    calendar_sessions: int = 0
    calendar_start: Optional[pd.Timestamp] = None
    calendar_end: Optional[pd.Timestamp] = None
    cache_dir: Optional[str] = None
    unreadable: List[str] = field(default_factory=list)
    min_sessions: int = DEFAULT_MIN_SESSIONS

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    @property
    def total_bars(self) -> int:
        return sum(s.sessions for s in self.symbols)

    @property
    def below_threshold(self) -> List[SymbolStatus]:
        """Symbols too short to train on."""
        return [s for s in self.symbols if s.sessions < self.min_sessions]

    @property
    def without_adjustment_columns(self) -> List[SymbolStatus]:
        return [s for s in self.symbols if not s.has_adjustment_columns]

    def span(self) -> tuple:
        """Earliest first bar and latest last bar across the store."""
        firsts = [s.first_bar for s in self.symbols if s.first_bar is not None]
        lasts = [s.last_bar for s in self.symbols if s.last_bar is not None]
        return (min(firsts) if firsts else None, max(lasts) if lasts else None)

    def actions_per_year(self) -> Dict[int, int]:
        """Corporate actions found per calendar year, across the store."""
        return dict(sorted(self._actions_by_year.items()))

    #: Filled by `collect_status`; kept off the dataclass signature because it
    #: is derived rather than supplied.
    _actions_by_year: Dict[int, int] = field(default_factory=dict, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        start, end = self.span()
        return {
            "cache_dir": self.cache_dir,
            "symbols": self.n_symbols,
            "total_bars": self.total_bars,
            "span_start": start.strftime("%Y-%m-%d") if start is not None else None,
            "span_end": end.strftime("%Y-%m-%d") if end is not None else None,
            "calendar_sessions": self.calendar_sessions,
            "below_threshold": len(self.below_threshold),
            "min_sessions": self.min_sessions,
            "without_adjustment_columns": len(self.without_adjustment_columns),
            "unreadable": list(self.unreadable),
            "corporate_actions_per_year": self.actions_per_year(),
        }

    def to_frame(self) -> pd.DataFrame:
        """Per-symbol table, for sorting and slicing in a notebook."""
        if not self.symbols:
            return pd.DataFrame()
        return pd.DataFrame([s.to_dict() for s in self.symbols])

    def render(self, worst: int = 10) -> str:
        """The report `data status` prints."""
        start, end = self.span()
        years = ((end - start).days / 365.25) if (start is not None and end is not None) else 0.0

        lines = [
            "Data store status",
            "=" * 62,
            f"  cache            {self.cache_dir or '(in memory)'}",
            f"  symbols          {self.n_symbols}",
            f"  bars             {self.total_bars:,}",
        ]
        if start is not None:
            lines.append(
                f"  span             {start:%Y-%m-%d} .. {end:%Y-%m-%d}  ({years:.1f} years)"
            )
        else:
            lines.append("  span             (no bars)")

        if self.calendar_sessions:
            lines.append(
                f"  sessions         {self.calendar_sessions} inferred trading days "
                f"({self.calendar_start:%Y-%m-%d} .. {self.calendar_end:%Y-%m-%d})"
            )
        else:
            lines.append(
                "  sessions         not inferred (needs at least two symbols)"
            )

        if self.symbols:
            coverage = np.mean([s.coverage for s in self.symbols])
            sessions = np.array([s.sessions for s in self.symbols])
            lines += [
                f"  coverage         {coverage:.1%} mean, "
                f"{min(s.coverage for s in self.symbols):.1%} worst",
                f"  history          median {int(np.median(sessions))} sessions, "
                f"min {sessions.min()}, max {sessions.max()}",
            ]

        short = self.below_threshold
        lines.append(
            f"  below {self.min_sessions:>4} bars  {len(short)} symbol(s) "
            f"({len(short) / max(self.n_symbols, 1):.0%}) — not usable for training"
        )

        missing_schema = self.without_adjustment_columns
        if missing_schema:
            lines += [
                "",
                f"  {len(missing_schema)} symbol(s) have no adjustment columns.",
                "  They were cached before the wider schema, so the adjustment-",
                "  provenance check cannot run on them and corporate actions are",
                "  invisible. Re-run `portfolio-agent download-data --force`.",
            ]

        actions = self.actions_per_year()
        if actions:
            lines += ["", "  Corporate actions per year"]
            for year, count in actions.items():
                lines.append(f"    {year}   {count}")
        elif not missing_schema:
            lines += [
                "",
                "  No corporate actions recorded anywhere in the store. Across an",
                "  Indian equity universe over several years that is not plausible;",
                "  the source is probably supplying adjusted prices only.",
            ]

        gaps = sorted(self.symbols, key=lambda s: -s.longest_gap)[:worst]
        gaps = [s for s in gaps if s.longest_gap > 1]
        if gaps:
            lines += ["", f"  Longest gaps (top {len(gaps)})"]
            for status in gaps:
                lines.append(
                    f"    {status.symbol:<20} {status.longest_gap:>4} consecutive "
                    f"sessions missing   ({status.coverage:.1%} coverage)"
                )

        if self.unreadable:
            lines += [
                "",
                f"  {len(self.unreadable)} file(s) could not be read: "
                f"{', '.join(self.unreadable[:5])}"
                + (" ..." if len(self.unreadable) > 5 else ""),
            ]

        return "\n".join(lines)


def _count_corporate_actions(frame: pd.DataFrame) -> pd.Series:
    """Dates carrying a dividend or a split, as a boolean series.

    A split of exactly 1.0 is "no split" in the upstream encoding, and counting
    it would report a corporate action on every single bar.
    """
    action = pd.Series(False, index=frame.index)
    if "dividends" in frame.columns:
        action |= frame["dividends"].fillna(0.0).astype(float) != 0.0
    if "stock_splits" in frame.columns:
        splits = frame["stock_splits"].fillna(0.0).astype(float)
        action |= (splits != 0.0) & ((splits - 1.0).abs() > 1e-9)
    return action


def _longest_gap(frame: pd.DataFrame, calendar: pd.DatetimeIndex) -> tuple:
    """(missing session count, longest consecutive run) inside a symbol's span."""
    if len(calendar) == 0 or frame.empty:
        return 0, 0
    expected = calendar[
        (calendar >= frame.index.min()) & (calendar <= frame.index.max())
    ]
    if len(expected) == 0:
        return 0, 0
    missing = expected.difference(frame.index)
    if len(missing) == 0:
        return 0, 0

    positions = np.searchsorted(expected, missing)
    longest = 1
    current = 1
    for previous, this in zip(positions, positions[1:]):
        current = current + 1 if this == previous + 1 else 1
        longest = max(longest, current)
    return int(len(missing)), int(longest)


def collect_status(
    frames: Mapping[str, pd.DataFrame],
    *,
    cache_dir: Optional[str] = None,
    min_sessions: int = DEFAULT_MIN_SESSIONS,
    quorum: float = DEFAULT_CALENDAR_QUORUM,
    unreadable: Optional[Sequence[str]] = None,
) -> StoreStatus:
    """Inventory a set of loaded frames."""
    calendar = infer_trading_calendar(frames, quorum)

    status = StoreStatus(
        cache_dir=cache_dir,
        calendar_sessions=len(calendar),
        calendar_start=calendar.min() if len(calendar) else None,
        calendar_end=calendar.max() if len(calendar) else None,
        min_sessions=min_sessions,
        unreadable=list(unreadable or []),
    )

    actions_by_year: Dict[int, int] = {}

    for symbol in sorted(frames):
        frame = frames[symbol]
        if frame is None or frame.empty:
            status.symbols.append(SymbolStatus(
                symbol=symbol, sessions=0, first_bar=None, last_bar=None,
                expected_sessions=0, missing_sessions=0, longest_gap=0,
                has_adjustment_columns=False, has_raw_columns=False,
                corporate_actions=0, years=0.0,
            ))
            continue

        frame = frame.copy()
        frame.columns = [str(column).lower() for column in frame.columns]

        first, last = frame.index.min(), frame.index.max()
        expected = calendar[(calendar >= first) & (calendar <= last)]
        missing, longest = _longest_gap(frame, calendar)

        actions = _count_corporate_actions(frame)
        for year, count in actions.groupby(actions.index.year).sum().items():
            if count:
                actions_by_year[int(year)] = actions_by_year.get(int(year), 0) + int(count)

        status.symbols.append(SymbolStatus(
            symbol=symbol,
            sessions=len(frame),
            first_bar=first,
            last_bar=last,
            expected_sessions=len(expected),
            missing_sessions=missing,
            longest_gap=longest,
            has_adjustment_columns=all(c in frame.columns for c in ADJUSTMENT_COLUMNS),
            has_raw_columns=all(c in frame.columns for c in RAW_PRICE_COLUMNS),
            corporate_actions=int(actions.sum()),
            years=(last - first).days / 365.25,
        ))

    status._actions_by_year = actions_by_year
    return status


def load_store(
    cache_dir: Optional[Path] = None,
    symbols: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> tuple:
    """Read the parquet cache into memory.

    Args:
        cache_dir: Directory to read. Defaults to the store's own `DATA_DIR`,
            so this looks at the same files everything else does.
        symbols: Specific tickers. None reads everything cached.
        limit: Read at most this many symbols, for an interactive look at a
            large store.

    Returns:
        `(frames, unreadable)` — loaded frames keyed by ticker, and the names
        of files that could not be read. Unreadable files are returned rather
        than raised: one corrupt parquet in 2,400 should be reported, not fatal.
    """
    from portfolio_agent.src.data_store import DATA_DIR, get_cached_tickers, read_cached_bars

    directory = Path(cache_dir) if cache_dir is not None else DATA_DIR
    names = list(symbols) if symbols is not None else get_cached_tickers(directory)
    names = sorted(names)
    if limit is not None:
        names = names[:limit]

    frames: Dict[str, pd.DataFrame] = {}
    unreadable: List[str] = []
    for ticker in names:
        try:
            # Deliberately not `load_ticker_data`: that reads only the default
            # directory, and `DataStore.load_ticker_data` forward-fills missing
            # days — a gap detector reading through a gap filler reports none.
            frame = read_cached_bars(ticker, directory)
        except Exception as exc:  # pragma: no cover - depends on cache state
            logger.debug("Could not read %s: %s", ticker, exc)
            unreadable.append(ticker)
            continue
        if frame is None or frame.empty:
            unreadable.append(ticker)
            continue
        frames[ticker] = frame

    return frames, unreadable
