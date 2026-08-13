"""Who was actually in the index on each date.

Every result this platform has produced so far was computed on the names that
survived to be downloaded. That is survivorship bias, and for Indian indices it
is not a rounding error: the 2026 study of Nifty constituents puts it at
**4.94 percentage points of annual return overstatement** against **82.5%
membership turnover** over the sample. Both strategies' neutralized rank IC is
smaller than that bias. The largest number in the pipeline is one nobody has
been measuring.

The shape of the problem
------------------------
A stock that was in the Nifty 500 in 2012 and was delisted in 2015 has no
parquet file today, so it never enters a cross-section, so the 2012 deciles are
formed from a universe that excludes exactly the names that went on to fail.
The bias is one-directional and it compounds: every date in the sample is
ranked against a forward-looking survivor set.

What this module does
---------------------
It holds the membership intervals and answers one question — `members_on(date)`
— then lets the panel builder intersect each date's cross-section with the
answer. It does **not** acquire the data; see `docs/OBTAINING_DATA.md`. What it
does guarantee is that a run without membership data *says so*, in the same
printed notes that carry the sector-map caveat from T05, rather than quietly
reporting a survivor-set number as if it were a universe number.

The interval format
-------------------
CSV, because that is the shape this data arrives in and because a human needs
to be able to fix a row by hand:

    symbol,index_name,start_date,end_date
    RELIANCE.NS,NIFTY50,2000-01-01,
    YESBANK.NS,NIFTY50,2017-03-27,2020-03-19

`end_date` empty means "still a member". Intervals are inclusive of both ends,
which is the convention the NSE change announcements use. A symbol may appear
more than once — names leave an index and come back, and treating a re-entry as
a correction rather than a second interval would silently extend membership
across the gap it was out.
"""

from __future__ import annotations

import csv
import logging
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

#: Columns a membership file must carry.
REQUIRED_COLUMNS = ("symbol", "index_name", "start_date", "end_date")

#: Default home for membership files, alongside the universe snapshots.
DEFAULT_MEMBERSHIP_DIR = Path("universe")

#: Printed on any evaluation that ran without membership data. Phrased as a
#: statement about the result rather than a to-do, because it *is* a property
#: of the number sitting next to it.
SURVIVORSHIP_NOTE = (
    "No point-in-time index membership was supplied, so every date was ranked "
    "against the names that survived to be downloaded. Indian index membership "
    "turns over heavily — a published study puts it at 82.5% over its sample, "
    "with roughly 4.94pp of annual return overstatement — so this result is "
    "biased upward by an amount plausibly larger than the alpha it reports. "
    "See docs/OBTAINING_DATA.md for how to supply one."
)


@dataclass(frozen=True)
class MembershipInterval:
    """One symbol's stay in one index, inclusive of both ends."""

    symbol: str
    index_name: str
    start: pd.Timestamp
    end: Optional[pd.Timestamp]

    def covers(self, date: pd.Timestamp) -> bool:
        if date < self.start:
            return False
        return self.end is None or date <= self.end


@dataclass
class IndexMembership:
    """Point-in-time constituents, queryable by date.

    Attributes:
        intervals: Every membership stay, in file order.
        index_name: The index these describe, or None when the file mixes
            several and the caller did not narrow it.
        source: Where it was loaded from, carried into run manifests so a
            result can name the membership data it used.
    """

    intervals: List[MembershipInterval]
    index_name: Optional[str] = None
    source: Optional[str] = None
    _by_symbol: Dict[str, List[MembershipInterval]] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        by_symbol: Dict[str, List[MembershipInterval]] = {}
        for interval in self.intervals:
            by_symbol.setdefault(interval.symbol, []).append(interval)
        for stays in by_symbol.values():
            stays.sort(key=lambda i: i.start)
        self._by_symbol = by_symbol

    def __len__(self) -> int:
        return len(self.intervals)

    @property
    def symbols(self) -> List[str]:
        """Every symbol that was ever a member, sorted."""
        return sorted(self._by_symbol)

    def members_on(self, date: Any) -> Set[str]:
        """Symbols in the index on `date`.

        Args:
            date: Anything `pd.Timestamp` accepts.

        Returns:
            The constituent set. Empty when the date precedes every interval,
            which is a real answer and not an error — it usually means the
            membership file starts after the evaluation window does.
        """
        moment = pd.Timestamp(date).normalize()
        return {
            symbol
            for symbol, stays in self._by_symbol.items()
            if any(stay.covers(moment) for stay in stays)
        }

    def was_member(self, symbol: str, date: Any) -> bool:
        moment = pd.Timestamp(date).normalize()
        return any(stay.covers(moment) for stay in self._by_symbol.get(symbol, ()))

    def coverage(self) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Earliest start and latest end, with None for an open-ended stay.

        A caller evaluating outside this range is asking the file a question it
        cannot answer, which is worth a warning rather than an empty filter.
        """
        if not self.intervals:
            return None, None
        start = min(i.start for i in self.intervals)
        if any(i.end is None for i in self.intervals):
            return start, None
        return start, max(i.end for i in self.intervals if i.end is not None)

    # ----------------------------------------------------------------------
    # Loading
    # ----------------------------------------------------------------------

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Mapping[str, Any]],
        index_name: Optional[str] = None,
        source: Optional[str] = None,
    ) -> "IndexMembership":
        """Build from parsed rows, validating as it goes.

        Raises:
            ValueError: On a missing column, an unparseable date, an end before
                its start, or two overlapping stays for one symbol. All four
                are silent corruptions of a constituent set — an overlapping
                pair in particular would just look like a longer membership.
        """
        intervals: List[MembershipInterval] = []
        for number, row in enumerate(rows, start=2):  # 1 is the header
            missing = [c for c in REQUIRED_COLUMNS if c not in row]
            if missing:
                raise ValueError(f"row {number}: missing column(s) {missing}")

            symbol = str(row["symbol"]).strip()
            name = str(row["index_name"]).strip()
            if not symbol:
                raise ValueError(f"row {number}: empty symbol")
            if index_name is not None and name != index_name:
                continue

            start = _parse_date(row["start_date"], number, "start_date")
            end = _parse_optional_date(row["end_date"], number, "end_date")
            if start is None:
                raise ValueError(f"row {number}: start_date is required")
            if end is not None and end < start:
                raise ValueError(
                    f"row {number}: end_date {end.date()} precedes start_date "
                    f"{start.date()}"
                )
            intervals.append(MembershipInterval(symbol, name, start, end))

        _reject_overlaps(intervals)
        return cls(intervals=intervals, index_name=index_name, source=source)

    @classmethod
    def load(
        cls, path: Any, index_name: Optional[str] = None
    ) -> "IndexMembership":
        """Read a membership CSV.

        Raises:
            FileNotFoundError: If the path does not exist. Deliberately not a
                silent empty membership: a typo'd path that quietly disabled
                the filter would restore the survivorship bias while the run
                claimed to have corrected it.
        """
        location = Path(path)
        if not location.exists():
            raise FileNotFoundError(
                f"no membership file at {location}. A missing file is an error "
                "rather than an empty filter, because silently skipping the "
                "point-in-time filter would restore the survivorship bias in a "
                "run that says it corrected for it."
            )
        with location.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"{location} has a header but no rows")
        return cls.from_rows(rows, index_name=index_name, source=str(location))

    def to_rows(self) -> List[Dict[str, str]]:
        """Round-trip form, for writing a file back out."""
        return [
            {
                "symbol": i.symbol,
                "index_name": i.index_name,
                "start_date": i.start.date().isoformat(),
                "end_date": "" if i.end is None else i.end.date().isoformat(),
            }
            for i in self.intervals
        ]

    def save(self, path: Any) -> Path:
        location = Path(path)
        location.parent.mkdir(parents=True, exist_ok=True)
        with location.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_COLUMNS))
            writer.writeheader()
            writer.writerows(self.to_rows())
        return location


# --------------------------------------------------------------------------
# Applying it to a panel
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MembershipFilterResult:
    """What the point-in-time filter removed, so the cost is visible.

    `removed_share` is the headline: it is the fraction of the survivor-set
    panel that was never eligible, and therefore a direct measure of how much
    of the original result was built on hindsight.
    """

    panel: pd.DataFrame = field(repr=False)
    n_before: int
    n_after: int
    dates_before: int
    dates_after: int
    symbols_dropped: int
    uncovered_dates: int

    @property
    def removed(self) -> int:
        return self.n_before - self.n_after

    @property
    def removed_share(self) -> float:
        return self.removed / self.n_before if self.n_before else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "membership_rows_before": self.n_before,
            "membership_rows_after": self.n_after,
            "membership_rows_removed": self.removed,
            "membership_removed_share": self.removed_share,
            "membership_dates_before": self.dates_before,
            "membership_dates_after": self.dates_after,
            "membership_symbols_dropped": self.symbols_dropped,
            "membership_uncovered_dates": self.uncovered_dates,
        }

    def note(self) -> str:
        if self.uncovered_dates:
            return (
                f"Point-in-time membership removed {self.removed} of "
                f"{self.n_before} observations ({self.removed_share:.1%}), but "
                f"{self.uncovered_dates} evaluation date(s) fall outside the "
                "membership file's coverage and were left unfiltered — those "
                "dates are still survivorship-biased."
            )
        return (
            f"Point-in-time membership removed {self.removed} of {self.n_before} "
            f"observations ({self.removed_share:.1%}) that were not index "
            f"constituents on their own date, across "
            f"{self.symbols_dropped} symbol(s)."
        )


def apply_membership(
    panel: pd.DataFrame, membership: IndexMembership
) -> MembershipFilterResult:
    """Keep only the rows whose symbol was a constituent on that row's date.

    Dates outside the membership file's coverage are left **unfiltered** rather
    than emptied. Emptying them would silently shorten the evaluation window,
    and a shorter window that looks like a clean one is worse than a longer
    window that admits which part of it is uncorrected — so the count travels
    into the result and into the printed note.
    """
    if panel.empty:
        return MembershipFilterResult(panel, 0, 0, 0, 0, 0, 0)

    dates = pd.to_datetime(panel["date"])
    first, last = membership.coverage()

    keep = pd.Series(True, index=panel.index)
    uncovered: Set[Any] = set()

    for date, positions in dates.groupby(dates).groups.items():
        moment = pd.Timestamp(date).normalize()
        outside = (first is not None and moment < first) or (
            last is not None and moment > last
        )
        if outside:
            uncovered.add(moment)
            continue
        members = membership.members_on(moment)
        keep.loc[positions] = panel.loc[positions, "symbol"].isin(members)

    filtered = panel[keep]
    return MembershipFilterResult(
        panel=filtered,
        n_before=int(len(panel)),
        n_after=int(len(filtered)),
        dates_before=int(dates.nunique()),
        dates_after=int(pd.to_datetime(filtered["date"]).nunique()) if len(filtered) else 0,
        symbols_dropped=int(
            panel["symbol"].nunique() - (filtered["symbol"].nunique() if len(filtered) else 0)
        ),
        uncovered_dates=len(uncovered),
    )


def load_membership(
    path: Optional[Any], index_name: Optional[str] = None
) -> Optional[IndexMembership]:
    """`IndexMembership.load` for a path that may be None.

    None means "none was asked for" and yields None, which the caller turns
    into `SURVIVORSHIP_NOTE`. A path that was given and does not resolve still
    raises.
    """
    return None if path is None else IndexMembership.load(path, index_name)


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------


def _parse_date(value: Any, row: int, column: str) -> Optional[pd.Timestamp]:
    text = str(value).strip() if value is not None else ""
    if not text:
        return None
    try:
        return pd.Timestamp(text).normalize()
    except (ValueError, TypeError) as error:
        raise ValueError(f"row {row}: {column} {text!r} is not a date") from error


def _parse_optional_date(value: Any, row: int, column: str) -> Optional[pd.Timestamp]:
    return _parse_date(value, row, column)


def _reject_overlaps(intervals: Sequence[MembershipInterval]) -> None:
    """Two stays for one symbol that overlap are a corrupted file.

    They read as a single longer membership, which is exactly the error this
    module exists to prevent — a name would appear eligible through a period it
    had actually left.
    """
    by_symbol: Dict[str, List[MembershipInterval]] = {}
    for interval in intervals:
        by_symbol.setdefault(interval.symbol, []).append(interval)

    for symbol, stays in by_symbol.items():
        stays = sorted(stays, key=lambda i: i.start)
        for earlier, later in zip(stays, stays[1:]):
            if earlier.end is None or earlier.end >= later.start:
                raise ValueError(
                    f"{symbol}: membership {earlier.start.date()}.."
                    f"{'open' if earlier.end is None else earlier.end.date()} "
                    f"overlaps {later.start.date()}.. — two overlapping stays "
                    "read as one longer membership, which would make the symbol "
                    "look eligible through a period it had left"
                )
