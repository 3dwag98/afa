"""Invariants a cached bar series has to satisfy, and what to do when it does not.

Why this exists
---------------
Bad bars become bad labels, and a bad label is indistinguishable from a
hard-to-forecast day in every metric downstream. A model trained through a
split that escaped adjustment learns that a 900% one-day return is a thing that
happens; nothing in a loss curve, a Sharpe ratio or an information coefficient
says so. The only place to catch it is at the data.

The five-year window is the cautionary example. `default_history_years: 5` was
a config default nobody had measured, so the sample began *after* the COVID
crash — an entire regime absent from every backtest on the platform — and it
went unnoticed until someone counted rows in the parquet files by hand. A
`data status` command would have printed it on day one.

Two severities, and the difference is the whole design
------------------------------------------------------
**Structural** violations are impossible in correct data: a high below the
open, a non-positive close, a duplicated date. Every one of them is a parser
bug or a corrupt source, so ingest refuses to write the frame and `data
validate` exits non-zero.

**Advisory** findings are *plausible but worth seeing*: an extreme return, a
long gap, a thin history. These are flagged and never dropped. That distinction
is deliberate and is a correction to how the platform behaved: the existing
`max_abs_target` filter silently discards labels beyond a threshold, which
throws away genuine corporate-action days along with the errors and leaves no
record that it did. A 20% move is an Indian circuit limit doing its job. A 900%
move is a split that escaped adjustment. Both survive here; only one gets
flagged, and a human decides.

The calendar is inferred, not imported
--------------------------------------
"Sessions reconcile against the trading calendar" needs a calendar, and NSE
holidays move every year. Rather than depend on a holiday package that will be
stale the moment it is pinned, the calendar is derived from the cross-section:
a date on which a quorum of covered symbols has a bar *is* a session, and a
date almost nobody traded is a holiday. That is self-maintaining, needs no
network, and — unlike a static list — cannot disagree with the data it is
checking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: A violation of one of these means the frame is not a price series. Ingest
#: refuses to write it rather than caching something every downstream feature
#: will silently consume.
STRUCTURAL = "structural"

#: Plausible but worth a look. Never a reason to drop a row.
ADVISORY = "advisory"

#: Daily moves beyond this are flagged. NSE price bands are typically 5%, 10%
#: or 20%, and a name can move further on the day a band is revised or on a
#: post-listing session — so 25% is above anything the bands permit routinely
#: and far below what an unadjusted split produces.
DEFAULT_EXTREME_RETURN = 0.25

#: Above this, a "return" is arithmetically almost certainly a corporate action
#: that was never adjusted: a 1:5 split shows up as -80%, a 5:1 bonus as +400%.
SPLIT_SCALE_RETURN = 0.60

#: Share of covered symbols that must have a bar for a date to count as a
#: trading session. Well above any plausible level of correlated outage and
#: well below the coverage a real holiday would produce (near zero).
DEFAULT_CALENDAR_QUORUM = 0.60

#: Sessions a symbol needs before it is usable for training at all: one year of
#: history, which is what the longest default feature lookback requires.
DEFAULT_MIN_SESSIONS = 252

OHLC = ("open", "high", "low", "close")


@dataclass(frozen=True)
class Violation:
    """One failed check on one symbol.

    `examples` carries the first few offending dates rather than a count alone.
    A report that says "37 bars have a high below the close" sends someone to
    write a script; one that names three dates sends them to look at those
    dates, which is where the answer is.
    """

    check: str
    severity: str
    symbol: str
    count: int
    detail: str
    examples: List[str] = field(default_factory=list)

    @property
    def structural(self) -> bool:
        return self.severity == STRUCTURAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check": self.check,
            "severity": self.severity,
            "symbol": self.symbol,
            "count": self.count,
            "detail": self.detail,
            "examples": list(self.examples),
        }

    def __str__(self) -> str:
        where = f" (first: {', '.join(self.examples)})" if self.examples else ""
        return f"[{self.severity}] {self.symbol}: {self.check} — {self.detail}{where}"


@dataclass
class ValidationReport:
    """Every violation found across a validation run.

    `ok` is deliberately keyed on structural violations only. An advisory
    finding must not fail a build — a store with three genuine 30% circuit days
    in it is a correct store — or the gate becomes something people disable.
    """

    violations: List[Violation] = field(default_factory=list)
    symbols_checked: int = 0
    rows_checked: int = 0

    @property
    def structural_violations(self) -> List[Violation]:
        return [v for v in self.violations if v.structural]

    @property
    def advisories(self) -> List[Violation]:
        return [v for v in self.violations if not v.structural]

    @property
    def ok(self) -> bool:
        """Whether the store is fit to use. Advisories do not fail the gate."""
        return not self.structural_violations

    @property
    def exit_code(self) -> int:
        return 0 if self.ok else 1

    def add(self, violation: Optional[Violation]) -> None:
        if violation is not None:
            self.violations.append(violation)

    def extend(self, violations: Iterable[Optional[Violation]]) -> None:
        for violation in violations:
            self.add(violation)

    def by_check(self) -> Dict[str, int]:
        """Violation counts per check, for a summary line."""
        counts: Dict[str, int] = {}
        for violation in self.violations:
            counts[violation.check] = counts.get(violation.check, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "symbols_checked": self.symbols_checked,
            "rows_checked": self.rows_checked,
            "structural": len(self.structural_violations),
            "advisory": len(self.advisories),
            "by_check": self.by_check(),
            "violations": [v.to_dict() for v in self.violations],
        }

    def render(self, examples_per_check: int = 4) -> str:
        """A report that says what is wrong and what it means.

        Grouped by check rather than listed flat. On a 2,400-symbol store a
        flat list is thousands of lines in which "118 symbols are missing the
        same session" — one partial trading day — is indistinguishable from 118
        separate problems. The grouping is what makes the shape visible.
        """
        lines = [
            f"Checked {self.symbols_checked} symbol(s), {self.rows_checked:,} bar(s)",
        ]
        if not self.violations:
            lines.append("  No violations.")
            return "\n".join(lines)

        structural = self.structural_violations
        advisories = self.advisories
        lines.append(f"  {len(structural)} structural, {len(advisories)} advisory")

        if structural:
            lines += [
                "",
                "  Structural — these bars are not a valid price series, and "
                "ingest would refuse them:",
            ]
            lines += self._grouped(structural, examples_per_check)
        if advisories:
            lines += ["", "  Advisory — plausible, but worth a look:"]
            lines += self._grouped(advisories, examples_per_check)

        lines += [
            "",
            f"  Verdict: {'PASS' if self.ok else 'FAIL'}"
            f"  ({len(structural)} structural violation(s); advisories do not "
            f"fail the gate)",
        ]
        return "\n".join(lines)

    def _grouped(self, violations: List[Violation], examples: int) -> List[str]:
        """Violations bucketed by check name, worst-affected check first."""
        by_check: Dict[str, List[Violation]] = {}
        for violation in violations:
            by_check.setdefault(violation.check, []).append(violation)

        lines: List[str] = []
        for check, found in sorted(by_check.items(), key=lambda kv: -len(kv[1])):
            bars = sum(v.count for v in found)
            lines.append(
                f"    {check}   {len(found)} symbol(s), {bars} bar(s)"
            )
            for violation in found[:examples]:
                where = (
                    f"  first {', '.join(violation.examples)}"
                    if violation.examples else ""
                )
                lines.append(f"        {violation.symbol}: {violation.detail}{where}")
            if len(found) > examples:
                lines.append(f"        ... and {len(found) - examples} more symbol(s)")
        return lines


def _examples(index: pd.Index, mask: np.ndarray, limit: int = 3) -> List[str]:
    """The first few offending dates, as ISO strings."""
    hits = index[mask][:limit]
    return [
        d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in hits
    ]


# --------------------------------------------------------------------------
# Per-symbol checks
# --------------------------------------------------------------------------


def check_ohlc_ordering(frame: pd.DataFrame, symbol: str) -> List[Violation]:
    """`high >= max(open, close)` and `low <= min(open, close)`.

    Structural because there is no market condition that produces a high below
    the price something traded at. A violation is a column swapped at the
    parser, or a source that has mixed adjusted and unadjusted legs in one row
    — which is exactly what happens when only some columns get back-adjusted.
    """
    if not all(column in frame.columns for column in OHLC):
        return []

    violations: List[Violation] = []
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    body_high = np.maximum(
        frame["open"].to_numpy(dtype=float), frame["close"].to_numpy(dtype=float)
    )
    body_low = np.minimum(
        frame["open"].to_numpy(dtype=float), frame["close"].to_numpy(dtype=float)
    )

    # A tiny relative tolerance: back-adjustment multiplies every leg by a
    # float factor, and an exact comparison turns rounding at the 15th digit
    # into thousands of false structural failures.
    tolerance = 1e-9 * np.maximum(np.abs(body_high), 1.0)

    above = high < body_high - tolerance
    if above.any():
        violations.append(Violation(
            check="high_below_body", severity=STRUCTURAL, symbol=symbol,
            count=int(above.sum()),
            detail=f"{int(above.sum())} bar(s) have high < max(open, close)",
            examples=_examples(frame.index, above),
        ))

    below = low > body_low + tolerance
    if below.any():
        violations.append(Violation(
            check="low_above_body", severity=STRUCTURAL, symbol=symbol,
            count=int(below.sum()),
            detail=f"{int(below.sum())} bar(s) have low > min(open, close)",
            examples=_examples(frame.index, below),
        ))

    crossed = high < low - tolerance
    if crossed.any():
        violations.append(Violation(
            check="high_below_low", severity=STRUCTURAL, symbol=symbol,
            count=int(crossed.sum()),
            detail=f"{int(crossed.sum())} bar(s) have high < low",
            examples=_examples(frame.index, crossed),
        ))

    return violations


def check_price_positivity(frame: pd.DataFrame, symbol: str) -> List[Violation]:
    """`close > 0` and `volume >= 0`.

    A zero close poisons every ratio feature at once — `return_1d` divides by
    it, `bollinger_pct_b` divides by a band built from it — and the resulting
    infinities propagate into a NaN loss several hundred training steps later,
    far from the cause.
    """
    violations: List[Violation] = []

    for column in OHLC:
        if column not in frame.columns:
            continue
        values = frame[column].to_numpy(dtype=float)
        bad = ~(values > 0.0)
        if bad.any():
            violations.append(Violation(
                check="non_positive_price", severity=STRUCTURAL, symbol=symbol,
                count=int(bad.sum()),
                detail=f"{int(bad.sum())} bar(s) have {column} <= 0 or missing",
                examples=_examples(frame.index, bad),
            ))

    if "volume" in frame.columns:
        volume = frame["volume"].to_numpy(dtype=float)
        negative = volume < 0.0
        if negative.any():
            violations.append(Violation(
                check="negative_volume", severity=STRUCTURAL, symbol=symbol,
                count=int(negative.sum()),
                detail=f"{int(negative.sum())} bar(s) have negative volume",
                examples=_examples(frame.index, negative),
            ))

    return violations


def check_duplicate_dates(frame: pd.DataFrame, symbol: str) -> List[Violation]:
    """One bar per symbol per session.

    Structural because a duplicated date silently doubles a day's weight in
    every fit and makes a rolling window span fewer real sessions than its
    length claims. Re-exports are the usual source: a corrected bar appended
    rather than replacing the original.
    """
    if not isinstance(frame.index, pd.DatetimeIndex):
        return []
    duplicated = np.asarray(frame.index.duplicated(keep=False))
    if not duplicated.any():
        return []
    return [Violation(
        check="duplicate_dates", severity=STRUCTURAL, symbol=symbol,
        count=int(frame.index.duplicated().sum()),
        detail=f"{int(frame.index.duplicated().sum())} duplicated date(s)",
        examples=_examples(frame.index, duplicated),
    )]


def check_monotonic_index(frame: pd.DataFrame, symbol: str) -> List[Violation]:
    """Bars in date order.

    Every rolling feature in the platform assumes it. An out-of-order frame
    produces values computed over a window that includes the future, which is
    the one failure mode nothing downstream can detect.
    """
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.is_monotonic_increasing:
        return []
    return [Violation(
        check="unsorted_index", severity=STRUCTURAL, symbol=symbol, count=1,
        detail="bars are not in ascending date order, so every rolling feature "
               "would be computed over a window containing later bars",
    )]


def check_extreme_returns(
    frame: pd.DataFrame,
    symbol: str,
    threshold: float = DEFAULT_EXTREME_RETURN,
    split_threshold: float = SPLIT_SCALE_RETURN,
) -> List[Violation]:
    """Flag implausible one-day moves. Never drop them.

    This is the check the spec calls out as mattering most, and the reason is
    the behaviour it replaces. `max_abs_target` *discards* labels beyond a
    bound, which removes genuine corporate-action days along with the errors
    and leaves nothing behind saying it happened. Here both survive: a move
    past the band limit is an advisory, and a move at split scale gets a
    separate check name because it has a specific, checkable cause.
    """
    if "close" not in frame.columns or len(frame) < 2:
        return []

    close = frame["close"].astype(float)
    returns = (close / close.shift(1) - 1.0).to_numpy()
    finite = np.isfinite(returns)

    violations: List[Violation] = []

    split_scale = finite & (np.abs(returns) >= split_threshold)
    if split_scale.any():
        worst = float(np.nanmax(np.abs(returns[split_scale])))
        violations.append(Violation(
            check="unadjusted_split_suspected", severity=ADVISORY, symbol=symbol,
            count=int(split_scale.sum()),
            detail=f"{int(split_scale.sum())} move(s) at or beyond "
                   f"{split_threshold:.0%} (worst {worst:+.1%}) — no price band "
                   f"permits this, so a split or bonus probably escaped adjustment",
            examples=_examples(frame.index, split_scale),
        ))

    extreme = finite & (np.abs(returns) >= threshold) & ~split_scale
    if extreme.any():
        worst = float(np.nanmax(np.abs(returns[extreme])))
        violations.append(Violation(
            check="extreme_return", severity=ADVISORY, symbol=symbol,
            count=int(extreme.sum()),
            detail=f"{int(extreme.sum())} move(s) beyond {threshold:.0%} "
                   f"(worst {worst:+.1%}) — above the usual NSE bands but within "
                   f"what a real session can do; kept, not dropped",
            examples=_examples(frame.index, extreme),
        ))

    return violations


def check_adjustment_factor(frame: pd.DataFrame, symbol: str) -> List[Violation]:
    """The adjustment factor may only move on a recorded corporate action.

    Needs the wider schema T01 introduced (`adj_factor`, `dividends`,
    `stock_splits`); a cache written before it simply skips this check rather
    than reporting a false pass, and `data status` reports how many symbols are
    in that state.

    A factor that steps on a day with no dividend and no split means the
    upstream source silently re-adjusted history. That is invisible in the
    prices — every bar looks internally consistent — but it means a series
    cached today and the same series cached last month describe different
    numbers, and any backtest comparing them is comparing two datasets.
    """
    if "adj_factor" not in frame.columns or len(frame) < 2:
        return []

    factor = frame["adj_factor"].astype(float)
    # `.to_numpy()` on a boolean Series can hand back a read-only view, and the
    # first element has to be cleared — `diff()` makes it NaN, which compares
    # False, but only by accident of the comparison rather than by intent.
    moved = np.array(factor.diff().abs() > 1e-9, dtype=bool, copy=True)
    moved[0] = False
    if not moved.any():
        return []

    action = np.zeros(len(frame), dtype=bool)
    for column in ("dividends", "stock_splits"):
        if column in frame.columns:
            values = frame[column].fillna(0.0).to_numpy(dtype=float)
            # A split is recorded as a ratio, so "no split" is 0 *or* 1.
            if column == "stock_splits":
                action |= (values != 0.0) & (np.abs(values - 1.0) > 1e-9)
            else:
                action |= values != 0.0

    unexplained = moved & ~action
    if not unexplained.any():
        return []

    return [Violation(
        check="unexplained_adjustment", severity=ADVISORY, symbol=symbol,
        count=int(unexplained.sum()),
        detail=f"{int(unexplained.sum())} date(s) where adj_factor changed with no "
               f"recorded dividend or split — the source may have re-adjusted "
               f"history, which makes this series incomparable with an earlier pull",
        examples=_examples(frame.index, unexplained),
    )]


def check_history_length(
    frame: pd.DataFrame, symbol: str, min_sessions: int = DEFAULT_MIN_SESSIONS
) -> List[Violation]:
    """Enough sessions for the longest default lookback to have filled."""
    if len(frame) >= min_sessions:
        return []
    return [Violation(
        check="short_history", severity=ADVISORY, symbol=symbol, count=len(frame),
        detail=f"{len(frame)} session(s), below the {min_sessions} a "
               f"nine-month formation feature needs before it stops being NaN",
    )]


def validate_frame(
    frame: pd.DataFrame,
    symbol: str,
    *,
    calendar: Optional[pd.DatetimeIndex] = None,
    extreme_return: float = DEFAULT_EXTREME_RETURN,
    min_sessions: int = DEFAULT_MIN_SESSIONS,
    structural_only: bool = False,
) -> List[Violation]:
    """Run every check against one symbol's bars.

    Args:
        frame: Date-indexed OHLCV, with the T01 adjustment columns when present.
        symbol: Name for the report.
        calendar: Inferred trading sessions. When given, missing sessions inside
            the symbol's own span are reported; when None that check is skipped
            rather than guessed at.
        extreme_return: Advisory threshold on a one-day move.
        min_sessions: Advisory threshold on history length.
        structural_only: Run only the checks that would refuse a write. This is
            what ingest uses: it is about to write one symbol and has no
            cross-section to infer a calendar from, and an advisory it cannot
            act on is noise at exactly the moment someone is watching a
            progress bar.

    Returns:
        Every violation found, structural first.
    """
    if frame is None or frame.empty:
        return [Violation(
            check="empty", severity=STRUCTURAL, symbol=symbol, count=0,
            detail="no bars",
        )]

    frame = frame.copy()
    frame.columns = [str(column).lower() for column in frame.columns]

    violations: List[Violation] = []
    violations += check_duplicate_dates(frame, symbol)
    violations += check_monotonic_index(frame, symbol)
    violations += check_ohlc_ordering(frame, symbol)
    violations += check_price_positivity(frame, symbol)

    if not structural_only:
        violations += check_extreme_returns(frame, symbol, extreme_return)
        violations += check_adjustment_factor(frame, symbol)
        violations += check_history_length(frame, symbol, min_sessions)
        if calendar is not None:
            violations += check_calendar_coverage(frame, symbol, calendar)

    return sorted(violations, key=lambda v: (not v.structural, v.check))


# --------------------------------------------------------------------------
# The inferred trading calendar
# --------------------------------------------------------------------------


def infer_trading_calendar(
    frames: Mapping[str, pd.DataFrame], quorum: float = DEFAULT_CALENDAR_QUORUM
) -> pd.DatetimeIndex:
    """Sessions the market actually traded, derived from the cross-section.

    A date on which a quorum of *covered* symbols has a bar is a session; a
    date almost nobody traded is a holiday. "Covered" is the load-bearing word:
    a symbol that listed in 2023 has no opinion about 2021, so it is excluded
    from the denominator on dates outside its own span. Without that, every
    session before the newest listing would fall below quorum and the whole
    early history would be classified as holiday.

    This is preferred to a holiday package on purpose. NSE's holiday list
    changes annually and includes ad-hoc closures; a pinned list is wrong the
    year after it is pinned, and it can disagree with the data it is checking.
    An inferred calendar cannot.

    Args:
        frames: Date-indexed bars keyed by symbol.
        quorum: Share of covered symbols that must have a bar.

    Returns:
        Sorted `DatetimeIndex` of inferred sessions. Empty when there is not
        enough cross-section to infer one — with a single symbol every date it
        has is trivially a session, which is not a check.
    """
    usable = {
        symbol: frame for symbol, frame in frames.items()
        if frame is not None and not frame.empty
        and isinstance(frame.index, pd.DatetimeIndex)
    }
    if len(usable) < 2:
        return pd.DatetimeIndex([])

    spans = {
        symbol: (frame.index.min(), frame.index.max())
        for symbol, frame in usable.items()
    }
    all_dates = pd.DatetimeIndex(
        sorted({date for frame in usable.values() for date in frame.index})
    )

    present = np.zeros(len(all_dates), dtype=int)
    covered = np.zeros(len(all_dates), dtype=int)
    position = {date: i for i, date in enumerate(all_dates)}

    for symbol, frame in usable.items():
        start, end = spans[symbol]
        in_span = (all_dates >= start) & (all_dates <= end)
        covered += in_span.astype(int)
        for date in frame.index:
            present[position[date]] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(covered > 0, present / np.maximum(covered, 1), 0.0)
    return all_dates[share >= quorum]


def check_calendar_coverage(
    frame: pd.DataFrame, symbol: str, calendar: pd.DatetimeIndex
) -> List[Violation]:
    """Sessions missing from a symbol's own span.

    Bounded to the span deliberately: a symbol that listed in 2023 is not
    "missing" 2021, and reporting it as a gap would bury the real finding under
    one entry per young name. What is left is a symbol that was trading, went
    quiet for sessions the rest of the market had, and came back — a suspension,
    or a hole in the download.
    """
    if len(calendar) == 0 or frame.empty:
        return []

    start, end = frame.index.min(), frame.index.max()
    expected = calendar[(calendar >= start) & (calendar <= end)]
    if len(expected) == 0:
        return []

    missing = expected.difference(frame.index)
    if len(missing) == 0:
        return []

    # Longest consecutive run of missing sessions, measured in *sessions* and
    # not calendar days — a four-day Diwali break is not a gap.
    positions = np.searchsorted(expected, missing)
    longest = 1
    current = 1
    for previous, this in zip(positions, positions[1:]):
        current = current + 1 if this == previous + 1 else 1
        longest = max(longest, current)

    coverage = 1.0 - len(missing) / len(expected)
    return [Violation(
        check="missing_sessions", severity=ADVISORY, symbol=symbol,
        count=int(len(missing)),
        detail=f"{len(missing)} session(s) missing inside its own span "
               f"({coverage:.1%} coverage, longest run {longest})",
        examples=[d.strftime("%Y-%m-%d") for d in missing[:3]],
    )]


# --------------------------------------------------------------------------
# Whole-store validation
# --------------------------------------------------------------------------


def validate_store(
    frames: Mapping[str, pd.DataFrame],
    *,
    extreme_return: float = DEFAULT_EXTREME_RETURN,
    min_sessions: int = DEFAULT_MIN_SESSIONS,
    quorum: float = DEFAULT_CALENDAR_QUORUM,
    use_calendar: bool = True,
) -> ValidationReport:
    """Validate every symbol, sharing one inferred calendar across them."""
    calendar = infer_trading_calendar(frames, quorum) if use_calendar else None
    if calendar is not None and len(calendar) == 0:
        calendar = None

    report = ValidationReport()
    for symbol in sorted(frames):
        frame = frames[symbol]
        report.symbols_checked += 1
        report.rows_checked += 0 if frame is None else len(frame)
        report.extend(validate_frame(
            frame, symbol, calendar=calendar,
            extreme_return=extreme_return, min_sessions=min_sessions,
        ))
    return report


class IngestRejected(ValueError):
    """A frame failed a structural check and was not written.

    Its own type so ingest can distinguish "this symbol is corrupt, skip it and
    keep going" from a bug in the ingest code itself, which should stop the run.
    """

    def __init__(self, symbol: str, violations: Sequence[Violation]):
        self.symbol = symbol
        self.violations = list(violations)
        detail = "; ".join(v.detail for v in violations)
        super().__init__(f"{symbol}: refusing to cache — {detail}")


def assert_writable(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Return `frame` if it is structurally sound; raise `IngestRejected` if not.

    The gate ingest calls before writing a parquet file. Structural checks only:
    caching a frame whose high is below its close is worse than not caching it,
    because everything downstream will consume it silently — but a symbol with
    one genuine 30% circuit day is a symbol worth having.
    """
    violations = validate_frame(frame, symbol, structural_only=True)
    if violations:
        raise IngestRejected(symbol, violations)
    return frame
