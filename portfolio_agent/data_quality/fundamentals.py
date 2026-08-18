"""Point-in-time fundamentals: what was *knowable* on each date, not what was true.

`docs/QUANT_RESEARCH.md` §8 and §9 both stop at the same sentence — the data
isn't ingested — and between them they cover size, value, profitability,
investment and quality. That is most of the cross-section of published equity
research, and none of it is expressible from OHLCV.

The bias this module exists to prevent
--------------------------------------
A company's results for the quarter ending **31 March** are published in late
May. A backtest that uses them from 31 March has read six to eight weeks into
the future, on every stock, every quarter, forever.

This is the single most common way a fundamentals backtest lies, and it is
invisible in the output: the numbers are real, the dates are real, and the
strategy simply knew them early. It does not look like a bug. It looks like
alpha.

So every fact here carries **two** dates and the store is keyed on the second:

- `fiscal_date` — the period the number describes.
- `report_date` — when it was published.

`as_of(date)` returns only facts with `report_date <= date`. A schema that
carried the fiscal date alone could not answer the question at all, which is
why the report date is required rather than optional.

The Indian specifics
--------------------
SEBI LODR Regulation 33 requires quarterly results within **45 days** of the
quarter end, and annual results within **60 days**. So a genuine report lag is
roughly 30-60 days, a lag under ~15 days is implausible enough to be worth
flagging, and a lag over ~120 days means either a late filer or a fiscal date
that has been mistaken for a report date. `validate_fundamentals` reports all
three, because a file whose report dates were reconstructed by adding a
constant to the fiscal date is *worse* than no file: it looks point-in-time
and is not.

What this module does not do
----------------------------
It does not acquire the data. See `docs/OBTAINING_DATA.md`. What it guarantees
is that a run **without** fundamentals says so — `FUNDAMENTALS_NOTE`, in the
same printed notes that carry T15's survivorship caveat and T05's sector-map
caveat — rather than quietly reporting a price-only result as though the
factor exposures had been controlled for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Columns every fundamentals file must carry. `report_date` is required, not
#: optional: a file without it cannot answer "what was knowable on date D", and
#: accepting one would mean silently falling back to the fiscal date — which is
#: precisely the look-ahead this module exists to prevent.
REQUIRED_COLUMNS = ("symbol", "fiscal_date", "report_date")

#: Balance-sheet and income-statement fields the derived characteristics read.
#: All optional: a file with only `total_equity` still yields book-to-price,
#: and demanding the full set would make a partial dataset useless.
FACT_COLUMNS = (
    "total_assets",
    "total_equity",
    "revenue",
    "cost_of_goods_sold",
    "operating_income",
    "net_income",
    "total_debt",
    "cash_flow_operating",
    "shares_outstanding",
)

#: Default home for fundamentals files, alongside the universe snapshots and
#: membership intervals.
DEFAULT_FUNDAMENTALS_DIR = Path("universe")

#: SEBI LODR Regulation 33 deadlines, in days after the period end.
QUARTERLY_DEADLINE_DAYS = 45
ANNUAL_DEADLINE_DAYS = 60

#: Below this, a reported lag is implausibly short for an audited filing and is
#: more likely a fiscal date copied into the report-date column.
IMPLAUSIBLY_FAST_DAYS = 15

#: Above this, either a genuinely late filer or — far more often — a fiscal
#: date mistaken for a report date one row at a time.
IMPLAUSIBLY_SLOW_DAYS = 120

#: Printed on any evaluation that ran without fundamentals. Phrased as a
#: statement about the result rather than a to-do, because it *is* a property
#: of the number sitting next to it.
FUNDAMENTALS_NOTE = (
    "No fundamentals data was supplied, so this result controls for no "
    "accounting characteristic. Size, value, profitability, investment and "
    "quality exposures are uncontrolled, and any of them could be producing "
    "the ranking measured here. See docs/OBTAINING_DATA.md."
)


@dataclass(frozen=True)
class FundamentalsValidation:
    """What a fundamentals file looks like, before anyone trusts it."""

    n_rows: int
    n_symbols: int
    n_facts: Dict[str, int]
    median_lag_days: float
    n_negative_lag: int
    n_implausibly_fast: int
    n_implausibly_slow: int
    n_duplicate_periods: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether the file can be used at all. Warnings do not block."""
        return not self.errors

    @property
    def lag_looks_synthetic(self) -> bool:
        """Whether the report dates were probably reconstructed, not observed.

        A file where every lag is identical was almost certainly produced by
        adding a constant to the fiscal date. That is worse than no file: it
        has the shape of point-in-time data and none of the content, so a
        backtest on it looks rigorous and is not.
        """
        return any("identical" in warning for warning in self.warnings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fundamentals_rows": self.n_rows,
            "fundamentals_symbols": self.n_symbols,
            "fundamentals_median_lag_days": self.median_lag_days,
            "fundamentals_negative_lag": self.n_negative_lag,
            "fundamentals_implausibly_fast": self.n_implausibly_fast,
            "fundamentals_implausibly_slow": self.n_implausibly_slow,
            "fundamentals_duplicate_periods": self.n_duplicate_periods,
            "fundamentals_lag_looks_synthetic": self.lag_looks_synthetic,
        }


def validate_fundamentals(frame: pd.DataFrame) -> FundamentalsValidation:
    """Check a fundamentals frame in the T02 style — invariants, not opinions.

    Args:
        frame: Rows of `(symbol, fiscal_date, report_date, *facts)`.

    Returns:
        A `FundamentalsValidation`. `errors` block use; `warnings` describe
        things worth knowing before believing a result.
    """
    errors: List[str] = []
    warnings: List[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        errors.append(
            f"Missing required column(s) {missing}. `report_date` in "
            f"particular is not optional: without it the store cannot answer "
            f"what was knowable on a date, and defaulting to the fiscal date "
            f"would build in a six-to-eight-week look-ahead."
        )
        return FundamentalsValidation(
            n_rows=len(frame), n_symbols=0, n_facts={}, median_lag_days=float("nan"),
            n_negative_lag=0, n_implausibly_fast=0, n_implausibly_slow=0,
            n_duplicate_periods=0, errors=errors,
        )

    fiscal = pd.to_datetime(frame["fiscal_date"], errors="coerce")
    report = pd.to_datetime(frame["report_date"], errors="coerce")

    unparseable = int(fiscal.isna().sum() + report.isna().sum())
    if unparseable:
        errors.append(f"{unparseable} row(s) have an unparseable date.")

    lag = (report - fiscal).dt.days
    usable = lag.dropna()

    n_negative = int((usable < 0).sum())
    if n_negative:
        errors.append(
            f"{n_negative} row(s) report *before* the period they describe. "
            f"That is not a late filing or a data-entry slip in one direction "
            f"— it is a column swap, and every value in the file is suspect."
        )

    n_fast = int(((usable >= 0) & (usable < IMPLAUSIBLY_FAST_DAYS)).sum())
    if n_fast:
        warnings.append(
            f"{n_fast} row(s) report within {IMPLAUSIBLY_FAST_DAYS} days of "
            f"the period end. SEBI LODR Regulation 33 allows "
            f"{QUARTERLY_DEADLINE_DAYS} days for quarterly results, so a lag "
            f"this short usually means a fiscal date in the report-date column."
        )

    n_slow = int((usable > IMPLAUSIBLY_SLOW_DAYS).sum())
    if n_slow:
        warnings.append(
            f"{n_slow} row(s) report more than {IMPLAUSIBLY_SLOW_DAYS} days "
            f"after the period end — either late filers or, more often, a "
            f"fiscal date mistaken for a report date."
        )

    if len(usable) > 5 and usable.nunique() == 1:
        warnings.append(
            f"Every report lag is identical ({int(usable.iloc[0])} days). The "
            f"dates were almost certainly reconstructed by adding a constant "
            f"to the fiscal date rather than observed. That is worse than no "
            f"file: it has the shape of point-in-time data and none of the "
            f"content, so a backtest on it looks rigorous and is not."
        )

    duplicated = frame.duplicated(subset=["symbol", "fiscal_date"], keep=False)
    n_duplicate = int(duplicated.sum())
    if n_duplicate:
        warnings.append(
            f"{n_duplicate} row(s) share a (symbol, fiscal_date). Restatements "
            f"are real and the store keeps the *earliest* report date for a "
            f"period, because a restatement published later was not knowable "
            f"at the original announcement."
        )

    present = {c: int(frame[c].notna().sum()) for c in FACT_COLUMNS if c in frame.columns}
    if not present:
        errors.append(
            f"No recognized fact column. Expected one or more of "
            f"{list(FACT_COLUMNS)}."
        )

    return FundamentalsValidation(
        n_rows=len(frame),
        n_symbols=int(frame["symbol"].nunique()),
        n_facts=present,
        median_lag_days=float(usable.median()) if len(usable) else float("nan"),
        n_negative_lag=n_negative,
        n_implausibly_fast=n_fast,
        n_implausibly_slow=n_slow,
        n_duplicate_periods=n_duplicate,
        errors=errors,
        warnings=warnings,
    )


@dataclass
class FundamentalsStore:
    """Point-in-time accounting facts, answering `as_of(date)`.

    Construct through `from_frame` or `load_fundamentals`, which validate
    first — a store built from an unvalidated frame is the thing this module
    exists to make hard.
    """

    facts: pd.DataFrame
    validation: Optional[FundamentalsValidation] = None

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "FundamentalsStore":
        """Build a store, refusing a frame that cannot be point-in-time.

        Raises:
            ValueError: If validation found errors. Warnings pass through onto
                the store, where a report can print them.
        """
        validation = validate_fundamentals(frame)
        if not validation.ok:
            raise ValueError(
                "Fundamentals file is not usable:\n  - "
                + "\n  - ".join(validation.errors)
            )

        facts = frame.copy()
        facts["fiscal_date"] = pd.to_datetime(facts["fiscal_date"])
        facts["report_date"] = pd.to_datetime(facts["report_date"])
        # Earliest report wins for a period: a restatement published later was
        # not knowable at the original announcement, and taking the latest
        # would quietly reintroduce the look-ahead by the back door.
        facts = (
            facts.sort_values(["symbol", "fiscal_date", "report_date"])
            .drop_duplicates(subset=["symbol", "fiscal_date"], keep="first")
            .sort_values(["symbol", "report_date"])
            .reset_index(drop=True)
        )
        return cls(facts=facts, validation=validation)

    @property
    def symbols(self) -> List[str]:
        return sorted(self.facts["symbol"].unique().tolist())

    @property
    def available_facts(self) -> List[str]:
        return [c for c in FACT_COLUMNS if c in self.facts.columns]

    def as_of(self, date: Any, symbols: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """The most recent fact per symbol that had been *published* by `date`.

        The whole point of the module in one line: `report_date <= date`, never
        `fiscal_date <= date`.

        Args:
            date: The decision date.
            symbols: Restrict to these. None takes every symbol in the store.

        Returns:
            One row per symbol, indexed by symbol, or an empty frame when
            nothing had been published yet.
        """
        cutoff = pd.Timestamp(date)
        known = self.facts[self.facts["report_date"] <= cutoff]
        if symbols is not None:
            known = known[known["symbol"].isin(list(symbols))]
        if known.empty:
            return pd.DataFrame(columns=self.facts.columns).set_index("symbol")
        return (
            known.sort_values("report_date")
            .drop_duplicates(subset=["symbol"], keep="last")
            .set_index("symbol")
        )

    def panel(
        self, dates: Sequence[Any], field_name: str,
        symbols: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """One fact as a `(date x symbol)` frame, forward-filled from report dates.

        Forward-filled rather than interpolated: a balance-sheet number is a
        step function. It is the last reported value until the next report, and
        smoothing between them would invent quarters that were never published.

        Args:
            dates: Trading dates to project onto.
            field_name: One of `FACT_COLUMNS`.
            symbols: Restrict to these.

        Returns:
            `(date x symbol)`, NaN before a symbol's first report.

        Raises:
            KeyError: If the store does not carry that field.
        """
        if field_name not in self.facts.columns:
            raise KeyError(
                f"Fundamentals store has no '{field_name}'. Available: "
                f"{self.available_facts}"
            )

        index = pd.DatetimeIndex(sorted(pd.to_datetime(list(dates))))
        wanted = list(symbols) if symbols is not None else self.symbols

        rows = self.facts[self.facts["symbol"].isin(wanted)]
        rows = rows[["symbol", "report_date", field_name]].dropna(subset=[field_name])
        if rows.empty:
            return pd.DataFrame(index=index, columns=wanted, dtype=float)

        wide = rows.pivot_table(
            index="report_date", columns="symbol", values=field_name, aggfunc="last"
        )
        # Union first so a report landing on a non-trading day still propagates
        # forward onto the next trading date rather than being dropped.
        combined = wide.reindex(wide.index.union(index)).ffill()
        return combined.reindex(index=index, columns=wanted)


def load_fundamentals(
    path: Any, directory: Optional[Path] = None
) -> FundamentalsStore:
    """Read a fundamentals CSV and validate it.

    Args:
        path: File path, or a bare name resolved inside `directory`.
        directory: Defaults to `DEFAULT_FUNDAMENTALS_DIR`.

    Returns:
        A validated `FundamentalsStore`.

    Raises:
        FileNotFoundError: With the resolved path, because the overwhelmingly
            common cause is a file in the wrong directory.
        ValueError: If validation found errors.
    """
    resolved = Path(path)
    if not resolved.is_absolute() and not resolved.exists():
        resolved = (directory or DEFAULT_FUNDAMENTALS_DIR) / resolved
    if not resolved.exists():
        raise FileNotFoundError(
            f"No fundamentals file at {resolved}. See docs/OBTAINING_DATA.md "
            f"for the format and where to get one."
        )
    return FundamentalsStore.from_frame(pd.read_csv(resolved))
