"""Reference data: the three inputs that are neither prices nor accounts.

Each closes a caveat the platform is already printing, and each is stated in
the code that prints it rather than tracked somewhere else.

**Free float, for a real size exposure.** `evaluation/neutralize.py` says it:

> size is a proxy: log rolling-median traded value, not log market cap. Market
> cap needs shares outstanding, which this platform does not have, and for
> Indian equities the correct figure is *free float* — promoter holdings run
> 50-75%, so total capitalisation is not what trades.

That parenthesis is the whole reason this is separate from T31's
`shares_outstanding`. A total-capitalisation size sort on Indian equities ranks
by promoter stake as much as by size: two firms with identical float and
different promoter holdings are the same size to a trader and three times apart
on paper. NSE's own indices are free-float weighted for exactly this reason.

**Sector, for a neutralization that is currently unavailable.** `src/sectors.py`
has had the loader since the concentration limits were written; nothing ships a
CSV, so `--neutralize sector` has never had anything to neutralize with. Indian
momentum concentrates hard by sector, which makes this the single most likely
place for an apparent alpha to turn out to be a sector bet.

**FII/DII flows** (§10) are a market-state variable rather than a per-stock
characteristic — the one input here that describes the market rather than the
names in it. Foreign flows dominate Indian index momentum and correlate with
global risk appetite; domestic flows offset them during selloffs. That makes
the *difference* between them more informative than either alone, which is why
`FlowSeries` exposes it directly.

Point-in-time, again
--------------------
Share counts change on splits, bonuses, buybacks and QIPs; promoter stakes
change on pledges and secondary sales; a stock moves sector on reclassification.
All three are keyed on an `effective_date` and read with `<= date`, for the
reason T31 spells out at length: the alternative is a backtest that knew things
early, and that failure is invisible in the output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Columns a free-float file must carry.
FLOAT_REQUIRED_COLUMNS = ("symbol", "effective_date", "free_float_shares")

#: Columns a flow file must carry. `fii_net` and `dii_net` in rupees crore, or
#: any consistent unit — every use is a sign, a ratio or a z-score.
FLOW_REQUIRED_COLUMNS = ("date", "fii_net", "dii_net")

#: Default home for reference files, alongside membership and fundamentals.
DEFAULT_REFERENCE_DIR = Path("universe")

#: Promoter holdings in Indian listed equity, as a sanity range. Free float
#: above 100% of issued shares is impossible; below 5% is possible but rare
#: enough to be worth a second look at the file.
IMPLAUSIBLE_FLOAT_FRACTION_LOW = 0.05

#: Sessions over which flows are accumulated for the regime read. One quarter,
#: matching `conditional.DEFAULT_TRAILING_WINDOW`, so a flow-conditioned split
#: and a return-conditioned one are looking at the same span.
DEFAULT_FLOW_WINDOW = 63

SIZE_PROXY_REPLACED_NOTE = (
    "size is free-float market capitalisation, not the traded-value proxy. "
    "Free float rather than total capitalisation because Indian promoter "
    "holdings run 50-75%: a total-cap sort ranks by promoter stake as much as "
    "by size, and NSE's own indices are free-float weighted for that reason."
)

NO_SECTOR_MAP_NOTE = (
    "No sector map was supplied, so this result is not sector-neutral. Indian "
    "momentum concentrates hard by sector, which makes this the most likely "
    "place for an apparent alpha to be a sector bet. See docs/OBTAINING_DATA.md."
)

NO_FLOWS_NOTE = (
    "No FII/DII flow data was supplied, so no flow-based regime conditioning "
    "was applied. Foreign flows dominate Indian index momentum, so a result "
    "measured across a period of sustained foreign selling and one measured "
    "across sustained buying are not the same result. See "
    "docs/OBTAINING_DATA.md."
)


# --------------------------------------------------------------------------
# Free float
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FloatValidation:
    """What a free-float file looks like before anyone sizes a book with it."""

    n_rows: int
    n_symbols: int
    n_non_positive: int
    n_exceeds_issued: int
    n_implausibly_small: int
    median_float_fraction: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "float_rows": self.n_rows,
            "float_symbols": self.n_symbols,
            "float_non_positive": self.n_non_positive,
            "float_exceeds_issued": self.n_exceeds_issued,
            "float_median_fraction": self.median_float_fraction,
        }


def validate_free_float(frame: pd.DataFrame) -> FloatValidation:
    """Check a free-float file. `total_shares` is optional but earns its keep.

    When both columns are present the ratio is checkable, and the ratio is
    where the errors are: a file that has quietly given total shares in the
    free-float column is not detectable from the float alone, and produces a
    size sort that is wrong by the promoter stake — largest exactly where
    promoter holdings are largest.
    """
    errors: List[str] = []
    warnings: List[str] = []

    missing = [c for c in FLOAT_REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        errors.append(f"Missing required column(s) {missing}.")
        return FloatValidation(
            n_rows=len(frame), n_symbols=0, n_non_positive=0, n_exceeds_issued=0,
            n_implausibly_small=0, median_float_fraction=float("nan"), errors=errors,
        )

    if pd.to_datetime(frame["effective_date"], errors="coerce").isna().any():
        errors.append("Some `effective_date` values could not be parsed.")

    floats = pd.to_numeric(frame["free_float_shares"], errors="coerce")
    n_non_positive = int((floats <= 0).sum() + floats.isna().sum())
    if n_non_positive:
        errors.append(
            f"{n_non_positive} row(s) have a non-positive or unparseable free "
            f"float. A zero float is not a very small company — it is a "
            f"missing value, and dividing by it would rank that name at an "
            f"extreme of the size sort."
        )

    fraction = pd.Series(dtype=float)
    n_exceeds = 0
    n_small = 0
    if "total_shares" in frame.columns:
        total = pd.to_numeric(frame["total_shares"], errors="coerce")
        fraction = (floats / total.replace(0.0, np.nan)).dropna()
        n_exceeds = int((fraction > 1.0).sum())
        if n_exceeds:
            errors.append(
                f"{n_exceeds} row(s) have free float exceeding issued shares, "
                f"which is impossible. The two columns are most likely swapped."
            )
        n_small = int((fraction < IMPLAUSIBLE_FLOAT_FRACTION_LOW).sum())
        if n_small:
            warnings.append(
                f"{n_small} row(s) have a free float below "
                f"{IMPLAUSIBLE_FLOAT_FRACTION_LOW:.0%} of issued shares. "
                f"Possible, but rare enough to be worth checking the file."
            )
        if len(fraction) > 5 and float(fraction.median()) > 0.95:
            warnings.append(
                "The median free float is above 95% of issued shares. Indian "
                "promoter holdings run 50-75%, so this file is probably "
                "reporting total capitalisation under a free-float heading — "
                "which produces a size sort wrong by exactly the promoter "
                "stake, and largest where promoter holdings are largest."
            )
    else:
        warnings.append(
            "No `total_shares` column, so the free-float fraction cannot be "
            "checked. A file that reports total shares under the free-float "
            "heading is undetectable without it."
        )

    return FloatValidation(
        n_rows=len(frame),
        n_symbols=int(frame["symbol"].nunique()),
        n_non_positive=n_non_positive,
        n_exceeds_issued=n_exceeds,
        n_implausibly_small=n_small,
        median_float_fraction=float(fraction.median()) if len(fraction) else float("nan"),
        errors=errors,
        warnings=warnings,
    )


@dataclass
class FreeFloatStore:
    """Free-float share counts, point-in-time on `effective_date`.

    Share counts move on splits, bonuses, buybacks and QIPs, and promoter
    stakes move on pledges and secondary sales. Applying today's float to a
    2015 date would restate a decade of market caps, so the store answers
    `as_of` the same way T31's fundamentals store does.
    """

    counts: pd.DataFrame
    validation: Optional[FloatValidation] = None

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "FreeFloatStore":
        """Build a store, refusing a frame that would misprice the size sort.

        Raises:
            ValueError: If validation found errors.
        """
        validation = validate_free_float(frame)
        if not validation.ok:
            raise ValueError(
                "Free-float file is not usable:\n  - " + "\n  - ".join(validation.errors)
            )
        counts = frame.copy()
        counts["effective_date"] = pd.to_datetime(counts["effective_date"])
        counts = (
            counts.sort_values(["symbol", "effective_date"])
            .drop_duplicates(subset=["symbol", "effective_date"], keep="last")
            .reset_index(drop=True)
        )
        return cls(counts=counts, validation=validation)

    @property
    def symbols(self) -> List[str]:
        return sorted(self.counts["symbol"].unique().tolist())

    def panel(
        self, dates: Sequence[Any], symbols: Optional[Sequence[str]] = None
    ) -> pd.DataFrame:
        """Free float as a `(date x symbol)` frame, stepped from effective dates.

        Forward-filled and never back-filled. Back-filling would apply a
        post-buyback share count to the years before it, which restates every
        market cap in the sample in the same direction.
        """
        index = pd.DatetimeIndex(sorted(pd.to_datetime(list(dates))))
        wanted = list(symbols) if symbols is not None else self.symbols

        rows = self.counts[self.counts["symbol"].isin(wanted)]
        if rows.empty:
            return pd.DataFrame(index=index, columns=wanted, dtype=float)

        wide = rows.pivot_table(
            index="effective_date", columns="symbol",
            values="free_float_shares", aggfunc="last",
        )
        combined = wide.reindex(wide.index.union(index)).ffill()
        return combined.reindex(index=index, columns=wanted)

    def market_cap(self, closes: pd.DataFrame) -> pd.DataFrame:
        """Free-float market capitalisation on a `(date x symbol)` close panel.

        The number `SIZE_PROXY_NOTE` says the platform does not have. NaN
        wherever the float is unknown, rather than falling back to the traded
        value proxy for some names and not others — a size column built from
        two different definitions ranks on which definition applied.
        """
        floats = self.panel(closes.index, list(closes.columns))
        return closes * floats


# --------------------------------------------------------------------------
# Sector
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SectorCoverage:
    """How much of a universe a sector map actually covers."""

    n_symbols: int
    n_mapped: int
    n_sectors: int
    largest_sector_share: float
    unmapped: List[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return self.n_mapped / self.n_symbols if self.n_symbols else 0.0

    def note(self) -> str:
        """One sentence for the report, stating coverage rather than implying it.

        A neutralization run against a map covering 40% of the universe is not
        a sector-neutral result, and reporting it without the fraction invites
        exactly that reading.
        """
        if self.n_symbols == 0:
            return NO_SECTOR_MAP_NOTE
        if self.n_mapped == 0:
            return NO_SECTOR_MAP_NOTE
        return (
            f"Sector map covers {self.n_mapped}/{self.n_symbols} names "
            f"({self.coverage:.0%}) across {self.n_sectors} sectors; the "
            f"largest holds {self.largest_sector_share:.0%} of mapped names. "
            + (
                "Unmapped names are pooled as UNKNOWN, which neutralizes them "
                "against each other rather than against their real peers."
                if self.n_mapped < self.n_symbols else ""
            )
        ).strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sector_symbols": self.n_symbols,
            "sector_mapped": self.n_mapped,
            "sector_coverage": self.coverage,
            "sector_count": self.n_sectors,
            "sector_largest_share": self.largest_sector_share,
        }


def sector_coverage(
    sector_map: Mapping[str, str], universe: Sequence[str]
) -> SectorCoverage:
    """Measure a sector map against the universe it will be used on.

    Coverage is the question, not existence. `src/sectors.py` has loaded these
    files since the concentration limits were written, and a map that resolves
    40% of a universe produces a "sector-neutral" result in which most names
    were neutralized against a pool called UNKNOWN.
    """
    symbols = list(universe)
    mapped = {s: sector_map[s] for s in symbols if s in sector_map}
    counts = pd.Series(list(mapped.values())).value_counts() if mapped else pd.Series(dtype=int)

    return SectorCoverage(
        n_symbols=len(symbols),
        n_mapped=len(mapped),
        n_sectors=int(counts.size),
        largest_sector_share=float(counts.iloc[0] / counts.sum()) if counts.size else 0.0,
        unmapped=[s for s in symbols if s not in sector_map],
    )


# --------------------------------------------------------------------------
# FII / DII flows
# --------------------------------------------------------------------------


@dataclass
class FlowSeries:
    """Daily net institutional flows — a property of the market, not a stock.

    The one reference input here that describes the market rather than the
    names in it, which is why it conditions a result rather than entering a
    cross-sectional ranking.
    """

    flows: pd.DataFrame

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "FlowSeries":
        """Build from `(date, fii_net, dii_net)` rows.

        Raises:
            ValueError: On a missing column or an unparseable date.
        """
        missing = [c for c in FLOW_REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(
                f"Flow file is missing required column(s) {missing}. Expected "
                f"{list(FLOW_REQUIRED_COLUMNS)}."
            )
        flows = frame.copy()
        flows["date"] = pd.to_datetime(flows["date"], errors="coerce")
        if flows["date"].isna().any():
            raise ValueError("Some flow `date` values could not be parsed.")
        for column in ("fii_net", "dii_net"):
            flows[column] = pd.to_numeric(flows[column], errors="coerce")
        return cls(
            flows=flows.sort_values("date").drop_duplicates("date", keep="last")
            .set_index("date")
        )

    @property
    def net(self) -> pd.Series:
        """FII minus DII — the number that carries the most information.

        Domestic flows systematically offset foreign ones during selloffs, so
        the two series are strongly negatively correlated and neither alone
        says whether the market was under pressure. Their difference does.
        """
        return (self.flows["fii_net"] - self.flows["dii_net"]).rename("net_flow")

    def trailing(self, window: int = DEFAULT_FLOW_WINDOW) -> pd.Series:
        """Accumulated net flow over a trailing window, ending at each date.

        Causal by construction: the window ends at the row it labels, so the
        value on date `t` was observable on `t`. That is what makes this usable
        as a *tradable* conditioner rather than only as attribution.
        """
        return self.net.rolling(window, min_periods=window).sum()

    def states(
        self, dates: Sequence[Any], window: int = DEFAULT_FLOW_WINDOW
    ) -> pd.Series:
        """Label each date `"inflow"` or `"outflow"` by trailing net flow.

        Shaped to drop straight into `evaluation/conditional.py` as an
        alternative conditioner: a result measured across sustained foreign
        selling and one measured across sustained buying are not the same
        result, and a pooled number describes neither.

        Dates without a full trailing window are omitted rather than labelled
        from a partial one — two conditioners under one name is the failure
        `trailing_states` avoids for the same reason.
        """
        accumulated = self.trailing(window)
        labels: Dict[Any, str] = {}
        for date in sorted(pd.to_datetime(list(dates))):
            usable = accumulated.loc[accumulated.index <= date].dropna()
            if usable.empty:
                continue
            labels[date] = "inflow" if float(usable.iloc[-1]) >= 0 else "outflow"
        return pd.Series(labels).sort_index()


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _resolve(path: Any, directory: Optional[Path], what: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute() and not resolved.exists():
        resolved = (directory or DEFAULT_REFERENCE_DIR) / resolved
    if not resolved.exists():
        raise FileNotFoundError(
            f"No {what} file at {resolved}. See docs/OBTAINING_DATA.md for the "
            f"format and where to get one."
        )
    return resolved


def load_free_float(path: Any, directory: Optional[Path] = None) -> FreeFloatStore:
    """Read and validate a free-float CSV."""
    return FreeFloatStore.from_frame(pd.read_csv(_resolve(path, directory, "free-float")))


def load_flows(path: Any, directory: Optional[Path] = None) -> FlowSeries:
    """Read and validate an FII/DII flow CSV."""
    return FlowSeries.from_frame(pd.read_csv(_resolve(path, directory, "flow")))
