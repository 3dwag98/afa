"""Sector classification and portfolio-level concentration limits.

Motivation (docs/QUANT_RESEARCH.md section 13). Cross-sectional strategies
rank stocks on a single characteristic and buy the extreme decile, with no
term in the objective that cares *what those stocks are*. In Indian equities
that regularly produces a portfolio which is nominally 10 names and
economically one bet: momentum concentrated in IT through 2020-21, then in
Banking/PSU through 2022-23. The factor exposure is intended; the sector
exposure is an accident, and it is what turns a factor drawdown into a
portfolio drawdown.

This module supplies:

- A ticker -> sector map loaded from a CSV (`data/sector_map.csv` by default),
  because OHLCV carries no sector information and this platform ingests
  nothing else. Tickers absent from the map are reported as "UNKNOWN".
- `sector_capacity_inr()`, which answers the only question the order path
  actually needs: given current exposure, how much more money may go into
  this ticker's sector before the cap is breached?

The cap is deliberately enforced at *order-creation* time rather than by
rejecting a filled position after the fact — a cap you can only discover
you've broken is not a cap.

Unmapped tickers are pooled under a single "UNKNOWN" sector and capped
together. That is the conservative reading: an unknown sector could be any
sector, including the one already at its limit.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, Mapping, Optional

logger = logging.getLogger(__name__)

UNKNOWN_SECTOR = "UNKNOWN"

# Column names accepted in the sector CSV, case-insensitive.
_TICKER_COLUMNS = ("ticker", "symbol")
_SECTOR_COLUMNS = ("sector", "industry", "gics_sector")


def normalize_ticker(ticker: str) -> str:
    """Normalize a ticker to the platform's canonical NSE form (UPPER + .NS)."""
    t = (ticker or "").strip().upper()
    if not t:
        return t
    return t if t.endswith(".NS") else f"{t}.NS"


def load_sector_map(csv_path: Optional[str] = None) -> Dict[str, str]:
    """Load a ticker -> sector mapping from CSV.

    The file needs a header with a ticker column (`ticker` or `symbol`) and a
    sector column (`sector`, `industry` or `gics_sector`); column order does
    not matter. A missing file is not an error — it yields an empty map, and
    callers fall back to treating every holding as UNKNOWN.

    Args:
        csv_path: Path to the sector CSV. Defaults to data/sector_map.csv
            relative to the current working directory (matching data_store's
            convention).

    Returns:
        Mapping of normalized ticker -> sector name. Empty if the file is
        absent or unreadable.
    """
    path = Path(csv_path) if csv_path else Path("data") / "sector_map.csv"
    if not path.exists():
        logger.info(
            "Sector map %s not found; sector concentration limits will treat every "
            "holding as %s. Provide a ticker,sector CSV to enable per-sector caps.",
            path, UNKNOWN_SECTOR,
        )
        return {}

    mapping: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return {}
            lower = {(name or "").strip().lower(): name for name in reader.fieldnames}
            ticker_col = next((lower[c] for c in _TICKER_COLUMNS if c in lower), None)
            sector_col = next((lower[c] for c in _SECTOR_COLUMNS if c in lower), None)
            if ticker_col is None or sector_col is None:
                logger.warning(
                    "Sector map %s must have a ticker column (%s) and a sector column (%s); "
                    "found %s. Ignoring the file.",
                    path, "/".join(_TICKER_COLUMNS), "/".join(_SECTOR_COLUMNS), reader.fieldnames,
                )
                return {}

            for row in reader:
                ticker = normalize_ticker(row.get(ticker_col, ""))
                sector = (row.get(sector_col) or "").strip()
                if ticker and sector:
                    mapping[ticker] = sector
    except (OSError, csv.Error):
        logger.warning("Could not read sector map %s; continuing without it", path, exc_info=True)
        return {}

    logger.info("Loaded sector map for %d tickers from %s", len(mapping), path)
    return mapping


def sector_of(ticker: str, sector_map: Mapping[str, str]) -> str:
    """Return a ticker's sector, or UNKNOWN when it isn't in the map."""
    return sector_map.get(normalize_ticker(ticker), UNKNOWN_SECTOR)


def sector_exposure_inr(
    position_values: Mapping[str, float],
    sector_map: Mapping[str, str],
) -> Dict[str, float]:
    """Aggregate per-ticker position values into per-sector exposure.

    Args:
        position_values: ticker -> current market value of the holding in INR.
        sector_map: ticker -> sector mapping (see load_sector_map).

    Returns:
        sector -> total exposure in INR.
    """
    by_sector: Dict[str, float] = {}
    for ticker, value in position_values.items():
        if value is None or value <= 0:
            continue
        sector = sector_of(ticker, sector_map)
        by_sector[sector] = by_sector.get(sector, 0.0) + float(value)
    return by_sector


def sector_capacity_inr(
    ticker: str,
    portfolio_value_inr: float,
    position_values: Mapping[str, float],
    sector_map: Mapping[str, str],
    max_sector_pct: float,
) -> float:
    """How much more capital may be added to this ticker's sector, in INR.

    Args:
        ticker: The ticker a new position is being sized for.
        portfolio_value_inr: Total portfolio value (cash + holdings).
        position_values: ticker -> current market value of each open holding.
        sector_map: ticker -> sector mapping.
        max_sector_pct: Cap on any one sector as a fraction of portfolio
            value. Values <= 0 or >= 1 disable the cap (returns infinity).

    Returns:
        Remaining INR capacity for the sector; 0.0 when it is already at or
        over the cap, and math.inf when the cap is disabled.
    """
    if max_sector_pct <= 0 or max_sector_pct >= 1 or portfolio_value_inr <= 0:
        return float("inf")

    sector = sector_of(ticker, sector_map)
    current = sector_exposure_inr(position_values, sector_map).get(sector, 0.0)
    allowance = portfolio_value_inr * max_sector_pct
    return max(0.0, allowance - current)
