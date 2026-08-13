#!/usr/bin/env python3
"""Probe the nsehistoricaldata.co.in extract API before building on it.

Run this locally — the host is not reachable from CI or the dev container.

    python tools/probe_nse_source.py

It answers the three questions that decide how much of the data problem this
source can actually fix, and it answers them empirically rather than by
assumption:

  1. Does PREV_CLOSE carry the corporate-action adjustment?
     If yes, the adjustment series can be derived from the price file alone —
     factor[t] = PREV_CLOSE[t] / CLOSE[t-1] deviates from 1.0 exactly on
     ex-dates. That means raw and adjusted prices can both be stored without a
     separate corporate-actions feed, which is what the circuit-band check
     needs and what the current pipeline throws away.

  2. Do delisted / renamed symbols return history?
     If they do not, the source is structurally survivorship-biased and cannot
     fix D9 on its own, no matter how many symbols are pulled.

  3. What does SERIES actually contain?
     EQ permits intraday netting; BE and BZ are trade-to-trade, where every
     trade settles by compulsory delivery and an intraday exit is not merely
     expensive but impossible. The backtest currently models none of this.

Nothing here writes to the repo's data directories. It prints findings and
exits.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional

API = "https://www.nsehistoricaldata.co.in/api/extract"

# The mirror serves a browser extension, so it expects browser-shaped headers.
# Kept in one place because this is the part most likely to need adjusting when
# the upstream changes.
HEADERS = {
    "accept": "*/*",
    "content-type": "application/json",
    "origin": "https://www.nsehistoricaldata.co.in",
    "referer": "https://www.nsehistoricaldata.co.in/",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"
    ),
}

# Liquid names that certainly exist, for the baseline shape check.
LIVE = ["TCS", "RELIANCE", "HDFCBANK"]

# Names that have left the market or changed identity. The point of the test is
# that a survivorship-free source returns history for these and a biased one
# returns nothing — so a 404 here is itself the finding.
GONE = [
    "DHFL",           # delisted after insolvency
    "BHARTIARTL",     # control: alive, should return data
    "BHARTIINFR",     # merged into Indus Towers
    "RCOM",           # suspended
    "VEDL",           # delisted then relisted
]


def fetch(symbol: str, from_date: str, to_date: str,
          timeout: int = 60) -> Optional[List[Dict[str, Any]]]:
    """One extract call. Returns rows, or None when the symbol yields nothing.

    A missing symbol is an ordinary outcome here — it is in fact the result the
    survivorship test is looking for — so it is reported rather than raised.
    """
    import requests

    payload = {
        "symbol": symbol,
        "from_date": from_date,
        "to_date": to_date,
        "progress_stream": False,
    }
    try:
        response = requests.post(API, headers=HEADERS, json=payload, timeout=timeout)
    except Exception as exc:
        print(f"    request failed: {type(exc).__name__}: {exc}")
        return None

    if response.status_code != 200:
        print(f"    HTTP {response.status_code}")
        return None

    try:
        body = response.json()
    except json.JSONDecodeError:
        # progress_stream responses may be newline-delimited JSON; take the last
        # complete object, which is where a streaming endpoint puts its payload.
        lines = [ln for ln in response.text.splitlines() if ln.strip()]
        for line in reversed(lines):
            try:
                body = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        else:
            print(f"    unparseable response ({len(response.text)} chars)")
            return None

    rows = _rows_from(body)
    if not rows:
        print(f"    no rows (keys: {list(body)[:8] if isinstance(body, dict) else type(body).__name__})")
    return rows


def _rows_from(body: Any) -> Optional[List[Dict[str, Any]]]:
    """Pull the record list out of whatever envelope the API used."""
    if isinstance(body, list):
        return body or None
    if isinstance(body, dict):
        for key in ("data", "rows", "records", "result"):
            value = body.get(key)
            if isinstance(value, list) and value:
                return value
    return None


def normalize(rows: List[Dict[str, Any]]):
    """Rows -> a date-indexed frame with the columns this probe reads."""
    import pandas as pd

    frame = pd.DataFrame(rows)
    frame.columns = [
        str(c).strip().lower().replace(" ", "_").replace(".", "").replace("%", "pct")
        for c in frame.columns
    ]

    date_col = next((c for c in ("date", "timestamp", "dt") if c in frame.columns), None)
    if date_col is None:
        raise ValueError(f"no date column in {list(frame.columns)}")

    # NSE renders dates as DD-MMM-YYYY; dayfirst covers the DD-MM-YYYY variant
    # the request itself uses.
    frame["date"] = pd.to_datetime(frame[date_col], errors="coerce", dayfirst=True)
    frame = frame.dropna(subset=["date"]).set_index("date").sort_index()

    for column in frame.columns:
        if column in ("series", "symbol"):
            continue
        frame[column] = pd.to_numeric(
            frame[column].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
    return frame


# ---------------------------------------------------------------------------
# Test 1 — is PREV_CLOSE adjusted?
# ---------------------------------------------------------------------------


def test_prev_close_adjustment(frame, symbol: str) -> None:
    """Look for dates where PREV_CLOSE disagrees with the prior CLOSE.

    On an unadjusted feed the two agree on every contiguous pair, and the ratio
    is 1.0 throughout. Deviations clustered on a handful of dates, at ratios
    near simple fractions, are corporate actions — which is the useful outcome.
    """
    import numpy as np

    prev_col = next((c for c in frame.columns if c.startswith("prev")), None)
    if prev_col is None or "close" not in frame.columns:
        print("    SKIP — no prev_close/close pair")
        return

    ratio = frame[prev_col] / frame["close"].shift(1)
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()

    # 0.5% tolerance absorbs rounding without hiding an ordinary dividend.
    breaks = ratio[(ratio - 1.0).abs() > 0.005]

    print(f"    {len(ratio)} contiguous pairs, {len(breaks)} with a factor != 1")
    if breaks.empty:
        print("    => PREV_CLOSE is NOT adjusted. A separate corporate-actions")
        print("       feed is required to build adjusted prices.")
        return

    print("    => PREV_CLOSE IS adjusted. Adjustment factors are derivable:")
    for date, value in breaks.sort_values().head(8).items():
        implied = 1.0 / value if value else float("nan")
        print(f"         {date.date()}  factor={value:.4f}  (~1:{implied:.2f})")
    if len(breaks) > 8:
        print(f"         ... and {len(breaks) - 8} more")


# ---------------------------------------------------------------------------
# Test 2 — survivorship
# ---------------------------------------------------------------------------


def test_survivorship(from_date: str, to_date: str, pause: float) -> None:
    """Ask for names that have left the market. Silence is the finding."""
    returned, missing = [], []
    for symbol in GONE:
        print(f"  {symbol}")
        rows = fetch(symbol, from_date, to_date)
        if rows:
            frame = normalize(rows)
            span = f"{frame.index.min().date()} -> {frame.index.max().date()}"
            print(f"    {len(frame)} rows  {span}")
            returned.append(symbol)
        else:
            missing.append(symbol)
        time.sleep(pause)

    print()
    print(f"  returned history: {returned or 'none'}")
    print(f"  no history:       {missing or 'none'}")
    if missing:
        print()
        print("  => The source does not serve every delisted/renamed name.")
        print("     It therefore CANNOT fix D9 on its own: a universe built by")
        print("     enumerating symbols it answers for is a survivor list, and")
        print("     every cross-sectional backtest over it stays biased upward.")
        print("     Point-in-time index membership is still required.")


# ---------------------------------------------------------------------------
# Test 3 — series
# ---------------------------------------------------------------------------


def test_series(frames: Dict[str, Any]) -> None:
    """Report which settlement series appear, and flag trade-to-trade."""
    seen: Dict[str, int] = {}
    for frame in frames.values():
        if "series" not in frame.columns:
            continue
        for value, count in frame["series"].value_counts().items():
            seen[str(value).strip().upper()] = seen.get(str(value).strip().upper(), 0) + int(count)

    if not seen:
        print("    SKIP — no series column")
        return

    for value, count in sorted(seen.items(), key=lambda kv: -kv[1]):
        note = ""
        if value in ("BE", "BZ"):
            note = "  <- TRADE-TO-TRADE: compulsory delivery, no intraday exit"
        elif value == "EQ":
            note = "  <- normal rolling settlement"
        print(f"    {value:6s} {count:7d} rows{note}")

    if {"BE", "BZ"} & set(seen):
        print()
        print("    => T2T sessions are present in this history. A backtest that")
        print("       exits intraday on these dates is simulating a trade the")
        print("       exchange does not permit.")


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-date", default="01-01-2015", help="DD-MM-YYYY")
    parser.add_argument("--to-date", default="08-08-2026", help="DD-MM-YYYY")
    parser.add_argument("--pause", type=float, default=1.5,
                        help="Seconds between requests. Be polite to a mirror "
                             "that is not yours (default: 1.5)")
    parser.add_argument("--skip-survivorship", action="store_true")
    args = parser.parse_args()

    try:
        import pandas  # noqa: F401
        import requests  # noqa: F401
    except ImportError as exc:
        print(f"needs pandas and requests: {exc}")
        return 1

    print("=" * 72)
    print("PROBE 0 — response shape")
    print("=" * 72)
    frames = {}
    for symbol in LIVE:
        print(f"  {symbol}")
        rows = fetch(symbol, args.from_date, args.to_date)
        if not rows:
            continue
        frame = normalize(rows)
        frames[symbol] = frame
        print(f"    {len(frame)} rows  "
              f"{frame.index.min().date()} -> {frame.index.max().date()}")
        print(f"    columns: {list(frame.columns)}")
        time.sleep(args.pause)

    if not frames:
        print("\nNo data returned at all — check headers, cookie and ToS before "
              "going further.")
        return 1

    print()
    print("=" * 72)
    print("PROBE 1 — is PREV_CLOSE adjusted for corporate actions?")
    print("=" * 72)
    for symbol, frame in frames.items():
        print(f"  {symbol}")
        test_prev_close_adjustment(frame, symbol)

    print()
    print("=" * 72)
    print("PROBE 2 — settlement series present")
    print("=" * 72)
    test_series(frames)

    if not args.skip_survivorship:
        print()
        print("=" * 72)
        print("PROBE 3 — are delisted / renamed names served?")
        print("=" * 72)
        test_survivorship(args.from_date, args.to_date, args.pause)

    print()
    print("=" * 72)
    print("Not answered by this source, regardless of the above:")
    print("  - point-in-time index membership (D9) — needs niftyindices")
    print("  - ISIN, so symbol renames still fragment history")
    print("  - price bands, ASM/GSM/ESM, F&O ban state")
    print("  - sector, free float, shares outstanding, fundamentals")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
