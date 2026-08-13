"""HuggingFace Hub as the historical OHLCV source for Indian equities.

The platform's original source is yfinance called live, which rate-limits,
changes its response shape between releases, and hands back whatever it has
today — so two runs a week apart can silently backtest on different bars. The
Hub dataset `vishnun0027/indian-market-historical-ohlcv` is the same upstream
data, but already collected, validated daily and *versioned*: pin
`data.hf_revision` and every run sees byte-identical history.

Layout (2,471 files, ~283 MB total):

    stocks/       NSE/BSE equities, one parquet per symbol (2,421)
    indices/      market indices (17)  <- the regime filter's benchmark lives here
    etfs/         ETFs (17)
    commodities/  commodity futures (8)
    forex/        currency pairs (8)
    metadata/     asset universe & catalog

One file per symbol is the reason this module downloads *per ticker* rather
than pulling the whole repo: a 30-name universe fetches 30 small files instead
of 283 MB.

Schema per file: date, open, high, low, close, adj_close, volume, dividends,
stock_splits, symbol.

**Price adjustment.** Both a raw `close` and an `adj_close` are present, and
which one is used is not cosmetic. On a 1:10 split the raw close drops 90% in a
single print — cross-sectional momentum reads that as a crash, ATR-derived
stops blow out, and the circuit-lock detector sees a limit-sized move. The
platform's previous yfinance path used `auto_adjust=True`, i.e. adjusted
prices, so `adjust_prices=True` (the default here) keeps the cache consistent
with everything already built on it: OHLC are back-adjusted by
`adj_close / close`, which removes split and dividend discontinuities while
leaving every intraday relationship (including high == low) intact.

Rows land in the same per-ticker parquet cache yfinance writes to
(src/data_store.py), so nothing downstream — universe discovery, the backtest
engine, the training panel — needs to know which source produced them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from portfolio_agent.data_quality.invariants import IngestRejected, assert_writable

logger = logging.getLogger(__name__)

DEFAULT_HF_DATASET_ID = "vishnun0027/indian-market-historical-ohlcv"

# Top-level directories in the repo, by asset class.
STOCKS_DIR = "stocks"
INDICES_DIR = "indices"
ASSET_DIRS = (STOCKS_DIR, INDICES_DIR, "etfs", "commodities", "forex")

# Accepted spellings for each canonical column. The dataset itself is already
# lowercase; the aliases cover hand-built fixtures and any future re-export.
_DATE_ALIASES = ("date", "timestamp", "datetime", "time", "trade_date")
_SYMBOL_ALIASES = ("symbol", "ticker", "symbols", "tickers", "scrip", "stock")
_COLUMN_ALIASES = {
    "open": ("open", "open_price", "o"),
    "high": ("high", "high_price", "h"),
    "low": ("low", "low_price", "l"),
    "close": ("close", "close_price", "c"),
    "adj_close": ("adj_close", "adjclose", "adj close", "adjusted_close"),
    "volume": ("volume", "vol", "v", "quantity", "traded_quantity"),
    "dividends": ("dividends", "dividend", "div"),
    "stock_splits": ("stock_splits", "stock split", "splits", "split_ratio"),
}

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")

# The price legs, kept unadjusted alongside the adjusted ones.
#
# Both are needed and they answer different questions. Returns must come from
# adjusted prices or a split reads as a crash. Anything about the *price level*
# must come from raw: a circuit band applies to the number that actually
# traded, so checking a band against a back-adjusted price compares against a
# price no exchange ever saw. Keeping only one of the two — which is what this
# module did until now — makes one of those two classes of question
# unanswerable, and silently wrong rather than absent.
RAW_PRICE_COLUMNS = ("open_raw", "high_raw", "low_raw", "close_raw")

# Corporate-action provenance. `adj_factor` is derived (adj_close / close);
# `dividends` and `stock_splits` are stated by the source. Keeping both lets
# them be cross-checked — where a stated split and the derived factor disagree,
# one of the two is wrong, and that is worth surfacing rather than resolving
# silently behind whichever happens to be read first.
ADJUSTMENT_COLUMNS = ("adj_close", "adj_factor", "dividends", "stock_splits")


class SchemaError(ValueError):
    """A dataset file's columns could not be mapped onto the OHLCV schema."""


def _pick(columns: Iterable[str], aliases: Sequence[str]) -> Optional[str]:
    """First column matching any alias, comparing case/separator-insensitively."""
    normalized = {
        str(c).strip().lower().replace(" ", "_").replace("-", "_"): c for c in columns
    }
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def normalize_ticker(ticker: str) -> str:
    """Canonical cache key for a symbol.

    Equities get the platform's `.NS` suffix. Index symbols (`^NSEI`,
    `^BSESN`, ...) are left exactly as written: they are looked up by the
    literal string in `data.benchmark_symbol`, and appending `.NS` to them
    produced a cache file (`^NSEI.NS.parquet`) that no reader ever asked for,
    silently disabling the benchmark-driven regime filter.
    """
    t = str(ticker).strip().upper()
    if not t or t.startswith("^"):
        return t
    return t if t.endswith(".NS") else f"{t}.NS"


def hub_symbol(ticker: str) -> str:
    """Bare symbol used for the Hub filename (`RELIANCE`, not `RELIANCE.NS`)."""
    t = str(ticker).strip().upper()
    return t[:-3] if t.endswith(".NS") else t


def normalize_frame(df: pd.DataFrame, adjust_prices: bool = True) -> pd.DataFrame:
    """Map one raw Hub parquet onto the platform's canonical OHLCV schema.

    Args:
        df: Raw frame as read from a Hub parquet file.
        adjust_prices: Back-adjust OHLC by `adj_close / close` so splits and
            dividends do not appear as price shocks. When False, or when the
            file carries no `adj_close`, raw prices pass through unchanged.

    Returns:
        Frame with lowercase `open/high/low/close/volume` and a tz-naive
        DatetimeIndex named `date`.

    Raises:
        SchemaError: when the date or close column cannot be identified. The
            columns actually present are named in the message, so a schema
            change is diagnosable without reading this source — the
            alternative is a cache full of NaN closes that only surfaces as a
            mysteriously empty backtest weeks later.
    """
    if df is None or df.empty:
        raise SchemaError("dataset file contained no rows")

    columns = list(df.columns)
    date_col = _pick(columns, _DATE_ALIASES)
    if date_col is None:
        raise SchemaError(
            f"no date column found (looked for {list(_DATE_ALIASES)}); columns present: {columns}"
        )

    mapping = {
        canonical: source
        for canonical, aliases in _COLUMN_ALIASES.items()
        if (source := _pick(columns, aliases)) is not None
    }

    if "close" not in mapping and "adj_close" not in mapping:
        raise SchemaError(
            f"no close column found (looked for {list(_COLUMN_ALIASES['close'])} or "
            f"{list(_COLUMN_ALIASES['adj_close'])}); columns present: {columns}"
        )

    out = pd.DataFrame(index=pd.RangeIndex(len(df)))
    for canonical, source in mapping.items():
        out[canonical] = pd.to_numeric(df[source].to_numpy(), errors="coerce")

    if "close" not in out.columns:
        out["close"] = out["adj_close"]

    # A file with only a close is still usable for every close-based signal;
    # keeping the column set uniform matters more than the missing legs. Done
    # before the raw legs are captured so they are uniform too.
    for leg in ("open", "high", "low"):
        if leg not in out.columns:
            out[leg] = out["close"]

    # Capture the traded prices before any adjustment touches them.
    for leg in ("open", "high", "low", "close"):
        out[f"{leg}_raw"] = out[leg]

    if "adj_close" in out.columns:
        # One factor per row, applied to all four legs together. Scaling them
        # as a set is what keeps intraday relationships intact: a locked
        # session (high == low) stays locked, and ATR keeps its proportion to
        # price. Volume is left as reported — the liquidity screen reads only
        # the trailing 60 sessions, where the factor is ~1.
        factor = (out["adj_close"] / out["close"]).replace(
            [float("inf"), float("-inf")], pd.NA
        )
        out["adj_factor"] = factor.fillna(1.0)
        if adjust_prices:
            for leg in ("open", "high", "low", "close"):
                out[leg] = out[leg] * out["adj_factor"]
    else:
        # Recorded as an explicit 1.0 rather than omitted, so a reader can tell
        # "no adjustment was needed" apart from "no adjustment was attempted".
        out["adj_factor"] = 1.0
    # Volume defaults to 0, never to a fabricated number: the liquidity screen
    # reads it, and an invented volume would defeat the screen entirely.
    if "volume" not in out.columns:
        out["volume"] = 0.0

    # Hub files carry date32; hand-built fixtures may carry tz-aware stamps.
    index = pd.to_datetime(df[date_col].to_numpy(), errors="coerce", utc=True)
    out.index = pd.DatetimeIndex(index).tz_localize(None)
    out.index.name = "date"

    out = out[out.index.notna() & out["close"].notna()]
    if out.empty:
        raise SchemaError("every row had an unparseable date or a missing close")

    out = out[~out.index.duplicated(keep="last")]

    # OHLCV first, so the frame reads the same as it always has and positional
    # access anywhere downstream still lands where it used to. The provenance
    # columns follow, and are optional by contract: a cache written before this
    # change simply lacks them, which every reader must tolerate.
    ordered = list(OHLCV_COLUMNS)
    ordered += [c for c in RAW_PRICE_COLUMNS if c in out.columns]
    ordered += [c for c in ADJUSTMENT_COLUMNS if c in out.columns]
    return out[ordered].sort_index()


def corporate_actions_from_frame(
    frame: pd.DataFrame, factor_tolerance: float = 0.005
) -> pd.DataFrame:
    """Extract corporate actions from a normalized frame.

    Two independent views of the same events, deliberately not merged:

    - **Stated** — the `dividends` and `stock_splits` columns the source
      publishes.
    - **Derived** — a change in `adj_factor` between consecutive sessions,
      which is what the adjustment actually applied.

    Where the two disagree, one of them is wrong, and which one matters. A
    split the source states but never adjusted for leaves a genuine 90% gap in
    the returns; an adjustment with no stated cause is usually a demerger or a
    rights issue, neither of which a naive split factor handles correctly. Both
    are reported rather than reconciled, because reconciling them here would
    hide exactly the cases worth looking at.

    Args:
        frame: Output of `normalize_frame`.
        factor_tolerance: Relative change in `adj_factor` treated as noise.

    Returns:
        A frame indexed by date with columns `kind`, `stated_value`,
        `factor_ratio` and `agrees`. Empty when the input carries no
        provenance columns, which is the case for caches written before these
        were preserved.
    """
    events = []

    if "stock_splits" in frame.columns:
        splits = frame["stock_splits"]
        for date, value in splits[splits.fillna(0) != 0].items():
            events.append({"date": date, "kind": "split", "stated_value": float(value)})

    if "dividends" in frame.columns:
        dividends = frame["dividends"]
        for date, value in dividends[dividends.fillna(0) != 0].items():
            events.append({"date": date, "kind": "dividend", "stated_value": float(value)})

    if not events and "adj_factor" not in frame.columns:
        return pd.DataFrame(columns=["kind", "stated_value", "factor_ratio", "agrees"])

    # The factor moves on every ex-date; a step between consecutive sessions is
    # the adjustment being applied, whatever caused it.
    ratios = pd.Series(dtype=float)
    if "adj_factor" in frame.columns:
        factor = frame["adj_factor"].astype(float)
        ratios = (factor / factor.shift(1)).replace([float("inf"), float("-inf")], pd.NA)
        moved = ratios[(ratios - 1.0).abs() > factor_tolerance].dropna()
        stated_dates = {e["date"] for e in events}
        for date, ratio in moved.items():
            if date not in stated_dates:
                events.append({
                    "date": date, "kind": "unexplained_adjustment", "stated_value": float("nan"),
                })

    if not events:
        return pd.DataFrame(columns=["kind", "stated_value", "factor_ratio", "agrees"])

    table = pd.DataFrame(events).set_index("date").sort_index()
    table["factor_ratio"] = [
        float(ratios.get(date, float("nan"))) if len(ratios) else float("nan")
        for date in table.index
    ]
    # A stated event the factor never moved for is the expensive case: the
    # source says a split happened and the prices were never adjusted for it.
    table["agrees"] = [
        bool(abs(r - 1.0) > factor_tolerance) if pd.notna(r) else False
        for r in table["factor_ratio"]
    ]
    return table


def _require_hub():
    """Import huggingface_hub, or explain how to install it."""
    try:
        import huggingface_hub
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise RuntimeError(
            "Loading market data from HuggingFace needs the `huggingface_hub` package. "
            "Install the extra with `uv sync --extra hf` (or `pip install huggingface_hub`), "
            "or set data.source: yfinance in config.yaml."
        ) from exc
    return huggingface_hub


def list_hub_symbols(
    dataset_id: str = DEFAULT_HF_DATASET_ID,
    revision: Optional[str] = None,
    asset_dir: str = STOCKS_DIR,
) -> List[str]:
    """List the symbols available in one asset directory of the Hub dataset.

    Args:
        dataset_id: Hub dataset repo id.
        revision: Optional git revision (branch, tag or commit) to pin.
        asset_dir: One of ASSET_DIRS.

    Returns:
        Sorted bare symbols (e.g. "RELIANCE"), without the .parquet suffix.
    """
    hub = _require_hub()
    files = hub.list_repo_files(dataset_id, repo_type="dataset", revision=revision)
    prefix = f"{asset_dir}/"
    return sorted(
        Path(name).stem
        for name in files
        if name.startswith(prefix) and name.endswith(".parquet")
    )


def load_hub_symbol(
    symbol: str,
    dataset_id: str = DEFAULT_HF_DATASET_ID,
    revision: Optional[str] = None,
    asset_dir: str = STOCKS_DIR,
    adjust_prices: bool = True,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Download and normalize one symbol's parquet from the Hub.

    Args:
        symbol: Ticker, with or without the .NS suffix.
        dataset_id: Hub dataset repo id.
        revision: Optional pinned git revision.
        asset_dir: Asset directory to read from.
        adjust_prices: Back-adjust OHLC by adj_close/close.
        start_date: Inclusive lower bound (YYYY-MM-DD).
        end_date: Inclusive upper bound (YYYY-MM-DD).

    Returns:
        Normalized OHLCV frame, or None when the symbol is absent from the
        dataset or its file cannot be read. A missing symbol is an ordinary
        outcome for a universe list that runs ahead of the dataset, not an
        error worth aborting an ingest over.
    """
    hub = _require_hub()
    filename = f"{asset_dir}/{hub_symbol(symbol)}.parquet"

    try:
        local_path = hub.hf_hub_download(
            repo_id=dataset_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
        )
    except Exception:
        logger.debug("Symbol %s not available in %s", symbol, dataset_id, exc_info=True)
        return None

    try:
        raw = pd.read_parquet(local_path)
        df = normalize_frame(raw, adjust_prices=adjust_prices)
    except SchemaError as e:
        logger.warning("Skipping %s: %s", symbol, e)
        return None
    except Exception:
        logger.warning("Could not read %s from %s", filename, dataset_id, exc_info=True)
        return None

    if start_date is not None:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df.index <= pd.to_datetime(end_date)]

    return df if not df.empty else None


def sync_hf_to_cache(
    dataset_id: str = DEFAULT_HF_DATASET_ID,
    revision: Optional[str] = None,
    tickers: Optional[Sequence[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    asset_dir: str = STOCKS_DIR,
    adjust_prices: bool = True,
    min_rows: int = 2,
    max_symbols: Optional[int] = None,
    progress: bool = False,
    skip_existing: bool = True,
    workers: int = 8,
) -> List[str]:
    """Write Hub symbols into the platform's per-ticker parquet cache.

    Afterwards every existing code path — `resolve_backtest_universe()`,
    `load_ticker_data()`, the training panel — reads Hub data without knowing
    it came from anywhere new.

    Args:
        dataset_id: Hub dataset repo id.
        revision: Optional pinned git revision.
        tickers: Symbols to fetch. When None, every symbol in `asset_dir` is
            fetched (2,421 files for stocks).
        start_date: Inclusive lower bound (YYYY-MM-DD).
        end_date: Inclusive upper bound (YYYY-MM-DD).
        cache_dir: Parquet cache directory (defaults to data_store.DATA_DIR).
        asset_dir: Asset directory to read from.
        adjust_prices: Back-adjust OHLC by adj_close/close.
        min_rows: Skip symbols with fewer rows than this in the window; a
            two-bar series is not worth a cache entry that later looks like a
            real ticker.
        max_symbols: Cap on symbols fetched (applied after listing).
        progress: Print a progress line as symbols complete.
        skip_existing: Skip symbols already present in the parquet cache. On by
            default, because this used to re-fetch all ~2,400 files on every
            invocation — a full re-download of data already on disk, which is
            most of the wall-clock time of a "did that finish?" re-run. Pass
            False (`--force` on the CLI) to refresh the cache.
        workers: Thread pool size for fetching. Threads rather than processes:
            this is network- and disk-bound, so the GIL is released for the
            part that takes the time, and threads avoid re-importing pandas and
            pyarrow per worker — which on Windows, where processes are spawned
            rather than forked, is what turns a download into a paging storm.

    Returns:
        Sorted list of tickers written, in the cache's .NS form. Includes
        symbols that were already cached and therefore skipped, so the return
        value describes the cache rather than this run's network activity.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from .data_store import DATA_DIR, DataStore

    if tickers is None:
        symbols = list_hub_symbols(dataset_id, revision, asset_dir)
        logger.info("Dataset %s/%s lists %d symbols", dataset_id, asset_dir, len(symbols))
    else:
        symbols = [hub_symbol(t) for t in tickers]

    if max_symbols is not None:
        symbols = symbols[:max_symbols]

    store = DataStore(cache_dir=cache_dir or DATA_DIR)

    already_cached: List[str] = []
    if skip_existing:
        pending = []
        for symbol in symbols:
            ticker = normalize_ticker(symbol)
            if store.has_ticker_data(ticker):
                already_cached.append(ticker)
            else:
                pending.append(symbol)
        if already_cached and progress:
            print(
                f"  {len(already_cached)} symbol(s) already cached, "
                f"{len(pending)} to fetch (use --force to re-download)"
            )
        symbols = pending

    def _fetch(symbol: str) -> Optional[tuple]:
        """Fetch one symbol. Returns (ticker, frame) or None."""
        try:
            df = load_hub_symbol(
                symbol,
                dataset_id=dataset_id,
                revision=revision,
                asset_dir=asset_dir,
                adjust_prices=adjust_prices,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as error:  # one bad symbol must not abort the sync
            logger.warning("Failed to fetch %s: %s", symbol, error)
            return None
        if df is None or len(df) < min_rows:
            return None
        return normalize_ticker(symbol), df

    written: List[str] = []
    # Span and corporate-action counts, accumulated as symbols arrive. Reported
    # at the end because the history window is a configuration choice whose
    # effect is otherwise invisible: a five-year window looks identical to a
    # source that only holds five years, and the difference decides whether a
    # regime model has ever seen a crash.
    spans: List[pd.Timestamp] = []
    action_count = 0
    # Symbols the structural gate refused. Counted rather than raised: one
    # corrupt series out of 2,400 should be skipped and named, not abort a
    # download that is otherwise fine.
    rejected: List[str] = []

    if symbols:
        # Fetching is parallel; the parquet write is not. DataStore is not
        # documented as thread-safe, and serializing the writes on this thread
        # costs nothing next to the network round trip they are waiting on.
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
            futures = {executor.submit(_fetch, symbol): symbol for symbol in symbols}
            for done, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                if result is not None:
                    ticker, df = result
                    try:
                        # The gate. A frame whose high is below its close is not
                        # a price series, and caching it is worse than not
                        # caching it — every feature downstream consumes it
                        # silently, and the resulting NaN loss surfaces hundreds
                        # of training steps away from the cause.
                        assert_writable(df, ticker)
                    except IngestRejected as rejection:
                        logger.warning("%s", rejection)
                        rejected.append(ticker)
                        continue
                    store.save_ticker_data(ticker, df.copy())
                    written.append(ticker)
                    if len(df):
                        spans.extend([df.index.min(), df.index.max()])
                    try:
                        action_count += len(corporate_actions_from_frame(df))
                    except Exception:  # provenance is a nicety, never a failure
                        pass
                if progress and done % 100 == 0:
                    print(f"  {done}/{len(symbols)} fetched, {len(written)} cached")

    logger.info(
        "Cached %d newly fetched and %d already-present symbols from %s/%s",
        len(written), len(already_cached), dataset_id, asset_dir,
    )
    if rejected:
        logger.warning(
            "Refused to cache %d symbol(s) that failed a structural check: %s%s. "
            "Run `portfolio-agent data validate` for the detail.",
            len(rejected), ", ".join(rejected[:5]),
            " ..." if len(rejected) > 5 else "",
        )
        if progress:
            print(
                f"  {len(rejected)} symbol(s) refused as structurally invalid "
                f"(see `portfolio-agent data validate`)"
            )
    if spans:
        earliest, latest = min(spans), max(spans)
        years = (latest - earliest).days / 365.25
        logger.info(
            "History obtained: %s to %s (%.1f years), %d corporate actions recorded. "
            "If this is shorter than expected, raise data.default_history_years — the "
            "source is trimmed to that window on ingest.",
            earliest.date(), latest.date(), years, action_count,
        )
        if progress:
            print(
                f"  history {earliest.date()} -> {latest.date()} "
                f"({years:.1f} years), {action_count} corporate actions"
            )
    return sorted(set(written) | set(already_cached))


def load_benchmark_series(
    symbol: str,
    dataset_id: str = DEFAULT_HF_DATASET_ID,
    revision: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pd.Series]:
    """Load a market index's close series from the dataset's `indices/` directory.

    The market-regime filter (src/regime.py) falls back to an equal-weighted
    composite of the traded universe when no index is available, which is a
    reasonable proxy but not the thing the research actually describes. This
    dataset ships the real indices, so the filter can key off the Nifty itself.

    Args:
        symbol: Index symbol as named in the dataset's indices/ directory
            (e.g. "^NSEI" for the Nifty 50).
        dataset_id: Hub dataset repo id.
        revision: Optional pinned git revision.
        start_date: Inclusive lower bound (YYYY-MM-DD).
        end_date: Inclusive upper bound (YYYY-MM-DD).

    Returns:
        Close-price Series indexed by date, or None when unavailable.
    """
    df = load_hub_symbol(
        symbol,
        dataset_id=dataset_id,
        revision=revision,
        asset_dir=INDICES_DIR,
        start_date=start_date,
        end_date=end_date,
    )
    return None if df is None else df["close"]
