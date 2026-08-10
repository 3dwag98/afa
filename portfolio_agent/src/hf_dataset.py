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
}

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


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

    if adjust_prices and "adj_close" in out.columns:
        # Back-adjust every price leg by the same per-row factor. Scaling all
        # four together is what keeps intraday relationships intact: a locked
        # session (high == low) stays locked, and ATR keeps its proportion to
        # price. Volume is left as reported — the liquidity screen reads only
        # the trailing 60 sessions, where the adjustment factor is ~1.
        factor = (out["adj_close"] / out["close"]).replace([float("inf"), float("-inf")], pd.NA)
        factor = factor.fillna(1.0)
        for leg in ("open", "high", "low", "close"):
            if leg in out.columns:
                out[leg] = out[leg] * factor

    out = out.drop(columns=[c for c in ("adj_close",) if c in out.columns])

    # A file with only a close is still usable for every close-based signal;
    # keeping the column set uniform matters more than the missing legs.
    for leg in ("open", "high", "low"):
        if leg not in out.columns:
            out[leg] = out["close"]
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
    return out[list(OHLCV_COLUMNS)].sort_index()


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
        progress: Print a progress line every 100 symbols.

    Returns:
        Sorted list of tickers written, in the cache's .NS form.
    """
    try:
        from .data_store import DATA_DIR, DataStore
    except ImportError:  # pragma: no cover - script-style import path
        from data_store import DATA_DIR, DataStore

    if tickers is None:
        symbols = list_hub_symbols(dataset_id, revision, asset_dir)
        logger.info("Dataset %s/%s lists %d symbols", dataset_id, asset_dir, len(symbols))
    else:
        symbols = [hub_symbol(t) for t in tickers]

    if max_symbols is not None:
        symbols = symbols[:max_symbols]

    store = DataStore(cache_dir=cache_dir or DATA_DIR)
    written: List[str] = []
    for i, symbol in enumerate(symbols, start=1):
        df = load_hub_symbol(
            symbol,
            dataset_id=dataset_id,
            revision=revision,
            asset_dir=asset_dir,
            adjust_prices=adjust_prices,
            start_date=start_date,
            end_date=end_date,
        )
        if df is None or len(df) < min_rows:
            continue
        ticker = normalize_ticker(symbol)
        store.save_ticker_data(ticker, df.copy())
        written.append(ticker)

        if progress and i % 100 == 0:
            print(f"  {i}/{len(symbols)} symbols processed, {len(written)} cached")

    logger.info("Cached %d/%d symbols from %s/%s", len(written), len(symbols), dataset_id, asset_dir)
    return sorted(written)


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
