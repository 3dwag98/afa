# --- Data: HuggingFace ingestion, cleaning, local cache -----------------------
# Self-contained. Nothing here imports the portfolio_agent package.

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

HF_DATASET_ID = "vishnun0027/indian-market-historical-ohlcv"
HF_REVISION: Optional[str] = None   # pin a commit sha for a reproducible run
STOCKS_DIR = "stocks"
INDICES_DIR = "indices"

CACHE_DIR = Path("data_cache")
OHLCV = ("open", "high", "low", "close", "volume")

# Dataset layout (2,471 files, ~283 MB): one parquet per symbol under
# stocks/ (2,421), indices/ (17), etfs/, commodities/, forex/, metadata/.
# Per-file schema: date, open, high, low, close, adj_close, volume,
# dividends, stock_splits, symbol.
#
# Downloading per symbol rather than snapshotting the repo is deliberate: a
# 30-name universe fetches 30 small files instead of 283 MB.


def hub_filename(symbol: str, asset_dir: str = STOCKS_DIR) -> str:
    """Path of a symbol's parquet inside the dataset repo.

    Index symbols (^NSEI) keep their caret and live under indices/; equities
    are stored bare, without the .NS suffix the exchange uses.
    """
    bare = str(symbol).strip().upper()
    if bare.endswith(".NS"):
        bare = bare[:-3]
    return f"{asset_dir}/{bare}.parquet"


def clean_ohlcv(raw: pd.DataFrame, adjust_prices: bool = True) -> pd.DataFrame:
    """Map one raw Hub parquet onto a canonical, analysis-ready OHLCV frame.

    Steps, each of which exists because skipping it produces a plausible-looking
    but wrong series:

    1. **Back-adjust by adj_close/close.** On a 1:10 split the raw close drops
       90% in a single print. Momentum reads that as a crash and ATR-derived
       stops blow out. Scaling all four price legs by the same per-row factor
       removes the discontinuity while leaving intraday relationships intact —
       a locked session (high == low) stays locked.
    2. **Coerce numerics.** A single object-dtype column silently poisons every
       rolling statistic computed from it.
    3. **Drop unparseable dates and missing closes** rather than forward-filling
       them, so a gap stays visible as a gap.
    4. **Drop duplicate dates**, keeping the last — re-exports occasionally
       carry a corrected duplicate of the same session.
    """
    if raw is None or len(raw) == 0:
        raise ValueError("empty frame")

    df = raw.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    if "date" not in df.columns:
        raise ValueError(f"no date column; got {list(df.columns)}")
    if "close" not in df.columns and "adj_close" not in df.columns:
        raise ValueError(f"no close column; got {list(df.columns)}")

    out = pd.DataFrame(index=pd.RangeIndex(len(df)))
    for column in ("open", "high", "low", "close", "adj_close", "volume"):
        if column in df.columns:
            out[column] = pd.to_numeric(df[column].to_numpy(), errors="coerce")

    if "close" not in out.columns:
        out["close"] = out["adj_close"]

    if adjust_prices and "adj_close" in out.columns:
        factor = (out["adj_close"] / out["close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        for leg in ("open", "high", "low", "close"):
            if leg in out.columns:
                out[leg] = out[leg] * factor

    out = out.drop(columns=[c for c in ("adj_close",) if c in out.columns])

    # A file carrying only a close is still usable for close-based signals;
    # a uniform column set matters more than the missing legs.
    for leg in ("open", "high", "low"):
        if leg not in out.columns:
            out[leg] = out["close"]
    # Volume defaults to 0, never to a fabricated number — a liquidity screen
    # reads it, and an invented volume defeats the screen entirely.
    if "volume" not in out.columns:
        out["volume"] = 0.0

    index = pd.to_datetime(df["date"].to_numpy(), errors="coerce", utc=True)
    out.index = pd.DatetimeIndex(index).tz_localize(None)
    out.index.name = "date"

    out = out[out.index.notna() & out["close"].notna()]
    if out.empty:
        raise ValueError("every row had an unparseable date or a missing close")

    out = out[~out.index.duplicated(keep="last")]
    return out[list(OHLCV)].sort_index()


def download_symbol(
    symbol: str,
    asset_dir: str = STOCKS_DIR,
    revision: Optional[str] = HF_REVISION,
    dataset_id: str = HF_DATASET_ID,
) -> Optional[pd.DataFrame]:
    """Fetch and clean one symbol. Returns None when the symbol is absent.

    A missing symbol is an ordinary outcome for a universe list that runs ahead
    of the dataset, not a reason to abort an ingest.
    """
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            repo_id=dataset_id,
            filename=hub_filename(symbol, asset_dir),
            repo_type="dataset",
            revision=revision,
        )
    except Exception:
        return None

    try:
        return clean_ohlcv(pd.read_parquet(path))
    except Exception as exc:
        warnings.warn(f"{symbol}: {exc}")
        return None


def list_hub_symbols(
    asset_dir: str = STOCKS_DIR,
    revision: Optional[str] = HF_REVISION,
    dataset_id: str = HF_DATASET_ID,
) -> List[str]:
    """Every symbol available in one asset directory."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(dataset_id, repo_type="dataset", revision=revision)
    prefix = f"{asset_dir}/"
    return sorted(
        Path(name).stem for name in files
        if name.startswith(prefix) and name.endswith(".parquet")
    )


def _synthetic_panel(
    symbols: Sequence[str], n_days: int = 1500, seed: int = 7
) -> Dict[str, pd.DataFrame]:
    """Offline stand-in with realistic structure, for when the Hub is unreachable.

    Deliberately *not* a plain random walk: it carries a common market factor,
    per-name idiosyncratic drift, volatility clustering and occasional gaps, so
    a cross-sectional strategy has something to rank and a risk model has
    something to estimate. It is still synthetic — any number produced from it
    describes the generator, not the market. Never read a result off this.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2018-01-01", periods=n_days, freq="B")

    market = rng.normal(0.0003, 0.010, n_days)
    vol_state = np.abs(rng.normal(1.0, 0.35, n_days))
    vol_state = pd.Series(vol_state).ewm(span=40).mean().to_numpy()

    panel: Dict[str, pd.DataFrame] = {}
    for i, symbol in enumerate(symbols):
        beta = 0.5 + 1.2 * rng.random()
        drift = rng.normal(0.0002, 0.0004)
        idio = rng.normal(0.0, 0.011 + 0.006 * rng.random(), n_days)
        returns = drift + beta * market * vol_state + idio * vol_state

        close = 100.0 * np.exp(np.cumsum(returns)) * (0.5 + 2.5 * rng.random())
        intraday = np.abs(rng.normal(0.008, 0.004, n_days))
        frame = pd.DataFrame(
            {
                "open": close * (1 + rng.normal(0, 0.003, n_days)),
                "high": close * (1 + intraday),
                "low": close * (1 - intraday),
                "close": close,
                "volume": rng.lognormal(13.0 + 0.4 * rng.random(), 0.6, n_days),
            },
            index=index,
        )
        frame["high"] = frame[["high", "open", "close"]].max(axis=1)
        frame["low"] = frame[["low", "open", "close"]].min(axis=1)

        # A few missing sessions, so the cleaning path is actually exercised.
        drop = rng.choice(n_days, size=max(1, n_days // 200), replace=False)
        frame = frame.drop(frame.index[drop])

        panel[symbol] = frame
        _ = i
    return panel


def load_panel(
    symbols: Sequence[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    cache_dir: Path = CACHE_DIR,
    use_cache: bool = True,
    allow_synthetic: bool = True,
    revision: Optional[str] = HF_REVISION,
) -> Dict[str, pd.DataFrame]:
    """Load a cleaned OHLCV panel, downloading only what is not already cached.

    Falls back to a synthetic panel when the Hub cannot be reached, so the rest
    of a notebook still runs offline. The fallback is announced loudly, because
    a synthetic result that is mistaken for a real one is the worst outcome here.

    Returns:
        `{symbol: DataFrame}` for every symbol that yielded usable data.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    panel: Dict[str, pd.DataFrame] = {}
    missing: List[str] = []

    for symbol in symbols:
        path = cache_dir / f"{symbol.replace('^', '_')}.parquet"
        if use_cache and path.exists():
            panel[symbol] = pd.read_parquet(path)
        else:
            missing.append(symbol)

    if missing:
        try:
            for symbol in missing:
                asset_dir = INDICES_DIR if symbol.startswith("^") else STOCKS_DIR
                frame = download_symbol(symbol, asset_dir=asset_dir, revision=revision)
                if frame is None:
                    continue
                panel[symbol] = frame
                frame.to_parquet(cache_dir / f"{symbol.replace('^', '_')}.parquet")
        except Exception as exc:
            if not allow_synthetic:
                raise
            print("=" * 74)
            print("HuggingFace unreachable — falling back to SYNTHETIC data.")
            print(f"  reason: {type(exc).__name__}: {str(exc)[:160]}")
            print("  Every number below describes the generator, not the market.")
            print("  Install `huggingface_hub` and re-run with network access for real data.")
            print("=" * 74)
            panel = _synthetic_panel(list(symbols))

    if not panel and allow_synthetic:
        print("=" * 74)
        print("No symbols resolved — falling back to SYNTHETIC data. Results are not real.")
        print("=" * 74)
        panel = _synthetic_panel(list(symbols))

    if start_date or end_date:
        lo = pd.Timestamp(start_date) if start_date else None
        hi = pd.Timestamp(end_date) if end_date else None
        for symbol, frame in list(panel.items()):
            if lo is not None:
                frame = frame[frame.index >= lo]
            if hi is not None:
                frame = frame[frame.index <= hi]
            if frame.empty:
                panel.pop(symbol)
            else:
                panel[symbol] = frame

    return panel


def panel_quality(panel: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-symbol data-quality summary — read this before trusting a backtest."""
    rows = []
    for symbol, frame in sorted(panel.items()):
        close = frame["close"]
        returns = close.pct_change()
        sessions = len(frame)
        expected = len(pd.date_range(frame.index.min(), frame.index.max(), freq="B"))
        rows.append({
            "symbol": symbol,
            "rows": sessions,
            "start": frame.index.min().date(),
            "end": frame.index.max().date(),
            "coverage": round(sessions / max(expected, 1), 3),
            "zero_vol_days": int((frame["volume"] <= 0).sum()),
            "flat_days": int((returns.abs() < 1e-9).sum()),
            "max_1d_move": round(float(returns.abs().max()), 4),
            "ann_vol": round(float(returns.std() * np.sqrt(252)), 4),
        })
    return pd.DataFrame(rows)


def align_close_matrix(panel: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide close-price matrix (dates x symbols) on the union of trading days.

    Left as NaN where a symbol has no print. Forward-filling here would invent
    prices a backtest could trade on; the strategies below handle NaN by simply
    not ranking a name that has none.
    """
    return pd.DataFrame({s: f["close"] for s, f in panel.items()}).sort_index()
