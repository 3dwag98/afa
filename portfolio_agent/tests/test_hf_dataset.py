"""Tests for the HuggingFace Hub OHLCV source.

The Hub is never contacted: `huggingface_hub` is stubbed, so these exercise the
parts that can actually go wrong locally — schema mapping, split/dividend
back-adjustment, date-window trimming, ticker normalization, and the write into
the parquet cache.
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.config.schema import AppConfig
from src.hf_dataset import (
    DEFAULT_HF_DATASET_ID,
    SchemaError,
    hub_symbol,
    list_hub_symbols,
    load_benchmark_series,
    load_hub_symbol,
    normalize_frame,
    normalize_ticker,
    sync_hf_to_cache,
)


def _hub_rows(n=6, symbol="RELIANCE", start=100.0, split_at=None):
    """A file in the dataset's real schema.

    date, open, high, low, close, adj_close, volume, dividends, stock_splits,
    symbol — all lowercase, one file per symbol.

    `split_at` inserts a 1:10 split at that row: the raw close drops 90% while
    adj_close stays on the continuous, back-adjusted path.
    """
    dates = pd.bdate_range("2024-01-01", periods=n)
    closes = np.array([start + i for i in range(n)], dtype=float)
    adj = closes.copy()
    splits = np.zeros(n)

    if split_at is not None:
        closes[split_at:] = closes[split_at:] / 10.0
        splits[split_at] = 10.0
        # adj_close is continuous: pre-split prices are divided by the ratio.
        adj = closes.copy()
        adj[:split_at] = adj[:split_at] / 10.0

    return pd.DataFrame({
        "date": dates.date,
        "open": closes - 1.0,
        "high": closes + 2.0,
        "low": closes - 2.0,
        "close": closes,
        "adj_close": adj,
        "volume": np.arange(n, dtype="int64") + 1_000_000,
        "dividends": np.zeros(n),
        "stock_splits": splits,
        "symbol": [symbol] * n,
    })


def _install_fake_hub(monkeypatch, tmp_path, files, recorder=None):
    """Stub huggingface_hub with a local file map: repo path -> DataFrame."""
    written = {}
    for repo_path, frame in files.items():
        local = tmp_path / repo_path.replace("/", "__")
        frame.to_parquet(local, index=False)
        written[repo_path] = str(local)

    module = types.ModuleType("huggingface_hub")

    def hf_hub_download(repo_id, filename, repo_type=None, revision=None):
        if recorder is not None:
            recorder.append({
                "repo_id": repo_id, "filename": filename,
                "repo_type": repo_type, "revision": revision,
            })
        if filename not in written:
            raise FileNotFoundError(filename)
        return written[filename]

    def list_repo_files(repo_id, repo_type=None, revision=None):
        return list(written)

    module.hf_hub_download = hf_hub_download
    module.list_repo_files = list_repo_files
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return written


class TestSymbolNaming:
    def test_cache_form_adds_the_nse_suffix(self):
        assert normalize_ticker("reliance") == "RELIANCE.NS"
        assert normalize_ticker("TCS.NS") == "TCS.NS"

    def test_hub_form_strips_it(self):
        """Files in the dataset are named RELIANCE.parquet, not RELIANCE.NS."""
        assert hub_symbol("RELIANCE.NS") == "RELIANCE"
        assert hub_symbol("reliance") == "RELIANCE"
        assert hub_symbol("^NSEI") == "^NSEI"


class TestNormalizeFrame:
    def test_maps_the_datasets_own_schema(self):
        out = normalize_frame(_hub_rows())

        # OHLCV still leads the frame, so nothing downstream moves.
        assert list(out.columns[:5]) == ["open", "high", "low", "close", "volume"]
        assert isinstance(out.index, pd.DatetimeIndex)
        assert out.index.name == "date"
        assert out.index.tz is None

    def test_keeps_the_adjustment_provenance(self):
        """This test used to assert the opposite, and the opposite was a bug.

        `adj_close`, `dividends` and `stock_splits` are the source's own record
        of every corporate action, and dropping them meant the platform threw
        away data it had already downloaded. Two things became impossible as a
        result: recovering the price that actually traded (needed for anything
        band- or level-related, since a back-adjusted price is not a price any
        exchange saw), and knowing that a large return was a split rather than
        a move. The second is the worse one — an unexplained 90% gap was
        silently discarded by the label filter instead of being recognised.

        `symbol` is still dropped: the ticker is the filename, and carrying it
        per row only invites the two to disagree.
        """
        out = normalize_frame(_hub_rows())

        for kept in ("adj_close", "adj_factor", "dividends", "stock_splits"):
            assert kept in out.columns
        for raw_leg in ("open_raw", "high_raw", "low_raw", "close_raw"):
            assert raw_leg in out.columns

        assert "symbol" not in out.columns

    def test_back_adjusts_a_split_out_of_the_return_series(self):
        """Unadjusted, a 1:10 split prints as a -90% day, which cross-sectional
        momentum reads as a crash and the circuit-lock detector reads as a
        limit move."""
        raw = _hub_rows(n=10, split_at=5)

        unadjusted = normalize_frame(raw, adjust_prices=False)
        adjusted = normalize_frame(raw, adjust_prices=True)

        assert unadjusted["close"].pct_change().min() < -0.85
        assert adjusted["close"].pct_change().min() > -0.10

    def test_adjustment_scales_every_leg_together(self):
        """High == low must survive adjustment, or the circuit-lock detector
        stops working on adjusted history."""
        raw = _hub_rows(n=10, split_at=5)
        raw["high"] = raw["close"]
        raw["low"] = raw["close"]

        out = normalize_frame(raw)

        assert np.allclose(out["high"], out["low"])

    def test_adjustment_leaves_an_unsplit_series_alone(self):
        raw = _hub_rows(n=6)

        assert np.allclose(
            normalize_frame(raw)["close"], normalize_frame(raw, adjust_prices=False)["close"]
        )

    def test_volume_is_reported_as_is(self):
        out = normalize_frame(_hub_rows(n=4))

        assert out["volume"].tolist() == [1_000_000, 1_000_001, 1_000_002, 1_000_003]

    def test_missing_volume_defaults_to_zero_not_a_fabricated_number(self):
        """The liquidity screen reads volume; inventing one would defeat it."""
        raw = _hub_rows().drop(columns=["volume"])

        assert (normalize_frame(raw)["volume"] == 0).all()

    def test_missing_ohl_legs_fall_back_to_close(self):
        raw = _hub_rows().drop(columns=["open", "high", "low"])

        out = normalize_frame(raw)

        assert (out["open"] == out["close"]).all()

    def test_adj_close_alone_is_enough(self):
        raw = _hub_rows().drop(columns=["close"])

        assert normalize_frame(raw)["close"].notna().all()

    def test_duplicate_dates_keep_the_last_row(self):
        raw = _hub_rows(n=3)
        duplicate = raw.iloc[[0]].copy()
        duplicate["close"] = 999.0
        duplicate["adj_close"] = 999.0

        out = normalize_frame(pd.concat([raw, duplicate], ignore_index=True))

        assert len(out) == 3
        assert out["close"].iloc[0] == pytest.approx(999.0)

    def test_drops_unparseable_dates_and_missing_closes(self):
        raw = _hub_rows(n=6)
        raw["date"] = raw["date"].astype(object)
        raw.loc[0, "date"] = "not-a-date"
        raw.loc[1, "close"] = None
        raw.loc[1, "adj_close"] = None

        assert len(normalize_frame(raw)) == 4

    def test_unmappable_schema_raises_naming_the_columns_it_saw(self):
        with pytest.raises(SchemaError) as exc:
            normalize_frame(pd.DataFrame({"foo": [1], "bar": [2]}))

        assert "foo" in str(exc.value) and "bar" in str(exc.value)

    def test_missing_close_and_adj_close_raises(self):
        raw = _hub_rows().drop(columns=["close", "adj_close"])

        with pytest.raises(SchemaError, match="no close column"):
            normalize_frame(raw)

    def test_empty_frame_raises(self):
        with pytest.raises(SchemaError):
            normalize_frame(pd.DataFrame())


class TestLoadHubSymbol:
    def test_reads_one_symbols_file(self, monkeypatch, tmp_path):
        calls = []
        _install_fake_hub(monkeypatch, tmp_path, {"stocks/RELIANCE.parquet": _hub_rows()}, calls)

        df = load_hub_symbol("RELIANCE.NS")

        assert df is not None and len(df) == 6
        assert calls[0]["filename"] == "stocks/RELIANCE.parquet"
        assert calls[0]["repo_type"] == "dataset"
        assert calls[0]["repo_id"] == DEFAULT_HF_DATASET_ID

    def test_pins_the_revision(self, monkeypatch, tmp_path):
        calls = []
        _install_fake_hub(monkeypatch, tmp_path, {"stocks/TCS.parquet": _hub_rows(symbol="TCS")}, calls)

        load_hub_symbol("TCS", revision="v1.2.3")

        assert calls[0]["revision"] == "v1.2.3"

    def test_absent_symbol_returns_none_rather_than_raising(self, monkeypatch, tmp_path):
        """A universe list running ahead of the dataset is ordinary, not an
        error worth aborting a 2,400-ticker ingest over."""
        _install_fake_hub(monkeypatch, tmp_path, {"stocks/TCS.parquet": _hub_rows(symbol="TCS")})

        assert load_hub_symbol("NOSUCH") is None

    def test_unreadable_file_returns_none(self, monkeypatch, tmp_path):
        _install_fake_hub(monkeypatch, tmp_path, {"stocks/BAD.parquet": pd.DataFrame({"x": [1]})})

        assert load_hub_symbol("BAD") is None

    def test_trims_to_the_requested_window(self, monkeypatch, tmp_path):
        _install_fake_hub(monkeypatch, tmp_path, {"stocks/TCS.parquet": _hub_rows(n=10, symbol="TCS")})

        df = load_hub_symbol("TCS", start_date="2024-01-03", end_date="2024-01-05")

        assert df.index.min() >= pd.Timestamp("2024-01-03")
        assert df.index.max() <= pd.Timestamp("2024-01-05")

    def test_reads_from_other_asset_directories(self, monkeypatch, tmp_path):
        _install_fake_hub(monkeypatch, tmp_path, {"indices/^NSEI.parquet": _hub_rows(symbol="^NSEI")})

        assert load_hub_symbol("^NSEI", asset_dir="indices") is not None

    def test_missing_hub_package_explains_the_fix(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)

        with pytest.raises(RuntimeError, match="huggingface_hub"):
            load_hub_symbol("TCS")


class TestListHubSymbols:
    def test_lists_only_the_requested_asset_directory(self, monkeypatch, tmp_path):
        _install_fake_hub(monkeypatch, tmp_path, {
            "stocks/TCS.parquet": _hub_rows(symbol="TCS"),
            "stocks/INFY.parquet": _hub_rows(symbol="INFY"),
            "indices/^NSEI.parquet": _hub_rows(symbol="^NSEI"),
        })

        assert list_hub_symbols() == ["INFY", "TCS"]
        assert list_hub_symbols(asset_dir="indices") == ["^NSEI"]


class TestLoadBenchmarkSeries:
    def test_returns_the_index_close_series(self, monkeypatch, tmp_path):
        _install_fake_hub(monkeypatch, tmp_path, {"indices/^NSEI.parquet": _hub_rows(symbol="^NSEI")})

        series = load_benchmark_series("^NSEI")

        assert series is not None
        assert isinstance(series, pd.Series)
        assert len(series) == 6

    def test_returns_none_when_the_index_is_absent(self, monkeypatch, tmp_path):
        _install_fake_hub(monkeypatch, tmp_path, {"stocks/TCS.parquet": _hub_rows(symbol="TCS")})

        assert load_benchmark_series("^NSEI") is None


class TestSyncHfToCache:
    def test_writes_parquet_the_normal_loader_can_read(self, monkeypatch, tmp_path):
        hub_dir = tmp_path / "hub"
        hub_dir.mkdir()
        cache_dir = tmp_path / "cache"
        _install_fake_hub(monkeypatch, hub_dir, {
            "stocks/TCS.parquet": _hub_rows(n=8, symbol="TCS"),
            "stocks/INFY.parquet": _hub_rows(n=8, symbol="INFY"),
        })

        written = sync_hf_to_cache(cache_dir=cache_dir)

        assert written == ["INFY.NS", "TCS.NS"]

        import src.data_store as data_store

        monkeypatch.setattr(data_store, "DATA_DIR", cache_dir)
        df = data_store.load_ticker_data("TCS.NS")
        assert df is not None and len(df) == 8
        assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)

    def test_fetches_only_the_requested_tickers(self, monkeypatch, tmp_path):
        hub_dir = tmp_path / "hub"
        hub_dir.mkdir()
        calls = []
        _install_fake_hub(monkeypatch, hub_dir, {
            "stocks/TCS.parquet": _hub_rows(symbol="TCS"),
            "stocks/INFY.parquet": _hub_rows(symbol="INFY"),
        }, calls)

        sync_hf_to_cache(cache_dir=tmp_path / "cache", tickers=["TCS.NS"])

        # One file downloaded, not the whole repo — the point of a per-symbol layout.
        assert [c["filename"] for c in calls] == ["stocks/TCS.parquet"]

    def test_max_symbols_caps_the_ingest(self, monkeypatch, tmp_path):
        hub_dir = tmp_path / "hub"
        hub_dir.mkdir()
        _install_fake_hub(monkeypatch, hub_dir, {
            f"stocks/SYM{i}.parquet": _hub_rows(symbol=f"SYM{i}") for i in range(5)
        })

        assert len(sync_hf_to_cache(cache_dir=tmp_path / "cache", max_symbols=2)) == 2

    def test_skips_symbols_with_too_few_rows(self, monkeypatch, tmp_path):
        hub_dir = tmp_path / "hub"
        hub_dir.mkdir()
        _install_fake_hub(monkeypatch, hub_dir, {"stocks/TCS.parquet": _hub_rows(n=1, symbol="TCS")})

        assert sync_hf_to_cache(cache_dir=tmp_path / "cache", min_rows=2) == []

    def test_honours_the_date_window(self, monkeypatch, tmp_path):
        hub_dir = tmp_path / "hub"
        hub_dir.mkdir()
        cache_dir = tmp_path / "cache"
        _install_fake_hub(monkeypatch, hub_dir, {"stocks/TCS.parquet": _hub_rows(n=10, symbol="TCS")})

        sync_hf_to_cache(cache_dir=cache_dir, start_date="2024-01-03", end_date="2024-01-05")

        import src.data_store as data_store

        monkeypatch.setattr(data_store, "DATA_DIR", cache_dir)
        assert len(data_store.load_ticker_data("TCS.NS")) == 3


class TestFetchAndCacheSourceSelection:
    """config.data.source is the single branch deciding where bars come from,
    used by both the CLI and the live agent's missing-ticker top-up."""

    def test_huggingface_source_uses_the_hub(self, monkeypatch):
        import src.data_store as data_store

        calls = []
        monkeypatch.setattr(
            "src.hf_dataset.sync_hf_to_cache",
            lambda **kwargs: calls.append(kwargs) or ["TCS.NS"],
        )

        config = AppConfig()
        config.data.source = "huggingface"
        ok = data_store.fetch_and_cache(
            config, ["TCS.NS"], start_date="2020-01-01", end_date="2025-01-01"
        )

        assert ok is True
        assert calls[0]["dataset_id"] == config.data.hf_dataset_id
        assert calls[0]["asset_dir"] == "stocks"
        assert calls[0]["adjust_prices"] is True
        assert calls[0]["start_date"] == "2020-01-01"

    def test_yfinance_source_uses_the_download_path(self, monkeypatch):
        import src.data_store as data_store

        calls = []
        monkeypatch.setattr(
            data_store, "batch_download_and_cache",
            lambda *args, **kwargs: calls.append((args, kwargs)) or True,
        )

        config = AppConfig()
        config.data.source = "yfinance"
        data_store.fetch_and_cache(
            config, ["TCS.NS"], start_date="2020-01-01", end_date="2025-01-01"
        )

        assert len(calls) == 1

    def test_hub_failure_falls_back_to_yfinance(self, monkeypatch):
        """A run must not end with no data because the Hub was unreachable —
        but the switch is logged, since silently changing source mid-experiment
        is how two 'identical' backtests end up disagreeing."""
        import src.data_store as data_store

        def boom(**kwargs):
            raise RuntimeError("hub unreachable")

        monkeypatch.setattr("src.hf_dataset.sync_hf_to_cache", boom)
        fallback = []
        monkeypatch.setattr(
            data_store, "batch_download_and_cache",
            lambda *args, **kwargs: fallback.append(args) or True,
        )

        config = AppConfig()
        config.data.source = "huggingface"
        ok = data_store.fetch_and_cache(
            config, ["TCS.NS"], start_date="2020-01-01", end_date="2025-01-01"
        )

        assert ok is True
        assert len(fallback) == 1

    def test_tickers_absent_from_the_dataset_are_reported(self, monkeypatch):
        import src.data_store as data_store

        monkeypatch.setattr("src.hf_dataset.sync_hf_to_cache", lambda **kwargs: ["TCS.NS"])

        config = AppConfig()
        config.data.source = "huggingface"
        ok = data_store.fetch_and_cache(
            config, ["TCS.NS", "NOSUCH.NS"], start_date="2020-01-01", end_date="2025-01-01"
        )

        assert ok is False


class TestDefaults:
    def test_config_points_at_the_indian_market_dataset(self):
        data = AppConfig().data

        assert data.source == "huggingface"
        assert data.hf_dataset_id == DEFAULT_HF_DATASET_ID
        assert data.hf_asset_dir == "stocks"
        assert data.hf_adjust_prices is True
        assert data.benchmark_symbol == "^NSEI"

    def test_history_window_is_long_enough_to_contain_a_crisis(self):
        """The window was five years, and the cost was invisible.

        Every cached file spanned exactly five years, so the sample began
        *after* the COVID crash: one bull run, one rate-hike correction, no
        crisis. Every tail estimate, regime model and drawdown forecast was
        fitted on data containing no crash. The source is trimmed to whatever
        it actually holds, so asking for more than exists costs nothing — which
        makes a short window pure downside.

        The bound below is deliberately loose. It pins the intent (long enough
        to reach more than one regime) without pinning a number someone would
        have to update to raise it further.
        """
        assert AppConfig().data.default_history_years >= 15


class TestBenchmarkCacheRoundTrip:
    """The benchmark is only useful if the name it is written under is the
    name every reader asks for."""

    def test_index_is_cached_under_the_symbol_readers_use(self, monkeypatch, tmp_path):
        hub_dir = tmp_path / "hub"
        hub_dir.mkdir()
        cache_dir = tmp_path / "cache"
        _install_fake_hub(monkeypatch, hub_dir, {
            "indices/^NSEI.parquet": _hub_rows(n=8, symbol="^NSEI"),
        })

        written = sync_hf_to_cache(
            cache_dir=cache_dir, tickers=["^NSEI"], asset_dir="indices"
        )

        # Not "^NSEI.NS": config.data.benchmark_symbol is read verbatim, so a
        # suffixed file would be written and then never found.
        assert written == ["^NSEI"]

        import src.data_store as data_store

        monkeypatch.setattr(data_store, "DATA_DIR", cache_dir)
        assert data_store.load_ticker_data("^NSEI") is not None

    def test_the_index_does_not_become_a_tradeable_ticker(self, monkeypatch, tmp_path):
        """Indices share the equity cache so the regime filter can read them.
        Letting one into the universe would rank the Nifty against individual
        stocks by momentum and queue orders against it."""
        hub_dir = tmp_path / "hub"
        hub_dir.mkdir()
        cache_dir = tmp_path / "cache"
        _install_fake_hub(monkeypatch, hub_dir, {
            "stocks/TCS.parquet": _hub_rows(n=8, symbol="TCS"),
            "indices/^NSEI.parquet": _hub_rows(n=8, symbol="^NSEI"),
        })

        sync_hf_to_cache(cache_dir=cache_dir, tickers=["TCS"], asset_dir="stocks")
        sync_hf_to_cache(cache_dir=cache_dir, tickers=["^NSEI"], asset_dir="indices")

        import src.data_store as data_store

        assert data_store.get_cached_tickers(cache_dir) == ["TCS.NS"]
