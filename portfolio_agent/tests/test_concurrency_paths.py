"""Concurrency in the download and live-agent paths must be speed-only.

Both paths gained parallelism: downloads run chunks on a thread pool
(network-bound), and the live agent prepares tickers on a process pool
(CPU-bound). Neither may change what the run produces.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import data_store as data_store_module
from data_store import DataStore
from portfolio_agent.config.schema import AppConfig
from src.monte_carlo import MonteCarloSettings
from portfolio_agent.execution.orchestrator import _prepare_all_tickers, _prepare_one_ticker


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    directory = tmp_path / "market_data"
    directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(data_store_module, "DATA_DIR", directory)
    return directory


def _fake_download_factory(record=None):
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    metrics = ["Open", "High", "Low", "Close", "Volume"]
    base = {"Open": 100.0, "High": 105.0, "Low": 99.0, "Close": 103.0, "Volume": 1000.0}

    def fake_download(tickers, start=None, end=None, **kwargs):
        symbols = tickers if isinstance(tickers, list) else [tickers]
        if record is not None:
            record.append(tuple(symbols))
        if len(symbols) == 1:
            return pd.DataFrame({m: [base[m]] * len(dates) for m in metrics}, index=dates)
        data = {(s, m): [base[m]] * len(dates) for s in symbols for m in metrics}
        df = pd.DataFrame(data, index=dates)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df

    return fake_download


class TestConcurrentDownloads:
    def test_threaded_and_serial_downloads_agree(self, cache_dir, monkeypatch):
        monkeypatch.setattr("yfinance.download", _fake_download_factory())
        tickers = [f"T{i}.NS" for i in range(12)]

        serial = DataStore(cache_dir=cache_dir / "serial").batch_download_and_cache(
            tickers, "2024-01-01", "2024-01-05", chunk_size=3,
            skip_existing=False, max_workers=1,
        )
        threaded = DataStore(cache_dir=cache_dir / "threaded").batch_download_and_cache(
            tickers, "2024-01-01", "2024-01-05", chunk_size=3,
            skip_existing=False, max_workers=4,
        )

        assert serial == threaded
        assert threaded['downloaded'] == len(tickers)
        assert threaded['failed'] == 0

    def test_every_ticker_is_written_to_the_cache(self, cache_dir, monkeypatch):
        monkeypatch.setattr("yfinance.download", _fake_download_factory())
        tickers = [f"T{i}.NS" for i in range(10)]

        store = DataStore(cache_dir=cache_dir)
        store.batch_download_and_cache(
            tickers, "2024-01-01", "2024-01-05", chunk_size=4,
            skip_existing=False, max_workers=3,
        )

        for ticker in tickers:
            assert (cache_dir / f"{ticker}.parquet").exists(), ticker

    def test_chunks_are_not_dropped_or_duplicated(self, cache_dir, monkeypatch):
        seen = []
        monkeypatch.setattr("yfinance.download", _fake_download_factory(seen))
        tickers = [f"T{i}.NS" for i in range(11)]

        DataStore(cache_dir=cache_dir).batch_download_and_cache(
            tickers, "2024-01-01", "2024-01-05", chunk_size=4,
            skip_existing=False, max_workers=4,
        )

        fetched = [t for chunk in seen for t in chunk]
        assert sorted(fetched) == sorted(tickers)

    def test_failed_chunk_is_reported_not_raised(self, cache_dir, monkeypatch):
        def exploding_download(tickers, start=None, end=None, **kwargs):
            raise RuntimeError("provider is down")

        monkeypatch.setattr("yfinance.download", exploding_download)
        monkeypatch.setattr(DataStore, "max_retries", 1, raising=False)

        stats = DataStore(cache_dir=cache_dir).batch_download_and_cache(
            ["A.NS", "B.NS"], "2024-01-01", "2024-01-05", chunk_size=2,
            skip_existing=False, max_workers=2,
        )

        assert stats['failed'] == 2
        assert sorted(stats['errors']) == ["A.NS", "B.NS"]


@pytest.fixture
def ticker_frames():
    np.random.seed(3)
    dates = pd.bdate_range("2022-01-03", periods=400)
    frames = {}
    for i in range(6):
        close = 100 + i * 10 + np.cumsum(np.random.normal(0.1, 1.0, len(dates)))
        frames[f"PREP{i}.NS"] = pd.DataFrame(
            {
                'open': close,
                'high': close * 1.01,
                'low': close * 0.99,
                'close': close,
                'volume': np.random.randint(100_000, 900_000, len(dates)).astype(float),
            },
            index=dates,
        )
    return frames


class TestParallelTickerPrep:
    """The live agent's per-ticker prep must be order-stable and identical."""

    FEATURES = ['close', 'sma_50', 'sma_200', 'donchian_upper_20',
                'volume_ratio_20', 'atr_14']

    def _config(self, parallel: bool) -> AppConfig:
        config = AppConfig()
        config.data.parallel_ticker_prep = parallel
        config.data.ticker_prep_workers = 2
        config.simulation.mc_simulations = 100
        return config

    def test_parallel_matches_serial(self, ticker_frames):
        logger = pytest.importorskip("logging").getLogger("test")

        serial = _prepare_all_tickers(
            ticker_frames, self.FEATURES, self._config(False), logger
        )
        parallel = _prepare_all_tickers(
            ticker_frames, self.FEATURES, self._config(True), logger
        )

        assert [r[0] for r in serial] == [r[0] for r in parallel]
        for (t1, _, mc1, f1), (t2, _, mc2, f2) in zip(serial, parallel):
            assert t1 == t2
            assert mc1 == mc2
            pd.testing.assert_frame_equal(f1, f2)

    def test_results_follow_input_order(self, ticker_frames):
        import logging

        prepared = _prepare_all_tickers(
            ticker_frames, self.FEATURES, self._config(True), logging.getLogger("test")
        )

        assert [r[0] for r in prepared] == list(ticker_frames)

    def test_unpreparable_ticker_is_skipped(self, ticker_frames):
        import logging

        frames = dict(ticker_frames)
        frames["BROKEN.NS"] = pd.DataFrame({'close': [1.0, 2.0]})

        prepared = _prepare_all_tickers(
            frames, self.FEATURES, self._config(True), logging.getLogger("test")
        )

        assert "BROKEN.NS" not in [r[0] for r in prepared]
        assert len(prepared) == len(ticker_frames)

    def test_single_ticker_helper_returns_none_on_failure(self):
        result = _prepare_one_ticker(
            "BAD.NS",
            pd.DataFrame({'close': [1.0]}),
            self.FEATURES,
            MonteCarloSettings(horizon_days=20, simulations=100, seed=42),
        )
        assert result is None
