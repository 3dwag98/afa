"""Tests for six defects found by actually running the platform.

Grouped in one file because they share a cause rather than a module: each is
somewhere the code worked on the developer's machine and failed on a real one —
a Windows box with a 6 GB GPU, a cache that was already populated, a UMA with
an untrained member.
"""

import sys

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.config.schema import AppConfig
from portfolio_agent.agents.trainer import _generate_synthetic_ohlcv, prepare_features
from portfolio_agent.src.universe import select_universe
from portfolio_agent.utils.workers import (
    MAX_PROCESS_WORKERS,
    MAX_PROCESS_WORKERS_WINDOWS,
    resolve_dataloader_workers,
    resolve_process_workers,
)


class TestTargetSanitization:
    """The real cause of NaN losses on CPU.

    Input features are standardized and clipped to +/-10 sigma before training.
    The *target* was passed through untouched, so a single bad cached bar could
    hand the optimizer a label of 111,300 and one gradient step against a loss
    that size moves the weights somewhere every later batch evaluates to NaN.
    That is why the mixed-precision fix did not cure it, and why it still
    appeared on CPU: the cause was the label, not fp16.
    """

    def _with_corrupt_bar(self, close_value):
        df = _generate_synthetic_ohlcv(600, seed=1)
        df.loc[df.index[300], "close"] = close_value
        return df

    def test_a_near_zero_close_used_to_produce_an_absurd_label(self):
        """The mechanism, stated so the regression is recognisable."""
        config = AppConfig()
        config.training.max_abs_target = 1e12  # effectively disable the guard

        features = prepare_features(self._with_corrupt_bar(0.001), config, verbose=False)
        target = features[features.columns[-1]].abs().max()

        assert target > 1000

    def test_the_guard_drops_the_poisoned_row(self):
        config = AppConfig()
        features = prepare_features(self._with_corrupt_bar(0.001), config, verbose=False)
        target = features[features.columns[-1]]

        assert target.abs().max() <= config.training.max_abs_target
        assert np.isfinite(target.to_numpy()).all()

    def test_genuinely_reachable_moves_are_kept(self):
        """Five consecutive 20% upper circuits compound to +149%, which is a
        real Indian small-cap outcome and must not be discarded as an error."""
        config = AppConfig()
        assert config.training.max_abs_target > 1.49

        df = _generate_synthetic_ohlcv(600, seed=2)
        # A genuine +140% move over the 5-day horizon.
        for offset, factor in enumerate([1.2, 1.2, 1.2, 1.2, 1.2], start=1):
            df.loc[df.index[300 + offset], "close"] = (
                df["close"].iloc[300 + offset - 1] * factor
            )

        features = prepare_features(df, config, verbose=False)
        assert features[features.columns[-1]].abs().max() > 1.0

    def test_rows_are_dropped_not_clipped(self):
        """Clipping would pile a spike of samples at the bound and teach the
        model that the bound is a common outcome."""
        config = AppConfig()
        features = prepare_features(self._with_corrupt_bar(0.001), config, verbose=False)
        target = features[features.columns[-1]]

        at_bound = (target.abs() == config.training.max_abs_target).sum()
        assert at_bound == 0

    def test_a_clean_series_loses_no_rows(self):
        config = AppConfig()
        clean = _generate_synthetic_ohlcv(600, seed=3)

        guarded = prepare_features(clean, config, verbose=False)
        config.training.max_abs_target = 1e12
        unguarded = prepare_features(clean, config, verbose=False)

        assert len(guarded) == len(unguarded)


class TestUniverseSelection:
    """An alphabetical truncation is a sample of the alphabet, not of the
    market — and it hands training and backtesting the identical names."""

    def _pool(self, n=2000):
        return [f"TICK{i:04d}.NS" for i in range(n)]

    def test_alphabetical_returns_the_front_of_the_cache(self):
        selected = select_universe(self._pool(), 50, "alphabetical")
        assert selected == [f"TICK{i:04d}.NS" for i in range(50)]

    def test_random_draws_from_across_the_cache(self):
        selected = select_universe(self._pool(), 50, "random", seed=42)

        assert len(selected) == 50
        assert len(set(selected)) == 50
        # A draw spread across 2,000 names, not the first 50.
        indices = [int(t[4:8]) for t in selected]
        assert max(indices) > 1000

    def test_training_and_backtesting_draw_different_names(self):
        """Evaluating a model on the very tickers it was fitted on is not
        out-of-sample in the cross-sectional dimension, however carefully the
        dates are split."""
        pool = self._pool()
        train = select_universe(pool, 50, "random", seed=42, purpose="train")
        backtest = select_universe(pool, 50, "random", seed=42, purpose="backtest")

        assert train != backtest
        assert len(set(train) & set(backtest)) < 10

    def test_the_same_config_reproduces_the_same_universe(self):
        """Seeded rather than truly random: two runs of one config must produce
        the same universe or nothing is reproducible."""
        pool = self._pool()
        first = select_universe(pool, 50, "random", seed=7, purpose="train")
        second = select_universe(pool, 50, "random", seed=7, purpose="train")

        assert first == second

    def test_reproducibility_survives_a_new_process(self):
        """hash() of a str is salted per process by PYTHONHASHSEED, so a
        purpose offset built on it would silently draw a different universe on
        every invocation. This is the regression guard for that."""
        import subprocess

        script = (
            "import sys; sys.path.insert(0, '.');"
            "from portfolio_agent.src.universe import select_universe;"
            "pool=[f'TICK{i:04d}.NS' for i in range(2000)];"
            "print(select_universe(pool, 5, 'random', 7, 'train'))"
        )
        runs = {
            subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, cwd=".",
            ).stdout.strip()
            for _ in range(2)
        }

        assert len(runs) == 1, f"universe differed between processes: {runs}"

    def test_a_seed_change_draws_a_different_sample(self):
        pool = self._pool()
        assert select_universe(pool, 50, "random", seed=1) != select_universe(
            pool, 50, "random", seed=2
        )

    def test_selection_is_sorted_regardless_of_draw_order(self):
        selected = select_universe(self._pool(), 50, "random", seed=42)
        assert selected == sorted(selected)

    def test_requesting_everything_returns_everything(self):
        pool = self._pool(30)
        for mode in ("alphabetical", "random"):
            assert len(select_universe(pool, None, mode)) == 30
            assert len(select_universe(pool, 0, mode)) == 30
            assert len(select_universe(pool, 500, mode)) == 30


class TestWorkerCaps:
    """On Windows every worker is a fresh interpreter that re-imports torch,
    so 'one per CPU' is how a 16 GB machine ends up in the page file."""

    def test_process_workers_are_capped(self):
        assert resolve_process_workers(1000) <= max(
            MAX_PROCESS_WORKERS, MAX_PROCESS_WORKERS_WINDOWS
        )
        assert resolve_process_workers(None) >= 1

    def test_an_explicit_request_is_still_capped(self):
        """A config written on a Linux box gets copied to a Windows one."""
        assert resolve_process_workers(64) < 64

    def test_a_smaller_explicit_request_is_honoured(self):
        assert resolve_process_workers(1) == 1

    def test_never_returns_zero_processes(self):
        assert resolve_process_workers(0) >= 1
        assert resolve_process_workers(-5) >= 1

    def test_dataloader_workers_are_zero_on_windows(self, monkeypatch):
        monkeypatch.setattr("portfolio_agent.utils.workers.is_windows", lambda: True)
        assert resolve_dataloader_workers(4) == 0
        assert resolve_dataloader_workers(None) == 0

    def test_dataloader_workers_are_used_where_fork_is_available(self, monkeypatch):
        monkeypatch.setattr("portfolio_agent.utils.workers.is_windows", lambda: False)
        assert resolve_dataloader_workers(None) == 2
        assert resolve_dataloader_workers(4) == 4
        assert resolve_dataloader_workers(99) == 4

    def test_windows_caps_processes_harder_than_posix(self):
        assert MAX_PROCESS_WORKERS_WINDOWS < MAX_PROCESS_WORKERS


class TestDownloadSkipsExistingData:
    """A plain re-run used to re-fetch all ~2,400 symbols already on disk."""

    def test_has_ticker_data_detects_a_cached_file(self, tmp_path):
        from portfolio_agent.src.data_store import DataStore

        store = DataStore(cache_dir=tmp_path)
        assert store.has_ticker_data("ABC.NS") is False

        frame = pd.DataFrame(
            {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]},
            index=pd.to_datetime(["2024-01-01"]),
        )
        store.save_ticker_data("ABC.NS", frame)

        assert store.has_ticker_data("ABC.NS") is True

    def test_a_zero_byte_file_does_not_count_as_cached(self, tmp_path):
        """A run interrupted mid-write leaves an empty file, and treating that
        as cached would permanently skip a ticker that never downloaded."""
        from portfolio_agent.src.data_store import DataStore

        store = DataStore(cache_dir=tmp_path)
        path = store._get_ticker_path("EMPTY.NS")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

        assert store.has_ticker_data("EMPTY.NS") is False

    def test_sync_skips_symbols_already_present(self, tmp_path, monkeypatch):
        import portfolio_agent.src.hf_dataset as hf

        fetched = []

        def fake_load_hub_symbol(symbol, **kwargs):
            fetched.append(symbol)
            return pd.DataFrame(
                {"open": [1.0] * 5, "high": [1.0] * 5, "low": [1.0] * 5,
                 "close": [1.0] * 5, "volume": [1.0] * 5},
                index=pd.date_range("2024-01-01", periods=5),
            )

        monkeypatch.setattr(hf, "load_hub_symbol", fake_load_hub_symbol)

        first = hf.sync_hf_to_cache(tickers=["AAA", "BBB"], cache_dir=tmp_path)
        assert len(fetched) == 2
        assert len(first) == 2

        # Second run: everything is cached, so nothing is fetched again.
        fetched.clear()
        second = hf.sync_hf_to_cache(tickers=["AAA", "BBB"], cache_dir=tmp_path)

        assert fetched == []
        assert sorted(second) == sorted(first)

    def test_force_refetches_everything(self, tmp_path, monkeypatch):
        import portfolio_agent.src.hf_dataset as hf

        fetched = []

        def fake_load_hub_symbol(symbol, **kwargs):
            fetched.append(symbol)
            return pd.DataFrame(
                {"open": [1.0] * 5, "high": [1.0] * 5, "low": [1.0] * 5,
                 "close": [1.0] * 5, "volume": [1.0] * 5},
                index=pd.date_range("2024-01-01", periods=5),
            )

        monkeypatch.setattr(hf, "load_hub_symbol", fake_load_hub_symbol)

        hf.sync_hf_to_cache(tickers=["AAA"], cache_dir=tmp_path)
        fetched.clear()
        hf.sync_hf_to_cache(tickers=["AAA"], cache_dir=tmp_path, skip_existing=False)

        assert fetched == ["AAA"]

    def test_a_failing_symbol_does_not_abort_the_sync(self, tmp_path, monkeypatch):
        """One bad symbol out of 2,400 must not lose the other 2,399."""
        import portfolio_agent.src.hf_dataset as hf

        def fake_load_hub_symbol(symbol, **kwargs):
            if symbol == "BAD":
                raise RuntimeError("hub is having a day")
            return pd.DataFrame(
                {"open": [1.0] * 5, "high": [1.0] * 5, "low": [1.0] * 5,
                 "close": [1.0] * 5, "volume": [1.0] * 5},
                index=pd.date_range("2024-01-01", periods=5),
            )

        monkeypatch.setattr(hf, "load_hub_symbol", fake_load_hub_symbol)

        written = hf.sync_hf_to_cache(
            tickers=["AAA", "BAD", "CCC"], cache_dir=tmp_path, workers=4
        )

        assert sorted(written) == ["AAA.NS", "CCC.NS"]

    def test_parallel_fetching_returns_the_same_set_as_serial(self, tmp_path, monkeypatch):
        """Threads are a performance change only."""
        import portfolio_agent.src.hf_dataset as hf

        def fake_load_hub_symbol(symbol, **kwargs):
            return pd.DataFrame(
                {"open": [1.0] * 5, "high": [1.0] * 5, "low": [1.0] * 5,
                 "close": [1.0] * 5, "volume": [1.0] * 5},
                index=pd.date_range("2024-01-01", periods=5),
            )

        monkeypatch.setattr(hf, "load_hub_symbol", fake_load_hub_symbol)
        symbols = [f"SYM{i}" for i in range(20)]

        serial = hf.sync_hf_to_cache(tickers=symbols, cache_dir=tmp_path / "a", workers=1)
        parallel = hf.sync_hf_to_cache(tickers=symbols, cache_dir=tmp_path / "b", workers=8)

        assert serial == parallel


class TestUnloadableUmaMembers:
    """'Strategy ensemble failed to load' names nothing a user can act on."""

    def test_the_message_names_the_failing_members_and_the_remedy(self):
        from portfolio_agent.agents.backtester import _load_failure_message

        class FakeUma:
            name = "meta_orchestrator"
            unloadable_members = ["lstm_forecaster"]

        message = _load_failure_message(FakeUma(), AppConfig())

        assert "lstm_forecaster" in message
        assert "portfolio-agent train" in message
        assert "drop_unavailable_members" in message

    def test_a_non_ensemble_strategy_still_gets_actionable_advice(self):
        from portfolio_agent.agents.backtester import _load_failure_message

        class FakeStrategy:
            name = "lstm"
            unloadable_members: list = []

        message = _load_failure_message(FakeStrategy(), AppConfig())

        assert "portfolio-agent train" in message
        assert "rule_based" in message
