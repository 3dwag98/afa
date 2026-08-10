"""Tests for the training pipeline (agents/trainer.py).

Focus: load_data() must load real cached tickers by default (the historical
bug being fixed here was that it always returned synthetic data), and must
fall back to synthetic data only when explicitly requested.
"""

import math

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.agents.trainer import (
    build_forward_return,
    evaluate_predictions,
    load_data,
    prepare_features,
    run_walk_forward_validation,
    target_column_name,
    _generate_synthetic_ohlcv,
    _target_horizon_days,
)
from portfolio_agent.utils.device import get_device
from portfolio_agent.config.schema import AppConfig


def _make_ohlcv(n_days: int = 300, seed: int = 1) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    close = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
    return pd.DataFrame({
        'open': close + np.random.randn(n_days) * 0.1,
        'high': close + np.abs(np.random.randn(n_days)) * 0.3,
        'low': close - np.abs(np.random.randn(n_days)) * 0.3,
        'close': close,
        'volume': np.random.randint(100000, 1000000, n_days).astype(float),
    }, index=dates)


class TestPrepareFeatures:
    def test_builds_features_and_target(self):
        config = AppConfig()
        df = _make_ohlcv()
        feature_df = prepare_features(df, config, verbose=False)

        # The target is namespaced away from the features: config.training.target
        # defaults to "return_5d", which is ALSO a registered feature. Before the
        # split, the trailing return silently became the label and the model was
        # trained to reproduce an input rather than forecast anything.
        assert target_column_name(config.training.target) == feature_df.columns[-1]
        assert config.training.target in feature_df.columns[:-1]
        assert len(feature_df) > 0
        assert not feature_df.isna().any().any()

    def test_target_is_a_forward_return_not_a_trailing_one(self):
        config = AppConfig()
        df = _make_ohlcv()
        feature_df = prepare_features(df, config, verbose=False)

        target_col = target_column_name(config.training.target)
        expected = build_forward_return(df['close'], config.training.target)

        aligned = expected.reindex(feature_df.index)
        assert np.allclose(feature_df[target_col].values, aligned.values)
        # A forward return is by definition unknown at the decision date, so it
        # must differ from the same-named trailing feature.
        assert not np.allclose(
            feature_df[target_col].values, feature_df[config.training.target].values
        )


class TestLoadDataUsesRealDataByDefault:
    """Regression test for the historical bug: load_data() always returned
    synthetic random-walk data instead of real cached tickers."""

    def test_synthetic_flag_returns_synthetic_data(self):
        config = AppConfig.model_validate({"training": {"use_synthetic_data": True}})
        df = load_data(config)
        assert len(df) > 0
        assert target_column_name(config.training.target) in df.columns

    def test_default_loads_real_cached_tickers(self, monkeypatch, tmp_path):
        """With use_synthetic_data unset (default False), load_data() must
        call into the real data store rather than fabricating data."""
        calls = []

        def fake_resolve_backtest_universe(max_tickers=None):
            calls.append(max_tickers)
            return ["FAKE1.NS", "FAKE2.NS"]

        def fake_load_ticker_data(ticker, start_date=None, end_date=None):
            return _make_ohlcv(seed=hash(ticker) % 1000)

        monkeypatch.setattr("portfolio_agent.agents.trainer.resolve_backtest_universe", fake_resolve_backtest_universe)
        monkeypatch.setattr("portfolio_agent.agents.trainer.load_ticker_data", fake_load_ticker_data)

        config = AppConfig.model_validate({
            "training": {"use_synthetic_data": False, "sequence_length": 10, "parallel_data_loading": False},
            "data": {"universe_size": 5, "min_history_days": 50},
        })

        df = load_data(config)

        assert calls, "load_data() must call resolve_backtest_universe() when not using synthetic data"
        assert len(df) > 0
        assert target_column_name(config.training.target) in df.columns

    def test_raises_when_no_cached_tickers_available(self, monkeypatch):
        monkeypatch.setattr("portfolio_agent.agents.trainer.resolve_backtest_universe", lambda max_tickers=None: [])

        config = AppConfig.model_validate({"training": {"use_synthetic_data": False}})

        with pytest.raises(RuntimeError, match="No cached tickers"):
            load_data(config)


class TestParallelDataLoading:
    """Tests for the multiprocessing training-panel construction path."""

    def test_parallel_loading_matches_serial_loading(self, monkeypatch):
        """parallel_data_loading=True is a performance change only — it must
        build an equivalent panel (same tickers contribute, same row count)
        as the serial path."""

        def fake_resolve_backtest_universe(max_tickers=None):
            return ["FAKE1.NS", "FAKE2.NS", "FAKE3.NS"]

        def fake_load_ticker_data(ticker, start_date=None, end_date=None):
            return _make_ohlcv(seed=hash(ticker) % 1000)

        monkeypatch.setattr("portfolio_agent.agents.trainer.resolve_backtest_universe", fake_resolve_backtest_universe)
        monkeypatch.setattr("portfolio_agent.agents.trainer.load_ticker_data", fake_load_ticker_data)

        base = {
            "training": {"use_synthetic_data": False, "sequence_length": 10},
            "data": {"universe_size": 5, "min_history_days": 50},
        }

        serial_config = AppConfig.model_validate({**base, "training": {**base["training"], "parallel_data_loading": False}})
        parallel_config = AppConfig.model_validate({**base, "training": {**base["training"], "parallel_data_loading": True}})

        serial_df = load_data(serial_config)
        parallel_df = load_data(parallel_config)

        assert len(serial_df) == len(parallel_df)
        assert list(serial_df.columns) == list(parallel_df.columns)


class TestEvaluatePredictions:
    """Out-of-sample scoring on the terms a trader cares about, not just MSE."""

    def test_perfect_foresight_beats_the_always_long_benchmark(self):
        actuals = np.array([0.02, -0.03, 0.01, -0.01, 0.04, -0.02])

        metrics = evaluate_predictions(actuals, actuals, horizon_days=5)

        assert metrics["directional_accuracy"] == 1.0
        assert metrics["excess_sharpe"] > 0
        assert metrics["strategy_sharpe"] > metrics["benchmark_sharpe"]

    def test_always_positive_prediction_reproduces_the_benchmark(self):
        """Predicting 'up' every day IS the always-long benchmark, so it can
        have no excess Sharpe — the guard against a model that looks good only
        because the market went up."""
        actuals = np.array([0.02, -0.03, 0.01, -0.01, 0.04, -0.02])
        predictions = np.ones_like(actuals)

        metrics = evaluate_predictions(predictions, actuals, horizon_days=5)

        assert metrics["strategy_sharpe"] == pytest.approx(metrics["benchmark_sharpe"])
        assert metrics["excess_sharpe"] == pytest.approx(0.0)

    def test_inverted_predictions_score_below_the_benchmark(self):
        actuals = np.array([0.02, -0.03, 0.01, -0.01, 0.04, -0.02])

        metrics = evaluate_predictions(-actuals, actuals, horizon_days=5)

        assert metrics["directional_accuracy"] == 0.0
        assert metrics["excess_sharpe"] < 0

    def test_sharpe_annualization_follows_the_horizon(self):
        actuals = np.array([0.02, -0.01, 0.03, -0.005, 0.01, 0.02])

        daily = evaluate_predictions(actuals, actuals, horizon_days=1)
        weekly = evaluate_predictions(actuals, actuals, horizon_days=5)

        assert daily["strategy_sharpe"] == pytest.approx(
            weekly["strategy_sharpe"] * math.sqrt(5), rel=1e-6
        )

    def test_non_finite_values_are_dropped(self):
        predictions = np.array([0.01, np.nan, 0.02, np.inf])
        actuals = np.array([0.01, 0.02, 0.03, 0.04])

        metrics = evaluate_predictions(predictions, actuals)

        assert metrics["n_samples"] == 2

    def test_empty_and_mismatched_inputs_return_zeros(self):
        assert evaluate_predictions(np.array([]), np.array([]))["n_samples"] == 0
        assert evaluate_predictions(np.array([1.0]), np.array([1.0, 2.0]))["n_samples"] == 0


class TestTargetHorizon:
    def test_parses_the_horizon_from_the_target_name(self):
        assert _target_horizon_days("return_5d") == 5
        assert _target_horizon_days("return_21d") == 21

    def test_defaults_to_one_day_for_unparseable_names(self):
        assert _target_horizon_days("next_close") == 1


class TestWalkForwardValidation:
    """Expanding-window folds are what turn 'it worked on 2023' into evidence."""

    @staticmethod
    def _config(**training):
        config = AppConfig()
        config.training.use_synthetic_data = True
        config.training.sequence_length = 10
        config.training.batch_size = 32
        config.training.walk_forward_splits = 3
        config.training.walk_forward_epochs = 1
        for key, value in training.items():
            setattr(config.training, key, value)
        return config

    def _panel(self, config, n=1000):
        return prepare_features(_generate_synthetic_ohlcv(n), config, verbose=False)

    def test_runs_every_fold_and_reports_the_benchmark_comparison(self):
        config = self._config()
        result = run_walk_forward_validation(
            self._panel(config), config, get_device("cpu")
        )

        assert result["n_folds"] == 3
        assert len(result["folds"]) == 3
        for key in ("mean_mse", "mean_directional_accuracy",
                    "mean_strategy_sharpe", "mean_benchmark_sharpe", "mean_excess_sharpe"):
            assert key in result

    def test_training_windows_expand_and_never_overlap_their_test_block(self):
        config = self._config()
        result = run_walk_forward_validation(
            self._panel(config), config, get_device("cpu")
        )

        train_rows = [f["train_rows"] for f in result["folds"]]
        assert train_rows == sorted(train_rows)
        assert len(set(train_rows)) == len(train_rows)
        # Each fold's training window ends exactly where its test block starts.
        for previous, current in zip(result["folds"], result["folds"][1:]):
            assert previous["train_rows"] + previous["test_rows"] == current["train_rows"]

    def test_disabled_by_zero_splits(self):
        config = self._config(walk_forward_splits=0)

        result = run_walk_forward_validation(
            self._panel(config), config, get_device("cpu")
        )

        assert "skipped" in result

    def test_short_panel_is_skipped_rather_than_raising(self):
        config = self._config(sequence_length=60, walk_forward_splits=5)
        panel = self._panel(config, n=400)

        result = run_walk_forward_validation(panel, config, get_device("cpu"))

        assert "skipped" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
