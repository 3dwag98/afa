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
    apply_cross_sectional_target,
    build_forward_return,
    evaluate_predictions,
    load_data,
    prepare_features,
    purge_and_embargo,
    run_walk_forward_validation,
    target_column_name,
    _generate_synthetic_ohlcv,
    _round_trip_cost,
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
        # The label is net of modelled round-trip friction, so the network
        # learns the sign of the move the portfolio actually keeps.
        expected = build_forward_return(
            df['close'], config.training.target, round_trip_cost=_round_trip_cost(config)
        )

        aligned = expected.reindex(feature_df.index)
        assert np.allclose(feature_df[target_col].values, aligned.values)

        # A forward return is by definition unknown at the decision date, so it
        # must differ from the same-named trailing feature.
        assert not np.allclose(
            feature_df[target_col].values, feature_df[config.training.target].values
        )

    def test_target_is_net_of_round_trip_friction(self):
        config = AppConfig()
        df = _make_ohlcv()

        gross = build_forward_return(df['close'], config.training.target)
        net = build_forward_return(
            df['close'], config.training.target, round_trip_cost=_round_trip_cost(config)
        )

        cost = _round_trip_cost(config)
        assert cost > 0
        assert np.allclose((gross - net).dropna().values, cost)

        config.training.target_net_of_costs = False
        assert _round_trip_cost(config) == 0.0


class TestCrossSectionalTarget:
    """Most of the variance of a 5-day equity return is the market factor,
    which this platform can neither forecast nor act on. Only the
    cross-sectional part is monetizable."""

    @staticmethod
    def _panel():
        idx = pd.bdate_range("2024-01-01", periods=6)
        return {
            "A": pd.DataFrame({"f": 1.0, "target_return_5d": [0.10, 0.02, -0.01, 0.05, 0.00, 0.03]}, index=idx),
            "B": pd.DataFrame({"f": 2.0, "target_return_5d": [0.09, 0.01, -0.02, 0.04, -0.01, 0.02]}, index=idx),
            "C": pd.DataFrame({"f": 3.0, "target_return_5d": [0.11, 0.03, 0.00, 0.06, 0.01, 0.04]}, index=idx),
        }

    def test_rank_target_maps_each_date_to_minus_one_to_one(self):
        out = apply_cross_sectional_target(self._panel(), "target_return_5d", "cross_sectional_rank")
        wide = pd.DataFrame({t: f["target_return_5d"] for t, f in out.items()})

        # C is the best name every day, B the worst: 2*rank/(N+1) - 1 for
        # N=3 gives exactly -0.5 / 0.0 / +0.5.
        assert np.allclose(wide["C"].values, 0.5)
        assert np.allclose(wide["A"].values, 0.0)
        assert np.allclose(wide["B"].values, -0.5)

    def test_demeaned_target_removes_the_common_component(self):
        out = apply_cross_sectional_target(self._panel(), "target_return_5d", "cross_sectional_demean")
        wide = pd.DataFrame({t: f["target_return_5d"] for t, f in out.items()})

        # A market-wide move is exactly what a demeaned target discards.
        assert np.allclose(wide.sum(axis=1).values, 0.0, atol=1e-12)

    def test_absolute_is_a_no_op_and_does_not_copy(self):
        panel = self._panel()
        assert apply_cross_sectional_target(panel, "target_return_5d", "absolute") is panel

    def test_target_stays_the_last_column(self):
        """create_dataloaders, the walk-forward splitter and the MLStrategy
        metadata all read the target positionally."""
        out = apply_cross_sectional_target(self._panel(), "target_return_5d", "cross_sectional_rank")
        for frame in out.values():
            assert frame.columns[-1] == "target_return_5d"

    def test_unknown_method_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown target transform"):
            apply_cross_sectional_target(self._panel(), "target_return_5d", "zscore")

    def test_single_ticker_panel_is_left_alone(self):
        panel = {"A": self._panel()["A"]}
        assert apply_cross_sectional_target(panel, "target_return_5d", "cross_sectional_rank") is panel


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

    def _panel_by_ticker(self, config, n=1000, n_tickers=2):
        """Per-ticker frames, each with its own DatetimeIndex."""
        return {
            f"SYM{i}.NS": prepare_features(
                _generate_synthetic_ohlcv(n, seed=i), config, verbose=False
            )
            for i in range(n_tickers)
        }

    def test_runs_every_fold_and_reports_the_benchmark_comparison(self):
        config = self._config()
        result = run_walk_forward_validation(
            self._panel_by_ticker(config), config, get_device("cpu")
        )

        assert result["n_folds"] == 3
        assert len(result["folds"]) == 3
        for key in ("mean_mse", "mean_directional_accuracy",
                    "mean_strategy_sharpe", "mean_benchmark_sharpe", "mean_excess_sharpe"):
            assert key in result

    def test_training_windows_expand_and_never_reach_the_test_period(self):
        config = self._config()
        result = run_walk_forward_validation(
            self._panel_by_ticker(config), config, get_device("cpu")
        )

        train_ends = [pd.Timestamp(f["train_end"]) for f in result["folds"]]
        test_ends = [pd.Timestamp(f["test_end"]) for f in result["folds"]]

        # Windows expand, and each fold's test period starts exactly where its
        # training window ends and finishes where the next fold's begins.
        assert train_ends == sorted(train_ends)
        assert len(set(train_ends)) == len(train_ends)
        for i, fold in enumerate(result["folds"]):
            assert test_ends[i] > train_ends[i]
            if i + 1 < len(result["folds"]):
                assert train_ends[i + 1] == test_ends[i]

    def test_training_rows_are_strictly_older_than_the_test_period(self):
        """The defect this replaced: an index split of the stacked panel put
        one ticker's 2019 rows in the 'future' test block while another
        ticker's 2019 rows trained the model."""
        config = self._config(walk_forward_splits=2)
        panel = self._panel_by_ticker(config)
        horizon = _target_horizon_days(config.training.target)

        result = run_walk_forward_validation(panel, config, get_device("cpu"))

        for fold in result["folds"]:
            train_end = pd.Timestamp(fold["train_end"])
            for frame in panel.values():
                history = frame[frame.index < train_end]
                # Every training row predates the boundary, and the embargo
                # removes the ones whose forward-return label reaches past it.
                assert len(history) >= fold["train_rows"] / len(panel) - horizon - 1

    def test_embargo_is_the_target_horizon(self):
        config = self._config()

        result = run_walk_forward_validation(
            self._panel_by_ticker(config), config, get_device("cpu")
        )

        assert result["embargo_days"] == _target_horizon_days(config.training.target)
        assert result["embargo_days"] == 5

    def test_disabled_by_zero_splits(self):
        config = self._config(walk_forward_splits=0)

        result = run_walk_forward_validation(
            self._panel_by_ticker(config), config, get_device("cpu")
        )

        assert "skipped" in result

    def test_empty_panel_is_skipped_rather_than_raising(self):
        config = self._config()

        assert "skipped" in run_walk_forward_validation({}, config, get_device("cpu"))

    def test_short_history_is_skipped_rather_than_raising(self):
        config = self._config(sequence_length=60, walk_forward_splits=5)
        panel = self._panel_by_ticker(config, n=400)

        result = run_walk_forward_validation(panel, config, get_device("cpu"))

        assert "skipped" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestPurgeAndEmbargo:
    """A row dated t carries a label computed from prices at t + horizon. If
    that window touches the test period, the label already encodes moves the
    model is about to be scored on."""

    @staticmethod
    def _frame(start="2024-01-01", periods=60):
        index = pd.bdate_range(start, periods=periods)
        return pd.DataFrame({"f": np.arange(len(index), dtype=float)}, index=index)

    def test_rows_inside_the_test_window_are_dropped(self):
        frame = self._frame()
        test_start, test_end = frame.index[20], frame.index[40]

        kept = purge_and_embargo(frame, test_start, test_end, horizon_days=0)

        assert not ((kept.index >= test_start) & (kept.index < test_end)).any()
        # Rows on both sides survive: this is a hole punched in the series,
        # not a truncation of it.
        assert (kept.index < test_start).any() and (kept.index >= test_end).any()

    def test_the_label_horizon_is_purged_before_the_window(self):
        frame = self._frame()
        test_start, test_end = frame.index[30], frame.index[45]

        kept = purge_and_embargo(frame, test_start, test_end, horizon_days=5)

        # Nothing kept before the window may have a 5-day label reaching into it.
        before = kept[kept.index < test_start]
        assert (before.index + pd.Timedelta(days=8)).max() < test_start

    def test_the_right_boundary_is_purged_too(self):
        """The defect this replaced: only the left boundary was handled. On a
        non-contiguous split, rows dated after the test window whose labels
        reach back into it leak just as badly."""
        frame = self._frame()
        test_start, test_end = frame.index[10], frame.index[25]

        kept = purge_and_embargo(frame, test_start, test_end, horizon_days=5)
        after = kept[kept.index >= test_end]

        # Every surviving row after the window starts at or past its end, and
        # rows dated at the boundary itself are kept only because their labels
        # run forward, away from the test period.
        assert len(after) > 0
        assert after.index.min() >= test_end

    def test_the_embargo_adds_a_gap_after_the_window(self):
        frame = self._frame()
        test_start, test_end = frame.index[10], frame.index[25]

        without = purge_and_embargo(frame, test_start, test_end, horizon_days=5)
        with_embargo = purge_and_embargo(
            frame, test_start, test_end, horizon_days=5, embargo_days=14
        )

        assert len(with_embargo) < len(without)
        after = with_embargo[with_embargo.index >= test_end]
        assert after.index.min() >= test_end + pd.Timedelta(days=14)

    def test_an_empty_frame_round_trips(self):
        empty = self._frame().iloc[:0]
        result = purge_and_embargo(empty, pd.Timestamp("2024-01-10"), pd.Timestamp("2024-02-01"), 5)
        assert result.empty

    def test_the_input_is_not_mutated(self):
        frame = self._frame()
        original = len(frame)
        purge_and_embargo(frame, frame.index[10], frame.index[25], horizon_days=5)
        assert len(frame) == original
