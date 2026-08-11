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
    apply_cost_to_target,
    apply_cross_sectional_target,
    build_forward_return,
    estimate_round_trip_cost,
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


def _with_flag(config: AppConfig, cost_adjusted: bool) -> AppConfig:
    """A copy of `config` with the cost-adjusted-label flag set."""
    updated = config.model_copy(deep=True)
    updated.training.cost_adjusted_target = cost_adjusted
    return updated


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
        config.training.cost_adjusted_target = False  # compare against the gross label
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

        def fake_resolve_backtest_universe(max_tickers=None, **kwargs):
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
        monkeypatch.setattr("portfolio_agent.agents.trainer.resolve_backtest_universe", lambda max_tickers=None, **kwargs: [])

        config = AppConfig.model_validate({"training": {"use_synthetic_data": False}})

        with pytest.raises(RuntimeError, match="No cached tickers"):
            load_data(config)


class TestParallelDataLoading:
    """Tests for the multiprocessing training-panel construction path."""

    def test_parallel_loading_matches_serial_loading(self, monkeypatch):
        """parallel_data_loading=True is a performance change only — it must
        build an equivalent panel (same tickers contribute, same row count)
        as the serial path."""

        def fake_resolve_backtest_universe(max_tickers=None, **kwargs):
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


    def test_overlapping_labels_widen_the_t_statistic_denominator(self):
        """A daily-sampled 5-day return shares 4 days with its neighbour.
        Treating the observations as independent understates the standard error
        by roughly sqrt(H), in the direction that manufactures significance.
        """
        rng = np.random.default_rng(0)
        daily = rng.normal(0.0004, 0.01, size=2000)
        overlapping = pd.Series(daily).rolling(5).sum().dropna().to_numpy()
        predictions = np.ones_like(overlapping)

        metrics = evaluate_predictions(predictions, overlapping, horizon_days=5)

        naive_t = np.mean(overlapping) / (
            np.std(overlapping, ddof=0) / math.sqrt(len(overlapping))
        )
        assert metrics["strategy_t_stat"] != 0.0
        assert abs(metrics["strategy_t_stat"]) < abs(naive_t)

    def test_a_relative_target_reports_no_t_statistic(self):
        """A rank is not a return, so neither is its mean."""
        actuals = np.array([0.5, -0.5, 0.25, -0.25, 1.0, -1.0])
        metrics = evaluate_predictions(actuals, actuals, horizon_days=5, relative_target=True)

        assert metrics["strategy_t_stat"] == 0.0
        assert metrics["rank_ic"] == pytest.approx(1.0)

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
        """An absolute return target is in return units, so the Sharpe-style
        comparison against always-long is the meaningful headline."""
        config = self._config(target_transform="absolute")
        result = run_walk_forward_validation(
            self._panel_by_ticker(config), config, get_device("cpu")
        )

        assert result["n_folds"] == 3
        assert len(result["folds"]) == 3
        for key in ("mean_mse", "mean_directional_accuracy",
                    "mean_strategy_sharpe", "mean_benchmark_sharpe", "mean_excess_sharpe"):
            assert key in result

    def test_a_relative_target_is_scored_on_rank_ic_not_sharpe(self):
        """A cross-sectional rank is not a return: +0.4 is a position in the
        ordering, not 40%. Reporting a Sharpe on it would be a confident number
        about the wrong quantity, so rank IC carries the evaluation instead."""
        config = self._config()
        assert config.training.target_transform == "cross_sectional_rank"

        result = run_walk_forward_validation(
            self._panel_by_ticker(config), config, get_device("cpu")
        )

        assert result["target_transform"] == "cross_sectional_rank"
        for key in ("mean_rank_ic", "rank_icir", "folds_with_positive_ic"):
            assert key in result
        for key in ("mean_strategy_sharpe", "mean_benchmark_sharpe", "mean_excess_sharpe"):
            assert key not in result
        assert all(-1.0 <= fold["rank_ic"] <= 1.0 for fold in result["folds"])

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



class TestCrossSectionalTarget:
    """The neural stack was aimed at the wrong quantity.

    Most of the variance of a 5-day equity return is the common market factor,
    which a long-only book with no index hedge cannot act on. Measuring the
    label against the cross-section leaves the idiosyncratic part — the only
    component the platform monetizes by choosing between stocks.
    """

    def _panel(self, values_by_date):
        """Build a tiny panel from {ticker: {date: target}}."""
        dates = pd.date_range("2024-01-01", periods=len(next(iter(values_by_date.values()))))
        return {
            ticker: pd.DataFrame(
                {"feature": np.arange(len(values), dtype=float), "target_return_5d": values},
                index=dates,
            )
            for ticker, values in values_by_date.items()
        }

    def test_rank_target_maps_the_cross_section_onto_minus_one_to_one(self):
        panel = self._panel({
            "A": [0.10, -0.05], "B": [0.05, 0.00],
            "C": [0.00, 0.05], "D": [-0.05, 0.10], "E": [-0.10, 0.15],
        })

        out = apply_cross_sectional_target(panel, "target_return_5d", "cross_sectional_rank")

        # Five names: ranks 1..5 map to 2*r/6 - 1 = -2/3, -1/3, 0, 1/3, 2/3.
        first_day = sorted(out[t]["target_return_5d"].iloc[0] for t in out)
        assert first_day == pytest.approx([-2 / 3, -1 / 3, 0.0, 1 / 3, 2 / 3])
        # The best performer on day one is the worst on day two.
        assert out["A"]["target_return_5d"].iloc[0] == pytest.approx(2 / 3)
        assert out["A"]["target_return_5d"].iloc[1] == pytest.approx(-2 / 3)

    def test_demeaned_target_removes_the_common_move(self):
        """A day when every name rose 10% carries no cross-sectional signal,
        and an absolute target would teach the model that it did."""
        panel = self._panel({
            "A": [0.10], "B": [0.10], "C": [0.10], "D": [0.10], "E": [0.10],
        })

        out = apply_cross_sectional_target(panel, "target_return_5d", "cross_sectional_demean")

        for ticker in out:
            assert out[ticker]["target_return_5d"].iloc[0] == pytest.approx(0.0)

    def test_rank_is_immune_to_a_circuit_limited_outlier(self):
        """Why rank beats demeaning on Indian data: one +20% upper-circuit
        print drags the cross-sectional mean and every other name's label with
        it, but moves the ranking by nothing at all."""
        base = {"A": [0.01], "B": [0.02], "C": [0.03], "D": [0.04], "E": [0.05]}
        shocked = dict(base, E=[0.20])

        ranks_base = apply_cross_sectional_target(
            self._panel(base), "target_return_5d", "cross_sectional_rank"
        )
        ranks_shocked = apply_cross_sectional_target(
            self._panel(shocked), "target_return_5d", "cross_sectional_rank"
        )
        demeaned_base = apply_cross_sectional_target(
            self._panel(base), "target_return_5d", "cross_sectional_demean"
        )
        demeaned_shocked = apply_cross_sectional_target(
            self._panel(shocked), "target_return_5d", "cross_sectional_demean"
        )

        for ticker in "ABCD":
            assert ranks_base[ticker]["target_return_5d"].iloc[0] == pytest.approx(
                ranks_shocked[ticker]["target_return_5d"].iloc[0]
            )
            assert demeaned_base[ticker]["target_return_5d"].iloc[0] != pytest.approx(
                demeaned_shocked[ticker]["target_return_5d"].iloc[0]
            )

    def test_uses_only_labels_dated_at_the_same_decision_point(self):
        """No look-ahead beyond what the forward return already carries.

        Changing what happens on day two must not alter any day-one label.
        """
        panel = self._panel({
            "A": [0.10, -0.05], "B": [0.05, 0.00], "C": [0.00, 0.05],
            "D": [-0.05, 0.10], "E": [-0.10, 0.15],
        })
        altered = self._panel({
            "A": [0.10, 9.99], "B": [0.05, -9.99], "C": [0.00, 9.99],
            "D": [-0.05, -9.99], "E": [-0.10, 9.99],
        })

        out = apply_cross_sectional_target(panel, "target_return_5d", "cross_sectional_rank")
        out_altered = apply_cross_sectional_target(
            altered, "target_return_5d", "cross_sectional_rank"
        )

        for ticker in out:
            assert out[ticker]["target_return_5d"].iloc[0] == pytest.approx(
                out_altered[ticker]["target_return_5d"].iloc[0]
            )

    def test_drops_dates_with_too_thin_a_cross_section(self):
        """Ranking three names on a day the rest of the universe has no history
        encodes which tickers were listed, not which ones outperformed."""
        dates = pd.date_range("2024-01-01", periods=3)
        panel = {
            ticker: pd.DataFrame(
                {"feature": [1.0, 2.0, 3.0], "target_return_5d": [np.nan, np.nan, 0.01 * i]},
                index=dates,
            )
            for i, ticker in enumerate("ABCDE")
        }
        # Only the last date has all five names.
        panel["A"].loc[dates[1], "target_return_5d"] = 0.02

        out = apply_cross_sectional_target(panel, "target_return_5d", "cross_sectional_rank")

        for frame in out.values():
            assert list(frame.index) == [dates[2]]

    def test_leaves_the_panel_alone_for_an_absolute_target(self):
        panel = self._panel({"A": [0.10], "B": [0.05]})
        assert apply_cross_sectional_target(panel, "target_return_5d", "absolute") is panel

    def test_a_single_ticker_cannot_be_ranked_against_anything(self):
        panel = self._panel({"A": [0.10, 0.20]})
        assert apply_cross_sectional_target(panel, "target_return_5d") is panel

    def test_rejects_an_unknown_transform(self):
        panel = self._panel({"A": [0.1], "B": [0.2]})
        with pytest.raises(ValueError, match="unknown target transform"):
            apply_cross_sectional_target(panel, "target_return_5d", "sideways")

    def test_preserves_features_and_column_order(self):
        """The target must stay the last column: everything downstream — the
        walk-forward splitter, the loaders, the checkpoint metadata — reads it
        positionally."""
        panel = self._panel({
            "A": [0.10], "B": [0.05], "C": [0.0], "D": [-0.05], "E": [-0.10],
        })

        out = apply_cross_sectional_target(panel, "target_return_5d", "cross_sectional_rank")

        for ticker, frame in out.items():
            assert list(frame.columns) == ["feature", "target_return_5d"]
            assert frame["feature"].to_numpy() == pytest.approx(
                panel[ticker]["feature"].to_numpy()
            )



if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestCostAdjustedTarget:
    """The model should learn the sign of the move the portfolio keeps.

    A gross forward return is a move the book never receives: brokerage, STT on
    both legs, exchange and SEBI charges, GST, stamp duty and the bid-ask
    spread all come out of it first.
    """

    def test_the_net_label_is_below_the_gross_one(self):
        config = AppConfig()
        df = _make_ohlcv()

        gross = prepare_features(df, _with_flag(config, False), verbose=False)
        net = prepare_features(df, _with_flag(config, True), verbose=False)

        target_col = target_column_name(config.training.target)
        shared = gross.index.intersection(net.index)
        assert len(shared) > 0
        assert (net.loc[shared, target_col] < gross.loc[shared, target_col]).all()

    def test_the_charge_is_the_modelled_round_trip(self):
        """Not an arbitrary haircut: the same rate schedule execution_sim
        charges realized fills, applied to both legs."""
        df = _make_ohlcv()
        costs = estimate_round_trip_cost(df)
        gross = build_forward_return(df['close'], "return_5d")

        net = apply_cost_to_target(gross, costs)

        expected = (1.0 + gross) * (1.0 - costs['sell']) / (1.0 + costs['buy']) - 1.0
        pd.testing.assert_series_equal(net.dropna(), expected.dropna())

    def test_both_legs_carry_statutory_costs_and_only_the_buy_leg_stamp_duty(self):
        from portfolio_agent.src.execution_sim import ExecutionSimulator

        df = _make_ohlcv()
        costs = estimate_round_trip_cost(df).dropna()

        assert (costs['buy'] > costs['sell']).all()
        assert np.allclose(
            (costs['buy'] - costs['sell']).to_numpy(),
            ExecutionSimulator.STAMP_DUTY_RATE,
        )

    def test_slippage_scales_with_the_name_s_own_range(self):
        """This is the part that survives a cross-sectional target. A constant
        cost is a level shift, and demeaning subtracts it back out while
        ranking is invariant to it — so if the charge did not vary by name,
        cost-adjusting the label under the default transform would do nothing
        at all."""
        calm = _make_ohlcv(seed=3)
        wild = calm.copy()
        # Same closes, four times the intraday range.
        wild['high'] = wild['close'] + (calm['high'] - calm['close']) * 4
        wild['low'] = wild['close'] - (calm['close'] - calm['low']) * 4

        calm_costs = estimate_round_trip_cost(calm)['buy'].dropna()
        wild_costs = estimate_round_trip_cost(wild)['buy'].dropna()

        assert (wild_costs > calm_costs.reindex(wild_costs.index)).all()

    def test_a_cross_sectional_rank_target_still_moves(self):
        """The end-to-end statement of the point above: after ranking within
        each date, the cost-adjusted panel is not the gross panel."""
        config = AppConfig()
        target_col = target_column_name(config.training.target)

        def _panel(cost_adjusted: bool):
            frames = {}
            for i, ticker in enumerate(["A.NS", "B.NS", "C.NS", "D.NS", "E.NS", "F.NS"]):
                df = _make_ohlcv(seed=20 + i)
                if i % 2:  # half the names are wide-spread
                    df['high'] = df['close'] + (df['high'] - df['close']) * 6
                    df['low'] = df['close'] - (df['close'] - df['low']) * 6
                frames[ticker] = prepare_features(
                    df, _with_flag(config, cost_adjusted), verbose=False
                )
            return apply_cross_sectional_target(frames, target_col, "cross_sectional_rank")

        gross = _panel(False)
        net = _panel(True)

        differences = [
            not np.allclose(
                gross[t][target_col].reindex(net[t].index).dropna().to_numpy(),
                net[t][target_col].reindex(gross[t].index).dropna().to_numpy(),
            )
            for t in gross
            if t in net
        ]
        assert any(differences), "the cost charge must survive cross-sectional ranking"

    def test_the_adjustment_can_be_switched_off(self):
        config = AppConfig()
        df = _make_ohlcv()
        target_col = target_column_name(config.training.target)

        feature_df = prepare_features(df, _with_flag(config, False), verbose=False)
        expected = build_forward_return(df['close'], config.training.target)

        assert np.allclose(
            feature_df[target_col].to_numpy(),
            expected.reindex(feature_df.index).to_numpy(),
        )
