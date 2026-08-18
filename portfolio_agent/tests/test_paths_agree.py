"""The three paths must read the same numbers on the same date.

`evaluate`, `backtest` and `train` each build their own feature panel. They are
supposed to be three views of one decision: on date D, with history through D,
what does this strategy think? Nothing enforced that, and they had drifted —
the backtest sliced `df.index < D` while the evaluation harness sliced
`frame.loc[:D]`, so the identical strategy on the identical date read inputs
one full session apart.

That is the expensive kind of divergence. Neither number is obviously wrong;
they are both plausible; and the platform reports one as the forecast skill of
the thing the other trades.

The tests here are deliberately about *equality between paths*, not about
whether any single path is right. A convention can be argued; a disagreement
cannot be, and a disagreement is what silently invalidates a comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.features.pipeline import build_features

FEATURES = ["sma_20", "rsi_14", "atr_14", "return_5d", "realized_vol_60", "close"]


@pytest.fixture
def ohlcv():
    """One ticker, long enough for every feature above to have warmed up."""
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2021-01-04", periods=500)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, len(index))))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, len(index))),
            "high": close * (1 + np.abs(rng.normal(0, 0.006, len(index)))),
            "low": close * (1 - np.abs(rng.normal(0, 0.006, len(index)))),
            "close": close,
            "volume": rng.integers(1e5, 1e6, len(index)).astype(float),
        },
        index=index,
    )


def backtest_style(frame: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """Slice first, then build — what `BacktestEngine` does every date."""
    history = frame[frame.index <= date]
    return build_features(history, FEATURES).iloc[-1]


def evaluation_style(frame: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """Build once over full history, then slice — what the harness does.

    `harness.py` builds features across the whole cache and hands each strategy
    `frame.loc[:date]`. That is a large speedup and it is only sound if it
    produces the same numbers as rebuilding per date.
    """
    return build_features(frame, FEATURES).loc[:date].iloc[-1]


# --------------------------------------------------------------------------
# The equality
# --------------------------------------------------------------------------


class TestBacktestAndEvaluationAgree:
    def test_the_feature_vector_is_identical_on_the_decision_date(self, ohlcv):
        """The assertion that would have caught T19's off-by-one."""
        date = ohlcv.index[400]
        pd.testing.assert_series_equal(
            backtest_style(ohlcv, date), evaluation_style(ohlcv, date)
        )

    @pytest.mark.parametrize("offset", [300, 350, 400, 450, 499])
    def test_it_holds_across_the_sample(self, ohlcv, offset):
        date = ohlcv.index[offset]
        pd.testing.assert_series_equal(
            backtest_style(ohlcv, date), evaluation_style(ohlcv, date)
        )

    def test_slicing_one_session_earlier_really_does_change_the_answer(self, ohlcv):
        """The divergence is material, not a rounding difference.

        Recorded so nobody re-introduces the exclusive slice on the grounds
        that it 'can't matter much'. On this fixture the 20-day moving average
        moves by a visible amount and `close` moves by a whole session.
        """
        date = ohlcv.index[400]
        inclusive = backtest_style(ohlcv, date)
        exclusive = build_features(
            ohlcv[ohlcv.index < date], FEATURES
        ).iloc[-1]

        assert inclusive["close"] != exclusive["close"]
        assert inclusive["sma_20"] != exclusive["sma_20"]


class TestTheEngineUsesTheSharedConvention:
    """The equality above is about `build_features`; this is about the engine."""

    def _engine(self, monkeypatch, ohlcv):
        from portfolio_agent.src import backtest_engine as module

        monkeypatch.setattr(
            module, "load_ticker_data",
            lambda ticker, start_date=None, end_date=None: ohlcv.copy(),
        )
        return module.BacktestEngine(
            start_date="2022-06-01", end_date="2022-12-30",
            initial_capital=1_000_000.0, universe_tickers=["A.NS", "B.NS"],
        )

    def test_history_through_includes_the_decision_date(self, monkeypatch, ohlcv):
        engine = self._engine(monkeypatch, ohlcv)
        date = engine.master_date_index[100]

        history = engine._history_through("A.NS", date)
        assert history is not None
        assert history.index.max() == date

    def test_it_excludes_everything_after(self, monkeypatch, ohlcv):
        engine = self._engine(monkeypatch, ohlcv)
        date = engine.master_date_index[100]

        history = engine._history_through("A.NS", date)
        assert not (history.index > date).any()

    def test_the_engine_slice_matches_the_harness_slice(self, monkeypatch, ohlcv):
        """End to end: engine slice -> features == harness build -> slice."""
        engine = self._engine(monkeypatch, ohlcv)
        date = engine.master_date_index[100]

        engine_features = build_features(
            engine._history_through("A.NS", date), FEATURES
        ).iloc[-1]
        pd.testing.assert_series_equal(
            engine_features, evaluation_style(ohlcv, date)
        )

    def test_the_benchmark_slices_are_the_same_length(self, monkeypatch, ohlcv):
        """A close series one bar longer than its own high/low range would let
        `assess_market_regime` read a trend from one session and an ADX from
        the one before it."""
        engine = self._engine(monkeypatch, ohlcv)
        engine.benchmark_close = ohlcv["close"]
        engine.benchmark_ohlcv = ohlcv
        date = engine.master_date_index[100]

        closes = engine._benchmark_up_to(date)
        bars = engine._benchmark_ohlcv_up_to(date)
        assert len(closes) == len(bars)
        assert closes.index.max() == bars.index.max() == date


# --------------------------------------------------------------------------
# The label side
# --------------------------------------------------------------------------


def test_the_forward_return_starts_from_the_decision_date(ohlcv):
    """The convention that makes the inclusive slice coherent.

    A feature vector describing the close of D must be paired with a return
    measured *from* D. If features said D and the label started at D-1, the
    label would overlap the information the features already carry.
    """
    from portfolio_agent.features.labels import build_forward_return

    horizon = 5
    forward = build_forward_return(ohlcv["close"], f"return_{horizon}d")
    date = ohlcv.index[400]

    expected = ohlcv["close"].iloc[405] / ohlcv["close"].iloc[400] - 1.0
    assert forward.loc[date] == pytest.approx(expected)


def test_the_harness_and_the_trainers_share_that_label(ohlcv):
    """`harness.py` and `features/labels.py` must not each define one.

    They agreed on every value and disagreed on coverage: `build_forward_return`
    was `shift(-h).pct_change(h)`, which lands on the same number but also NaNs
    the first `h` rows, because `pct_change` has nothing to difference against
    there. Training therefore labelled less of each ticker than evaluation did,
    at the start of the sample — exactly where the long-lookback features have
    only just warmed up.
    """
    from portfolio_agent.evaluation.harness import forward_return
    from portfolio_agent.features.labels import build_forward_return

    for horizon in (1, 5, 21):
        pd.testing.assert_series_equal(
            forward_return(ohlcv["close"], horizon).dropna(),
            build_forward_return(ohlcv["close"], f"return_{horizon}d").dropna(),
            check_names=False,
        )


@pytest.mark.parametrize("horizon", [1, 5, 21])
def test_the_label_keeps_the_first_rows_of_the_sample(ohlcv, horizon):
    """The regression guard for the coverage bug above.

    Only the final `horizon` rows may be NaN — their outcome has not happened
    yet. Anything missing at the *start* is thrown-away training data.
    """
    from portfolio_agent.features.labels import build_forward_return

    label = build_forward_return(ohlcv["close"], f"return_{horizon}d")

    assert label.head(horizon).notna().all()
    assert label.tail(horizon).isna().all()
    assert label.notna().sum() == len(ohlcv) - horizon


class TestTheEngineLoadsItsOwnWarmup:
    """Raising the eligibility bar exposed why it had been low.

    `_load_all_data` requested data from `start_date`, so a run beginning
    2023-01-02 had no bar before it. `sma_200` was undefined for the first 200
    sessions regardless of what the cache held, and `mom_9m_skip1m` for the
    first 211 — and the 20-row bar let those tickers through, which is how the
    opening months of every backtest came to rank the universe on NaN.
    """

    def _engine(self, monkeypatch, ohlcv, strategy=None, **kwargs):
        from portfolio_agent.src import backtest_engine as module

        requested = {}

        def fake_load(ticker, start_date=None, end_date=None):
            requested["start"] = start_date
            return ohlcv.copy()

        monkeypatch.setattr(module, "load_ticker_data", fake_load)
        engine = module.BacktestEngine(
            start_date="2022-06-01", end_date="2022-12-30",
            initial_capital=1_000_000.0, universe_tickers=["A.NS"],
            strategy=strategy, **kwargs,
        )
        return engine, requested

    def test_it_reaches_back_before_the_window(self, monkeypatch, ohlcv):
        from portfolio_agent.config.schema import StrategyConfig
        from portfolio_agent.strategies.cross_sectional import MomentumStrategy

        strategy = MomentumStrategy(StrategyConfig(type="momentum", params={}))
        engine, requested = self._engine(monkeypatch, ohlcv, strategy=strategy)

        assert pd.Timestamp(requested["start"]) < engine.start_date

    def test_the_reach_covers_the_warm_up_in_sessions(self, monkeypatch, ohlcv):
        """211 sessions of warm-up cannot be 211 calendar days."""
        from portfolio_agent.config.schema import StrategyConfig
        from portfolio_agent.strategies.cross_sectional import MomentumStrategy

        strategy = MomentumStrategy(StrategyConfig(type="momentum", params={}))
        engine, requested = self._engine(monkeypatch, ohlcv, strategy=strategy)

        needed = engine._required_history_rows()
        sessions_available = len(
            pd.bdate_range(pd.Timestamp(requested["start"]), engine.start_date)
        )
        assert needed > 200
        assert sessions_available >= needed

    def test_the_scored_window_is_unchanged(self, monkeypatch, ohlcv):
        """Extra bars are warm-up; the run must not start scoring earlier."""
        from portfolio_agent.config.schema import StrategyConfig
        from portfolio_agent.strategies.cross_sectional import MomentumStrategy

        strategy = MomentumStrategy(StrategyConfig(type="momentum", params={}))
        engine, _ = self._engine(monkeypatch, ohlcv, strategy=strategy)

        assert engine.master_date_index.min() >= engine.start_date
        assert engine.master_date_index.max() <= engine.end_date

    def test_the_default_strategy_gets_its_own_warm_up_too(self, monkeypatch, ohlcv):
        """Passing no strategy is not passing no features.

        The constructor substitutes the default strategy, so a caller who omits
        one still gets that strategy's warm-up — 201 rows for `rule_based`, not
        a floor. This is the case the old 20-row bar was widest against.
        """
        engine, requested = self._engine(monkeypatch, ohlcv)

        assert engine.strategy is not None
        assert engine._required_history_rows() > 200
        assert pd.Timestamp(requested["start"]) < engine.start_date

    def test_the_warm_up_is_the_strategy_s_own(self, monkeypatch, ohlcv):
        """Two strategies with different lookbacks get different reaches."""
        from portfolio_agent.config.schema import StrategyConfig
        from portfolio_agent.strategies.cross_sectional import (
            LowVolatilityStrategy,
            MomentumStrategy,
        )

        momentum, _ = self._engine(
            monkeypatch, ohlcv,
            strategy=MomentumStrategy(StrategyConfig(type="momentum", params={})),
        )
        low_vol, _ = self._engine(
            monkeypatch, ohlcv,
            strategy=LowVolatilityStrategy(
                StrategyConfig(type="low_volatility_idio", params={})
            ),
        )

        # `mom_9m_skip1m` needs three quarters of history; a 60-session
        # volatility does not. A single constant could serve only one of them.
        assert momentum._required_history_rows() > low_vol._required_history_rows()
        assert low_vol._required_history_rows() > 1


class TestBothLabelForksFilterOutliers:
    """The ±5.0 filter sat on one side of a fork nobody noticed was a fork.

    `agents/trainer.prepare_features` has dropped absurd labels since a run was
    poisoned by one bad bar: a split that escapes adjustment produces an
    eleven-million-percent "return", and one such row dominates a squared-error
    objective completely. `build_gbm_panel` assembles its label itself rather
    than going through `prepare_features`, so the boosting trainers kept them.
    """

    def _frame(self, values):
        return pd.DataFrame(
            {"x": range(len(values)), "target_5d": values},
            index=pd.bdate_range("2023-01-02", periods=len(values)),
        )

    def test_it_drops_what_cannot_be_a_price_move(self):
        from portfolio_agent.features.labels import drop_absurd_labels

        frame = self._frame([0.01, -0.02, 11_000.0, 0.03])
        kept = drop_absurd_labels(frame, "target_5d")

        assert len(kept) == 3
        assert 11_000.0 not in set(kept["target_5d"])

    def test_it_keeps_moves_that_are_merely_extreme(self):
        """+149% is five consecutive 20% upper circuits — reachable, so kept."""
        from portfolio_agent.features.labels import drop_absurd_labels

        frame = self._frame([1.49, -0.60, 2.5, 0.0])
        assert len(drop_absurd_labels(frame, "target_5d")) == 4

    def test_it_drops_rather_than_clips(self):
        """A clip piles a spike of samples at the bound and teaches the model
        that the bound is a common outcome — one distortion for a subtler."""
        from portfolio_agent.features.labels import DEFAULT_MAX_ABS_LABEL, drop_absurd_labels

        frame = self._frame([9.0, 9.0, 9.0, 0.02])
        kept = drop_absurd_labels(frame, "target_5d")

        assert len(kept) == 1
        assert DEFAULT_MAX_ABS_LABEL not in set(kept["target_5d"].abs())

    def test_it_cuts_both_tails(self):
        from portfolio_agent.features.labels import drop_absurd_labels

        kept = drop_absurd_labels(self._frame([-9.0, 0.01, 9.0]), "target_5d")
        assert list(kept["target_5d"]) == [0.01]

    def test_a_clean_frame_comes_back_unchanged(self):
        from portfolio_agent.features.labels import drop_absurd_labels

        frame = self._frame([0.01, -0.02, 0.03])
        assert drop_absurd_labels(frame, "target_5d") is frame

    def test_a_missing_target_is_not_an_error(self):
        """The filter runs on panels that may not carry the column yet."""
        from portfolio_agent.features.labels import drop_absurd_labels

        frame = self._frame([0.01, 0.02]).drop(columns=["target_5d"])
        assert drop_absurd_labels(frame, "target_5d") is frame

    def test_the_supervised_pipeline_routes_through_it(self):
        """`prepare_features` must not keep its own copy of the rule.

        The filter it inlined and this one agreed on the threshold. They would
        not have stayed agreed — which is the whole shape of this round.
        """
        import inspect

        from portfolio_agent.agents import trainer

        source = inspect.getsource(trainer.prepare_features)
        assert "drop_absurd_labels(" in source
        assert ".abs() > config.training.max_abs_target" not in source
