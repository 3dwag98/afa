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
