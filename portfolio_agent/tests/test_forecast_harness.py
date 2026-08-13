"""The forecast evaluation harness: metrics, panel construction, discipline.

Two kinds of test here, and the split matters.

The metric tests use synthetic panels with a *known* answer — a score that is
the future return must produce an IC of exactly 1.0, a reversed one exactly
-1.0, noise something indistinguishable from zero. Those are the tests that
would catch a sign error, a pooled-instead-of-per-date correlation, or a
standard error that forgot the overlap.

The harness tests use a recording strategy against a synthetic cache and assert
the properties that make the numbers trustworthy at all: that a strategy is
never handed a row dated after the decision, that features built once and
sliced equal features rebuilt from truncated history, and that no
`BacktestEngine` is constructed anywhere in the path.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.config.loader import load_config
from portfolio_agent.evaluation import (
    build_forecast_panel,
    compare_forecasts,
    evaluate_forecast,
    evaluate_panel,
    forward_return,
    rank_ic,
    rank_ic_series,
    signal_decay,
    summarize_ic,
)
from portfolio_agent.evaluation.metrics import (
    assign_buckets,
    bucket_analysis,
    cross_sectional_percentile,
    directional_hit_rate,
    overlap_lags,
    rank_error_summary,
    score_dispersion,
    validate_panel,
)
from portfolio_agent.strategies.base import BaseStrategy
from portfolio_agent.strategies.types import StrategyContext, StrategySignal


@pytest.fixture
def app_config():
    return load_config()


# --------------------------------------------------------------------------
# Panel fixtures with a known answer
# --------------------------------------------------------------------------


def synthetic_returns(n_dates: int = 200, n_symbols: int = 50, seed: int = 0):
    """A panel of pure noise returns, with no score column yet."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n_dates, freq="B")
    symbols = [f"S{i:03d}" for i in range(n_symbols)]
    rows = [
        {"date": date, "symbol": symbol, "forward_return": value}
        for date in dates
        for symbol, value in zip(symbols, rng.normal(0.0, 0.03, n_symbols))
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def noise_panel():
    return synthetic_returns()


@pytest.fixture
def oracle_panel(noise_panel):
    """The score *is* the realized forward return — a perfect forecast."""
    return noise_panel.assign(score=noise_panel["forward_return"])


# --------------------------------------------------------------------------
# The two acceptance signals
# --------------------------------------------------------------------------


def test_a_signal_built_from_future_returns_scores_perfect_ic(oracle_panel):
    result = evaluate_panel(oracle_panel, horizon=5, strategy="oracle")
    assert result.ic.mean == pytest.approx(1.0)
    assert result.ic.positive_share == 1.0
    assert result.buckets.monotonicity == pytest.approx(1.0)
    assert result.buckets.monotone_steps == pytest.approx(1.0)
    assert result.buckets.spread > 0.0
    assert result.errors.mean_abs_error == pytest.approx(0.0, abs=1e-12)


def test_a_perfect_signal_is_reported_as_significant(oracle_panel):
    """The degenerate case that reads backwards if it is not handled.

    An oracle's IC is exactly 1.0 on every date, so the series has zero
    dispersion and the Newey–West standard error is zero. Returning t=0, p=1
    there would label the strongest evidence available "not significant".
    """
    result = evaluate_panel(oracle_panel, horizon=5, strategy="oracle")
    assert math.isinf(result.ic.t_stat) and result.ic.t_stat > 0
    assert result.ic.p_value == 0.0
    assert result.ic.significant


def test_a_random_signal_scores_near_zero_and_does_not_reject(noise_panel):
    # Not seed 0. `noise_panel` draws its returns from `default_rng(0)`, so a
    # score drawn from a fresh `default_rng(0)` of the same shape walks the
    # identical standard-normal stream and *is* the forward return up to a
    # scale factor — a "random" signal that scores an IC of exactly 1.0.
    rng = np.random.default_rng(12345)
    panel = noise_panel.assign(score=rng.normal(0.0, 1.0, len(noise_panel)))
    result = evaluate_panel(panel, horizon=5, strategy="random")

    assert abs(result.ic.mean) < 0.02
    assert not result.ic.significant
    assert abs(result.buckets.spread) < 0.002
    assert 0.45 < result.hit_rate < 0.55


def test_the_t_statistic_is_correctly_sized_under_the_null():
    """One seed proves nothing; the rejection *rate* is the real claim.

    A single random panel that fails to reject is consistent with a
    t-statistic that never rejects anything, and a single one that does reject
    is consistent with a correctly-sized test having an ordinary bad day —
    seed 99 above produces p=0.004 on pure noise, which is exactly the 1-in-250
    draw you should see roughly once in 250 tries.

    What the acceptance criterion actually asks is whether the statistic is
    calibrated. Across 60 independent null panels the nominal 5% test should
    reject about 5% of the time; the bound here is loose enough not to be
    flaky and tight enough to catch a standard error that is wrong by the
    sqrt(horizon) factor the overlap correction exists to supply.
    """
    rejections = 0
    means: List[float] = []
    for seed in range(60):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2022-01-03", periods=150, freq="B")
        symbols = [f"S{i}" for i in range(40)]
        rows = []
        for date in dates:
            returns = rng.normal(0.0, 0.03, len(symbols))
            scores = rng.normal(0.0, 1.0, len(symbols))
            rows.extend(
                {"date": date, "symbol": symbol, "score": score, "forward_return": value}
                for symbol, score, value in zip(symbols, scores, returns)
            )
        summary = summarize_ic(rank_ic_series(pd.DataFrame(rows)), horizon=5)
        rejections += int(summary.significant)
        means.append(summary.mean)

    assert rejections / 60 <= 0.10, f"{rejections}/60 null panels rejected"
    assert abs(float(np.mean(means))) < 0.01


def test_a_reversed_signal_scores_minus_one(oracle_panel):
    """A sign error anywhere in the chain shows up here and nowhere else."""
    panel = oracle_panel.assign(score=-oracle_panel["forward_return"])
    result = evaluate_panel(panel, horizon=5, strategy="reversed")
    assert result.ic.mean == pytest.approx(-1.0)
    assert result.buckets.monotonicity == pytest.approx(-1.0)
    assert result.buckets.spread < 0.0
    assert result.hit_rate == pytest.approx(0.0, abs=0.05)


def test_two_runs_of_one_configuration_agree_exactly(noise_panel):
    rng = np.random.default_rng(7)
    panel = noise_panel.assign(score=rng.normal(0.0, 1.0, len(noise_panel)))
    first = evaluate_panel(panel, horizon=5, strategy="s")
    again = evaluate_panel(panel, horizon=5, strategy="s")

    # NaN-aware, because some metrics are legitimately undefined and `nan !=
    # nan`. On a noise panel the gross decile spread comes out negative, which
    # makes "what share of it did costs eat" a ratio with no readable sign —
    # reported as NaN. Two runs producing the same NaN *is* agreement; plain
    # dict equality would call it a reproducibility failure.
    left, right = first.to_dict(), again.to_dict()
    assert left.keys() == right.keys()
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            continue
        assert a == b, key

    pd.testing.assert_series_equal(first.ic_series, again.ic_series)


# --------------------------------------------------------------------------
# Rank IC
# --------------------------------------------------------------------------


def test_ic_is_computed_within_a_date_not_across_the_pool():
    """Pooling would mostly measure whether the score tracks the market level.

    Here the score orders each date perfectly but its *level* is anticorrelated
    with the date's mean return. A per-date IC sees the perfect ordering; a
    pooled correlation would be dragged toward zero or below.
    """
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    rows = []
    for offset, date in enumerate(dates):
        level = -10.0 * offset
        for rank in range(6):
            rows.append(
                {
                    "date": date, "symbol": f"S{rank}",
                    "score": level + rank,
                    "forward_return": 0.10 * offset + 0.01 * rank,
                }
            )
    panel = pd.DataFrame(rows)

    per_date = rank_ic_series(panel, min_names=5)
    assert per_date.to_numpy() == pytest.approx(1.0)

    pooled = panel["score"].corr(panel["forward_return"], method="spearman")
    assert pooled < 0.5  # what pooling would have reported instead


def test_rank_ic_is_undefined_rather_than_zero_for_a_constant_score():
    """A constant score makes no ordering claim; 0.0 would average in as a miss."""
    assert math.isnan(rank_ic([1.0, 1.0, 1.0], [0.1, 0.2, 0.3]))
    assert math.isnan(rank_ic([1.0, 2.0, 3.0], [0.5, 0.5, 0.5]))


def test_rank_ic_series_drops_thin_cross_sections():
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    rows = [
        {"date": dates[0], "symbol": f"S{i}", "score": float(i), "forward_return": 0.01 * i}
        for i in range(3)
    ] + [
        {"date": dates[1], "symbol": f"S{i}", "score": float(i), "forward_return": 0.01 * i}
        for i in range(8)
    ]
    ic = rank_ic_series(pd.DataFrame(rows), min_names=5)
    assert list(ic.index) == [dates[1]]


# --------------------------------------------------------------------------
# The Newey-West adjustment
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "horizon,stride,expected",
    [(5, 1, 4), (1, 1, 0), (21, 1, 20), (5, 5, 0), (5, 2, 2), (10, 3, 3)],
)
def test_overlap_lags_follows_the_sampling_frequency(horizon, stride, expected):
    """Sampled every 5th day, a 5-day label overlaps none of its neighbours.

    Correcting for an overlap that is not there is conservative but wrong, and
    wrong exactly on the fast runs someone strode to make cheap.
    """
    assert overlap_lags(horizon, stride) == expected


def test_the_overlap_adjustment_widens_the_standard_error():
    """Treating overlapping observations as independent manufactures significance.

    A positively autocorrelated IC series has a genuinely wider standard error
    than its independent-sample counterpart, so the adjusted t must be smaller.
    """
    rng = np.random.default_rng(3)
    innovations = rng.normal(0.02, 0.05, 400)
    # Impose persistence, which is what an overlapping label produces.
    values = pd.Series(innovations).rolling(5).mean().dropna()

    naive = summarize_ic(values, horizon=1)
    adjusted = summarize_ic(values, horizon=5)
    assert adjusted.newey_west_lags == 4
    assert abs(adjusted.t_stat) < abs(naive.t_stat)


def test_an_ic_series_of_all_zeros_is_not_significant():
    result = summarize_ic(pd.Series([0.0] * 50), horizon=5)
    assert result.t_stat == 0.0
    assert result.p_value == 1.0
    assert not result.significant


def test_a_single_date_measures_nothing():
    result = summarize_ic(pd.Series([0.4]), horizon=5)
    assert result.n_dates == 1
    assert result.p_value == 1.0
    assert not result.significant


# --------------------------------------------------------------------------
# Buckets
# --------------------------------------------------------------------------


def test_dates_are_weighted_equally_not_observations():
    """A 2,000-name date would otherwise count ten times a 200-name date.

    Four narrow dates say the signal is perfect and one twenty-times-wider date
    says it is reversed. Both span the same return magnitudes, so the only
    thing that can decide the sign of the spread is how the two are weighted:
    by date the majority wins, by observation the single wide date does.
    """
    rows = []
    for date in pd.date_range("2024-01-01", periods=4, freq="B"):
        for rank in range(10):
            rows.append({"date": date, "symbol": f"N{rank}", "score": float(rank),
                         "forward_return": 0.09 * rank / 9})
    wide_date = pd.Timestamp("2024-01-08")
    for rank in range(200):
        rows.append({"date": wide_date, "symbol": f"W{rank}", "score": float(rank),
                     "forward_return": -0.09 * rank / 199})

    result = bucket_analysis(pd.DataFrame(rows), n_buckets=5)
    assert result.spread > 0.0
    assert result.monotonicity > 0.0
    # Observation weighting would have inverted it: 200 reversed rows against 40.
    assert sum(result.counts) == 240


def test_heavy_ties_leave_a_bucket_empty_without_warning_or_crash():
    """A screen emitting one floor value for half the universe is normal.

    `pd.qcut` raises on duplicate bin edges here; average-ranking spreads the
    tied block and keeps the date usable. The bucket that ends up unoccupied is
    reported as NaN with a count of zero rather than averaged away.
    """
    import warnings

    rows = []
    for date in pd.date_range("2024-01-01", periods=20, freq="B"):
        for i in range(40):
            score = 0.0 if i < 30 else float(i)  # 30 names tied at the floor
            rows.append({"date": date, "symbol": f"S{i}", "score": score,
                         "forward_return": 0.001 * i})

    with warnings.catch_warnings():
        # A numpy "Mean of empty slice" here would mean the empty bucket is
        # being averaged rather than recognised.
        warnings.simplefilter("error", RuntimeWarning)
        result = bucket_analysis(pd.DataFrame(rows), n_buckets=10)

    assert result.counts[0] == 0
    assert math.isnan(result.mean_returns[0])
    assert sum(result.counts) == 800


def test_assign_buckets_spans_the_full_range_without_ties():
    buckets = assign_buckets(np.arange(100.0), n_buckets=10)
    assert buckets.min() == 0
    assert buckets.max() == 9
    assert len(set(buckets.tolist())) == 10


def test_monotonicity_separates_a_broad_signal_from_a_tail_driven_one():
    """The check the spec exists for: a tail-only signal is not breadth.

    Both panels below have a positive top-minus-bottom spread. Only one has a
    profile that climbs, and a book holding the top two deciles gets nothing
    from the other.
    """
    broad_rows, tail_rows = [], []
    for date in pd.date_range("2024-01-01", periods=40, freq="B"):
        for rank in range(50):
            broad_rows.append({"date": date, "symbol": f"S{rank}", "score": float(rank),
                               "forward_return": 0.0004 * rank})
            # Flat everywhere except the very top names.
            tail = 0.02 if rank >= 48 else 0.0
            tail_rows.append({"date": date, "symbol": f"S{rank}", "score": float(rank),
                              "forward_return": tail})

    broad = bucket_analysis(pd.DataFrame(broad_rows), n_buckets=10)
    tail = bucket_analysis(pd.DataFrame(tail_rows), n_buckets=10)

    assert broad.spread > 0 and tail.spread > 0
    assert broad.monotone_steps == pytest.approx(1.0)
    assert tail.monotone_steps < 0.2


# --------------------------------------------------------------------------
# Hit rate, errors, dispersion
# --------------------------------------------------------------------------


def test_hit_rate_is_relative_not_absolute(oracle_panel):
    """A 0-100 canonical score has no sign, so the call is against the field."""
    assert directional_hit_rate(oracle_panel) > 0.9
    reversed_panel = oracle_panel.assign(score=-oracle_panel["forward_return"])
    assert directional_hit_rate(reversed_panel) < 0.1


def test_rank_error_is_zero_for_an_oracle_and_large_when_reversed(oracle_panel):
    perfect = rank_error_summary(oracle_panel)
    assert perfect.mean_abs_error == pytest.approx(0.0, abs=1e-12)

    reversed_panel = oracle_panel.assign(score=-oracle_panel["forward_return"])
    wrong = rank_error_summary(reversed_panel)
    assert wrong.mean_abs_error > 0.3
    assert set(wrong.quantiles) == {"p05", "p25", "p50", "p75", "p95"}


def test_score_dispersion_distinguishes_a_flat_signal_from_a_wrong_one():
    """A near-zero IC has two very different causes; this tells them apart."""
    rows_flat, rows_distinct = [], []
    for date in pd.date_range("2024-01-01", periods=10, freq="B"):
        for i in range(20):
            rows_flat.append({"date": date, "symbol": f"S{i}", "score": 50.0,
                              "forward_return": 0.001 * i})
            rows_distinct.append({"date": date, "symbol": f"S{i}", "score": float(i),
                                  "forward_return": 0.001 * i})

    assert score_dispersion(pd.DataFrame(rows_flat)) == pytest.approx(1 / 20)
    assert score_dispersion(pd.DataFrame(rows_distinct)) == pytest.approx(1.0)


def test_cross_sectional_percentile_stays_off_the_boundary():
    values = cross_sectional_percentile([1.0, 2.0, 3.0])
    assert values.min() > 0.0 and values.max() < 1.0
    assert list(values) == pytest.approx([0.25, 0.5, 0.75])


def test_signal_decay_reports_one_ic_per_horizon(noise_panel):
    panels = {
        1: noise_panel.assign(score=noise_panel["forward_return"]),
        5: noise_panel.assign(score=-noise_panel["forward_return"]),
    }
    decay = signal_decay(panels)
    assert list(decay.index) == [1, 5]
    assert decay[1] == pytest.approx(1.0)
    assert decay[5] == pytest.approx(-1.0)


# --------------------------------------------------------------------------
# Panel validation and the result object
# --------------------------------------------------------------------------


def test_a_misspelled_column_raises_rather_than_scoring_empty():
    """An empty result reads as "no skill", not as "wrong input"."""
    panel = pd.DataFrame({"date": [1], "symbol": ["A"], "score": [1.0], "ret": [0.1]})
    with pytest.raises(ValueError, match="forward_return"):
        validate_panel(panel)


def test_non_finite_rows_are_dropped(noise_panel):
    panel = noise_panel.assign(score=1.0)
    panel.loc[0, "score"] = np.inf
    panel.loc[1, "forward_return"] = np.nan
    assert len(validate_panel(panel)) == len(panel) - 2


def test_result_renders_and_serializes(oracle_panel):
    result = evaluate_panel(oracle_panel, horizon=5, strategy="oracle")
    text = result.render()
    assert "mean rank IC" in text
    assert "monotonicity" in text
    assert "oracle" in text

    document = result.to_dict()
    assert document["strategy"] == "oracle"
    assert document["mean_ic"] == pytest.approx(1.0)
    assert len(document["bucket_mean_returns"]) == 10

    frame = result.to_frame()
    assert len(frame) == 1
    # Nested values stay out of the flat table, or every row becomes a list.
    assert "bucket_mean_returns" not in frame.columns


def test_compare_forecasts_ranks_by_mean_ic(oracle_panel):
    good = evaluate_panel(oracle_panel, horizon=5, strategy="good")
    bad = evaluate_panel(
        oracle_panel.assign(score=-oracle_panel["forward_return"]),
        horizon=5, strategy="bad",
    )
    table = compare_forecasts([bad, good])
    assert list(table["strategy"]) == ["good", "bad"]


def test_worst_and_best_dates_come_from_the_ic_series(noise_panel):
    rng = np.random.default_rng(11)
    panel = noise_panel.assign(score=rng.normal(0.0, 1.0, len(noise_panel)))
    result = evaluate_panel(panel, horizon=5, strategy="s")
    assert len(result.worst_dates(3)) == 3
    assert result.worst_dates(3).iloc[0] <= result.best_dates(3).iloc[0]


# --------------------------------------------------------------------------
# Walk-forward folds
# --------------------------------------------------------------------------


def test_folds_report_per_fold_ic_and_what_was_purged(oracle_panel):
    from portfolio_agent.validation.purged import PurgedWalkForward

    result = evaluate_panel(
        oracle_panel, horizon=5, strategy="oracle",
        splitter=PurgedWalkForward(n_splits=3, horizon=5),
    )
    assert len(result.folds) == 3
    for fold in result.folds:
        assert fold.ic.mean == pytest.approx(1.0)
        assert fold.n_purged == 5     # exactly the horizon, at daily sampling
        assert fold.n_dates > 0
    # Folds are chronological and do not overlap.
    starts = [fold.test_start for fold in result.folds]
    assert starts == sorted(starts)


def test_a_leaky_fold_stops_the_run(oracle_panel, monkeypatch):
    """The assertion is the point: a leaky fold's IC looks like skill."""
    from portfolio_agent.validation import purged as purged_module

    def explode(fold, dates, horizon):
        raise AssertionError("leak")

    monkeypatch.setattr(purged_module, "assert_no_leakage", explode)
    with pytest.raises(AssertionError, match="leak"):
        evaluate_panel(
            oracle_panel, horizon=5,
            splitter=purged_module.PurgedWalkForward(n_splits=2, horizon=5),
        )


def test_an_ineffective_embargo_says_why(oracle_panel):
    """Zero embargoed rows in an expanding scheme is correct, not broken."""
    from portfolio_agent.validation.purged import PurgedWalkForward

    result = evaluate_panel(
        oracle_panel, horizon=5,
        splitter=PurgedWalkForward(n_splits=3, horizon=5, embargo=3),
    )
    assert all(fold.n_embargoed == 0 for fold in result.folds)
    assert any("embargo" in note for note in result.notes)
    assert "Note:" in result.render()


# --------------------------------------------------------------------------
# Building a panel from a strategy
# --------------------------------------------------------------------------


def _ohlcv(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    return pd.DataFrame(
        {
            "open": close, "high": close * 1.02, "low": close * 0.98,
            "close": close, "volume": rng.integers(1e5, 1e6, n).astype(float),
        },
        index=index,
    )


class RecordingStrategy(BaseStrategy):
    """Scores the last RSI value and records what it was shown.

    Deliberately trivial: the point is not what it predicts but that the
    harness can be interrogated about exactly which rows reached it.
    """

    def __init__(self) -> None:
        self.seen_max_dates: List[pd.Timestamp] = []
        self.calls = 0

    @property
    def name(self) -> str:
        return "recording"

    @property
    def requires_full_batch(self) -> bool:
        return True

    def required_features(self) -> List[str]:
        return ["rsi_14"]

    def score(self, symbol, features, context):  # pragma: no cover - batch only
        raise NotImplementedError

    def score_batch(
        self, features_by_symbol: Dict[str, pd.DataFrame], context: StrategyContext
    ) -> Dict[str, StrategySignal]:
        self.calls += 1
        self.seen_max_dates.extend(frame.index.max() for frame in features_by_symbol.values())
        return {
            symbol: StrategySignal(
                symbol=symbol, signal="BUY" if frame["rsi_14"].iloc[-1] > 50 else "HOLD",
                score=float(frame["rsi_14"].iloc[-1]), trigger="Model",
                entry_price=100.0, stop_price=95.0, target_price=110.0,
                reward_risk=2.0, probability_profit=0.5,
            )
            for symbol, frame in features_by_symbol.items()
        }


@pytest.fixture
def fake_cache(monkeypatch):
    frames = {f"T{i}": _ohlcv(seed=i) for i in range(8)}

    def fake_load(ticker, start_date=None, end_date=None):
        return frames.get(ticker)

    monkeypatch.setattr(
        "portfolio_agent.src.data_store.load_ticker_data", fake_load, raising=True
    )
    return frames


def test_features_built_once_and_sliced_equal_features_rebuilt(fake_cache):
    """The assumption the harness's speed rests on, checked rather than trusted.

    If a non-causal feature is ever registered, this fails here instead of the
    harness silently handing a strategy information from after the decision.
    """
    from portfolio_agent.features.pipeline import build_features
    from portfolio_agent.features.registry import _FEATURE_REGISTRY

    # Every registered feature, not just the ones the strategies happen to use
    # today: the harness's guarantee is about the pipeline, and a feature added
    # later is exactly the one nobody would think to re-check.
    names = sorted(_FEATURE_REGISTRY)
    raw = fake_cache["T0"]
    cut = raw.index[400]

    full = build_features(raw, names)
    truncated = build_features(raw.loc[:cut], names)
    common = full.index.intersection(truncated.index)
    assert len(common) > 100

    for column in names:
        np.testing.assert_allclose(
            full.loc[common, column].to_numpy(dtype=float),
            truncated.loc[common, column].to_numpy(dtype=float),
            equal_nan=True,
            err_msg=f"{column} is not causal: building it over the full history "
                    f"changes its value at earlier dates",
        )


def test_a_strategy_is_never_shown_a_row_after_the_decision_date(
    app_config, fake_cache
):
    """Point-in-time discipline, asserted from inside the strategy."""
    strategy = RecordingStrategy()
    panel = build_forecast_panel(
        app_config, strategy, list(fake_cache),
        horizon=5, stride=20, min_history=260, use_benchmark=False,
    )
    assert strategy.calls > 0

    # Every frame handed over ends exactly on the date being scored, never after.
    scored_dates = set(pd.to_datetime(panel["date"]).unique())
    assert set(strategy.seen_max_dates) <= scored_dates


def test_the_recorded_score_is_the_value_at_the_decision_date(app_config, fake_cache):
    """A panel whose scores came from the wrong row would still look plausible."""
    from portfolio_agent.features.pipeline import build_features

    strategy = RecordingStrategy()
    panel = build_forecast_panel(
        app_config, strategy, list(fake_cache),
        horizon=5, stride=40, min_history=260, use_benchmark=False,
    )

    expected = build_features(fake_cache["T0"], ["rsi_14"])["rsi_14"]
    rows = panel[panel["symbol"] == "T0"]
    assert len(rows) > 1
    for _, row in rows.iterrows():
        assert row["score"] == pytest.approx(float(expected.loc[row["date"]]))


def test_the_label_is_the_return_after_the_decision_date(app_config, fake_cache):
    strategy = RecordingStrategy()
    panel = build_forecast_panel(
        app_config, strategy, list(fake_cache),
        horizon=5, stride=40, min_history=260, use_benchmark=False,
    )
    close = fake_cache["T0"]["close"].astype(float)
    rows = panel[panel["symbol"] == "T0"]
    for _, row in rows.iterrows():
        position = close.index.get_loc(row["date"])
        expected = close.iloc[position + 5] / close.iloc[position] - 1.0
        assert row["forward_return"] == pytest.approx(expected)


def test_forward_return_is_dated_at_the_decision_point():
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
    values = forward_return(close, 2)
    assert values.iloc[0] == pytest.approx(102.0 / 100.0 - 1.0)
    assert math.isnan(values.iloc[-1])
    assert math.isnan(values.iloc[-2])


def test_no_backtest_engine_is_constructed(app_config, fake_cache, monkeypatch):
    """The acceptance criterion, asserted literally.

    Routing around the engine is the whole point of this task; a harness that
    quietly imported and built one would inherit the cost and the
    portfolio-construction filtering it exists to avoid.
    """
    from portfolio_agent.src import backtest_engine

    def refuse(*args, **kwargs):
        raise AssertionError("the harness must not construct a BacktestEngine")

    monkeypatch.setattr(backtest_engine.BacktestEngine, "__init__", refuse)

    panel = build_forecast_panel(
        app_config, RecordingStrategy(), list(fake_cache),
        horizon=5, stride=40, min_history=260, use_benchmark=False,
    )
    assert not panel.empty


def test_an_empty_cache_raises_rather_than_scoring_nothing(app_config, monkeypatch):
    monkeypatch.setattr(
        "portfolio_agent.src.data_store.load_ticker_data",
        lambda ticker, start_date=None, end_date=None: None,
        raising=True,
    )
    with pytest.raises(ValueError, match="usable history"):
        build_forecast_panel(app_config, RecordingStrategy(), ["A", "B"], horizon=5)


def test_a_universe_too_thin_to_rank_raises(app_config, monkeypatch):
    frames = {"A": _ohlcv(seed=1), "B": _ohlcv(seed=2)}
    monkeypatch.setattr(
        "portfolio_agent.src.data_store.load_ticker_data",
        lambda ticker, start_date=None, end_date=None: frames.get(ticker),
        raising=True,
    )
    with pytest.raises(ValueError, match="cross-section"):
        build_forecast_panel(
            app_config, RecordingStrategy(), list(frames),
            horizon=5, stride=40, min_history=260, min_names=5, use_benchmark=False,
        )


def test_stride_must_be_positive(app_config, fake_cache):
    with pytest.raises(ValueError, match="stride"):
        build_forecast_panel(
            app_config, RecordingStrategy(), list(fake_cache), stride=0
        )


def test_max_dates_keeps_the_most_recent_window(app_config, fake_cache):
    """A truncated run should describe the regime a model would deploy into."""
    strategy = RecordingStrategy()
    full = build_forecast_panel(
        app_config, strategy, list(fake_cache),
        horizon=5, stride=10, min_history=260, use_benchmark=False,
    )
    capped = build_forecast_panel(
        app_config, RecordingStrategy(), list(fake_cache),
        horizon=5, stride=10, min_history=260, max_dates=3, use_benchmark=False,
    )
    assert capped["date"].nunique() == 3
    assert capped["date"].max() == full["date"].max()


# --------------------------------------------------------------------------
# End to end through a registered strategy
# --------------------------------------------------------------------------


def test_evaluate_forecast_runs_a_registered_strategy(app_config, fake_cache, tmp_path):
    result = evaluate_forecast(
        app_config, "momentum", universe=list(fake_cache),
        horizon=5, stride=40, min_history=260, min_names=5,
        n_buckets=4, use_benchmark=False, runs_dir=str(tmp_path),
    )
    assert result.strategy == "momentum"
    assert result.n_observations > 0
    assert result.n_symbols == len(fake_cache)
    assert -1.0 <= result.ic.mean <= 1.0
    assert "momentum" in result.render()


def test_buys_only_narrows_the_panel_and_says_so(app_config, fake_cache, tmp_path):
    result = evaluate_forecast(
        app_config, RecordingStrategy(), universe=list(fake_cache),
        horizon=5, stride=40, min_history=260, min_names=5,
        n_buckets=4, use_benchmark=False, buys_only=True, runs_dir=str(tmp_path),
    )
    assert any("BUY" in note for note in result.notes)
    assert "Note:" in result.render()
