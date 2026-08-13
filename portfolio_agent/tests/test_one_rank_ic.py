"""One definition of the information coefficient, asserted so it stays one.

The repository had four. Two of them agreed; the one the neural trainer used
for model selection disagreed *about the sign*, because it pooled every
observation into a single rank correlation instead of correlating within each
date and averaging. Those are different quantities: pooled, the number mostly
says whether the model's level tracks the market's from day to day, which a
long-only ranking book cannot trade.

The gap is not subtle and it is not a rounding difference. The first test here
builds the panel that separates them — a signal that orders every date
perfectly while its level runs against the market's — and the two conventions
come out at +1.00 and -1.00. Everything after it checks that every entry point
in the package now routes to the same arithmetic.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.evaluation.metrics import (
    MIN_CROSS_SECTION_NAMES,
    rank_ic_from_arrays,
    rank_ic_series,
)

REPO = Path(__file__).resolve().parent.parent.parent


# --------------------------------------------------------------------------
# The panel that tells the two conventions apart
# --------------------------------------------------------------------------


def adversarial_panel(n_dates: int = 40, n_names: int = 30):
    """A signal with perfect per-date ordering and an anticorrelated level.

    Each date gets a market-wide level that walks down while the signal's level
    walks up. Within every date the signal orders the names exactly right. A
    per-date IC sees the ordering and reports +1; a pooled IC sees the two
    opposing trends in the levels and reports about -1.
    """
    rows = []
    for t in range(n_dates):
        market_level = -0.02 * t
        signal_level = 0.10 * t
        spread = np.linspace(-0.01, 0.01, n_names)
        for i in range(n_names):
            rows.append(
                {
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=t),
                    "symbol": f"S{i:03d}",
                    "score": signal_level + spread[i],
                    "forward_return": market_level + spread[i],
                }
            )
    return pd.DataFrame(rows)


def pooled_rank_correlation(scores, labels) -> float:
    """The discarded convention, kept here so the difference stays measurable."""
    return float(
        pd.Series(np.asarray(scores, dtype=float)).corr(
            pd.Series(np.asarray(labels, dtype=float)), method="spearman"
        )
    )


class TestTheTwoConventionsDisagree:
    def test_per_date_ic_reads_the_ordering(self):
        panel = adversarial_panel()
        assert rank_ic_series(panel).mean() == pytest.approx(1.0)

    def test_pooled_correlation_reads_the_level_and_flips_the_sign(self):
        """Not a bug in this test — the reason the pooled version was removed."""
        panel = adversarial_panel()
        pooled = pooled_rank_correlation(panel["score"], panel["forward_return"])
        assert pooled < -0.9

    def test_the_gap_is_two_whole_units_of_correlation(self):
        panel = adversarial_panel()
        per_date = rank_ic_series(panel).mean()
        pooled = pooled_rank_correlation(panel["score"], panel["forward_return"])
        assert per_date - pooled > 1.9


# --------------------------------------------------------------------------
# Every entry point routes to the same arithmetic
# --------------------------------------------------------------------------


class TestEveryEntryPointAgrees:
    def _arrays(self):
        panel = adversarial_panel()
        return (
            panel["score"].to_numpy(),
            panel["forward_return"].to_numpy(),
            panel["date"].to_numpy(),
        )

    def test_the_array_adapter_matches_the_panel_function(self):
        panel = adversarial_panel()
        from_panel = rank_ic_series(panel)
        from_arrays = rank_ic_from_arrays(
            panel["score"], panel["forward_return"], panel["date"]
        )
        pd.testing.assert_series_equal(from_panel, from_arrays)

    def test_the_gbm_trainer_agrees(self):
        pytest.importorskip("sklearn")
        from portfolio_agent.training.trainers.gbm import rank_ic_by_date

        scores, labels, dates = self._arrays()
        pd.testing.assert_series_equal(
            rank_ic_by_date(scores, labels, dates),
            rank_ic_from_arrays(scores, labels, dates),
        )

    def test_performance_stats_agrees(self):
        from portfolio_agent.src.performance_stats import rank_information_coefficient

        scores, labels, dates = self._arrays()
        result = rank_information_coefficient(
            pd.Series(scores), pd.Series(labels), dates=dates
        )
        assert result["mean_ic"] == pytest.approx(
            float(rank_ic_from_arrays(scores, labels, dates).mean())
        )

    def test_the_neural_trainer_agrees(self):
        pytest.importorskip("torch")
        from portfolio_agent.agents.trainer import evaluate_predictions

        scores, labels, dates = self._arrays()
        metrics = evaluate_predictions(
            scores, labels, horizon_days=1, relative_target=True, dates=dates
        )
        assert metrics["rank_ic"] == pytest.approx(1.0)
        assert metrics["n_ic_dates"] == 40

    def test_the_neural_trainer_no_longer_reports_the_pooled_sign(self):
        """The regression this task exists to prevent."""
        pytest.importorskip("torch")
        from portfolio_agent.agents.trainer import evaluate_predictions

        scores, labels, dates = self._arrays()
        metrics = evaluate_predictions(
            scores, labels, horizon_days=1, relative_target=True, dates=dates
        )
        assert metrics["rank_ic"] > 0


# --------------------------------------------------------------------------
# Refusing to answer beats answering the wrong question
# --------------------------------------------------------------------------


class TestWithoutDatesThereIsNoIC:
    def test_rank_ic_is_nan_when_no_dates_are_given(self):
        pytest.importorskip("torch")
        from portfolio_agent.agents.trainer import evaluate_predictions

        rng = np.random.default_rng(0)
        metrics = evaluate_predictions(
            rng.normal(size=200), rng.normal(size=200), horizon_days=1
        )
        assert math.isnan(metrics["rank_ic"])
        assert metrics["n_ic_dates"] == 0

    def test_the_other_metrics_are_still_reported_without_dates(self):
        """Only IC needs a cross-section; MSE and hit rate do not."""
        pytest.importorskip("torch")
        from portfolio_agent.agents.trainer import evaluate_predictions

        rng = np.random.default_rng(1)
        metrics = evaluate_predictions(
            rng.normal(size=200), rng.normal(size=200), horizon_days=1
        )
        assert metrics["n_samples"] == 200
        assert metrics["mse"] > 0

    def test_misaligned_dates_raise_rather_than_regroup(self):
        pytest.importorskip("torch")
        from portfolio_agent.agents.trainer import evaluate_predictions

        rng = np.random.default_rng(2)
        with pytest.raises(ValueError, match="dates has"):
            evaluate_predictions(
                rng.normal(size=100), rng.normal(size=100),
                dates=pd.date_range("2024-01-01", periods=99),
            )


# --------------------------------------------------------------------------
# The ICIR the two modules used to disagree about
# --------------------------------------------------------------------------


class TestOneICIRConvention:
    def _signal(self, n_dates=120, n_names=20, seed=7):
        rng = np.random.default_rng(seed)
        dates = np.repeat(pd.date_range("2024-01-01", periods=n_dates), n_names)
        realized = rng.normal(0, 0.05, size=n_dates * n_names)
        signal = realized + rng.normal(0, 0.05, size=realized.size)
        return signal, realized, dates

    def test_icir_is_the_raw_ratio_and_matches_the_evaluation_layer(self):
        """It used to be annualized here and raw there, a factor of 16 apart."""
        from portfolio_agent.evaluation.metrics import summarize_ic
        from portfolio_agent.src.performance_stats import rank_information_coefficient

        signal, realized, dates = self._signal()
        stats = rank_information_coefficient(
            pd.Series(signal), pd.Series(realized), dates=dates, horizon_days=1
        )
        evaluation = summarize_ic(
            rank_ic_from_arrays(signal, realized, dates), horizon=1
        )
        assert stats["icir"] == pytest.approx(evaluation.icir)

    def test_the_annualized_figure_is_still_available_under_its_own_name(self):
        from portfolio_agent.src.performance_stats import rank_information_coefficient

        signal, realized, dates = self._signal()
        result = rank_information_coefficient(
            pd.Series(signal), pd.Series(realized), dates=dates, horizon_days=5
        )
        assert result["icir_annualized"] == pytest.approx(
            result["icir"] * math.sqrt(252 / 5)
        )

    def test_an_oracle_signal_is_infinitely_stable_not_zero(self):
        """IC is 1.0 on every date, so the dispersion is zero.

        Reporting an ICIR of 0.0 there — which this did — labels the strongest
        possible evidence as no evidence. Same fix as T04 made in the
        evaluation layer, now reachable from both entry points.
        """
        from portfolio_agent.src.performance_stats import rank_information_coefficient

        _, realized, dates = self._signal()
        result = rank_information_coefficient(
            pd.Series(realized), pd.Series(realized), dates=dates
        )
        assert result["mean_ic"] == pytest.approx(1.0)
        assert result["icir"] == math.inf


# --------------------------------------------------------------------------
# The structural check, in the shape T10 used for RSI
# --------------------------------------------------------------------------


def test_only_one_module_computes_a_rank_correlation_by_date():
    """Searched by shape, because the failure mode is a re-added local copy.

    A per-date IC has a recognizable form: group by date, then take a Spearman
    correlation or correlate two `.rank()` calls. Outside `evaluation/metrics`
    nothing should contain one — the trainers call into it instead.
    """
    grouping = re.compile(r"groupby\(\s*[\"']date[\"']", re.M)
    correlating = re.compile(r"method\s*=\s*[\"']spearman[\"']|\.rank\(\)", re.M)

    offenders = []
    for path in (REPO / "portfolio_agent").rglob("*.py"):
        if "tests" in path.parts:
            continue
        if path.relative_to(REPO).as_posix() == "portfolio_agent/evaluation/metrics.py":
            continue
        text = path.read_text()
        if grouping.search(text) and correlating.search(text):
            offenders.append(path.relative_to(REPO).as_posix())
    assert offenders == [], offenders


def test_the_min_cross_section_threshold_is_shared_not_copied():
    """`gbm.py` carried its own `min_names: int = 5` default.

    Two thresholds that happen to be equal today are two thresholds, and the
    one nobody remembers is the one that stops matching.
    """
    pytest.importorskip("sklearn")
    import inspect

    from portfolio_agent.training.trainers.gbm import rank_ic_by_date

    default = inspect.signature(rank_ic_by_date).parameters["min_names"].default
    assert default is MIN_CROSS_SECTION_NAMES


def test_the_split_boundaries_are_shared_not_copied():
    """The date slice and the array slice have to agree, or IC scores the wrong days."""
    from portfolio_agent.data.dataset import chronological_split_bounds, test_split_dates

    index = pd.date_range("2024-01-01", periods=100)
    train_end, val_end = chronological_split_bounds(100)
    assert (train_end, val_end) == (70, 85)

    dates = test_split_dates(index, sequence_length=5)
    # 15 test rows, the first 5 consumed as history for the first prediction.
    assert len(dates) == 10
    assert pd.Timestamp(dates[0]) == index[90]


# --------------------------------------------------------------------------
# Alignment: the dates have to name the rows the predictions came from
# --------------------------------------------------------------------------


def test_stacked_dates_line_up_with_what_the_dataset_predicts():
    """The IC is only meaningful if row i's date is row i's date.

    Builds two ticker blocks, runs the real `TimeSeriesDataset` over the same
    concatenation, and checks that the target the dataset hands back at
    position i belongs to the row `_stacked_dates` names at position i.
    """
    pytest.importorskip("torch")
    from portfolio_agent.agents.trainer import _stack_blocks, _stacked_dates
    from portfolio_agent.data.dataset import TimeSeriesDataset

    sequence_length = 3
    blocks = []
    for offset, ticker in enumerate(("A", "B")):
        index = pd.date_range("2024-01-01", periods=12) + pd.Timedelta(days=offset)
        blocks.append(
            pd.DataFrame(
                {"feature": np.arange(12, dtype=float),
                 "target": np.arange(12, dtype=float) + 100 * offset},
                index=index,
            )
        )

    features, targets = _stack_blocks(blocks)
    dates = _stacked_dates(blocks, sequence_length)
    dataset = TimeSeriesDataset(features, targets, sequence_length)

    assert len(dataset) == len(dates)
    stacked = pd.concat(blocks)
    for i in range(len(dataset)):
        _, target = dataset[i]
        row = stacked.index.get_indexer_for([dates[i]])
        assert float(target) in set(stacked["target"].to_numpy()[row])
