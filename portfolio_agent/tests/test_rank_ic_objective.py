"""Training on the metric the model is judged on.

Every trainer here fits a pointwise loss and is then ranked by rank IC. The
`gbm` baseline went further and chose *which iteration ships* by validation
MSE, so even model selection never looked at the number in the summary table.

Two changes, tested separately because they are separable: the baseline can now
select on IC (and does by default), and there is a trainer whose gradient is
the IC surrogate's.

The central correctness test is `test_the_gradient_matches_a_numerical_derivative`.
A closed-form gradient that is subtly wrong still trains — it just trains
towards something else — so it is checked against finite differences rather
than against another formula.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from portfolio_agent.evaluation.metrics import MIN_CROSS_SECTION_NAMES
from portfolio_agent.training.registry import get_trainer, list_trainers
from portfolio_agent.training.trainers.gbm import SELECTION_METRICS, rank_ic_by_date
from portfolio_agent.training.trainers.rank_ic import (
    AdditiveEnsemble,
    RankICTrainer,
    RankICTrainerConfig,
    date_ic_gradient,
    ic_gradient,
    mean_date_correlation,
)


def correlation(a, b) -> float:
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 0 else 0.0


# --------------------------------------------------------------------------
# The gradient
# --------------------------------------------------------------------------


class TestTheGradient:
    def test_the_gradient_matches_a_numerical_derivative(self):
        """Checked against finite differences, not against another formula.

        A closed form that is subtly wrong still trains — it trains towards
        something else — and no downstream metric would reveal which.
        """
        rng = np.random.default_rng(0)
        scores, labels = rng.normal(size=12), rng.normal(size=12)

        analytic = date_ic_gradient(scores, labels)

        epsilon = 1e-6
        numeric = np.zeros_like(scores)
        for i in range(len(scores)):
            up, down = scores.copy(), scores.copy()
            up[i] += epsilon
            down[i] -= epsilon
            numeric[i] = (correlation(up, labels) - correlation(down, labels)) / (
                2 * epsilon
            )

        np.testing.assert_allclose(analytic, numeric, atol=1e-8)

    def test_it_points_uphill(self):
        """A small step along the gradient must raise the correlation."""
        rng = np.random.default_rng(1)
        scores, labels = rng.normal(size=30), rng.normal(size=30)
        gradient = date_ic_gradient(scores, labels)

        before = correlation(scores, labels)
        after = correlation(scores + 1e-3 * gradient, labels)
        assert after > before

    def test_constant_scores_give_the_centred_label_direction(self):
        """Iteration one, where a naive implementation divides by zero.

        Every score is identical at initialization, so the correlation is
        undefined. The limit is not: with no score dispersion the direction
        that most increases correlation is the centred label.
        """
        gradient = date_ic_gradient(np.zeros(5), np.array([1.0, 2, 3, 4, 5]))
        assert not np.isnan(gradient).any()
        # Proportional to the centred label.
        centred = np.array([-2.0, -1, 0, 1, 2])
        assert correlation(gradient, centred) == pytest.approx(1.0)

    def test_a_constant_label_gives_no_direction(self):
        """No ordering claim to move towards, so no move."""
        rng = np.random.default_rng(2)
        gradient = date_ic_gradient(rng.normal(size=5), np.ones(5))
        assert np.all(gradient == 0.0)

    def test_the_gradient_sums_to_zero_within_a_date(self):
        """Correlation is invariant to a constant shift of the scores, so the
        gradient must have no component along that direction."""
        rng = np.random.default_rng(3)
        gradient = date_ic_gradient(rng.normal(size=20), rng.normal(size=20))
        assert gradient.sum() == pytest.approx(0.0, abs=1e-12)

    def test_it_is_invariant_to_rescaling_the_scores(self):
        """Correlation does not change if scores are doubled; the gradient
        scales as 1/scale, which is what keeps the step size sane."""
        rng = np.random.default_rng(4)
        scores, labels = rng.normal(size=15), rng.normal(size=15)
        base = date_ic_gradient(scores, labels)
        scaled = date_ic_gradient(2 * scores, labels)
        np.testing.assert_allclose(scaled, base / 2, rtol=1e-9)


class TestThePanelGradient:
    def _panel(self, n_dates=10, n_names=20, seed=5):
        rng = np.random.default_rng(seed)
        dates = np.repeat(np.arange(n_dates), n_names)
        return (
            rng.normal(size=n_dates * n_names),
            rng.normal(size=n_dates * n_names),
            dates,
        )

    def test_each_date_is_handled_independently(self):
        scores, labels, dates = self._panel()
        gradient = ic_gradient(scores, labels, dates)
        for date in np.unique(dates):
            rows = dates == date
            expected = date_ic_gradient(scores[rows], labels[rows]) / 10
            np.testing.assert_allclose(gradient[rows], expected, rtol=1e-9)

    def test_thin_dates_contribute_nothing(self):
        """The same threshold the evaluation layer drops them at, so the
        objective is not optimized on cross-sections the metric will refuse."""
        rng = np.random.default_rng(6)
        dates = np.array([0] * 20 + [1] * (MIN_CROSS_SECTION_NAMES - 1))
        scores, labels = rng.normal(size=len(dates)), rng.normal(size=len(dates))

        gradient = ic_gradient(scores, labels, dates)
        assert np.all(gradient[dates == 1] == 0.0)
        assert np.any(gradient[dates == 0] != 0.0)

    def test_it_averages_over_dates_rather_than_summing(self):
        """Otherwise the step size depends on how long the panel is."""
        scores, labels, dates = self._panel(n_dates=10)
        short = ic_gradient(scores[:100], labels[:100], dates[:100])
        full = ic_gradient(scores, labels, dates)
        # The first date's rows appear in both; averaging over 5 dates vs 10
        # halves the contribution.
        np.testing.assert_allclose(full[:20], short[:20] / 2, rtol=1e-9)

    def test_an_all_thin_panel_returns_zeros_without_dividing_by_zero(self):
        dates = np.arange(4)
        gradient = ic_gradient(np.zeros(4), np.arange(4.0), dates)
        assert np.all(gradient == 0.0)


class TestTheObjectiveValue:
    def test_it_is_the_mean_of_the_per_date_correlations(self):
        rng = np.random.default_rng(7)
        dates = np.repeat(np.arange(6), 15)
        scores, labels = rng.normal(size=90), rng.normal(size=90)

        expected = np.mean(
            [correlation(scores[dates == d], labels[dates == d]) for d in range(6)]
        )
        assert mean_date_correlation(scores, labels, dates) == pytest.approx(expected)

    def test_a_perfect_ordering_scores_one(self):
        dates = np.repeat(np.arange(4), 10)
        labels = np.tile(np.arange(10.0), 4)
        assert mean_date_correlation(labels, labels, dates) == pytest.approx(1.0)

    def test_a_reversed_ordering_scores_minus_one(self):
        dates = np.repeat(np.arange(4), 10)
        labels = np.tile(np.arange(10.0), 4)
        assert mean_date_correlation(-labels, labels, dates) == pytest.approx(-1.0)


# --------------------------------------------------------------------------
# What the objective buys, measured
# --------------------------------------------------------------------------


def market_dominated_panel(n_dates=150, n_names=40, seed=1):
    """A label whose variance is almost all market, as an equity panel's is.

    96% of the label's variance is the day's market move, shared by every name
    and contributing nothing to the ordering. This is the shape that separates
    the two objectives: squared error must explain the market before it can get
    to the cross-section, and the correlation objective never sees it because
    it centres within each date.
    """
    rng = np.random.default_rng(seed)
    dates = np.repeat(np.arange(n_dates), n_names)
    signal = rng.normal(size=n_dates * n_names)
    market = np.repeat(rng.normal(0.0, 0.03, n_dates), n_names)
    features = np.column_stack(
        [
            signal,
            market + rng.normal(0.0, 0.001, n_dates * n_names),
            rng.normal(size=n_dates * n_names),
        ]
    )
    labels = market + 0.004 * signal + rng.normal(0.0, 0.004, n_dates * n_names)
    return features, labels, dates, market


def fit_ic_objective(features, labels, dates, iterations, learning_rate=0.1):
    from sklearn.tree import DecisionTreeRegressor

    scores = np.zeros(len(labels))
    ensemble = AdditiveEnsemble(learning_rate)
    for i in range(iterations):
        gradient = ic_gradient(scores, labels, dates)
        tree = DecisionTreeRegressor(
            max_depth=3, min_samples_leaf=50, random_state=i
        ).fit(features, gradient)
        ensemble.append(tree)
        scores += learning_rate * tree.predict(features)
    return ensemble


def fit_squared_error(features, labels, iterations, learning_rate=0.1):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_iter=iterations, learning_rate=learning_rate,
        max_depth=3, min_samples_leaf=50, random_state=0,
    ).fit(features, labels)


class TestWhatTheObjectiveBuys:
    def test_the_panel_really_is_market_dominated(self):
        """Guard on the fixture: if this stops holding the comparison is
        measuring something else."""
        _, labels, _, market = market_dominated_panel()
        assert np.var(market) / np.var(labels) > 0.9

    def test_it_converges_in_far_fewer_trees(self):
        """The finding, and it is about *rate*, not ceiling.

        Squared error must explain the market before it reaches the
        cross-section, so its early capacity buys no ordering skill at all. At
        five trees the IC objective is already at 0.52 while squared error is
        at 0.06. This matters precisely because early stopping cuts the budget.
        """
        features, labels, dates, _ = market_dominated_panel()

        ic_model = fit_ic_objective(features, labels, dates, iterations=5)
        mse_model = fit_squared_error(features, labels, iterations=5)

        ic_skill = rank_ic_by_date(ic_model.predict(features), labels, dates).mean()
        mse_skill = rank_ic_by_date(mse_model.predict(features), labels, dates).mean()

        assert ic_skill > 0.4
        assert mse_skill < 0.2
        assert ic_skill > 3 * mse_skill

    def test_given_enough_trees_squared_error_catches_up(self):
        """Reported because it is true, and because the honest claim is
        narrower than "the new objective is better"."""
        features, labels, dates, _ = market_dominated_panel()

        ic_model = fit_ic_objective(features, labels, dates, iterations=80)
        mse_model = fit_squared_error(features, labels, iterations=80)

        ic_skill = rank_ic_by_date(ic_model.predict(features), labels, dates).mean()
        mse_skill = rank_ic_by_date(mse_model.predict(features), labels, dates).mean()
        assert abs(ic_skill - mse_skill) < 0.02

    def test_the_ic_objective_is_much_worse_at_squared_error(self):
        """It is not fitting the level at all, and should not be mistaken for
        a model that does. Its output is an ordering, not a return forecast."""
        features, labels, dates, _ = market_dominated_panel()

        ic_model = fit_ic_objective(features, labels, dates, iterations=80)
        mse_model = fit_squared_error(features, labels, iterations=80)

        ic_error = np.mean((ic_model.predict(features) - labels) ** 2)
        mse_error = np.mean((mse_model.predict(features) - labels) ** 2)
        assert ic_error > 10 * mse_error


# --------------------------------------------------------------------------
# The ensemble
# --------------------------------------------------------------------------


class TestAdditiveEnsemble:
    def _ensemble(self, n=5):
        from sklearn.tree import DecisionTreeRegressor

        rng = np.random.default_rng(8)
        x = rng.normal(size=(200, 3))
        ensemble = AdditiveEnsemble(0.1)
        for i in range(n):
            ensemble.append(
                DecisionTreeRegressor(max_depth=2, random_state=i).fit(
                    x, rng.normal(size=200)
                )
            )
        return ensemble, x

    def test_prediction_is_the_shrunk_sum(self):
        ensemble, x = self._ensemble()
        expected = sum(0.1 * tree.predict(x) for tree in ensemble.trees)
        np.testing.assert_allclose(ensemble.predict(x), expected)

    def test_truncation_is_exact(self):
        """Unlike a warm-started histogram ensemble, where truncating in place
        is not part of the public API. Here the prediction is a plain sum, so
        dropping the tail reproduces the model that existed at that size."""
        ensemble, x = self._ensemble(n=8)
        at_three = AdditiveEnsemble(0.1, list(ensemble.trees[:3]))
        ensemble.truncate(3)
        np.testing.assert_array_equal(ensemble.predict(x), at_three.predict(x))
        assert len(ensemble) == 3

    def test_an_empty_ensemble_predicts_zero(self):
        empty = AdditiveEnsemble(0.1)
        np.testing.assert_array_equal(empty.predict(np.zeros((4, 2))), np.zeros(4))

    def test_it_survives_joblib(self):
        """The checkpoint writer is joblib, same as the gbm baseline's."""
        import io

        import joblib

        ensemble, x = self._ensemble()
        buffer = io.BytesIO()
        joblib.dump(ensemble, buffer)
        buffer.seek(0)
        np.testing.assert_allclose(joblib.load(buffer).predict(x), ensemble.predict(x))


# --------------------------------------------------------------------------
# Registration and config
# --------------------------------------------------------------------------


class TestRegistration:
    def test_the_trainer_is_registered(self):
        assert "rank_ic" in list_trainers()
        assert get_trainer("rank_ic") is RankICTrainer

    def test_it_writes_joblib_not_torch(self):
        assert RankICTrainer.checkpoint_suffix == ".joblib"

    def test_its_config_inherits_the_baselines_panel_settings(self):
        """The panel, split, purge and label must be identical to `gbm`'s, or
        a comparison between them is not a comparison of objectives."""
        cfg = RankICTrainerConfig()
        assert cfg.target_transform == "cross_sectional_rank"
        assert cfg.feature_normalization == "cross_sectional"
        assert cfg.purge_days is None
        assert cfg.train_fraction == 0.8

    def test_a_gbm_config_still_validates_against_it(self):
        """So a config written for the baseline can be pointed at this trainer."""
        cfg = RankICTrainerConfig(
            epochs=30, learning_rate=0.05, max_bins=128, min_samples_leaf=50
        )
        assert cfg.epochs == 30

    def test_date_subsampling_is_the_only_subsample_offered(self):
        cfg = RankICTrainerConfig(subsample=0.5)
        assert cfg.subsample == 0.5
        with pytest.raises(ValueError):
            RankICTrainerConfig(subsample=0.0)

    def test_unknown_settings_are_still_refused(self):
        with pytest.raises(ValueError):
            RankICTrainerConfig(nonexistent_knob=1)


class TestSelectionMetric:
    def test_the_baseline_now_selects_on_rank_ic_by_default(self):
        """The mismatch at its source: the iteration that ships is chosen by
        the metric the model is reported on."""
        from portfolio_agent.training.trainers.gbm import GBMTrainerConfig

        assert GBMTrainerConfig().selection_metric == "rank_ic"

    def test_mse_selection_is_still_available(self):
        """Kept so the difference between the two stays measurable rather than
        becoming a claim in a commit message."""
        from portfolio_agent.training.trainers.gbm import GBMTrainerConfig

        assert GBMTrainerConfig(selection_metric="mse").selection_metric == "mse"

    def test_an_unknown_selection_metric_is_refused(self):
        from portfolio_agent.training.trainers.gbm import GBMTrainerConfig

        with pytest.raises(ValueError, match="selection_metric must be one of"):
            GBMTrainerConfig(selection_metric="sharpe")

    def test_both_documented_options_are_the_ones_implemented(self):
        assert set(SELECTION_METRICS) == {"rank_ic", "mse"}
