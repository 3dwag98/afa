"""Tests for the Markov-switching regime model."""

import functools
import math

import numpy as np
import pytest

from src.markov_regime import (
    GaussianHMM,
    assess_markov_regime,
    current_regime_state,
    filtered_probabilities,
    fit_gaussian_hmm,
    forecast_state_distribution,
    select_n_states,
    sleeve_weights,
    smoothed_probabilities,
)


@functools.lru_cache(maxsize=None)
def _fitted(n_states=2, n_periods=3000, seed=0, persistence=0.98):
    """Cached (returns, model) pair.

    Baum-Welch over a few thousand observations is not free, and most tests
    below want the same fit. Caching keeps the file fast without weakening any
    assertion — determinism is itself under test above, so a shared model and a
    freshly fitted one are the same object by construction.
    """
    returns, states = _two_state_series(n_periods=n_periods, seed=seed, persistence=persistence)
    return returns, states, fit_gaussian_hmm(returns, n_states=n_states)


def _two_state_series(n_periods=3000, seed=0, persistence=0.98):
    """Simulate a genuinely regime-switching series with known parameters.

    Calm: 0.8%/day volatility. Crisis: 3%/day and a negative drift — the
    asymmetry equity regimes actually show.
    """
    rng = np.random.default_rng(seed)
    means = np.array([0.0006, -0.0015])
    sigmas = np.array([0.008, 0.030])

    states = np.zeros(n_periods, dtype=int)
    for t in range(1, n_periods):
        stay = rng.random() < persistence
        states[t] = states[t - 1] if stay else 1 - states[t - 1]

    returns = rng.normal(means[states], sigmas[states])
    return returns, states


class TestFitGaussianHMM:
    def test_recovers_known_state_parameters(self):
        returns, _, model = _fitted(2)

        assert model is not None
        assert model.n_states == 2
        # States are volatility-ordered, so index 0 is the calm one.
        assert model.annualized_volatility[0] == pytest.approx(
            0.008 * math.sqrt(252), rel=0.15
        )
        assert model.annualized_volatility[1] == pytest.approx(
            0.030 * math.sqrt(252), rel=0.15
        )
        assert model.annualized_drift[0] > 0 > model.annualized_drift[1]

    def test_recovers_the_persistence_of_the_generating_process(self):
        """The property a threshold cascade cannot express at all: how long a
        regime lasts, estimated rather than implied by input autocorrelation."""
        returns, _ = _two_state_series(persistence=0.98)
        model = fit_gaussian_hmm(returns, n_states=2)

        assert model is not None
        assert np.all(model.self_transition_probabilities > 0.90)
        assert model.expected_durations.min() > 10

    def test_states_are_ordered_by_volatility_for_identifiability(self):
        """HMM states are exchangeable, so without a convention two runs can
        produce the same model with the labels swapped and anything keyed to a
        state index becomes unreproducible."""
        returns, _, model = _fitted(3)

        assert model is not None
        assert list(model.variances) == sorted(model.variances)
        assert list(model.annualized_volatility) == sorted(model.annualized_volatility)

    def test_is_deterministic_on_identical_input(self):
        """EM finds a local optimum, so a random start would make the regime
        map — and therefore which sleeves may trade — differ between two runs
        on identical data."""
        returns, _ = _two_state_series()

        first = fit_gaussian_hmm(returns, n_states=3)
        second = fit_gaussian_hmm(returns, n_states=3)

        assert first is not None and second is not None
        assert first.means == pytest.approx(second.means)
        assert first.variances == pytest.approx(second.variances)
        assert first.transition_matrix == pytest.approx(second.transition_matrix)
        assert first.log_likelihood == pytest.approx(second.log_likelihood)

    def test_transition_matrix_rows_are_distributions(self):
        returns, _, model = _fitted(3)

        assert model is not None
        assert model.transition_matrix.sum(axis=1) == pytest.approx(np.ones(3))
        assert np.all(model.transition_matrix >= 0)

    def test_log_likelihood_increases_monotonically_under_em(self):
        """A Baum-Welch iteration that lowers the likelihood is a bug in the
        M-step, and it is easy to write one that looks right."""
        returns, _ = _two_state_series(n_periods=1200, seed=3)

        likelihoods = [
            fit_gaussian_hmm(returns, n_states=2, max_iterations=n).log_likelihood
            for n in (2, 4, 8, 16, 32)
        ]

        assert likelihoods == sorted(likelihoods)

    def test_refuses_to_fit_too_little_history(self):
        """A transition matrix estimated from one year of data mostly describes
        that year."""
        returns, _ = _two_state_series(n_periods=100)
        assert fit_gaussian_hmm(returns, n_states=3) is None

    def test_refuses_a_constant_series(self):
        assert fit_gaussian_hmm(np.zeros(1000), n_states=2) is None

    def test_stationary_distribution_sums_to_one(self):
        returns, _, model = _fitted(2)

        stationary = model.stationary_distribution()
        assert stationary.sum() == pytest.approx(1.0)
        assert np.all(stationary >= 0)
        # And it is genuinely stationary under P.
        assert stationary @ model.transition_matrix == pytest.approx(stationary, abs=1e-8)


class TestNoLookAhead:
    """The distinction the whole module turns on."""

    def test_filtered_probabilities_ignore_every_future_observation(self):
        """P(S_t | F_t) must be identical no matter what happens after t.

        This is the property that makes a regime probability usable as a
        signal, and it is the one a smoothed probability silently violates.
        """
        returns, _, model = _fitted(2)
        assert model is not None

        cutoff = 2000
        original = filtered_probabilities(model, returns)

        corrupted = returns.copy()
        corrupted[cutoff:] = np.random.default_rng(99).normal(0.5, 0.5, size=len(returns) - cutoff)
        after_corruption = filtered_probabilities(model, corrupted)

        assert after_corruption[:cutoff] == pytest.approx(original[:cutoff])

    def test_truncating_the_series_does_not_change_earlier_filtered_rows(self):
        returns, _, model = _fitted(2)

        full = filtered_probabilities(model, returns)
        truncated = filtered_probabilities(model, returns[:1500])

        assert truncated == pytest.approx(full[:1500])

    def test_smoothed_probabilities_do_depend_on_the_future(self):
        """Stated as a test so the difference is impossible to miss: this is
        the quantity that must never reach a trading decision."""
        returns, _, model = _fitted(2)

        full = smoothed_probabilities(model, returns)
        truncated = smoothed_probabilities(model, returns[:1500])

        assert truncated[:1500] != pytest.approx(full[:1500])

    def test_smoothed_is_more_confident_than_filtered(self):
        """Hindsight looks better on every plot, which is what makes it
        dangerous — a backtest wired to it looks excellent and is worthless."""
        returns, _, model = _fitted(2)

        filtered = filtered_probabilities(model, returns)
        smoothed = smoothed_probabilities(model, returns)

        assert float(np.mean(np.max(smoothed, axis=1))) > float(
            np.mean(np.max(filtered, axis=1))
        )


class TestFilteredProbabilities:
    def test_rows_are_probability_distributions(self):
        returns, _, model = _fitted(3)

        probabilities = filtered_probabilities(model, returns)

        assert probabilities.shape == (len(returns), 3)
        assert probabilities.sum(axis=1) == pytest.approx(np.ones(len(returns)))
        assert np.all(probabilities >= 0)

    def test_identifies_the_true_state_better_than_chance(self):
        returns, states, model = _fitted(2)

        probabilities = filtered_probabilities(model, returns)
        predicted = np.argmax(probabilities, axis=1)

        assert float(np.mean(predicted == states)) > 0.85

    def test_an_empty_series_is_not_an_error(self):
        returns, _, model = _fitted(2)

        assert filtered_probabilities(model, []).shape == (0, 2)


class TestStateForecast:
    def test_projects_toward_the_stationary_distribution(self):
        """What a threshold cascade cannot do at all: say where the regime is
        likely to be, not merely where it is."""
        returns, _, model = _fitted(2)

        certain_calm = np.array([1.0, 0.0])
        near = forecast_state_distribution(model, certain_calm, horizon_days=1)
        far = forecast_state_distribution(model, certain_calm, horizon_days=2000)

        assert near[0] > far[0]
        assert far == pytest.approx(model.stationary_distribution(), abs=1e-4)

    def test_a_zero_horizon_returns_the_input(self):
        returns, _, model = _fitted(2)

        assert forecast_state_distribution(model, [0.3, 0.7], horizon_days=0) == pytest.approx(
            [0.3, 0.7]
        )

    def test_rejects_a_wrong_length_distribution(self):
        returns, _, model = _fitted(2)

        with pytest.raises(ValueError):
            forecast_state_distribution(model, [0.5, 0.3, 0.2])


class TestModelSelection:
    def test_bic_prefers_two_states_for_a_two_state_process(self):
        """The number of regimes becomes an estimate with a stated criterion,
        rather than four states named in advance."""
        returns, _ = _two_state_series(n_periods=4000, seed=1)

        best, scores = select_n_states(returns, candidates=(1, 2, 3, 4))

        assert best is not None
        assert best.n_states == 2
        assert len(scores) == 4
        assert min(scores, key=lambda s: s["bic"])["n_states"] == 2

    def test_reports_every_candidate_so_the_margin_is_visible(self):
        returns, _ = _two_state_series(n_periods=2000, seed=2)
        _, scores = select_n_states(returns, candidates=(2, 3))

        assert [s["n_states"] for s in scores] == [2.0, 3.0]
        for entry in scores:
            # No sign constraint: for continuous densities the log-likelihood
            # is positive (a density exceeds 1 wherever sigma is small), so BIC
            # is routinely negative. Only the ordering between candidates
            # carries meaning.
            assert math.isfinite(entry["bic"])
            assert "log_likelihood" in entry

    def test_bic_penalizes_parameters_harder_than_aic(self):
        returns, _ = _two_state_series(n_periods=2000, seed=4)
        model = fit_gaussian_hmm(returns, n_states=4)

        assert model.bic() > model.aic()
        assert model.n_parameters == 4 * 3 + 2 * 4 + 3

    def test_rejects_an_unknown_criterion(self):
        with pytest.raises(ValueError):
            select_n_states(np.zeros(500), criterion="rmse")

    def test_returns_none_when_nothing_can_be_fitted(self):
        best, scores = select_n_states(np.zeros(50), candidates=(2, 3))
        assert best is None
        assert scores == []


class TestSleeveWeights:
    def test_blends_sleeve_affinities_by_state_probability(self):
        weights = sleeve_weights(
            [0.7, 0.3],
            {"momentum": [1.0, 0.0], "low_volatility": [0.0, 1.0]},
        )

        assert weights["momentum"] == pytest.approx(0.7)
        assert weights["low_volatility"] == pytest.approx(0.3)

    def test_varies_smoothly_where_a_hard_label_would_flip(self):
        """The defect this fixes: a benchmark oscillating around its 200-day
        average flipped the whole permission map day to day."""
        affinity = {"momentum": [1.0, 0.0]}
        crossing = [
            sleeve_weights([p, 1 - p], affinity)["momentum"]
            for p in (0.52, 0.51, 0.50, 0.49, 0.48)
        ]

        # Monotone and small steps — no discontinuity at the 50% boundary a
        # threshold rule would have snapped across.
        assert crossing == sorted(crossing, reverse=True)
        assert max(abs(a - b) for a, b in zip(crossing, crossing[1:])) < 0.02

    def test_normalizes_an_unnormalized_probability_vector(self):
        assert sleeve_weights([2.0, 2.0], {"s": [1.0, 0.0]})["s"] == pytest.approx(0.5)

    def test_a_degenerate_distribution_mutes_everything(self):
        assert sleeve_weights([0.0, 0.0], {"s": [1.0, 1.0]}) == {"s": 0.0}

    def test_rejects_a_misaligned_affinity_row(self):
        """A silently truncated row would enable a strategy in the state it was
        meant to be muted in."""
        with pytest.raises(ValueError, match="momentum"):
            sleeve_weights([0.5, 0.3, 0.2], {"momentum": [1.0, 0.0]})


class TestRegimeState:
    def test_describes_the_final_date(self):
        returns, _, model = _fitted(2)

        state = current_regime_state(model, returns)

        assert state is not None
        assert state.probabilities.sum() == pytest.approx(1.0)
        assert state.most_likely_name in ("CALM", "TURBULENT")
        assert 0.0 <= state.confidence <= 1.0
        assert state.expected_volatility > 0

    def test_expected_volatility_blends_states_rather_than_picking_one(self):
        """The practical value of a probabilistic regime: a 60/40 split between
        calm and crisis is a materially riskier market than a confident calm
        reading, and a hard label reports both as 'calm'."""
        model = GaussianHMM(
            means=np.array([0.0, 0.0]),
            variances=np.array([(0.01) ** 2, (0.04) ** 2]),
            transition_matrix=np.array([[0.98, 0.02], [0.02, 0.98]]),
            initial_distribution=np.array([0.5, 0.5]),
            log_likelihood=0.0,
            n_observations=1000,
            iterations=1,
            converged=True,
        )
        low, high = model.annualized_volatility

        blended = float(np.dot([0.6, 0.4], model.annualized_volatility))

        assert low < blended < high
        assert blended == pytest.approx(0.6 * low + 0.4 * high)

    def test_summary_is_report_ready(self):
        returns, _, model = _fitted(2)

        summary = model.summary()

        assert summary["n_states"] == 2
        assert len(summary["expected_durations_days"]) == 2
        assert summary["bic"] == pytest.approx(model.bic())
        assert set(current_regime_state(model, returns).as_dict()) == {
            "probabilities", "expected_volatility", "most_likely_state", "confidence",
        }


class TestAssessMarkovRegime:
    """The end-to-end entry point a caller actually uses."""

    def _close(self, n_periods=2500, seed=0):
        import pandas as pd

        returns, _ = _two_state_series(n_periods=n_periods, seed=seed)
        return pd.Series(100 * np.cumprod(1 + returns))

    def test_reads_a_regime_from_a_price_series(self):
        state = assess_markov_regime(self._close())

        assert state is not None
        assert state.probabilities.sum() == pytest.approx(1.0)
        assert state.most_likely_name in ("CALM", "TURBULENT", "CRISIS", "NORMAL")
        assert state.expected_volatility > 0
        assert state.model_summary["n_states"] in (2, 3, 4)

    def test_recovers_the_generating_persistence(self):
        """A self-transition diagonal well below ~0.9 means the fit is
        describing noise rather than regimes."""
        state = assess_markov_regime(self._close())

        persistence = state.model_summary["self_transition_probabilities"]
        assert min(persistence) > 0.90

    def test_parameters_are_estimated_only_on_the_training_head(self):
        """Fitting on the full sample would let the parameters see the period
        the filter is later read over — a subtler leak than using smoothed
        probabilities, and just as real.
        """
        import pandas as pd

        returns, _ = _two_state_series(n_periods=2500, seed=0)
        close = pd.Series(100 * np.cumprod(1 + returns))

        baseline = assess_markov_regime(close, train_fraction=0.6)

        # Rewrite the final third — everything past the training window — and
        # the *fitted parameters* must not move.
        corrupted = returns.copy()
        tail = int(len(returns) * 0.6)
        corrupted[tail:] = np.random.default_rng(5).normal(0.0, 0.05, len(returns) - tail)
        altered = assess_markov_regime(
            pd.Series(100 * np.cumprod(1 + corrupted)), train_fraction=0.6
        )

        assert altered.model_summary["annualized_volatility"] == pytest.approx(
            baseline.model_summary["annualized_volatility"]
        )
        assert altered.model_summary["self_transition_probabilities"] == pytest.approx(
            baseline.model_summary["self_transition_probabilities"]
        )
        # The *state read* is allowed to differ — it should, the tail changed.
        assert altered.probabilities != pytest.approx(baseline.probabilities)

    def test_accepts_returns_directly(self):
        returns, _ = _two_state_series()
        assert assess_markov_regime(returns=returns) is not None

    def test_returns_none_on_too_little_history(self):
        """Callers fall back to the threshold classification rather than trade
        a model they could not estimate."""
        assert assess_markov_regime(self._close(n_periods=200)) is None
        assert assess_markov_regime(None) is None

    def test_a_non_series_input_is_refused_rather_than_guessed_at(self):
        assert assess_markov_regime(market_close=[1.0, 2.0, 3.0]) is None
