"""Tests for the RL exposure policy.

The important ones are the no-look-ahead test and the "can it learn a signal it
is actually given" test. Everything else is scaffolding.
"""

import numpy as np
import pytest

from src.rl import (
    DEFAULT_EXPOSURE_LEVELS,
    EnvironmentConfig,
    ExposureEnvironment,
    LinearSoftmaxPolicy,
    evaluate_policy,
    train_policy,
    walk_forward_policy,
)


def _regime_switching_path(n_periods=1200, seed=0):
    """A path with a genuinely learnable exposure signal.

    Two regimes: a calm one with positive drift, and a crisis one with negative
    drift and triple the volatility. The regime is *observable* in the state, so
    a correct agent learns to be invested in one and flat in the other. If the
    policy cannot find this, the implementation is broken — no amount of market
    realism would rescue it.
    """
    rng = np.random.default_rng(seed)
    states = np.zeros(n_periods, dtype=int)
    for t in range(1, n_periods):
        states[t] = states[t - 1] if rng.random() < 0.98 else 1 - states[t - 1]

    returns = rng.normal(
        np.array([0.0012, -0.0025])[states],
        np.array([0.008, 0.025])[states],
    )
    # One-hot the true regime as the observable probability vector.
    probabilities = np.zeros((n_periods, 2))
    probabilities[np.arange(n_periods), states] = 1.0
    return returns, probabilities, states


class TestEnvironment:
    def test_state_dimensions_match_the_feature_contract(self):
        returns, probabilities, _ = _regime_switching_path(200)
        env = ExposureEnvironment(returns, probabilities)

        state = env.reset()
        # 2 regime probabilities + exposure + trailing volatility + bias
        assert state.size == env.n_features == 5

    def test_works_without_regime_probabilities(self):
        returns, _, _ = _regime_switching_path(200)
        env = ExposureEnvironment(returns)

        assert env.n_features == 3
        assert env.reset().size == 3

    def test_the_reward_comes_from_the_return_after_the_decision(self):
        """Action at t is rewarded by r_{t+1}. Getting this backwards produces
        an agent that appears to predict the market perfectly."""
        returns = np.array([0.0, 0.10, -0.10])
        env = ExposureEnvironment(
            returns, config=EnvironmentConfig(risk_aversion=0.0, turnover_cost=0.0)
        )

        env.reset()
        _, reward, _, info = env.step(env.n_actions - 1)  # full exposure

        assert info["portfolio_return"] == pytest.approx(0.10)
        assert reward == pytest.approx(0.10 * env.config.reward_scale)

    def test_no_look_ahead_in_the_observed_state(self):
        """Corrupting the future must not change any earlier observation."""
        returns, probabilities, _ = _regime_switching_path(400, seed=1)

        def observations(series):
            env = ExposureEnvironment(series, probabilities)
            state = env.reset()
            seen = [state]
            for _ in range(200):
                state, _, done, _ = env.step(0)  # zero exposure: no feedback
                seen.append(state)
                if done:
                    break
            return np.array(seen)

        corrupted = returns.copy()
        corrupted[250:] = 5.0

        assert observations(returns) == pytest.approx(observations(corrupted))

    def test_turnover_is_charged_on_every_change_of_exposure(self):
        returns = np.zeros(5)
        env = ExposureEnvironment(
            returns, config=EnvironmentConfig(risk_aversion=0.0, turnover_cost=0.01)
        )
        env.reset()

        _, _, _, info = env.step(env.n_actions - 1)  # 0 -> 100%
        assert info["turnover"] == pytest.approx(1.0)
        assert info["cost"] == pytest.approx(0.01)

        _, _, _, info = env.step(env.n_actions - 1)  # 100% -> 100%
        assert info["turnover"] == pytest.approx(0.0)
        assert info["cost"] == pytest.approx(0.0)

    def test_risk_aversion_penalizes_large_moves_in_either_direction(self):
        """The quadratic term is what stops the policy collapsing to
        always-maximum-exposure."""
        calm = ExposureEnvironment(
            np.array([0.0, 0.001]),
            config=EnvironmentConfig(risk_aversion=50.0, turnover_cost=0.0),
        )
        wild = ExposureEnvironment(
            np.array([0.0, 0.10]),
            config=EnvironmentConfig(risk_aversion=50.0, turnover_cost=0.0),
        )
        calm.reset(); wild.reset()

        _, calm_reward, _, _ = calm.step(calm.n_actions - 1)
        _, wild_reward, _, _ = wild.step(wild.n_actions - 1)

        # The wild path returns 100x more but is penalized for the variance.
        assert wild_reward < calm_reward * 100

    def test_rejects_a_misaligned_regime_matrix(self):
        with pytest.raises(ValueError):
            ExposureEnvironment(np.zeros(100), np.zeros((50, 2)))

    def test_rejects_non_finite_returns(self):
        with pytest.raises(ValueError):
            ExposureEnvironment(np.array([0.01, np.nan, 0.02]))

    def test_rejects_an_out_of_range_action(self):
        env = ExposureEnvironment(np.zeros(10))
        env.reset()
        with pytest.raises(ValueError):
            env.step(99)

    def test_exhausted_environment_refuses_to_step(self):
        env = ExposureEnvironment(np.zeros(3))
        env.reset()
        for _ in range(env.n_steps):
            env.step(0)
        with pytest.raises(RuntimeError):
            env.step(0)


class TestPolicy:
    def test_starts_uniform_so_nothing_is_assumed(self):
        policy = LinearSoftmaxPolicy(5, 4)
        probabilities = policy.probabilities(np.ones(5))

        assert probabilities == pytest.approx(np.full(4, 0.25))

    def test_probabilities_are_a_distribution(self):
        policy = LinearSoftmaxPolicy(3, 5, seed=1)
        policy.weights = np.random.default_rng(0).normal(0, 3, size=(3, 5))

        probabilities = policy.probabilities(np.array([1.0, -2.0, 0.5]))
        assert probabilities.sum() == pytest.approx(1.0)
        assert np.all(probabilities >= 0)

    def test_survives_extreme_logits_without_overflow(self):
        policy = LinearSoftmaxPolicy(1, 3)
        policy.weights = np.array([[1e4, -1e4, 0.0]])

        probabilities = policy.probabilities(np.array([500.0]))
        assert np.all(np.isfinite(probabilities))
        assert probabilities.sum() == pytest.approx(1.0)

    def test_the_score_function_matches_a_numerical_gradient(self):
        """A wrong sign here trains the policy to do the opposite of what
        works, which looks like 'RL does not work on finance' rather than
        like a bug."""
        policy = LinearSoftmaxPolicy(3, 4, seed=2)
        policy.weights = np.random.default_rng(1).normal(0, 0.5, size=(3, 4))
        state = np.array([0.7, -0.2, 1.0])
        action = 2

        analytic = policy.gradient(state, action)

        step = 1e-6
        numerical = np.zeros_like(analytic)
        for i in range(policy.n_features):
            for j in range(policy.n_actions):
                original = policy.weights[i, j]
                policy.weights[i, j] = original + step
                up = np.log(policy.probabilities(state)[action])
                policy.weights[i, j] = original - step
                down = np.log(policy.probabilities(state)[action])
                policy.weights[i, j] = original
                numerical[i, j] = (up - down) / (2 * step)

        assert analytic == pytest.approx(numerical, abs=1e-6)

    def test_greedy_action_is_the_most_likely_one(self):
        policy = LinearSoftmaxPolicy(2, 3)
        policy.weights = np.array([[0.0, 5.0, 0.0], [0.0, 0.0, 0.0]])

        assert policy.act_greedily(np.array([1.0, 0.0])) == 1


class TestTraining:
    def test_learns_to_stand_aside_in_the_bad_regime(self):
        """The signal is observable in the state, so a working implementation
        must find it. This is the test that says the plumbing is right."""
        returns, probabilities, _ = _regime_switching_path(1200, seed=3)
        env = ExposureEnvironment(returns, probabilities)

        result = train_policy(env, episodes=300, learning_rate=0.05, seed=0)

        levels = np.array(DEFAULT_EXPOSURE_LEVELS)
        calm = levels[result.policy.act_greedily(np.array([1.0, 0.0, 0.0, 0.15, 1.0]))]
        crisis = levels[result.policy.act_greedily(np.array([0.0, 1.0, 0.0, 0.40, 1.0]))]

        assert calm > crisis

    def test_training_is_reproducible(self):
        returns, probabilities, _ = _regime_switching_path(400, seed=4)

        first = train_policy(
            ExposureEnvironment(returns, probabilities), episodes=30, seed=7
        )
        second = train_policy(
            ExposureEnvironment(returns, probabilities), episodes=30, seed=7
        )

        assert first.policy.weights == pytest.approx(second.policy.weights)

    def test_a_different_seed_explores_differently(self):
        returns, probabilities, _ = _regime_switching_path(400, seed=5)

        first = train_policy(ExposureEnvironment(returns, probabilities), episodes=30, seed=1)
        second = train_policy(ExposureEnvironment(returns, probabilities), episodes=30, seed=2)

        assert first.policy.weights != pytest.approx(second.policy.weights)

    def test_turnover_cost_reduces_trading(self):
        """Without a turnover charge the agent rebalances on noise, which is
        free in simulation and ruinous in a delivery account."""
        returns, probabilities, _ = _regime_switching_path(800, seed=6)

        def total_turnover(cost):
            config = EnvironmentConfig(turnover_cost=cost)
            env = ExposureEnvironment(returns, probabilities, config)
            result = train_policy(env, episodes=150, learning_rate=0.05, seed=0)
            return evaluate_policy(
                result.policy, ExposureEnvironment(returns, probabilities, config)
            )["total_turnover"]

        assert total_turnover(0.05) <= total_turnover(0.0)

    def test_reports_its_action_distribution(self):
        returns, probabilities, _ = _regime_switching_path(300, seed=7)
        result = train_policy(ExposureEnvironment(returns, probabilities), episodes=20)

        assert set(result.final_action_distribution) == {
            f"{level:.0%}" for level in DEFAULT_EXPOSURE_LEVELS
        }
        assert sum(result.final_action_distribution.values()) == pytest.approx(1.0)


class TestEvaluation:
    def test_reports_the_always_invested_baseline_alongside(self):
        """An agent that cannot beat holding the book at full exposure has
        learned nothing except to reduce exposure."""
        returns, probabilities, _ = _regime_switching_path(600, seed=8)
        env = ExposureEnvironment(returns, probabilities)
        result = train_policy(env, episodes=100, learning_rate=0.05, seed=0)

        metrics = evaluate_policy(
            result.policy, ExposureEnvironment(returns, probabilities)
        )

        assert "always_invested_return" in metrics
        assert "always_invested_sharpe" in metrics
        assert metrics["n_steps"] > 0
        assert 0.0 <= metrics["mean_exposure"] <= 1.0

    def test_greedy_evaluation_is_deterministic(self):
        returns, probabilities, _ = _regime_switching_path(400, seed=9)
        result = train_policy(
            ExposureEnvironment(returns, probabilities), episodes=30, seed=0
        )

        runs = [
            evaluate_policy(result.policy, ExposureEnvironment(returns, probabilities))
            for _ in range(2)
        ]
        assert runs[0] == runs[1]


class TestWalkForward:
    def test_fits_on_the_head_and_evaluates_on_the_tail(self):
        returns, probabilities, _ = _regime_switching_path(1200, seed=10)

        result = walk_forward_policy(
            returns, probabilities, episodes=200, seed=0, train_fraction=0.6
        )

        assert "train" in result and "test" in result
        # The windows partition the path: roughly 60/40, minus one step each
        # for the terminal return that has no successor to reward it.
        assert result["train"]["n_steps"] == pytest.approx(1200 * 0.6 - 1, abs=2)
        assert result["test"]["n_steps"] == pytest.approx(1200 * 0.4 - 1, abs=2)

    def test_beats_the_baseline_on_held_out_data_when_the_signal_is_real(self):
        """With an observable regime the policy should generalize. This is a
        sanity check on the machinery, NOT evidence that RL works on markets —
        the signal here is handed to the agent noiselessly."""
        returns, probabilities, _ = _regime_switching_path(1500, seed=11)

        result = walk_forward_policy(
            returns, probabilities, episodes=300, seed=0, train_fraction=0.6
        )

        assert result["test"]["sharpe"] > result["test"]["always_invested_sharpe"]

    def test_refuses_a_window_too_short_to_split(self):
        assert "skipped" in walk_forward_policy(np.zeros(40))
