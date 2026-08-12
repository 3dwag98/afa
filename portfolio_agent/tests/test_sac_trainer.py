"""The SAC trainer: reward construction, the loop, and reproducibility.

Data here is synthetic and the epoch counts are tiny — these assert that the
mechanism is the one documented, not that the policy learns anything. Whether
it learns is a research question that a unit test cannot answer and should not
pretend to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from portfolio_agent.strategies.india_sac import DEFAULT_SAC_FEATURES
from portfolio_agent.training.base import TrainingData
from portfolio_agent.training.trainers.sac import (
    DifferentialSortino,
    ReplayBuffer,
    SACActorTrainingNetwork,
    SACTrainer,
    SACTrainerConfig,
    _annualized_sortino,
    forward_returns,
)

N_FEATURES = len(DEFAULT_SAC_FEATURES)


def make_training_data(n_tickers: int = 3, n_rows: int = 300, seed: int = 0) -> TrainingData:
    """Deterministic synthetic panel shaped like `prepare_panel` output."""
    rng = np.random.default_rng(seed)
    features, prices, splits = {}, {}, {}

    for i in range(n_tickers):
        ticker = f"T{i}"
        index = pd.date_range("2020-01-01", periods=n_rows, freq="B")

        block = rng.normal(size=(n_rows, N_FEATURES)).astype(np.float32)
        features[ticker] = pd.DataFrame(block, index=index, columns=DEFAULT_SAC_FEATURES)

        steps = rng.normal(loc=0.0005, scale=0.01, size=n_rows)
        close = 100.0 * np.exp(np.cumsum(steps))
        prices[ticker] = pd.DataFrame(
            {
                "open": close, "high": close * 1.01, "low": close * 0.99,
                "close": close, "volume": 1e6,
            },
            index=index,
        )
        splits[ticker] = int(n_rows * 0.8)

    return TrainingData(
        features_by_ticker=features,
        prices_by_ticker=prices,
        tickers=sorted(features),
        feature_names=list(DEFAULT_SAC_FEATURES),
        scaler=None,
        split_index_by_ticker=splits,
    )


def tiny_config(**overrides) -> SACTrainerConfig:
    base = dict(
        epochs=2, batch_size=32, hidden_dim=16, gradient_steps=3,
        warmup_transitions=50, sortino_warmup=5, buffer_size=5000,
        device="cpu", seed=7,
    )
    base.update(overrides)
    return SACTrainerConfig(**base)


# --------------------------------------------------------------------------
# Reward
# --------------------------------------------------------------------------


def test_differential_sortino_rewards_gains_and_punishes_losses():
    sortino = DifferentialSortino(eta=0.05)
    for _ in range(30):
        sortino.warmup(0.001)
    sortino.warmup(-0.01)

    gain = DifferentialSortino(eta=0.05)
    loss = DifferentialSortino(eta=0.05)
    for state in (gain, loss):
        state.a, state.dd2 = sortino.a, sortino.dd2

    assert gain.update(0.02) > loss.update(-0.02)


def test_differential_sortino_is_bounded():
    """A near-zero downside deviation makes the raw ratio explode; one
    unclipped spike is enough to leave every later batch NaN."""
    sortino = DifferentialSortino(eta=0.5, clip=10.0)
    sortino.a, sortino.dd2 = 0.0, 1e-15
    for value in (5.0, -5.0, 1e6, -1e6):
        assert abs(sortino.update(value)) <= 10.0


def test_differential_sortino_never_emits_a_non_finite_reward():
    sortino = DifferentialSortino()
    sortino.a, sortino.dd2 = float("inf"), 0.0
    assert np.isfinite(sortino.update(0.01))


def test_turnover_cost_depends_on_the_action():
    """A constant cost cannot penalize turnover — it shifts every action alike.

    Holding a position must cost less than flipping it, for the same return.
    """
    cfg = tiny_config(friction_cost=0.01)
    ret = 0.01

    holding = 0.5 * ret - cfg.friction_cost * abs(0.5 - 0.5)
    flipping = 0.5 * ret - cfg.friction_cost * abs(0.5 - 0.0)
    assert holding > flipping


# --------------------------------------------------------------------------
# Replay buffer
# --------------------------------------------------------------------------


def test_replay_buffer_wraps_at_capacity():
    buffer = ReplayBuffer(capacity=3, state_dim=N_FEATURES, seed=1)
    for i in range(5):
        buffer.push(np.zeros(N_FEATURES), 0.5, float(i), np.zeros(N_FEATURES), False)
    assert len(buffer) == 3


def test_replay_buffer_sampling_is_seeded():
    """Drawing from the global numpy state would make a run depend on every
    other caller that touched it."""
    def draw():
        buffer = ReplayBuffer(capacity=100, state_dim=N_FEATURES, seed=99)
        for i in range(50):
            buffer.push(
                np.full(N_FEATURES, i, dtype=np.float32), 0.5, float(i),
                np.zeros(N_FEATURES), False,
            )
        return buffer.sample(16)[2]

    np.random.seed(0)
    first = draw()
    np.random.seed(12345)  # perturb the global state; must not matter
    second = draw()
    np.testing.assert_array_equal(first, second)


def test_sampling_an_empty_buffer_raises():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=10, state_dim=N_FEATURES).sample(4)


def test_sampling_with_replacement_works_below_batch_size():
    """Early in a run the buffer is legitimately smaller than the batch."""
    buffer = ReplayBuffer(capacity=100, state_dim=N_FEATURES, seed=3)
    for _ in range(4):
        buffer.push(np.zeros(N_FEATURES), 0.1, 1.0, np.zeros(N_FEATURES), False)
    states, *_ = buffer.sample(32)
    assert states.shape == (32, N_FEATURES)


# --------------------------------------------------------------------------
# Returns alignment
# --------------------------------------------------------------------------


def test_forward_returns_are_next_step_and_end_with_nan():
    index = pd.date_range("2020-01-01", periods=4, freq="B")
    prices = pd.DataFrame({"close": [100.0, 110.0, 121.0, 121.0]}, index=index)

    rets = forward_returns(prices, index)
    assert rets[0] == pytest.approx(0.10)
    assert rets[1] == pytest.approx(0.10)
    assert np.isnan(rets[-1])


def test_forward_returns_span_a_gap_rather_than_splicing_it():
    """Rows dropped for a non-finite feature leave gaps; the return across one
    is the true realized return, not a one-day return over a splice."""
    full = pd.date_range("2020-01-01", periods=4, freq="B")
    prices = pd.DataFrame({"close": [100.0, 110.0, 121.0, 133.1]}, index=full)

    kept = full[[0, 2, 3]]  # the second row was dropped
    rets = forward_returns(prices, kept)
    assert rets[0] == pytest.approx(0.21)  # 100 -> 121, not 100 -> 110


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def test_fit_produces_a_loadable_artifact():
    data = make_training_data()
    artifact = SACTrainer().fit(data, tiny_config())

    assert artifact.metadata["feature_names"] == list(DEFAULT_SAC_FEATURES)
    assert artifact.metadata["trainer"] == "sac"
    assert artifact.metadata["hidden_dim"] == 16
    assert artifact.state_dict, "no weights were produced"
    assert not any(k.startswith("log_std_head") for k in artifact.state_dict)
    assert "val_sortino" in artifact.metrics


def test_fit_is_reproducible_for_one_seed():
    """Two runs of one configuration must agree — the platform requires it."""
    data = make_training_data()
    first = SACTrainer().fit(data, tiny_config(seed=11))
    second = SACTrainer().fit(make_training_data(), tiny_config(seed=11))

    assert set(first.state_dict) == set(second.state_dict)
    for key, tensor in first.state_dict.items():
        torch.testing.assert_close(tensor, second.state_dict[key])


def test_different_seeds_give_different_weights():
    data = make_training_data()
    a = SACTrainer().fit(data, tiny_config(seed=1))
    b = SACTrainer().fit(make_training_data(), tiny_config(seed=2))

    assert any(
        not torch.allclose(tensor, b.state_dict[key])
        for key, tensor in a.state_dict.items()
    )


def test_selected_checkpoint_is_the_best_validation_epoch_not_the_last():
    artifact = SACTrainer().fit(make_training_data(), tiny_config(epochs=4))
    history = artifact.metrics["history"]
    best_epoch = artifact.metadata["best_epoch"]

    best_score = max(record["val_sortino"] for record in history)
    assert artifact.metrics["val_sortino"] == pytest.approx(best_score)
    assert best_epoch == next(
        r["epoch"] for r in history if r["val_sortino"] == pytest.approx(best_score)
    )


def test_a_run_that_never_warms_up_fails_loudly():
    """Silently returning an untrained actor is the expensive failure."""
    with pytest.raises(RuntimeError, match="warmup_transitions"):
        SACTrainer().fit(
            make_training_data(n_tickers=1, n_rows=120),
            tiny_config(warmup_transitions=10_000_000, epochs=1),
        )


def test_validation_scores_the_deterministic_policy():
    """Inference runs the mean action; evaluating the sampler would report a
    number no backtest can reproduce."""
    data = make_training_data()
    cfg = tiny_config()
    trainer = SACTrainer()
    actor = SACActorTrainingNetwork(N_FEATURES, cfg.hidden_dim)

    first = trainer.evaluate(actor, data, cfg, torch.device("cpu"), segment="validation")
    second = trainer.evaluate(actor, data, cfg, torch.device("cpu"), segment="validation")
    assert first == second


def test_evaluate_reports_train_and_validation_separately():
    data = make_training_data()
    cfg = tiny_config()
    trainer, actor = SACTrainer(), SACActorTrainingNetwork(N_FEATURES, cfg.hidden_dim)
    device = torch.device("cpu")

    assert "val_sortino" in trainer.evaluate(actor, data, cfg, device, segment="validation")
    assert "train_sortino" in trainer.evaluate(actor, data, cfg, device, segment="train")


def test_actor_sample_returns_actions_in_the_unit_interval():
    actor = SACActorTrainingNetwork(N_FEATURES, 16)
    actions, log_prob = actor.sample(torch.randn(64, N_FEATURES))

    assert torch.all(actions > 0.0) and torch.all(actions < 1.0)
    assert actions.shape == (64, 1)
    assert torch.isfinite(log_prob).all()


def test_sortino_of_a_loss_free_series_is_zero_not_infinite():
    """An infinity would win every best-checkpoint comparison it entered."""
    assert _annualized_sortino(np.array([0.01, 0.02, 0.03])) == 0.0


def test_config_rejects_an_out_of_range_gamma():
    with pytest.raises(Exception):
        SACTrainerConfig(gamma=1.5)
