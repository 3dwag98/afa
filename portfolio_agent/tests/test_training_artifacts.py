"""The checkpoint contract.

The load-bearing test in this file is `test_saved_checkpoint_loads_into_the_strategy`:
it trains nothing, but it asserts that what a trainer writes is what
`IndiaSACStrategy.load()` reads — feature order, scaler and all.

That assertion exists because the failure it catches is silent. A trainer that
serializes the standardizer under the wrong key, or reaches for scikit-learn's
`.mean_`/`.scale_` spelling on a `FeatureScaler` that exposes `.mean`/`.std`,
writes `None` and raises nothing. The model then trains on standardized inputs
and scores raw ones, and every number downstream looks plausible.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from portfolio_agent.config.schema import StrategyConfig
from portfolio_agent.features.scaling import FeatureScaler
from portfolio_agent.strategies.india_sac import (
    DEFAULT_SAC_FEATURES,
    IndiaSACStrategy,
    SACActorNetwork,
)
from portfolio_agent.training.artifacts import build_metadata, load_artifact, save_artifact
from portfolio_agent.training.base import TrainingArtifact
from portfolio_agent.training.trainers.sac import SACActorTrainingNetwork


@pytest.fixture
def scaler():
    rng = np.random.default_rng(0)
    block = rng.normal(size=(500, len(DEFAULT_SAC_FEATURES)))
    return FeatureScaler.fit(block)


def _artifact(scaler, hidden_dim=32):
    actor = SACActorTrainingNetwork(len(DEFAULT_SAC_FEATURES), hidden_dim)
    return TrainingArtifact(
        state_dict=actor.inference_state_dict(),
        metadata=build_metadata(
            feature_names=list(DEFAULT_SAC_FEATURES),
            scaler=scaler,
            trainer="sac",
            extra={"hidden_dim": hidden_dim},
        ),
        metrics={"val_sortino": 1.25},
    )


# --------------------------------------------------------------------------
# The contract that matters
# --------------------------------------------------------------------------


def test_saved_checkpoint_loads_into_the_strategy(tmp_path, scaler):
    """End to end: what save_artifact writes, IndiaSACStrategy.load() reads."""
    save_artifact(_artifact(scaler), tmp_path / "india_sac_best.pt")

    strategy = IndiaSACStrategy(
        StrategyConfig(
            type="india_sac",
            params={"models_dir": str(tmp_path), "model_name": "india_sac", "hidden_dim": 32},
        )
    )
    assert strategy.load() is True

    # The standardizer survived. This is the assertion that fails when a
    # trainer writes `scaler_params` / `.mean_` instead of `feature_scaler` /
    # `.to_dict()`, and nothing else in the stack would have noticed.
    assert strategy._scaler is not None
    np.testing.assert_allclose(strategy._scaler.mean, scaler.mean)
    np.testing.assert_allclose(strategy._scaler.std, scaler.std)

    assert strategy._feature_names == list(DEFAULT_SAC_FEATURES)


def test_checkpoint_survives_weights_only_loading(tmp_path, scaler):
    """The strategy loaders pass `weights_only=True`, which rejects pickles.

    Metadata assembled from pandas or numpy carries `np.float32`/`np.int64`
    values a caller never notices writing, and they fail to load there.
    """
    artifact = _artifact(scaler)
    artifact.metadata["numpy_float"] = np.float32(1.5)
    artifact.metadata["numpy_int"] = np.int64(7)
    artifact.metadata["numpy_array"] = np.arange(3)
    path = save_artifact(artifact, tmp_path / "m_best.pt")

    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["metadata"]["numpy_float"] == pytest.approx(1.5)
    assert payload["metadata"]["numpy_int"] == 7
    assert payload["metadata"]["numpy_array"] == [0, 1, 2]


def test_inference_weights_load_strictly_into_the_inference_class(scaler):
    """The training actor is a strict superset of the inference one.

    If the two ever drift — a renamed layer, an extra head left in — this fails
    here rather than at the first backtest that tries to score with it.
    """
    hidden = 32
    training_actor = SACActorTrainingNetwork(len(DEFAULT_SAC_FEATURES), hidden)
    inference_actor = SACActorNetwork(
        state_dim=len(DEFAULT_SAC_FEATURES), action_dim=1, hidden_dim=hidden
    )

    inference_actor.load_state_dict(training_actor.inference_state_dict(), strict=True)

    # And the deterministic action is identical across the two implementations.
    state = torch.randn(4, len(DEFAULT_SAC_FEATURES))
    with torch.no_grad():
        np.testing.assert_allclose(
            training_actor(state).numpy(), inference_actor(state).numpy(), rtol=1e-6
        )


def test_log_std_head_is_stripped(scaler):
    actor = SACActorTrainingNetwork(len(DEFAULT_SAC_FEATURES), 16)
    keys = set(actor.inference_state_dict())
    assert not any(k.startswith("log_std_head") for k in keys)
    assert any(k.startswith("mean_head") for k in keys)


# --------------------------------------------------------------------------
# build_metadata / save / load
# --------------------------------------------------------------------------


def test_metadata_records_the_scaler_in_the_shape_from_dict_expects(scaler):
    metadata = build_metadata(
        feature_names=["a", "b"], scaler=scaler, trainer="sac"
    )
    payload = metadata["feature_scaler"]
    assert set(payload) == {"mean", "std", "clip"}
    assert FeatureScaler.from_dict(payload) is not None


def test_absent_scaler_is_recorded_explicitly_as_none():
    """`None` distinguishes 'fitted on raw features' from 'someone forgot'."""
    metadata = build_metadata(feature_names=["a"], scaler=None, trainer="sac")
    assert "feature_scaler" in metadata
    assert metadata["feature_scaler"] is None


def test_metadata_without_feature_names_is_refused(scaler):
    with pytest.raises(ValueError, match="feature names"):
        build_metadata(feature_names=[], scaler=scaler, trainer="sac")


def test_save_refuses_an_artifact_missing_feature_names(tmp_path):
    artifact = TrainingArtifact(state_dict={}, metadata={"trainer": "sac"})
    with pytest.raises(ValueError, match="feature_names"):
        save_artifact(artifact, tmp_path / "x_best.pt")


def test_extra_cannot_overwrite_the_fields_inference_needs(scaler):
    metadata = build_metadata(
        feature_names=["a", "b"],
        scaler=scaler,
        trainer="sac",
        extra={"feature_names": ["wrong"], "feature_scaler": None, "hidden_dim": 8},
    )
    assert metadata["feature_names"] == ["a", "b"]
    assert metadata["feature_scaler"] is not None
    assert metadata["hidden_dim"] == 8


def test_artifact_round_trips(tmp_path, scaler):
    original = _artifact(scaler)
    path = save_artifact(original, tmp_path / "r_best.pt")

    loaded = load_artifact(path)
    assert loaded.metadata["feature_names"] == list(DEFAULT_SAC_FEATURES)
    assert loaded.metrics["val_sortino"] == pytest.approx(1.25)
    assert set(loaded.state_dict) == set(original.state_dict)


def test_sidecar_is_written_when_requested(tmp_path, scaler):
    sidecar = tmp_path / "metadata.json"
    save_artifact(_artifact(scaler), tmp_path / "s_best.pt", sidecar_path=sidecar)

    payload = json.loads(sidecar.read_text())
    assert payload["feature_names"] == list(DEFAULT_SAC_FEATURES)
    assert payload["metrics"]["val_sortino"] == pytest.approx(1.25)


def test_loading_a_missing_checkpoint_raises(tmp_path):
    """Never 'start from random weights' — an untrained actor scores nonsense."""
    with pytest.raises(FileNotFoundError):
        load_artifact(tmp_path / "absent_best.pt")


def test_flat_legacy_checkpoints_still_load(tmp_path):
    """`agents/trainer.py` writes metadata at the top level, not under `metadata`."""
    path = tmp_path / "legacy_best.pt"
    torch.save(
        {
            "model_state_dict": {"w": torch.zeros(2)},
            "feature_names": ["a", "b"],
            "sequence_length": 60,
        },
        path,
    )
    loaded = load_artifact(path)
    assert loaded.metadata["feature_names"] == ["a", "b"]
    assert loaded.metadata["sequence_length"] == 60


def test_primary_metric_prefers_validation_numbers():
    artifact = TrainingArtifact(
        state_dict={}, metrics={"final_loss": 9.0, "val_sortino": 2.0}
    )
    assert artifact.primary_metric() == pytest.approx(2.0)


def test_primary_metric_is_none_when_nothing_comparable():
    assert TrainingArtifact(state_dict={}, metrics={"note": "x"}).primary_metric() is None
