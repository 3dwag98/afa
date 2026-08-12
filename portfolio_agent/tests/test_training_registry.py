"""The pluggable training layer: registry, config resolution, universes.

These cover the seams rather than the learning: that a trainer can be looked
up, that the right hyperparameters reach it, that a wrong one stops the run,
and that two runs can be pinned to the same names.
"""

from __future__ import annotations

import json

import pytest

from portfolio_agent.config.loader import load_config
from portfolio_agent.training import registry
from portfolio_agent.training.base import BaseTrainer, TrainerConfig, TrainingData
from portfolio_agent.training.config import (
    load_strategy_training_block,
    parse_overrides,
    resolve_trainer_name,
    resolve_training_config,
)
from portfolio_agent.training.universe import UniverseSnapshot, resolve_universe


@pytest.fixture
def app_config():
    return load_config()


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_builtin_trainers_are_registered():
    names = registry.list_trainers()
    assert "supervised" in names
    assert "sac" in names


def test_get_trainer_returns_the_class():
    from portfolio_agent.training.trainers.sac import SACTrainer

    assert registry.get_trainer("sac") is SACTrainer


def test_unknown_trainer_names_what_is_available():
    with pytest.raises(KeyError) as excinfo:
        registry.get_trainer("definitely_not_a_trainer")
    message = str(excinfo.value)
    assert "definitely_not_a_trainer" in message
    # The overwhelmingly common cause is a typo, so the message has to carry
    # the correct spellings rather than just reporting absence.
    assert "sac" in message


def test_register_trainer_adds_to_the_registry():
    @registry.register_trainer("_test_only")
    class _Dummy(BaseTrainer):
        name = "_test_only"

        def prepare(self, app_config, universe, cfg):  # pragma: no cover
            raise NotImplementedError

        def fit(self, data, cfg):  # pragma: no cover
            raise NotImplementedError

    try:
        assert registry.is_trainer_registered("_test_only")
        assert registry.get_trainer("_test_only") is _Dummy
    finally:
        registry._TRAINER_REGISTRY.pop("_test_only", None)


# --------------------------------------------------------------------------
# Config resolution
# --------------------------------------------------------------------------


def test_strategy_yaml_selects_the_trainer(app_config):
    """india_sac declares `trainer: sac`, so it must not get the default."""
    name, trainer_class, _ = resolve_training_config(app_config, "india_sac")
    assert name == "sac"
    assert trainer_class.__name__ == "SACTrainer"


def test_no_strategy_falls_back_to_supervised(app_config):
    """Plain `portfolio-agent train` must behave exactly as it did before."""
    name, _, cfg = resolve_training_config(app_config, None)
    assert name == "supervised"
    # Global config.yaml `training:` values still reach the supervised trainer.
    assert cfg.epochs == app_config.training.epochs


def test_explicit_trainer_beats_the_yaml(app_config):
    name, _, _ = resolve_training_config(app_config, "india_sac", trainer="supervised")
    assert name == "supervised"


def test_overrides_beat_the_yaml(app_config):
    _, _, cfg = resolve_training_config(
        app_config, "india_sac", overrides={"epochs": 3}
    )
    assert cfg.epochs == 3


def test_yaml_beats_the_class_defaults(app_config):
    """india_sac.yaml sets epochs: 200; the class default agrees, so probe a
    key only the YAML sets."""
    _, _, cfg = resolve_training_config(app_config, "india_sac")
    assert cfg.gradient_steps == 200
    assert cfg.auto_entropy is True


def test_string_values_are_coerced_by_the_schema(app_config):
    """`--set` produces strings; the schema decides what they mean."""
    _, _, cfg = resolve_training_config(
        app_config, "india_sac", overrides=parse_overrides(["epochs=12", "gamma=0.5"])
    )
    assert cfg.epochs == 12 and isinstance(cfg.epochs, int)
    assert cfg.gamma == 0.5


def test_unknown_hyperparameter_is_an_error_not_a_silent_drop(app_config):
    """The failure PR-era trainers had: a flag parsed, printed and ignored."""
    with pytest.raises(ValueError) as excinfo:
        resolve_training_config(app_config, "india_sac", overrides={"buffer_sizee": 10})
    message = str(excinfo.value)
    assert "buffer_sizee" in message
    assert "buffer_size" in message  # the correct spelling is offered


def test_out_of_range_hyperparameter_is_rejected(app_config):
    with pytest.raises(ValueError) as excinfo:
        resolve_training_config(app_config, "india_sac", overrides={"epochs": 0})
    assert "epochs" in str(excinfo.value)


def test_supervised_only_keys_do_not_leak_into_the_sac_schema(app_config):
    """config.yaml carries `sequence_length`, which SAC does not accept.

    If the global block were merged wholesale, `extra="forbid"` would make
    config.yaml un-loadable for every non-supervised trainer.
    """
    assert hasattr(app_config.training, "sequence_length")
    _, _, cfg = resolve_training_config(app_config, "india_sac")
    assert not hasattr(cfg, "sequence_length")


def test_shared_keys_are_taken_from_the_global_block(app_config):
    """A key both schemas declare still flows from config.yaml."""
    _, _, cfg = resolve_training_config(app_config, None)
    assert cfg.batch_size == app_config.training.batch_size


def test_parse_overrides_rejects_malformed_pairs():
    with pytest.raises(ValueError):
        parse_overrides(["epochs"])
    with pytest.raises(ValueError):
        parse_overrides(["=5"])
    assert parse_overrides(["a=1", "b=x"]) == {"a": "1", "b": "x"}


def test_resolve_trainer_name_precedence():
    assert resolve_trainer_name(None) == "supervised"
    assert resolve_trainer_name(None, explicit="sac") == "sac"
    assert resolve_trainer_name(None, strategy_block={"trainer": "sac"}) == "sac"
    # Explicit beats the block.
    assert (
        resolve_trainer_name(None, explicit="supervised", strategy_block={"trainer": "sac"})
        == "supervised"
    )


def test_missing_strategy_yaml_is_not_an_error():
    assert load_strategy_training_block("no_such_strategy_anywhere") == {}


# --------------------------------------------------------------------------
# Universe snapshots
# --------------------------------------------------------------------------


def test_snapshot_fingerprint_is_order_insensitive():
    a = UniverseSnapshot.from_tickers(["RELIANCE", "TCS", "INFY"])
    b = UniverseSnapshot.from_tickers(["INFY", "RELIANCE", "TCS"])
    assert a.fingerprint == b.fingerprint


def test_snapshot_fingerprint_changes_with_membership():
    a = UniverseSnapshot.from_tickers(["RELIANCE", "TCS"])
    b = UniverseSnapshot.from_tickers(["RELIANCE", "INFY"])
    assert a.fingerprint != b.fingerprint


def test_snapshot_deduplicates_and_sorts():
    snap = UniverseSnapshot.from_tickers(["TCS", "RELIANCE", "TCS"])
    assert snap.tickers == ["RELIANCE", "TCS"]
    assert len(snap) == 2


def test_empty_snapshot_is_refused():
    with pytest.raises(ValueError, match="at least one ticker"):
        UniverseSnapshot.from_tickers([])


def test_snapshot_round_trips_through_disk(tmp_path):
    snap = UniverseSnapshot.from_tickers(["A", "B", "C"], name="q1")
    path = snap.save(tmp_path / "u.json")

    loaded = UniverseSnapshot.load(path)
    assert loaded.tickers == snap.tickers
    assert loaded.fingerprint == snap.fingerprint
    assert loaded.name == "q1"


def test_edited_snapshot_warns_but_still_loads(tmp_path, caplog):
    """Hand-editing to drop a delisted name is legitimate; silence is not."""
    snap = UniverseSnapshot.from_tickers(["A", "B", "C"])
    path = snap.save(tmp_path / "u.json")

    payload = json.loads(path.read_text())
    payload["tickers"] = ["A", "B"]
    path.write_text(json.dumps(payload))

    with caplog.at_level("WARNING"):
        loaded = UniverseSnapshot.load(path)
    assert loaded.tickers == ["A", "B"]
    assert "edited" in caplog.text


def test_missing_snapshot_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        UniverseSnapshot.load(tmp_path / "nope.json")


def test_resolve_universe_prefers_explicit_tickers(app_config, tmp_path):
    snap = UniverseSnapshot.from_tickers(["X", "Y"]).save(tmp_path / "u.json")
    resolved = resolve_universe(app_config, tickers=["A", "B"], snapshot=snap)
    assert resolved.tickers == ["A", "B"]


def test_resolve_universe_uses_the_snapshot_when_no_tickers(app_config, tmp_path):
    path = UniverseSnapshot.from_tickers(["X", "Y"]).save(tmp_path / "u.json")
    resolved = resolve_universe(app_config, snapshot=path)
    assert resolved.tickers == ["X", "Y"]


# --------------------------------------------------------------------------
# TrainingData
# --------------------------------------------------------------------------


def test_training_data_rejects_an_empty_panel():
    with pytest.raises(ValueError):
        TrainingData(
            features_by_ticker={}, prices_by_ticker={}, tickers=[], feature_names=["a"]
        )


def test_trainer_config_forbids_extras():
    with pytest.raises(Exception):
        TrainerConfig(nonsense=1)
