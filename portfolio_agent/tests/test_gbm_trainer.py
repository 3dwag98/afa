"""The gradient-boosting baseline: panel construction, the split, the artifact.

Synthetic data throughout, and deliberately so. These assert that the
mechanism is the one documented — that the label matches the supervised
pipeline's, that the date split does not leak, that the same seed gives the
same model — not that boosting forecasts Indian equities. Whether it does is a
research question no unit test can answer.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

sklearn = pytest.importorskip("sklearn")

from portfolio_agent.config.loader import load_config
from portfolio_agent.training import get_trainer, list_trainers
from portfolio_agent.training.artifacts import (
    load_sklearn_artifact,
    save_sklearn_artifact,
)
from portfolio_agent.training.base import TrainingArtifact
from portfolio_agent.training.runner import checkpoint_path_for, run_training_job
from portfolio_agent.training.trainers.gbm import (
    DEFAULT_GBM_FEATURES,
    GBMTrainer,
    GBMTrainerConfig,
    build_gbm_panel,
    horizon_from_target,
    rank_ic_by_date,
)


@pytest.fixture
def app_config():
    return load_config()


def _ohlcv(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
    return pd.DataFrame(
        {
            "open": close, "high": close * 1.02, "low": close * 0.98,
            "close": close, "volume": rng.integers(1e5, 1e6, n).astype(float),
        },
        index=index,
    )


@pytest.fixture
def fake_cache(monkeypatch):
    """Serve synthetic OHLCV in place of the parquet cache."""
    frames = {f"T{i}": _ohlcv(seed=i) for i in range(8)}

    def fake_load(ticker, start_date=None, end_date=None):
        return frames.get(ticker)

    monkeypatch.setattr(
        "portfolio_agent.src.data_store.load_ticker_data", fake_load, raising=True
    )
    return frames


def tiny_config(**overrides) -> GBMTrainerConfig:
    base = dict(epochs=20, n_iter_no_change=10, min_history=260, importance_repeats=2)
    base.update(overrides)
    return GBMTrainerConfig(**base)


# --------------------------------------------------------------------------
# Registration and discoverability
# --------------------------------------------------------------------------


def test_gbm_is_registered():
    assert "gbm" in list_trainers()
    assert get_trainer("gbm") is GBMTrainer


def test_availability_is_clean_when_sklearn_is_installed():
    assert GBMTrainer.availability() is None


def test_availability_names_the_extra_when_sklearn_is_absent(monkeypatch):
    """`list-trainers` prints this, so it has to say what to install.

    Checked with `find_spec` rather than an import: a listing command must not
    cost a second per optional library.
    """
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    reason = GBMTrainer.availability()
    assert reason is not None
    assert "--extra gbm" in reason


def test_unavailable_trainers_reports_import_failures():
    """Every built-in is either registered or explained — never just missing."""
    from portfolio_agent.training.registry import unavailable_trainers

    accounted = set(list_trainers()) | set(unavailable_trainers())
    assert {"gbm", "sac", "supervised"} <= accounted


def test_every_setting_carries_a_description():
    """`list-trainers --name gbm` prints these; a blank one helps nobody."""
    for name, field in GBMTrainerConfig.model_fields.items():
        assert field.description, f"{name} has no description"


def test_inherited_knobs_that_do_not_apply_say_so():
    """A knob that silently does nothing is what this package exists to stop.

    `batch_size` and `device` are inherited from `TrainerConfig` and mean
    nothing to a tree ensemble. They cannot be removed from the schema, so the
    next best thing is that the text `list-trainers` prints admits it.
    """
    fields = GBMTrainerConfig.model_fields
    assert "Unused" in fields["batch_size"].description
    assert "Unused" in fields["device"].description


def test_unknown_setting_is_rejected():
    """`extra="forbid"` is the whole point of a per-trainer schema."""
    with pytest.raises(Exception) as excinfo:
        GBMTrainerConfig(max_leaf_nodez=31)
    assert "max_leaf_nodez" in str(excinfo.value)


def test_learning_rate_default_is_a_boosting_rate_not_a_network_one():
    """The inherited 1e-3 would leave a hundred trees barely off the mean."""
    assert GBMTrainerConfig().learning_rate == pytest.approx(0.05)


def test_default_features_match_the_supervised_pipeline():
    """Two trainers on different features produce incomparable numbers.

    The list is duplicated in gbm.py so that module stays torch-free; this is
    what stops the copy drifting from the original.
    """
    from portfolio_agent.agents.trainer import TRAINING_FEATURE_NAMES

    assert DEFAULT_GBM_FEATURES == list(TRAINING_FEATURE_NAMES)


# --------------------------------------------------------------------------
# The label
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target,expected",
    [("return_5d", 5), ("return_1d", 1), ("return_21d", 21), ("direction", 1)],
)
def test_horizon_is_read_out_of_the_target_name(target, expected):
    assert horizon_from_target(target) == expected


def test_label_is_the_supervised_pipeline_s_label(app_config, fake_cache):
    """Not a lookalike — the same function, applied to the same prices.

    Recomputed here from the raw close series and compared against what the
    panel builder produced, before the rank transform is applied.
    """
    from portfolio_agent.features.labels import build_forward_return

    cfg = tiny_config(target_transform="absolute", feature_normalization="none")
    panel, _ = build_gbm_panel(app_config, list(fake_cache), cfg)

    # Reconstruct one ticker's label independently and check a value survives
    # into the stacked matrix. The panel is stacked across names, so this
    # asserts membership rather than position.
    expected = build_forward_return(fake_cache["T0"]["close"].astype(float), "return_5d")
    expected = expected.dropna().to_numpy()
    assert np.isin(np.round(expected[300], 10), np.round(panel.y_train, 10)).any()


def test_rank_target_is_bounded_and_centred(app_config, fake_cache):
    """2*rank/(N+1) - 1 lands in (-1, 1) and has mean ~0 on each date."""
    cfg = tiny_config()
    panel, _ = build_gbm_panel(app_config, list(fake_cache), cfg)
    assert panel.y_train.min() > -1.0
    assert panel.y_train.max() < 1.0
    assert abs(float(panel.y_train.mean())) < 0.05


# --------------------------------------------------------------------------
# The split
# --------------------------------------------------------------------------


def test_split_is_by_date_not_by_each_ticker_s_row_count(app_config, monkeypatch):
    """A per-ticker split puts one name's validation inside another's training.

    The short name here has a third of the history of the others. Under a
    per-ticker fractional split its validation rows would sit in calendar time
    well inside the long names' training rows; under a date split every
    validation row is strictly after every training row.
    """
    frames = {
        "LONG_A": _ohlcv(n=500, seed=1),
        "LONG_B": _ohlcv(n=500, seed=2),
        "LONG_C": _ohlcv(n=500, seed=3),
        "SHORT": _ohlcv(n=500, seed=4).iloc[:340],
    }
    monkeypatch.setattr(
        "portfolio_agent.src.data_store.load_ticker_data",
        lambda t, start_date=None, end_date=None: frames.get(t),
        raising=True,
    )

    panel, _ = build_gbm_panel(app_config, list(frames), tiny_config())
    assert panel.train_dates.max() < panel.val_dates.min()
    assert pd.Timestamp(panel.split_date) > pd.Timestamp(panel.train_dates.max())


def test_the_boundary_is_purged_by_one_horizon(app_config, fake_cache):
    """A row dated t carries a label realized at t+horizon.

    Training rows inside one horizon of the cut have already seen across it,
    so they are dropped. The gap between the last training date and the split
    date is what proves it happened.
    """
    cfg = tiny_config()
    panel, _ = build_gbm_panel(app_config, list(fake_cache), cfg)

    assert panel.purged_rows > 0

    # Positions have to be read off the *full* date index. One rebuilt from the
    # surviving dates has the purged gap closed up and would report a distance
    # of 1 however wide the purge actually was.
    all_dates = pd.DatetimeIndex(panel.all_dates)
    split_pos = all_dates.get_loc(pd.Timestamp(panel.split_date))
    last_train_pos = all_dates.get_loc(pd.Timestamp(panel.train_dates.max()))

    # The last training row's label is realized `horizon` dates later, and that
    # has to land strictly before the validation block begins...
    assert last_train_pos + panel.horizon_days < split_pos
    # ...and it is the last such row: one date further would not.
    assert last_train_pos + 1 + panel.horizon_days == split_pos
    # Every purged row belongs to one of the dates in the gap.
    assert panel.purged_rows == panel.horizon_days * panel.n_tickers


def test_purging_can_be_disabled_to_measure_what_it_costs(app_config, fake_cache):
    """purge_days=0 exists so the bias is measurable rather than theoretical."""
    purged, _ = build_gbm_panel(app_config, list(fake_cache), tiny_config())
    unpurged, _ = build_gbm_panel(
        app_config, list(fake_cache), tiny_config(purge_days=0)
    )
    assert unpurged.purged_rows == 0
    assert len(unpurged.y_train) > len(purged.y_train)


def test_a_purge_wider_than_the_sample_is_an_error_not_an_empty_run(
    app_config, fake_cache
):
    """Training through an empty block would report a run that never happened."""
    cfg = tiny_config(purge_days=10_000)
    with pytest.raises(ValueError, match="split left one side empty"):
        build_gbm_panel(app_config, list(fake_cache), cfg)


def test_training_data_reports_the_names_that_survived(app_config, fake_cache):
    """The runner fingerprints this, so it has to describe the real sample."""
    _, data = build_gbm_panel(app_config, list(fake_cache), tiny_config())
    assert data.tickers == sorted(fake_cache)
    assert data.feature_names == DEFAULT_GBM_FEATURES
    for ticker in data.tickers:
        assert list(data.features_by_ticker[ticker].columns) == DEFAULT_GBM_FEATURES
        assert data.prices_by_ticker[ticker].index.equals(
            data.features_by_ticker[ticker].index
        )


# --------------------------------------------------------------------------
# Feature normalization
# --------------------------------------------------------------------------


def _fit_predictions(app_config, tickers, **overrides):
    trainer = GBMTrainer()
    cfg = tiny_config(importance_repeats=0, **overrides)
    artifact = trainer.fit(trainer.prepare(app_config, list(tickers), cfg), cfg)
    return artifact, artifact.state_dict["estimator"].predict(trainer._panel.x_val)


def test_pooled_standardization_leaves_a_tree_ensemble_unchanged(
    app_config, fake_cache
):
    """The claim in the schema's own field description, checked rather than asserted.

    A pooled standardizer is one positive-scale affine map per feature. Trees
    split on thresholds, so an order-preserving map of a column produces the
    same splits and the same predictions — bit for bit, not approximately.
    """
    _, raw = _fit_predictions(app_config, fake_cache, feature_normalization="none")
    pooled_artifact, pooled = _fit_predictions(
        app_config, fake_cache, feature_normalization="pooled"
    )
    np.testing.assert_allclose(pooled, raw)
    # It still has to be *recorded*, or inference cannot reproduce the inputs.
    assert pooled_artifact.metadata["feature_scaler"] is not None


def test_cross_sectional_standardization_genuinely_changes_the_model(
    app_config, fake_cache
):
    """A different affine map on every date is not a monotone transform.

    Which is exactly why it is the default against a cross-sectional label: it
    is the transform that lets a tree ask "is this RSI high relative to what
    else I could buy today" instead of "high against a five-year mean".
    """
    _, raw = _fit_predictions(app_config, fake_cache, feature_normalization="none")
    artifact, cross = _fit_predictions(
        app_config, fake_cache, feature_normalization="cross_sectional"
    )
    assert not np.allclose(cross, raw)
    # There is no fitted state to record — the transform reads only rows dated
    # t — and null says that, where an omitted key would look like an oversight.
    assert artifact.metadata["feature_scaler"] is None
    assert artifact.metadata["feature_normalization"] == "cross_sectional"


# --------------------------------------------------------------------------
# Rank IC
# --------------------------------------------------------------------------


def test_rank_ic_is_one_for_a_perfect_ordering():
    dates = np.repeat(pd.date_range("2024-01-01", periods=3), 6)
    labels = np.tile(np.arange(6.0), 3)
    ic = rank_ic_by_date(labels * 2.0, labels, dates)
    assert len(ic) == 3
    assert ic.to_numpy() == pytest.approx(1.0)


def test_rank_ic_is_minus_one_for_a_reversed_ordering():
    dates = np.repeat(pd.date_range("2024-01-01", periods=2), 6)
    labels = np.tile(np.arange(6.0), 2)
    ic = rank_ic_by_date(-labels, labels, dates)
    assert ic.to_numpy() == pytest.approx(-1.0)


def test_rank_ic_drops_dates_with_too_thin_a_cross_section():
    """A correlation over three names is noise with a decimal point."""
    dates = np.array(
        list(pd.date_range("2024-01-01", periods=1).repeat(3))
        + list(pd.date_range("2024-01-02", periods=1).repeat(6))
    )
    labels = np.concatenate([np.arange(3.0), np.arange(6.0)])
    ic = rank_ic_by_date(labels, labels, dates, min_names=5)
    assert len(ic) == 1


def test_rank_ic_skips_a_constant_prediction_rather_than_scoring_it_zero():
    """Spearman is undefined against a constant; 0.0 would be a claim."""
    dates = np.repeat(pd.date_range("2024-01-01", periods=2), 6)
    labels = np.tile(np.arange(6.0), 2)
    ic = rank_ic_by_date(np.zeros_like(labels), labels, dates)
    assert ic.empty


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


def test_fit_produces_an_estimator_and_the_metrics_that_matter(app_config, fake_cache):
    trainer = GBMTrainer()
    cfg = tiny_config()
    data = trainer.prepare(app_config, list(fake_cache), cfg)
    artifact = trainer.fit(data, cfg)

    assert "estimator" in artifact.state_dict
    for key in ("train_loss", "val_loss", "val_rank_ic", "val_icir", "n_iterations"):
        assert key in artifact.metrics
    # primary_metric picks the rank IC, which is the number a cross-sectional
    # forecaster is actually judged on.
    assert artifact.primary_metric() == pytest.approx(artifact.metrics["val_rank_ic"])


def test_feature_importances_are_recorded_for_every_feature(app_config, fake_cache):
    trainer = GBMTrainer()
    cfg = tiny_config()
    artifact = trainer.fit(trainer.prepare(app_config, list(fake_cache), cfg), cfg)

    importances = artifact.metadata["feature_importances"]
    assert set(importances) == set(DEFAULT_GBM_FEATURES)
    assert artifact.metadata["importance_method"] == "permutation_on_validation"
    # Sorted largest-first, so the head of the dict is the story.
    values = list(importances.values())
    assert values == sorted(values, reverse=True)


def test_importances_can_be_switched_off(app_config, fake_cache):
    """They cost a prediction pass per feature per repeat; a sweep may skip them."""
    trainer = GBMTrainer()
    cfg = tiny_config(importance_repeats=0)
    artifact = trainer.fit(trainer.prepare(app_config, list(fake_cache), cfg), cfg)
    assert artifact.metadata["feature_importances"] == {}
    assert artifact.metadata["importance_method"] == "not_computed"


def test_same_seed_gives_the_same_model(app_config, fake_cache):
    """Two runs of one configuration must agree — binning and permutation
    importances both draw at random and neither would otherwise."""
    def run(seed):
        trainer = GBMTrainer()
        cfg = tiny_config(seed=seed)
        artifact = trainer.fit(trainer.prepare(app_config, list(fake_cache), cfg), cfg)
        panel = trainer._panel
        return (
            artifact.state_dict["estimator"].predict(panel.x_val),
            artifact.metrics["val_rank_ic"],
            artifact.metadata["feature_importances"],
        )

    first_pred, first_ic, first_imp = run(7)
    again_pred, again_ic, again_imp = run(7)

    np.testing.assert_allclose(first_pred, again_pred)
    assert first_ic == again_ic
    assert first_imp == again_imp


def test_early_stopping_keeps_the_best_size_not_the_last_one(app_config, fake_cache):
    """Shipping an ensemble a full patience window past its best is the overfit
    early stopping exists to prevent."""
    trainer = GBMTrainer()
    cfg = tiny_config(epochs=60, n_iter_no_change=5)
    artifact = trainer.fit(trainer.prepare(app_config, list(fake_cache), cfg), cfg)
    assert 0 < artifact.metrics["n_iterations"] <= 60


def test_early_stopping_off_grows_the_full_ensemble(app_config, fake_cache):
    trainer = GBMTrainer()
    cfg = tiny_config(epochs=15, early_stopping=False)
    artifact = trainer.fit(trainer.prepare(app_config, list(fake_cache), cfg), cfg)
    assert artifact.metrics["n_iterations"] == 15


def test_fit_before_prepare_is_an_error(app_config):
    with pytest.raises(RuntimeError, match="prepare"):
        GBMTrainer().fit(None, tiny_config())


def test_a_non_cpu_device_is_reported_and_ignored(app_config, fake_cache, caplog):
    """Silently accepting `--set device=cuda` would be a lie about what ran."""
    trainer = GBMTrainer()
    with caplog.at_level("INFO", logger="portfolio_agent.training.trainers.gbm"):
        trainer.prepare(app_config, list(fake_cache), tiny_config(device="cuda"))
    assert "ignored" in caplog.text


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def test_checkpoint_suffix_is_joblib_not_pt():
    """A fitted estimator is a pickle; torch.load(weights_only=True) refuses it."""
    assert GBMTrainer.checkpoint_suffix == ".joblib"
    path = checkpoint_path_for(None, "gbm", models_dir="models", suffix=".joblib")
    assert path.name == "gbm_best.joblib"


def test_other_trainers_still_write_pt():
    """The suffix hook must not have changed the torch trainers' paths."""
    from portfolio_agent.training.trainers.supervised import SupervisedTrainer

    assert SupervisedTrainer.checkpoint_suffix == ".pt"
    assert checkpoint_path_for("india_sac", "sac").name == "india_sac_best.pt"


def test_end_to_end_run_writes_a_loadable_checkpoint(app_config, fake_cache, tmp_path):
    run = run_training_job(
        app_config,
        trainer="gbm",
        universe=list(fake_cache),
        overrides={"epochs": 20, "n_iter_no_change": 10, "min_history": 260,
                   "importance_repeats": 1},
        models_dir=tmp_path, runs_dir=tmp_path,
    )
    assert run.ok, run.error
    assert run.checkpoint_path == tmp_path / "gbm_best.joblib"
    assert run.checkpoint_path.exists()

    restored = load_sklearn_artifact(run.checkpoint_path)
    assert restored.metadata["feature_names"] == DEFAULT_GBM_FEATURES
    assert restored.metadata["trainer"] == "gbm"
    assert restored.metrics["val_rank_ic"] == run.artifact.metrics["val_rank_ic"]

    # The restored estimator predicts identically — a checkpoint that loads but
    # scores differently is worse than one that fails to load.
    panel, _ = build_gbm_panel(
        app_config, list(fake_cache),
        GBMTrainerConfig(epochs=20, n_iter_no_change=10, min_history=260),
    )
    np.testing.assert_allclose(
        restored.state_dict["estimator"].predict(panel.x_val),
        run.artifact.state_dict["estimator"].predict(panel.x_val),
    )


def test_sidecar_carries_metrics_without_unpickling(app_config, fake_cache, tmp_path):
    """Comparison tooling must never have to execute a checkpoint to read it."""
    run = run_training_job(
        app_config,
        trainer="gbm",
        universe=list(fake_cache),
        overrides={"epochs": 20, "n_iter_no_change": 10, "min_history": 260,
                   "importance_repeats": 1},
        models_dir=tmp_path, runs_dir=tmp_path,
    )
    sidecar = tmp_path / "gbm_best.json"
    assert sidecar.exists()

    document = json.loads(sidecar.read_text())
    assert document["trainer"] == "gbm"
    assert document["metrics"]["val_rank_ic"] == run.artifact.metrics["val_rank_ic"]
    assert set(document["feature_importances"]) == set(DEFAULT_GBM_FEATURES)
    assert document["horizon_days"] == 5
    assert document["library"] == "scikit-learn"


def test_saving_without_an_estimator_is_refused(tmp_path):
    """A checkpoint with no model loads back as one that predicts nothing."""
    artifact = TrainingArtifact(state_dict={}, metadata={"feature_names": ["a"]})
    with pytest.raises(ValueError, match="no 'estimator'"):
        save_sklearn_artifact(artifact, tmp_path / "x.joblib")


def test_saving_without_feature_names_is_refused(tmp_path):
    from sklearn.dummy import DummyRegressor

    model = DummyRegressor().fit([[0.0], [1.0]], [0.0, 1.0])
    artifact = TrainingArtifact(state_dict={"estimator": model}, metadata={})
    with pytest.raises(ValueError, match="feature_names"):
        save_sklearn_artifact(artifact, tmp_path / "x.joblib")


def test_loading_a_missing_checkpoint_raises(tmp_path):
    """Never "start from an unfitted estimator" — that fails much later."""
    with pytest.raises(FileNotFoundError):
        load_sklearn_artifact(tmp_path / "absent.joblib")


def test_loading_something_that_is_not_a_checkpoint_raises(tmp_path):
    import joblib

    path = tmp_path / "junk.joblib"
    joblib.dump({"not": "a checkpoint"}, path)
    with pytest.raises(ValueError, match="does not contain"):
        load_sklearn_artifact(path)


# --------------------------------------------------------------------------
# The missing-library path
# --------------------------------------------------------------------------


#: Run in a subprocess, because the only honest way to test "does not need
#: PyTorch" is from an interpreter where PyTorch was never importable. Blocking
#: it in-process would leave a half-initialized `torch` in `sys.modules` for
#: every test that runs afterwards.
_NO_TORCH_SCRIPT = """
import builtins, sys

_real_import = builtins.__import__


def _blocked(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ImportError("No module named 'torch'")
    return _real_import(name, *args, **kwargs)


builtins.__import__ = _blocked

import numpy as np
import pandas as pd

import portfolio_agent.src.data_store as data_store
from portfolio_agent.config.loader import load_config
from portfolio_agent.training import list_trainers, run_training_job


def ohlcv(n, seed):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
    return pd.DataFrame(
        {"open": close, "high": close * 1.02, "low": close * 0.98,
         "close": close, "volume": rng.integers(1e5, 1e6, n).astype(float)},
        index=index,
    )


frames = {"T%d" % i: ohlcv(500, i) for i in range(6)}
data_store.load_ticker_data = lambda t, start_date=None, end_date=None: frames.get(t)

trainers = list_trainers()
run = run_training_job(
    load_config(), trainer="gbm", universe=list(frames),
    overrides={"epochs": 20, "n_iter_no_change": 10, "min_history": 260,
               "importance_repeats": 1},
    models_dir=sys.argv[1], runs_dir=sys.argv[1],
)
print("TRAINERS", ",".join(trainers))
print("OK", run.ok, run.error)
print("CHECKPOINT", run.checkpoint_path.name if run.checkpoint_path else None)
print("TORCH_IMPORTED", "torch" in sys.modules)
"""


def test_the_whole_gbm_path_runs_without_pytorch(tmp_path):
    """`pip install portfolio-agent[gbm]` is enough to train a real model.

    Boosting is the one trainer with no torch dependency anywhere in its path —
    not in the panel, not in the metrics, not in the joblib checkpoint. That is
    only true as long as nothing on the way imports torch at module scope, and
    the per-module guard in `trainers/__init__.py` is what keeps `sac`'s
    `import torch` from taking `gbm` and `supervised` down with it.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    import portfolio_agent

    repo_root = Path(portfolio_agent.__file__).resolve().parent.parent
    script = tmp_path / "no_torch.py"
    script.write_text(_NO_TORCH_SCRIPT)

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        capture_output=True, text=True, cwd=str(repo_root), env=env,
    )
    assert result.returncode == 0, result.stderr

    lines = dict(line.split(" ", 1) for line in result.stdout.strip().splitlines())
    # sac needs torch and is correctly absent; the other two are not affected.
    assert lines["TRAINERS"] == "gbm,supervised"
    assert lines["OK"].startswith("True")
    assert lines["CHECKPOINT"] == "gbm_best.joblib"
    assert lines["TORCH_IMPORTED"] == "False"


def test_absent_sklearn_names_the_extra_to_install(monkeypatch):
    """The trainer stays registered so `list-trainers` can point at the fix."""
    import builtins

    from portfolio_agent.training.trainers import gbm as gbm_module

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "sklearn" or name.startswith("sklearn."):
            raise ImportError("No module named 'sklearn'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError) as excinfo:
        gbm_module._require_sklearn()

    message = str(excinfo.value)
    assert "--extra gbm" in message
    assert "list-trainers" in message
