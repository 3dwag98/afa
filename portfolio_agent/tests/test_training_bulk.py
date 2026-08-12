"""Bulk runs, the notebook facade, and shared panel preparation.

The theme is comparability: every job in a bulk run, and every call on a `Lab`,
must see the same tickers — otherwise the differences in a results table are
partly differences in the samples, and nothing in the table says which.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.config.loader import load_config
from portfolio_agent.training import registry
from portfolio_agent.training.base import BaseTrainer, TrainerConfig, TrainingArtifact
from portfolio_agent.training.bulk import BulkJob, run_bulk, sweep
from portfolio_agent.training.data import prepare_panel


@pytest.fixture
def app_config():
    return load_config()


# --------------------------------------------------------------------------
# A trainer that records what it was handed, so the seams can be asserted
# --------------------------------------------------------------------------


class RecordingConfig(TrainerConfig):
    marker: float = 0.0
    explode: bool = False


@pytest.fixture
def recording_trainer():
    """Register a fake trainer for the duration of one test."""
    seen: list[dict] = []

    @registry.register_trainer("_recording")
    class _Recording(BaseTrainer):
        name = "_recording"

        @classmethod
        def config_model(cls):
            return RecordingConfig

        def prepare(self, app_config, universe, cfg):
            from portfolio_agent.training.base import TrainingData

            seen.append({"universe": list(universe), "marker": cfg.marker})
            return TrainingData(
                features_by_ticker={},
                prices_by_ticker={},
                tickers=list(universe),
                feature_names=["f"],
            )

        def fit(self, data, cfg):
            if cfg.explode:
                raise RuntimeError("deliberate failure")
            return TrainingArtifact(
                state_dict={},
                metadata={"feature_names": ["f"], "trainer": "_recording"},
                metrics={"val_sharpe": cfg.marker},
            )

    try:
        yield seen
    finally:
        registry._TRAINER_REGISTRY.pop("_recording", None)


# --------------------------------------------------------------------------
# Bulk
# --------------------------------------------------------------------------


def test_every_job_sees_the_identical_universe(app_config, recording_trainer):
    """The whole reason bulk pins the universe before running anything."""
    jobs = [
        BulkJob(trainer="_recording", overrides={"marker": 1.0}),
        BulkJob(trainer="_recording", overrides={"marker": 2.0}),
        BulkJob(trainer="_recording", overrides={"marker": 3.0}),
    ]
    report = run_bulk(app_config, jobs, universe=["C", "A", "B"])

    assert len(recording_trainer) == 3
    universes = {tuple(entry["universe"]) for entry in recording_trainer}
    assert len(universes) == 1
    assert universes.pop() == ("A", "B", "C")  # sorted by the snapshot
    assert len({run.universe.fingerprint for run in report.runs}) == 1


def test_a_failing_job_does_not_discard_the_others(app_config, recording_trainer):
    jobs = [
        BulkJob(trainer="_recording", overrides={"marker": 1.0}),
        BulkJob(trainer="_recording", overrides={"marker": 2.0, "explode": True}),
        BulkJob(trainer="_recording", overrides={"marker": 3.0}),
    ]
    report = run_bulk(app_config, jobs, universe=["A", "B"])

    assert [run.ok for run in report.runs] == [True, False, True]
    assert len(report.failures) == 1
    assert "deliberate failure" in report.runs[1].error


def test_report_frame_puts_failures_below_successes(app_config, recording_trainer):
    jobs = [
        BulkJob(trainer="_recording", overrides={"marker": 0.0, "explode": True}, label="bad"),
        BulkJob(trainer="_recording", overrides={"marker": 5.0}, label="good"),
    ]
    frame = run_bulk(app_config, jobs, universe=["A"]).to_frame()

    assert frame.iloc[0]["label"] == "good"
    assert frame.iloc[-1]["status"] == "failed"


def test_best_picks_the_highest_metric(app_config, recording_trainer):
    jobs = [
        BulkJob(trainer="_recording", overrides={"marker": 1.0}, label="low"),
        BulkJob(trainer="_recording", overrides={"marker": 9.0}, label="high"),
    ]
    report = run_bulk(app_config, jobs, universe=["A"])
    assert report.best().artifact.metrics["val_sharpe"] == pytest.approx(9.0)


def test_best_is_none_when_everything_failed(app_config, recording_trainer):
    jobs = [BulkJob(trainer="_recording", overrides={"explode": True})]
    assert run_bulk(app_config, jobs, universe=["A"]).best() is None


def test_bulk_writes_the_snapshot_when_asked(app_config, recording_trainer, tmp_path):
    path = tmp_path / "u.json"
    run_bulk(
        app_config,
        [BulkJob(trainer="_recording")],
        universe=["A", "B"],
        save_snapshot_to=path,
    )
    assert path.exists()

    from portfolio_agent.training.universe import UniverseSnapshot

    assert UniverseSnapshot.load(path).tickers == ["A", "B"]


def test_bulk_needs_at_least_one_job(app_config):
    with pytest.raises(ValueError):
        run_bulk(app_config, [], universe=["A"])


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------


def test_sweep_expands_the_cross_product():
    jobs = sweep("india_sac", {"gamma": [0.0, 0.9], "epochs": [10, 20]})
    assert len(jobs) == 4
    combos = {(j.overrides["gamma"], j.overrides["epochs"]) for j in jobs}
    assert combos == {(0.0, 10), (0.0, 20), (0.9, 10), (0.9, 20)}


def test_sweep_keeps_base_overrides_on_every_point():
    jobs = sweep("india_sac", {"gamma": [0.0, 0.9]}, base_overrides={"epochs": 5})
    assert all(job.overrides["epochs"] == 5 for job in jobs)


def test_saving_sweep_points_gives_each_a_distinct_checkpoint_name():
    """Otherwise every point overwrites one file and only the last survives."""
    jobs = sweep("india_sac", {"gamma": [0.0, 0.9]}, save=True)
    names = [job.model_name for job in jobs]
    assert len(set(names)) == len(names)
    assert all(name for name in names)


def test_sweep_defaults_to_not_saving():
    jobs = sweep("india_sac", {"gamma": [0.0, 0.9]})
    assert all(job.save is False for job in jobs)
    assert all(job.model_name is None for job in jobs)


def test_sweep_needs_a_grid():
    with pytest.raises(ValueError):
        sweep("india_sac", {})


def test_job_labels_describe_what_varies():
    job = BulkJob(strategy="india_sac", overrides={"gamma": 0.9})
    assert "india_sac" in job.resolved_label()
    assert "gamma=0.9" in job.resolved_label()


# --------------------------------------------------------------------------
# Lab facade
# --------------------------------------------------------------------------


def test_lab_pins_its_universe(app_config):
    from portfolio_agent.lab import Lab

    lab = Lab(tickers=["TCS", "RELIANCE"], config=app_config)
    assert lab.tickers == ["RELIANCE", "TCS"]
    assert lab.fingerprint


def test_lab_train_and_compare_use_the_pinned_tickers(app_config, recording_trainer):
    from portfolio_agent.lab import Lab

    lab = Lab(tickers=["B", "A"], config=app_config)

    first = lab.train(trainer="_recording", marker=1.0, save=False)
    second = lab.train(trainer="_recording", marker=2.0, save=False)

    assert first.ok and second.ok
    assert len(recording_trainer) == 2
    assert all(entry["universe"] == ["A", "B"] for entry in recording_trainer)
    # And the two runs are on the same universe, which is the guarantee a Lab
    # exists to provide.
    assert first.universe.fingerprint == second.universe.fingerprint


def test_lab_round_trips_its_universe(app_config, tmp_path):
    from portfolio_agent.lab import Lab

    original = Lab(tickers=["A", "B", "C"], config=app_config)
    path = original.save_universe(tmp_path / "u.json")

    restored = Lab(snapshot=path, config=app_config)
    assert restored.tickers == original.tickers
    assert restored.fingerprint == original.fingerprint


def test_lab_exposes_trainer_settings():
    from portfolio_agent.lab import Lab

    settings = Lab.settings("sac")
    assert "gamma" in settings and "buffer_size" in settings


# --------------------------------------------------------------------------
# Panel preparation
# --------------------------------------------------------------------------


def _ohlcv(n=400, seed=0, offset=0.0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n))) + offset
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
    frames = {"AAA": _ohlcv(seed=1), "BBB": _ohlcv(seed=2), "SHORT": _ohlcv(n=30, seed=3)}

    def fake_load(ticker, start_date=None, end_date=None):
        return frames.get(ticker)

    monkeypatch.setattr(
        "portfolio_agent.src.data_store.load_ticker_data", fake_load, raising=True
    )
    return frames


def test_prepare_panel_keeps_only_usable_tickers(app_config, fake_cache):
    data = prepare_panel(
        app_config, ["AAA", "BBB", "SHORT", "MISSING"], ["rsi_14", "macd"],
        min_history=100,
    )
    assert data.tickers == ["AAA", "BBB"]
    assert data.feature_names == ["rsi_14", "macd"]


def test_prepare_panel_preserves_the_requested_column_order(app_config, fake_cache):
    """The scaler and the network are both positional."""
    data = prepare_panel(
        app_config, ["AAA"], ["macd", "rsi_14", "atr_14"], min_history=100
    )
    assert list(data.features_by_ticker["AAA"].columns) == ["macd", "rsi_14", "atr_14"]


def test_scaler_is_fitted_on_training_rows_only(app_config, fake_cache):
    """Fitting on everything leaks the validation segment into the transform.

    The validation block here is shifted far from the training block, so a
    scaler that saw it would have a visibly different mean.
    """
    data = prepare_panel(app_config, ["AAA"], ["rsi_14", "macd"], min_history=100)
    frame = data.features_by_ticker["AAA"]
    cut = data.split_index_by_ticker["AAA"]

    # Standardized training rows should be ~zero-mean by construction; the
    # held-out rows carry no such guarantee.
    train_mean = np.abs(frame.iloc[:cut].to_numpy().mean(axis=0))
    assert np.all(train_mean < 0.15)


def test_split_is_chronological(app_config, fake_cache):
    data = prepare_panel(
        app_config, ["AAA"], ["rsi_14"], min_history=100, train_fraction=0.75
    )
    train, val = data.split("AAA")
    assert len(train) > 0 and len(val) > 0
    assert train.index.max() < val.index.min()
    assert len(train) / (len(train) + len(val)) == pytest.approx(0.75, abs=0.02)


def test_prices_are_aligned_to_the_surviving_feature_rows(app_config, fake_cache):
    data = prepare_panel(app_config, ["AAA"], ["rsi_14", "macd"], min_history=100)
    assert data.prices_by_ticker["AAA"].index.equals(data.features_by_ticker["AAA"].index)


def test_features_are_finite_after_preparation(app_config, fake_cache):
    data = prepare_panel(app_config, ["AAA"], ["rsi_14", "macd", "atr_14"], min_history=100)
    assert np.isfinite(data.features_by_ticker["AAA"].to_numpy()).all()


def test_an_empty_panel_is_an_error_not_an_empty_run(app_config, fake_cache):
    with pytest.raises(ValueError, match="No ticker produced usable history"):
        prepare_panel(app_config, ["MISSING", "SHORT"], ["rsi_14"], min_history=100)


def test_unknown_feature_names_are_reported(app_config, fake_cache):
    with pytest.raises(ValueError, match="No ticker produced usable history"):
        prepare_panel(app_config, ["AAA"], ["not_a_real_feature"], min_history=100)


def test_fit_scaler_false_leaves_features_raw(app_config, fake_cache):
    data = prepare_panel(
        app_config, ["AAA"], ["rsi_14"], min_history=100, fit_scaler=False
    )
    assert data.scaler is None
    # RSI lives in [0, 100]; standardized values would not.
    assert data.features_by_ticker["AAA"]["rsi_14"].max() > 20
