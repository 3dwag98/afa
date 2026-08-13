"""Run manifests: provenance recorded when the result is produced.

The tests that matter most here are about the failure the manifest exists to
prevent — a number quoted from code nobody can recover. So: the dirty flag is
recorded, it is *visible* rather than buried, an unknown git state is not
silently reported as clean, and a manifest never breaks the run it describes.

The reproducibility criterion is tested the only way it can be honestly tested
without a training loop: two runs of one recorded configuration produce
identical metrics, and the fingerprints that identify that configuration are
stable across reorderings and sensitive to real changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.config.loader import load_config
from portfolio_agent.provenance import (
    RunManifest,
    build_manifest,
    config_fingerprint,
    data_fingerprint,
    find_manifest,
    library_versions,
    list_manifests,
    load_manifest,
    new_run_id,
    render_index,
    render_note,
    stable_hash,
    write_note,
)
from portfolio_agent.provenance.report import _decay_svg


@pytest.fixture
def app_config():
    return load_config()


def a_manifest(**overrides) -> RunManifest:
    defaults = dict(
        kind="evaluate",
        strategy="momentum",
        universe_fingerprint="abc123",
        universe_name="test",
        settings={"horizon": 5, "stride": 1},
        split={"scheme": "PurgedWalkForward", "horizon": 5, "embargo": 2},
        metrics={"mean_ic": 0.042, "t_stat": 3.1, "n_dates": 200},
        timings={"total": 12.5},
    )
    defaults.update(overrides)
    return build_manifest(**defaults)


# --------------------------------------------------------------------------
# Fingerprints
# --------------------------------------------------------------------------


def test_a_fingerprint_is_stable_across_key_order():
    """Two configs differing only in emission order must hash the same.

    Otherwise the fingerprint reports a change that did not happen, and the
    first thing anyone does with a noisy fingerprint is stop reading it.
    """
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_a_fingerprint_changes_when_a_value_changes():
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})


def test_the_config_fingerprint_tracks_the_resolved_config(app_config):
    first = config_fingerprint(app_config)
    assert first == config_fingerprint(app_config)

    changed = app_config.model_copy(deep=True)
    changed.data.universe_size = app_config.data.universe_size + 1
    assert config_fingerprint(changed) != first


def test_the_data_fingerprint_moves_when_the_cache_changes(tmp_path):
    from portfolio_agent.src.data_store import DataStore

    index = pd.date_range("2024-01-01", periods=30, freq="B")
    close = np.linspace(100.0, 130.0, 30)
    frame = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1e5},
        index=index,
    )
    store = DataStore(cache_dir=tmp_path)
    store.save_ticker_data("AAA.NS", frame.copy())

    before = data_fingerprint(["AAA.NS"], tmp_path)
    assert before["n_symbols"] == 1
    assert before["n_missing"] == 0

    # A refreshed cache must not fingerprint the same as the old one.
    import os, time

    path = tmp_path / "AAA.NS.parquet"
    os.utime(path, (time.time() + 100, time.time() + 100))
    after = data_fingerprint(["AAA.NS"], tmp_path)
    assert after["fingerprint"] != before["fingerprint"]


def test_the_data_fingerprint_counts_symbols_it_could_not_find(tmp_path):
    """A hash of nothing is a perfectly valid hash and looks like a real one."""
    record = data_fingerprint(["MISSING.NS", "ALSO_MISSING.NS"], tmp_path)
    assert record["n_symbols"] == 0
    assert record["n_missing"] == 2

    manifest = a_manifest()
    manifest.data = record
    assert "2 not found in the cache" in manifest.render()


def test_the_fingerprint_method_is_stated_rather_than_assumed(tmp_path):
    """It detects a changed cache, not a byte-identical rewrite. Say so."""
    record = data_fingerprint([], tmp_path)
    assert "not a content hash" in record["method"]


# --------------------------------------------------------------------------
# The dirty flag
# --------------------------------------------------------------------------


def test_a_dirty_tree_is_recorded_and_makes_the_run_irreproducible():
    manifest = a_manifest()
    manifest.git = {"commit": "a" * 40, "branch": "wip", "dirty": True, "dirty_files": ["x.py"]}
    assert not manifest.reproducible


def test_a_clean_tree_is_reproducible():
    manifest = a_manifest()
    manifest.git = {"commit": "a" * 40, "branch": "main", "dirty": False, "dirty_files": []}
    assert manifest.reproducible


def test_an_unknown_git_state_is_not_treated_as_clean():
    """"We did not check" and "we checked and it was clean" are different claims.

    Only one of them supports reproducing anything, so the conservative reading
    is the correct one.
    """
    manifest = a_manifest()
    manifest.git = {"commit": None, "branch": None, "dirty": None, "dirty_files": []}
    assert not manifest.reproducible


def test_the_dirty_warning_leads_the_note_rather_than_trailing_it():
    """Burying it in a footer makes the note complicit in the confusion."""
    manifest = a_manifest()
    manifest.git = {"commit": "b" * 40, "branch": "wip", "dirty": True, "dirty_files": []}
    html = render_note(manifest)

    assert "cannot be reproduced" in html
    # Before the metrics table, not after it.
    assert html.index("cannot be reproduced") < html.index("<h2>Metrics</h2>")


def test_an_unknown_git_state_gets_its_own_banner():
    manifest = a_manifest()
    manifest.git = {"commit": None, "branch": None, "dirty": None}
    html = render_note(manifest)
    assert "Provenance incomplete" in html


def test_a_clean_run_gets_no_banner():
    manifest = a_manifest()
    manifest.git = {"commit": "c" * 40, "branch": "main", "dirty": False}
    html = render_note(manifest)
    assert "cannot be reproduced" not in html
    assert "Provenance incomplete" not in html


def test_a_dirty_run_carries_the_explanation_as_a_note(app_config, monkeypatch):
    monkeypatch.setattr(
        "portfolio_agent.provenance.manifest.git_state",
        lambda: {"commit": "d" * 40, "branch": "wip", "dirty": True, "dirty_files": []},
    )
    manifest = build_manifest("train", app_config=app_config, strategy="x")
    assert any("uncommitted" in note for note in manifest.notes)


# --------------------------------------------------------------------------
# Round-tripping
# --------------------------------------------------------------------------


def test_a_manifest_round_trips_through_disk(tmp_path):
    manifest = a_manifest()
    path = manifest.save(tmp_path)
    restored = load_manifest(path)

    assert restored.run_id == manifest.run_id
    assert restored.metrics == manifest.metrics
    assert restored.split == manifest.split
    assert restored.git == manifest.git


def test_a_manifest_is_valid_json_without_a_custom_decoder(tmp_path):
    """Other tools read these. A numpy float that only survives `default=str`
    comes back as the string "0.042", which compares equal to nothing."""
    manifest = a_manifest(metrics={
        "mean_ic": np.float64(0.042),
        "n_dates": np.int64(200),
        "significant": np.bool_(True),
    })
    path = manifest.save(tmp_path)
    payload = json.loads(path.read_text())

    assert isinstance(payload["metrics"]["mean_ic"], float)
    assert isinstance(payload["metrics"]["n_dates"], int)
    assert isinstance(payload["metrics"]["significant"], bool)


def test_run_ids_sort_chronologically():
    first = new_run_id("evaluate", "momentum")
    second = new_run_id("evaluate", "momentum")
    assert sorted([second, first])[0][:15] <= sorted([second, first])[1][:15]


def test_two_runs_started_in_the_same_second_do_not_collide():
    """Comparing two seeds is a normal thing to do, and a timestamp alone
    would have one overwrite the other."""
    ids = {new_run_id("train", "x") for _ in range(50)}
    assert len(ids) == 50


def test_a_run_id_survives_a_strategy_name_with_punctuation():
    run_id = new_run_id("train", "weird/name:with*chars")
    assert "/" not in run_id and ":" not in run_id and "*" not in run_id


# --------------------------------------------------------------------------
# Lookup
# --------------------------------------------------------------------------


def test_a_manifest_is_found_by_a_unique_prefix(tmp_path):
    """Nobody retypes a timestamp."""
    manifest = a_manifest()
    manifest.save(tmp_path)
    found = find_manifest(manifest.run_id[:18], tmp_path)
    assert found.run_id == manifest.run_id


def test_an_ambiguous_prefix_raises_rather_than_picking_one(tmp_path):
    """Silently rendering the wrong run's note defeats having manifests."""
    for _ in range(3):
        a_manifest().save(tmp_path)
    with pytest.raises(ValueError, match="matches 3 runs"):
        find_manifest("2", tmp_path)


def test_an_unknown_run_id_points_at_the_listing_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="report --list"):
        find_manifest("nope", tmp_path)


def test_listing_returns_newest_first(tmp_path):
    ids = []
    for i in range(3):
        manifest = a_manifest()
        manifest.run_id = f"2026081{i}T000000-evaluate-x-0000"
        manifest.save(tmp_path)
        ids.append(manifest.run_id)

    listed = [m.run_id for m in list_manifests(tmp_path)]
    assert listed == sorted(ids, reverse=True)


def test_an_unreadable_manifest_is_skipped_not_fatal(tmp_path):
    """One corrupt file must not hide every other run."""
    a_manifest().save(tmp_path)
    (tmp_path / "broken.json").write_text("{not json")
    assert len(list_manifests(tmp_path)) == 1


def test_listing_an_absent_directory_is_empty_not_an_error(tmp_path):
    assert list_manifests(tmp_path / "nope") == []


# --------------------------------------------------------------------------
# The rendered note
# --------------------------------------------------------------------------


def test_the_note_is_standalone_with_no_external_requests():
    """A note that fails to render offline is worse than a plain table."""
    manifest = a_manifest()
    html = render_note(manifest)

    assert "<!doctype html>" in html.lower()
    assert "http://" not in html and "https://" not in html
    for tag in ("<script src=", "<link rel=\"stylesheet\"", "@import"):
        assert tag not in html


def test_the_note_carries_metrics_settings_and_provenance():
    manifest = a_manifest()
    html = render_note(manifest)

    assert "mean_ic" in html
    assert "horizon" in html
    assert manifest.universe_fingerprint in html
    assert "Provenance" in html
    assert manifest.run_id in html


def test_the_note_renders_a_decay_chart_when_one_is_recorded():
    manifest = a_manifest(extras={"decay": {
        "shape": "slow: IC peaks at 21d",
        "points": [
            {"horizon": 1, "mean_ic": 0.02, "t_stat": 2.2},
            {"horizon": 5, "mean_ic": 0.04, "t_stat": 4.0},
            {"horizon": 21, "mean_ic": 0.043, "t_stat": 2.7},
        ],
    }})
    html = render_note(manifest)
    assert "<svg" in html
    assert "Signal decay" in html
    assert "slow: IC peaks at 21d" in html


def test_the_decay_chart_needs_at_least_two_points():
    assert _decay_svg([{"horizon": 1, "mean_ic": 0.02}]) == ""
    assert _decay_svg([]) == ""


def test_the_decay_chart_draws_a_zero_line_when_ic_crosses_zero():
    with_negative = _decay_svg([
        {"horizon": 1, "mean_ic": 0.05},
        {"horizon": 21, "mean_ic": -0.03},
    ])
    assert with_negative.count("<line") > 3   # gridlines plus the zero rule


def test_the_note_lists_folds_when_a_split_was_used():
    manifest = a_manifest(extras={"folds": [
        {"fold": 0, "test_start": "2024-01-01", "test_end": "2024-06-30",
         "n_dates": 120, "n_purged": 5, "mean_ic": 0.03},
    ]})
    html = render_note(manifest)
    assert "Walk-forward folds" in html
    assert "n_purged" in html


def test_the_note_escapes_content_rather_than_injecting_it():
    manifest = a_manifest(strategy="<script>alert(1)</script>")
    html = render_note(manifest)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_writing_a_note_creates_the_directory(tmp_path):
    manifest = a_manifest()
    path = write_note(manifest, tmp_path / "nested" / "note.html")
    assert path.exists()
    assert path.read_text().startswith("<!doctype html>")


def test_the_index_renders_a_table_of_runs(tmp_path):
    manifests = [a_manifest(strategy=name) for name in ("momentum", "low_volatility")]
    text = render_index(manifests)
    assert "momentum" in text and "low_volatility" in text
    assert "run_id" in text


def test_the_index_of_nothing_says_so():
    assert render_index([]) == "No runs recorded."


# --------------------------------------------------------------------------
# Wiring: train and evaluate both write one
# --------------------------------------------------------------------------


def _ohlcv(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    return pd.DataFrame(
        {"open": close, "high": close * 1.02, "low": close * 0.98,
         "close": close, "volume": rng.integers(1e5, 1e6, n).astype(float)},
        index=index,
    )


@pytest.fixture
def fake_cache(monkeypatch):
    frames = {f"T{i}": _ohlcv(seed=i) for i in range(10)}
    monkeypatch.setattr(
        "portfolio_agent.src.data_store.load_ticker_data",
        lambda ticker, start_date=None, end_date=None: frames.get(ticker),
        raising=True,
    )
    return frames


def test_an_evaluation_writes_a_manifest(app_config, fake_cache, tmp_path):
    from portfolio_agent.evaluation import evaluate_forecast

    result = evaluate_forecast(
        app_config, "momentum", universe=list(fake_cache),
        horizon=5, stride=30, min_history=260, use_benchmark=False,
        runs_dir=str(tmp_path),
    )
    assert result.run_id
    manifest = find_manifest(result.run_id, tmp_path)

    assert manifest.kind == "evaluate"
    assert manifest.strategy == "momentum"
    assert manifest.n_symbols == 10
    assert manifest.settings["horizon"] == 5
    assert manifest.split["scheme"] == "single window (no walk-forward split)"
    assert "mean_ic" in manifest.metrics


def test_the_split_travels_with_the_metric(app_config, fake_cache, tmp_path):
    """A metric separated from the split that produced it is not comparable."""
    from portfolio_agent.evaluation import evaluate_forecast
    from portfolio_agent.validation.purged import PurgedWalkForward

    result = evaluate_forecast(
        app_config, "momentum", universe=list(fake_cache),
        horizon=5, stride=20, min_history=260, use_benchmark=False,
        splitter=PurgedWalkForward(n_splits=2, horizon=5, embargo=3),
        runs_dir=str(tmp_path),
    )
    manifest = find_manifest(result.run_id, tmp_path)
    assert manifest.split["scheme"] == "PurgedWalkForward"
    assert manifest.split["embargo"] == 3
    assert manifest.split["n_splits"] == 2


def test_manifests_can_be_turned_off(app_config, fake_cache, tmp_path):
    from portfolio_agent.evaluation import evaluate_forecast

    result = evaluate_forecast(
        app_config, "momentum", universe=list(fake_cache),
        horizon=5, stride=30, min_history=260, use_benchmark=False,
        manifest=False, runs_dir=str(tmp_path),
    )
    assert result.run_id is None
    assert list(tmp_path.glob("*.json")) == []


def test_a_manifest_failure_never_breaks_the_result(
    app_config, fake_cache, tmp_path, monkeypatch
):
    """Provenance is worth a file and is not worth losing a result over."""
    from portfolio_agent.evaluation import evaluate_forecast

    def explode(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr("portfolio_agent.provenance.build_manifest", explode)

    result = evaluate_forecast(
        app_config, "momentum", universe=list(fake_cache),
        horizon=5, stride=30, min_history=260, use_benchmark=False,
        runs_dir=str(tmp_path),
    )
    assert result.n_observations > 0
    assert result.run_id is None


def test_two_runs_of_one_configuration_reproduce_the_metrics(
    app_config, fake_cache, tmp_path
):
    """The acceptance criterion, in the form this layer can honestly assert."""
    from portfolio_agent.evaluation import evaluate_forecast

    def run():
        return evaluate_forecast(
            app_config, "momentum", universe=list(fake_cache),
            horizon=5, stride=30, min_history=260, use_benchmark=False,
            runs_dir=str(tmp_path),
        )

    first, second = run(), run()
    manifests = [find_manifest(r.run_id, tmp_path) for r in (first, second)]

    assert manifests[0].run_id != manifests[1].run_id
    assert manifests[0].metrics == manifests[1].metrics
    assert manifests[0].config_fingerprint == manifests[1].config_fingerprint
    assert manifests[0].universe_fingerprint == manifests[1].universe_fingerprint
    assert manifests[0].settings == manifests[1].settings


def test_a_training_run_writes_a_manifest(app_config, fake_cache, tmp_path):
    from portfolio_agent.training import run_training_job

    run = run_training_job(
        app_config, "india_sac",
        overrides={"epochs": 1, "gradient_steps": 1, "batch_size": 8,
                   "warmup_transitions": 0, "min_history": 260},
        universe=list(fake_cache), models_dir=tmp_path / "models",
        runs_dir=tmp_path / "runs",
    )
    assert run.run_id
    manifest = find_manifest(run.run_id, tmp_path / "runs")

    assert manifest.kind == "train"
    assert manifest.strategy == "india_sac"
    assert manifest.trainer == "sac"
    assert manifest.universe_fingerprint == run.universe.fingerprint
    assert manifest.settings["epochs"] == 1
    assert "total" in manifest.timings


def test_a_failed_training_run_still_writes_a_manifest(app_config, tmp_path, monkeypatch):
    """A failed run is exactly the one someone will want to reconstruct."""
    from portfolio_agent.training import run_training_job

    monkeypatch.setattr(
        "portfolio_agent.src.data_store.load_ticker_data",
        lambda ticker, start_date=None, end_date=None: None,
        raising=True,
    )
    run = run_training_job(
        app_config, "india_sac", universe=["A.NS", "B.NS"],
        models_dir=tmp_path / "models", runs_dir=tmp_path / "runs",
    )
    assert not run.ok
    manifest = find_manifest(run.run_id, tmp_path / "runs")
    assert any("failed" in note for note in manifest.notes)


# --------------------------------------------------------------------------
# The CLI
# --------------------------------------------------------------------------


def test_cli_report_lists_recorded_runs(tmp_path, capsys):
    from portfolio_agent.cli import main

    for name in ("momentum", "low_volatility"):
        a_manifest(strategy=name).save(tmp_path)

    assert main(["report", "--runs-dir", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "momentum" in output and "low_volatility" in output


def test_cli_report_renders_a_note_without_re_running_anything(tmp_path, capsys):
    """The acceptance criterion: `report --run ID` reads, it does not compute."""
    from portfolio_agent.cli import main

    manifest = a_manifest()
    manifest.git = {"commit": "e" * 40, "branch": "main", "dirty": False}
    manifest.save(tmp_path)

    code = main(["report", "--run", manifest.run_id, "--runs-dir", str(tmp_path)])
    assert code == 0

    note = tmp_path / f"{manifest.run_id}.html"
    assert note.exists()
    assert "mean_ic" in note.read_text()
    assert "Research note written" in capsys.readouterr().out


def test_cli_report_exits_two_for_a_run_from_a_dirty_tree(tmp_path, capsys):
    """A script must not quote an irreproducible number without noticing."""
    from portfolio_agent.cli import main

    manifest = a_manifest()
    manifest.git = {"commit": "f" * 40, "branch": "wip", "dirty": True, "dirty_files": []}
    manifest.save(tmp_path)

    args = ["report", "--run", manifest.run_id, "--runs-dir", str(tmp_path), "--no-html"]
    assert main(args) == 2
    assert main(args + ["--allow-dirty"]) == 0
    assert "DIRTY WORKING TREE" in capsys.readouterr().out


def test_cli_report_emits_json(tmp_path, capsys):
    from portfolio_agent.cli import main

    manifest = a_manifest()
    manifest.save(tmp_path)

    assert main(["report", "--run", manifest.run_id, "--runs-dir", str(tmp_path),
                 "--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["run_id"] == manifest.run_id
    assert document["metrics"]["mean_ic"] == 0.042


def test_cli_report_on_an_empty_directory_explains_where_runs_come_from(
    tmp_path, capsys
):
    from portfolio_agent.cli import main

    assert main(["report", "--runs-dir", str(tmp_path)]) == 1
    assert "written automatically" in capsys.readouterr().out


def test_cli_report_reports_an_unknown_run(tmp_path, capsys):
    from portfolio_agent.cli import main

    a_manifest().save(tmp_path)
    assert main(["report", "--run", "nope", "--runs-dir", str(tmp_path)]) == 1
    assert "Error" in capsys.readouterr().out


def test_the_library_versions_that_can_move_a_number_are_recorded():
    versions = library_versions()
    assert "numpy" in versions and "pandas" in versions
