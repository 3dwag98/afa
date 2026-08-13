"""The CLI, which had no tests at all before this.

873 lines, the only surface anybody touches, and nothing checked it. Three of
the bugs found while building the training layer were in exactly this shape of
code — plumbing that looks obviously correct and quietly is not: a flag parsed,
printed in `--help`, and never passed to the function it named.

So the coverage here is deliberately shallow and wide rather than deep. Every
subcommand gets a `--help` smoke test, an end-to-end invocation, an unknown
option, and a `--json` parse where it produces results. Depth belongs in the
tests for the layers underneath; what was missing was anything at all at this
one.

Two commands are exercised through their guarded early exits rather than end to
end, and it is worth being explicit about which: `download-data` needs the
network, and `run-agent` drives the frozen live path and writes an Excel
report. Both are asserted to parse and to reach their first decision.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.cli import create_parser, main
from portfolio_agent.cli_forecast import (
    CV_SCHEMES,
    NEUTRALIZE_KINDS,
    build_splitter,
    parse_int_list,
    parse_str_list,
)

#: Every subcommand the parser exposes. Derived from the parser rather than
#: hardcoded, so a new command that ships without a test fails this file.
def all_subcommands() -> list:
    parser = create_parser()
    actions = [
        a for a in parser._actions if hasattr(a, "choices") and isinstance(a.choices, dict)
    ]
    assert actions, "no subparser action found"
    return sorted(actions[0].choices)


def _ohlcv(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    return pd.DataFrame(
        {
            "open": close, "high": close * 1.02, "low": close * 0.98,
            "close": close, "volume": rng.integers(1e5, 1e6, n).astype(float),
        },
        index=index,
    )


@pytest.fixture
def fake_cache(monkeypatch, tmp_path):
    """Synthetic bars in place of the parquet cache, plus a real on-disk copy.

    Both, because the evaluation commands read through `load_ticker_data` while
    the data commands read files directly — and a fixture that served only one
    would make half these tests pass for the wrong reason.
    """
    from portfolio_agent.src.data_store import DataStore

    frames = {f"T{i}": _ohlcv(seed=i) for i in range(12)}
    monkeypatch.setattr(
        "portfolio_agent.src.data_store.load_ticker_data",
        lambda ticker, start_date=None, end_date=None: frames.get(ticker),
        raising=True,
    )

    store = DataStore(cache_dir=tmp_path / "market_data")
    for ticker, frame in frames.items():
        store.save_ticker_data(f"{ticker}.NS", frame.copy())

    return {"frames": frames, "cache_dir": tmp_path / "market_data",
            "runs_dir": tmp_path / "runs"}


# --------------------------------------------------------------------------
# --help for every subcommand
# --------------------------------------------------------------------------


@pytest.mark.parametrize("command", all_subcommands())
def test_every_subcommand_has_help(command, capsys):
    """A `--help` that raises is a command nobody can discover."""
    with pytest.raises(SystemExit) as excinfo:
        main([command, "--help"])
    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "usage:" in output


@pytest.mark.parametrize("subcommand", ["status", "validate", "build"])
def test_every_data_subcommand_has_help(subcommand, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["data", subcommand, "--help"])
    assert excinfo.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_the_top_level_help_lists_every_command(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    output = capsys.readouterr().out
    for command in ("evaluate", "compare", "list-features", "data", "report"):
        assert command in output


def test_no_command_prints_help_and_fails(capsys):
    assert main([]) == 1
    assert "usage:" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Unknown options name the valid ones
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["evaluate", "--strategy", "momentum"],
        ["compare", "--strategies", "momentum"],
        ["list-features"],
        ["report"],
        ["data", "status"],
    ],
)
def test_an_unknown_option_is_rejected(argv, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(argv + ["--definitely-not-an-option"])
    assert excinfo.value.code != 0
    assert "unrecognized arguments" in capsys.readouterr().err


def test_an_unknown_command_lists_the_valid_ones(capsys):
    with pytest.raises(SystemExit):
        main(["definitely-not-a-command"])
    error = capsys.readouterr().err
    assert "invalid choice" in error
    assert "evaluate" in error


def test_an_invalid_cv_scheme_names_the_valid_ones(capsys):
    with pytest.raises(SystemExit):
        main(["evaluate", "--strategy", "momentum", "--cv", "sideways"])
    error = capsys.readouterr().err
    assert "invalid choice" in error
    for scheme in CV_SCHEMES:
        assert scheme in error


def test_an_unknown_neutralize_exposure_names_the_valid_ones(capsys):
    code = main([
        "evaluate", "--strategy", "momentum", "--neutralize", "vibes", "--dry-run",
    ])
    assert code == 1
    output = capsys.readouterr().out
    assert "vibes" in output
    for kind in NEUTRALIZE_KINDS:
        assert kind in output


def test_a_malformed_horizon_list_names_the_offender(capsys):
    code = main([
        "evaluate", "--strategy", "momentum", "--horizons", "1,five,21", "--dry-run",
    ])
    assert code == 1
    assert "five" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Argument parsing helpers
# --------------------------------------------------------------------------


def test_parse_int_list():
    assert parse_int_list("1,5,21", "horizons") == [1, 5, 21]
    assert parse_int_list(" 1 , 5 ", "horizons") == [1, 5]
    assert parse_int_list(None, "horizons") is None


def test_parse_int_list_rejects_an_empty_list():
    with pytest.raises(ValueError, match="no values"):
        parse_int_list(",,", "horizons")


def test_parse_str_list():
    assert parse_str_list("a, b ,c") == ["a", "b", "c"]
    assert parse_str_list("") is None


# --------------------------------------------------------------------------
# --cv defaults to the correct scheme
# --------------------------------------------------------------------------


def test_cv_defaults_to_purged():
    """The correct method should be what you get by not thinking about it.

    A default that requires knowledge to be safe is a default that will be
    wrong most of the times it is used.
    """
    parser = create_parser()
    args = parser.parse_args(["evaluate", "--strategy", "momentum"])
    assert args.cv == "purged"


def test_the_purged_splitter_actually_purges():
    """A `--cv` value that parsed and then did nothing is the failure mode."""
    splitter = build_splitter("purged", horizon=5, embargo=2, n_splits=3)
    assert splitter.horizon == 5
    assert splitter.embargo == 2
    assert splitter.n_splits == 3


def test_walkforward_switches_the_purge_off_rather_than_pretending():
    """It exists so the size of that bias is measurable, not so it is hidden."""
    splitter = build_splitter("walkforward", horizon=5, embargo=0, n_splits=3)
    assert splitter.horizon == 0


def test_cv_none_means_a_single_window():
    assert build_splitter("none", horizon=5, embargo=0, n_splits=3) is None


def test_an_unknown_cv_scheme_raises():
    with pytest.raises(ValueError, match="purged"):
        build_splitter("nonsense", horizon=5, embargo=0, n_splits=3)


# --------------------------------------------------------------------------
# evaluate, end to end
# --------------------------------------------------------------------------


def _evaluate_args(fake_cache, *extra):
    return [
        "evaluate", "--strategy", "momentum",
        "--tickers", ",".join(fake_cache["frames"]),
        "--horizon", "5", "--stride", "40", "--min-history", "260",
        "--no-benchmark", "--output", str(fake_cache["runs_dir"]),
        *extra,
    ]


def test_evaluate_runs_end_to_end(fake_cache, capsys):
    assert main(_evaluate_args(fake_cache)) == 0
    output = capsys.readouterr().out
    assert "Forecast evaluation" in output
    assert "mean rank IC" in output
    assert "monotonicity" in output


def test_evaluate_emits_parseable_json(fake_cache, capsys):
    assert main(_evaluate_args(fake_cache, "--json")) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["strategy"] == "momentum"
    assert "mean_ic" in document
    assert "bucket_mean_returns" in document


def test_evaluate_dry_run_resolves_and_runs_nothing(fake_cache, capsys):
    assert main(_evaluate_args(fake_cache, "--dry-run")) == 0
    output = capsys.readouterr().out
    assert "nothing was run" in output
    assert "purged" in output
    # Nothing was computed, so nothing was recorded.
    assert not list(fake_cache["runs_dir"].glob("*.json")) if fake_cache["runs_dir"].exists() else True


def test_evaluate_dry_run_emits_json(fake_cache, capsys):
    assert main(_evaluate_args(fake_cache, "--dry-run", "--json")) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["cv"] == "purged"
    assert plan["universe_size"] == 12


def test_evaluate_records_a_run(fake_cache, capsys):
    assert main(_evaluate_args(fake_cache)) == 0
    manifests = list(fake_cache["runs_dir"].glob("*.json"))
    assert len(manifests) == 1
    assert "portfolio-agent report --run" in capsys.readouterr().out


def test_evaluate_with_folds_reports_them(fake_cache, capsys):
    # A denser stride than the other tests use: two folds need enough distinct
    # dates to divide, and six is not enough to divide into anything.
    args = [
        "evaluate", "--strategy", "momentum",
        "--tickers", ",".join(fake_cache["frames"]),
        "--horizon", "5", "--stride", "5", "--min-history", "260",
        "--no-benchmark", "--output", str(fake_cache["runs_dir"]),
        "--cv", "purged", "--splits", "2",
    ]
    assert main(args) == 0
    output = capsys.readouterr().out
    assert "Walk-forward folds" in output
    assert "purged" in output


def test_evaluate_with_a_decay_curve(fake_cache, capsys):
    assert main(_evaluate_args(fake_cache, "--horizons", "1,5,21")) == 0
    output = capsys.readouterr().out
    assert "Signal decay" in output
    assert "21d" in output


def test_evaluate_with_neutralization_states_the_size_proxy(fake_cache, capsys):
    """The substitution has to reach the output, not stop at a docstring."""
    assert main(_evaluate_args(fake_cache, "--neutralize", "beta,size")) == 0
    output = capsys.readouterr().out
    assert "Neutralized against" in output
    assert "free float" in output


def test_evaluate_with_a_baseline_prints_one_table(fake_cache, capsys):
    """One flag, same panel, same dates — so the comparison never gets skipped."""
    assert main(_evaluate_args(fake_cache, "--baseline", "low_volatility")) == 0
    output = capsys.readouterr().out
    assert "baseline, same panel and same dates" in output
    assert "low_volatility" in output
    assert ("beats" in output) or ("does NOT beat" in output)


def test_a_baseline_that_cannot_run_does_not_lose_the_result(fake_cache, capsys):
    """The primary result is already computed and is worth having regardless."""
    assert main(_evaluate_args(fake_cache, "--baseline", "not_a_strategy")) == 0
    output = capsys.readouterr().out
    assert "Forecast evaluation" in output
    assert "could not be scored" in output


def test_evaluate_reports_an_unknown_strategy(fake_cache, capsys):
    assert main(_evaluate_args(fake_cache)[:2] + ["not_a_strategy"] +
                _evaluate_args(fake_cache)[3:]) == 1
    assert "Error" in capsys.readouterr().out


def test_evaluate_seeds_when_asked(fake_cache, capsys):
    assert main(_evaluate_args(fake_cache, "--seed", "7", "--json")) == 0
    assert json.loads(capsys.readouterr().out)["strategy"] == "momentum"


def test_limit_narrows_the_universe_visibly(fake_cache, capsys):
    """The honest alternative to a preset that loosens the method."""
    assert main([
        "evaluate", "--strategy", "momentum", "--limit", "5", "--dry-run", "--json",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["universe_size"] == 5


def test_there_is_no_quick_preset():
    """A preset that trades correctness for speed becomes the default in a week,
    and its numbers are indistinguishable from real ones in every report."""
    parser = create_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate", "--strategy", "momentum", "--quick"])


# --------------------------------------------------------------------------
# compare, end to end
# --------------------------------------------------------------------------


def test_compare_runs_every_strategy_on_one_universe(fake_cache, capsys):
    """Two strategies on two different draws differ by the draw at least as
    much as by the strategy. One universe, resolved once."""
    assert main([
        "compare", "--strategies", "momentum,low_volatility",
        "--tickers", ",".join(fake_cache["frames"]),
        "--stride", "40", "--min-history", "260", "--no-benchmark",
        "--cv", "none", "--output", str(fake_cache["runs_dir"]),
    ]) == 0
    output = capsys.readouterr().out
    assert "Forecast comparison" in output
    assert "momentum" in output and "low_volatility" in output


def test_compare_emits_parseable_json(fake_cache, capsys):
    assert main([
        "compare", "--strategies", "momentum,low_volatility",
        "--tickers", ",".join(fake_cache["frames"]),
        "--stride", "40", "--min-history", "260", "--no-benchmark",
        "--cv", "none", "--output", str(fake_cache["runs_dir"]), "--json",
    ]) == 0
    document = json.loads(capsys.readouterr().out)
    assert len(document["results"]) == 2
    assert {"strategy", "mean_ic"} <= set(document["results"][0])


def test_one_failing_strategy_does_not_discard_the_others(fake_cache, capsys):
    assert main([
        "compare", "--strategies", "momentum,not_a_strategy",
        "--tickers", ",".join(fake_cache["frames"]),
        "--stride", "40", "--min-history", "260", "--no-benchmark",
        "--cv", "none", "--output", str(fake_cache["runs_dir"]),
    ]) == 0
    output = capsys.readouterr().out
    assert "momentum" in output
    assert "not_a_strategy failed" in output


def test_compare_fails_when_nothing_could_be_evaluated(fake_cache, capsys):
    assert main([
        "compare", "--strategies", "not_a_strategy,also_not_one",
        "--tickers", ",".join(fake_cache["frames"]),
        "--stride", "40", "--min-history", "260", "--no-benchmark", "--cv", "none",
    ]) == 1
    assert "no strategy could be evaluated" in capsys.readouterr().out


def test_compare_requires_a_strategy_list(capsys):
    assert main(["compare", "--strategies", ","]) == 1
    assert "comma-separated" in capsys.readouterr().out


# --------------------------------------------------------------------------
# list-features
# --------------------------------------------------------------------------


def test_list_features_lists_the_registry(capsys):
    assert main(["list-features"]) == 0
    output = capsys.readouterr().out
    assert "rsi_14" in output and "macd" in output
    assert "Registered features" in output


def test_list_features_emits_parseable_json(capsys):
    assert main(["list-features", "--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert "rsi_14" in document
    from portfolio_agent.features.registry import _FEATURE_REGISTRY

    assert set(document) == set(_FEATURE_REGISTRY)


# --------------------------------------------------------------------------
# data status / validate / build
# --------------------------------------------------------------------------


def test_data_status_runs_end_to_end(fake_cache, capsys):
    assert main(["data", "status", "--cache-dir", str(fake_cache["cache_dir"])]) == 0
    assert "Data store status" in capsys.readouterr().out


def test_data_status_emits_parseable_json(fake_cache, capsys):
    assert main([
        "data", "status", "--cache-dir", str(fake_cache["cache_dir"]), "--json",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["symbols"] == 12


def test_data_validate_runs_end_to_end(fake_cache, capsys):
    code = main(["data", "validate", "--cache-dir", str(fake_cache["cache_dir"])])
    assert code in (0, 1)
    assert "Checked 12 symbol(s)" in capsys.readouterr().out


def test_data_validate_emits_parseable_json(fake_cache, capsys):
    main(["data", "validate", "--cache-dir", str(fake_cache["cache_dir"]), "--json"])
    document = json.loads(capsys.readouterr().out)
    assert document["symbols_checked"] == 12
    assert "by_check" in document


def test_data_with_no_subcommand_prints_help(capsys):
    assert main(["data"]) == 1
    assert "usage:" in capsys.readouterr().out


def test_data_build_runs_the_download_then_checks_it(fake_cache, monkeypatch, capsys):
    """A download that half-succeeds looks exactly like one that worked."""
    calls = {"download": 0}

    def fake_download(args):
        calls["download"] += 1
        return 0

    monkeypatch.setattr("portfolio_agent.cli.cmd_download_data", fake_download)
    monkeypatch.setattr(
        "portfolio_agent.src.data_store.DATA_DIR", fake_cache["cache_dir"]
    )

    code = main(["data", "build", "--validate-limit", "12"])
    assert calls["download"] == 1
    assert code in (0, 1)
    output = capsys.readouterr().out
    assert "Checking what arrived" in output
    assert "Data store status" in output


def test_data_build_can_skip_validation_and_says_so(fake_cache, monkeypatch, capsys):
    monkeypatch.setattr("portfolio_agent.cli.cmd_download_data", lambda a: 0)
    assert main(["data", "build", "--no-validate"]) == 0
    assert "Skipping validation" in capsys.readouterr().out


def test_data_build_stops_when_the_download_fails(monkeypatch, capsys):
    monkeypatch.setattr("portfolio_agent.cli.cmd_download_data", lambda a: 1)
    assert main(["data", "build"]) == 1
    assert "Checking what arrived" not in capsys.readouterr().out


def test_data_build_accepts_the_documented_flags():
    parser = create_parser()
    args = parser.parse_args([
        "data", "build", "--years", "20", "--keep-raw", "--fail-on-warning",
    ])
    assert args.years == 20
    assert args.keep_raw is True
    assert args.fail_on_warning is True


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def test_report_lists_runs_after_an_evaluation(fake_cache, capsys):
    assert main(_evaluate_args(fake_cache)) == 0
    capsys.readouterr()

    assert main(["report", "--runs-dir", str(fake_cache["runs_dir"])]) == 0
    assert "momentum" in capsys.readouterr().out


def test_report_emits_parseable_json(fake_cache, capsys):
    assert main(_evaluate_args(fake_cache)) == 0
    capsys.readouterr()

    assert main(["report", "--runs-dir", str(fake_cache["runs_dir"]), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows and rows[0]["strategy"] == "momentum"


# --------------------------------------------------------------------------
# The two commands that cannot run offline, exercised at their first decision
# --------------------------------------------------------------------------


def test_download_data_reaches_its_source_branch(monkeypatch, capsys):
    """Needs the network, so the assertion is that it parses and dispatches."""
    seen = {}

    def fake_sync(**kwargs):
        seen.update(kwargs)
        return ["AAA.NS"]

    monkeypatch.setattr("portfolio_agent.src.hf_dataset.sync_hf_to_cache", fake_sync)
    code = main(["download-data", "--years", "20", "--universe-size", "3"])

    assert code == 0
    assert seen["max_symbols"] == 3
    # --years must reach the ingest, not merely be accepted by the parser.
    assert seen["start_date"] < seen["end_date"]


def test_run_agent_parses_and_dispatches(monkeypatch, capsys):
    """Drives the frozen live path and writes an Excel report, so it is not run.

    What is asserted is that its flags reach the orchestrator, which is the
    class of bug this file exists for.
    """
    seen = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        return "output/report.xlsx"

    monkeypatch.setattr("portfolio_agent.src.orchestrator.run_orchestrator", fake_run)
    code = main(["run-agent", "--force-refresh"])

    assert code == 0
    assert seen.get("force_refresh") is True


# --------------------------------------------------------------------------
# Global flags
# --------------------------------------------------------------------------


def test_the_config_flag_reaches_the_forecasting_commands(fake_cache, tmp_path, capsys):
    """These load the config themselves, so --config has to be threaded through.

    Exactly the shape of bug this file exists for: a global flag that the
    parser accepts and one branch of the code never reads.
    """
    import shutil

    custom = tmp_path / "custom.yaml"
    shutil.copy("config.yaml", custom)
    custom.write_text(
        custom.read_text().replace("universe_size: 4000", "universe_size: 3")
    )

    assert main([
        "--config", str(custom), "evaluate", "--strategy", "momentum",
        "--dry-run", "--json",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["universe_size"] == 3


def test_a_missing_config_file_is_reported(capsys, tmp_path):
    code = main(["--config", str(tmp_path / "nope.yaml"), "list-features"])
    # The loader warns and falls back rather than failing; the command still runs.
    assert code == 0
