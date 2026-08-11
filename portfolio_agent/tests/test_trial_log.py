"""Tests for the append-only trial log.

The Deflated Sharpe Ratio's denominator lives here. If this file is wrong, a
reported Sharpe is deflated against the wrong N and the statistic is worse
than useless — it is confidently wrong.
"""

import json

import pytest

from portfolio_agent.config.schema import AppConfig
from portfolio_agent.src.trial_log import (
    Trial,
    config_hash,
    iter_trials,
    read_trials,
    record_trial,
    trial_statistics,
)


def _trial(run_id: str, config_hash_value: str, sharpe: float) -> Trial:
    return Trial(
        run_id=run_id,
        timestamp="2026-08-11T00:00:00+00:00",
        config_hash=config_hash_value,
        sharpe=sharpe,
    )


class TestConfigHash:
    def test_identical_configs_hash_identically(self):
        assert config_hash(AppConfig()) == config_hash(AppConfig())

    def test_a_changed_parameter_changes_the_hash(self):
        changed = AppConfig()
        changed.risk.atr_stop_multiplier += 1.0
        assert config_hash(changed) != config_hash(AppConfig())

    def test_key_order_does_not_matter(self):
        assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


class TestRecordAndRead:
    def test_round_trips(self, tmp_path):
        path = str(tmp_path / "trials.jsonl")
        assert record_trial(_trial("r1", "abc", 0.8), path)
        assert record_trial(_trial("r2", "def", -0.4), path)

        trials = read_trials(path)
        assert [t.run_id for t in trials] == ["r1", "r2"]
        assert [t.sharpe for t in trials] == [0.8, -0.4]

    def test_appends_rather_than_overwrites(self, tmp_path):
        path = str(tmp_path / "nested" / "trials.jsonl")
        for i in range(5):
            record_trial(_trial(f"r{i}", f"h{i}", float(i)), path)
        assert len(read_trials(path)) == 5

    def test_missing_file_reads_as_no_trials(self, tmp_path):
        assert read_trials(str(tmp_path / "absent.jsonl")) == []
        assert list(iter_trials(str(tmp_path / "absent.jsonl"))) == []

    def test_a_corrupt_line_does_not_lose_the_rest(self, tmp_path):
        """A half-written record from a killed process is one lost trial, not
        a lost history."""
        path = tmp_path / "trials.jsonl"
        path.write_text(
            json.dumps({"run_id": "good1", "timestamp": "t", "config_hash": "a", "sharpe": 1.0})
            + "\n{ this is not json\n"
            + json.dumps({"run_id": "good2", "timestamp": "t", "config_hash": "b", "sharpe": 2.0})
            + "\n\n"
        )
        assert [t.run_id for t in read_trials(str(path))] == ["good1", "good2"]

    def test_unknown_fields_are_ignored(self, tmp_path):
        """A newer writer must not break an older reader."""
        path = tmp_path / "trials.jsonl"
        path.write_text(
            json.dumps({
                "run_id": "r", "timestamp": "t", "config_hash": "a", "sharpe": 0.5,
                "some_future_field": 42,
            }) + "\n"
        )
        assert read_trials(str(path))[0].sharpe == 0.5

    def test_a_failed_write_is_reported_not_raised(self, tmp_path):
        # A directory where the file should be: open() will fail.
        blocked = tmp_path / "trials.jsonl"
        blocked.mkdir()
        assert record_trial(_trial("r", "a", 1.0), str(blocked)) is False


class TestTrialStatistics:
    def test_supplies_n_and_variance(self, tmp_path):
        path = str(tmp_path / "trials.jsonl")
        for i, sharpe in enumerate([0.1, 0.5, -0.3, 1.2]):
            record_trial(_trial(f"r{i}", f"h{i}", sharpe), path)

        stats = trial_statistics(path)

        assert stats["n_trials"] == 4
        # ddof=1 over [0.1, 0.5, -0.3, 1.2], mean 0.375.
        assert stats["sharpe_variance"] == pytest.approx(0.409167, abs=1e-5)
        assert stats["best_sharpe"] == 1.2

    def test_repeat_runs_of_one_config_count_once(self, tmp_path):
        """Re-running the same backtest is not an independent trial; letting
        it inflate N would deflate the Sharpe for the wrong reason."""
        path = str(tmp_path / "trials.jsonl")
        record_trial(_trial("r1", "same", 0.4), path)
        record_trial(_trial("r2", "same", 0.9), path)
        record_trial(_trial("r3", "other", 0.2), path)

        stats = trial_statistics(path)

        assert stats["n_trials"] == 2
        assert stats["n_records"] == 3
        # The FIRST result for a config is kept, so a re-run cannot quietly
        # replace a recorded number with a luckier one.
        assert stats["best_sharpe"] == 0.4

    def test_empty_log_reports_zero_trials_not_one(self, tmp_path):
        """'No recorded trials' and 'one trial' are different claims, and the
        DSR is not computable under the first."""
        stats = trial_statistics(str(tmp_path / "absent.jsonl"))
        assert stats["n_trials"] == 0
        assert stats["sharpe_variance"] == 0.0

    def test_non_finite_sharpes_are_dropped(self, tmp_path):
        path = tmp_path / "trials.jsonl"
        path.write_text(
            json.dumps({"run_id": "r1", "timestamp": "t", "config_hash": "a", "sharpe": 1.0})
            + "\n"
            + json.dumps({"run_id": "r2", "timestamp": "t", "config_hash": "b", "sharpe": None})
            + "\n"
        )
        # sharpe=None fails the isfinite check without raising.
        assert trial_statistics(str(path))["n_trials"] == 1


class TestDeflationEndToEnd:
    def test_a_larger_search_deflates_the_same_sharpe_further(self, tmp_path):
        """The behaviour the whole file exists to support."""
        import numpy as np
        import pandas as pd

        from portfolio_agent.src.risk_analytics import RiskAnalyzer

        rng = np.random.default_rng(0)
        index = pd.bdate_range("2021-01-01", periods=1000)
        equity = pd.Series(
            1_000_000 * np.cumprod(1 + rng.normal(0.0012, 0.011, 1000)), index=index
        )

        small_log = str(tmp_path / "small.jsonl")
        large_log = str(tmp_path / "large.jsonl")
        for i in range(4):
            record_trial(_trial(f"s{i}", f"s{i}", 0.2 * i), small_log)
        for i in range(400):
            record_trial(_trial(f"l{i}", f"l{i}", 0.2 * (i % 4)), large_log)

        small = RiskAnalyzer(equity, [], trial_log_path=small_log)
        large = RiskAnalyzer(equity, [], trial_log_path=large_log)

        small_dsr = small.calculate_deflated_sharpe_ratio()
        large_dsr = large.calculate_deflated_sharpe_ratio()

        assert small_dsr["computable"] and large_dsr["computable"]
        assert small_dsr["n_trials"] == 4 and large_dsr["n_trials"] == 400
        assert large_dsr["expected_max_sharpe"] > small_dsr["expected_max_sharpe"]
        assert large_dsr["dsr"] < small_dsr["dsr"]

    def test_no_log_means_the_dsr_is_not_computable(self, tmp_path):
        import numpy as np
        import pandas as pd

        from portfolio_agent.src.risk_analytics import RiskAnalyzer

        index = pd.bdate_range("2021-01-01", periods=300)
        equity = pd.Series(np.linspace(1e6, 1.2e6, 300), index=index)
        result = RiskAnalyzer(equity, [], trial_log_path=str(tmp_path / "none.jsonl")) \
            .calculate_deflated_sharpe_ratio()

        assert result["computable"] is False
        assert result["n_trials"] == 0
