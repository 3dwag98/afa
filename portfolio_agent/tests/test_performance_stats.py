"""Tests for the selection-bias-aware performance statistics."""

import math

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.src.performance_stats import (
    TRADING_DAYS_PER_YEAR,
    Trial,
    deflated_sharpe_ratio,
    evaluate_sharpe,
    excess_returns,
    expected_maximum_sharpe,
    log_trial,
    newey_west_standard_error,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    rank_information_coefficient,
    read_trials,
    return_moments,
    sharpe_ratio,
    to_daily_risk_free,
    trial_sharpe_variance,
)


class TestSharpeRatio:
    """The Sharpe has to be arithmetic-over-arithmetic to mean what its
    threshold means."""

    def test_matches_the_textbook_definition(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(0.0008, 0.01, size=2000)

        expected = np.mean(returns) / np.std(returns, ddof=1) * math.sqrt(252)
        assert sharpe_ratio(returns) == pytest.approx(expected)

    def test_risk_free_rate_is_deducted_per_period_and_compounded(self):
        returns = np.full(500, 0.001)
        daily_rf = to_daily_risk_free(0.065)

        # A constant series has no dispersion, so check the excess directly.
        excess = excess_returns(returns, risk_free_rate=0.065)
        assert excess == pytest.approx(0.001 - daily_rf)
        # Compounded, not divided: (1+r)^(1/252)-1 sits below r/252.
        assert daily_rf < 0.065 / 252

    def test_accepts_a_time_varying_risk_free_series(self):
        rng = np.random.default_rng(3)
        returns = rng.normal(0.0008, 0.01, size=300)
        rf_series = np.full(300, to_daily_risk_free(0.065))

        assert sharpe_ratio(returns, rf_series) == pytest.approx(
            sharpe_ratio(returns, 0.065)
        )

    def test_rejects_a_misaligned_risk_free_series(self):
        with pytest.raises(ValueError):
            excess_returns(np.zeros(10), risk_free_rate=np.zeros(9))

    def test_geometric_hybrid_is_biased_low_by_half_the_volatility(self):
        """The defect the arithmetic definition fixes, measured.

        Dividing CAGR by an arithmetic sigma returns approximately
        (mu-rf)/sigma - sigma/2, so the two definitions diverge by half the
        annualized volatility — a penalty that grows with volatility and so
        charges volatile strategies twice.
        """
        rng = np.random.default_rng(1)
        n = 252 * 8
        returns = rng.normal(0.0006, 0.0126, size=n)  # ~20% annualized vol

        arithmetic = sharpe_ratio(returns, risk_free_rate=0.0)

        equity = pd.Series(100000 * np.cumprod(1 + returns))
        years = n / 252
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
        annual_vol = np.std(returns, ddof=1) * math.sqrt(252)
        geometric = cagr / annual_vol

        assert arithmetic - geometric == pytest.approx(annual_vol / 2, abs=0.03)

    def test_no_dispersion_returns_zero_rather_than_dividing_by_zero(self):
        assert sharpe_ratio(np.full(100, 0.001), risk_free_rate=0.0) == 0.0
        assert sharpe_ratio([0.01]) == 0.0


class TestNeweyWest:
    """Overlapping labels understate the standard error in the direction that
    manufactures significance."""

    def test_matches_the_plain_standard_error_at_zero_lags(self):
        rng = np.random.default_rng(2)
        values = rng.normal(0.0, 1.0, size=500)

        plain = np.std(values, ddof=0) / math.sqrt(len(values))
        assert newey_west_standard_error(values, lags=0) == pytest.approx(plain)

    def test_overlapping_observations_widen_the_standard_error(self):
        """A daily-sampled 5-day return shares 4 days with its neighbour. The
        naive standard error treats those as 5 independent observations."""
        rng = np.random.default_rng(4)
        daily = rng.normal(0.0, 0.01, size=3000)
        overlapping = pd.Series(daily).rolling(5).sum().dropna().to_numpy()

        naive = np.std(overlapping, ddof=0) / math.sqrt(len(overlapping))
        corrected = newey_west_standard_error(overlapping, lags=4)

        assert corrected > naive
        # Overlap inflates the variance of the mean by roughly the horizon.
        assert corrected / naive == pytest.approx(math.sqrt(5), rel=0.35)

    def test_degenerate_input_returns_zero(self):
        assert newey_west_standard_error([], lags=4) == 0.0
        assert newey_west_standard_error([1.0], lags=4) == 0.0


class TestProbabilisticSharpe:
    def test_is_one_half_when_the_observed_sharpe_equals_the_benchmark(self):
        psr = probabilistic_sharpe_ratio(
            observed_sharpe=1.2, n_observations=1000, benchmark_sharpe=1.2
        )
        assert psr == pytest.approx(0.5)

    def test_rises_with_sample_length(self):
        short = probabilistic_sharpe_ratio(1.0, n_observations=100)
        long = probabilistic_sharpe_ratio(1.0, n_observations=2000)
        assert 0.5 < short < long < 1.0

    def test_negative_skew_and_fat_tails_weaken_the_same_sharpe(self):
        """The whole point of the statistic: identical headline numbers are not
        identical evidence. Equity strategies are negatively skewed and
        fat-tailed, which inflates the standard error of a Sharpe estimate."""
        gaussian = probabilistic_sharpe_ratio(1.0, 1000, skewness=0.0, kurtosis=3.0)
        skewed = probabilistic_sharpe_ratio(1.0, 1000, skewness=-1.5, kurtosis=3.0)
        fat_tailed = probabilistic_sharpe_ratio(1.0, 1000, skewness=0.0, kurtosis=9.0)

        assert skewed < gaussian
        assert fat_tailed < gaussian

    def test_a_negative_sharpe_is_unlikely_but_not_impossible_to_be_real(self):
        """Worth stating precisely, because the intuition runs the other way.

        Five years of daily data showing an annualized Sharpe of -0.4 still
        leaves roughly a one-in-five chance the true Sharpe is positive. A
        backtest is a small sample even when it covers a long calendar.
        """
        psr = probabilistic_sharpe_ratio(-0.4, n_observations=1250)
        assert 0.1 < psr < 0.25
        # Twenty years of the same result would settle it.
        assert probabilistic_sharpe_ratio(-0.4, n_observations=5000) < 0.05

    def test_too_short_a_sample_returns_zero(self):
        assert probabilistic_sharpe_ratio(2.0, n_observations=1) == 0.0


class TestDeflatedSharpe:
    def test_expected_maximum_grows_with_the_number_of_trials(self):
        few = expected_maximum_sharpe(n_trials=10, sharpe_variance=0.25)
        many = expected_maximum_sharpe(n_trials=1000, sharpe_variance=0.25)
        assert 0 < few < many

    def test_a_single_trial_needs_no_deflation(self):
        assert expected_maximum_sharpe(n_trials=1, sharpe_variance=0.25) == 0.0

    def test_deflation_lowers_the_probability(self):
        psr = probabilistic_sharpe_ratio(1.0, n_observations=1250)
        dsr = deflated_sharpe_ratio(
            1.0, n_observations=1250, n_trials=200, sharpe_variance=0.25
        )
        assert dsr < psr

    def test_a_lucky_winner_from_a_wide_search_does_not_survive_deflation(self):
        """Search enough zero-edge configurations and the best one still prints
        a respectable Sharpe. DSR is what says so."""
        dsr = deflated_sharpe_ratio(
            observed_sharpe=0.9,
            n_observations=1250,
            n_trials=500,
            sharpe_variance=0.5,
        )
        assert dsr < 0.95

    def test_evaluate_sharpe_reports_the_whole_set(self):
        rng = np.random.default_rng(5)
        returns = rng.normal(0.0008, 0.012, size=1500)

        report = evaluate_sharpe(returns, risk_free_rate=0.065, n_trials=50)

        assert set(report) >= {
            "sharpe_ratio", "psr", "dsr", "deflation_threshold_sharpe",
            "skewness", "kurtosis", "n_observations", "n_trials",
        }
        assert report["dsr"] <= report["psr"]
        assert report["n_trials"] == 50
        assert report["deflation_threshold_sharpe"] > 0


class TestReturnMoments:
    def test_kurtosis_is_raw_not_excess(self):
        rng = np.random.default_rng(6)
        n, skew, kurt = return_moments(rng.normal(0, 1, size=200_000))

        assert n == 200_000
        assert skew == pytest.approx(0.0, abs=0.05)
        assert kurt == pytest.approx(3.0, abs=0.1)

    def test_detects_negative_skew(self):
        _, skew, _ = return_moments([-0.10] + [0.01] * 99)
        assert skew < -1.0

    def test_short_samples_fall_back_to_gaussian_moments(self):
        assert return_moments([0.01, 0.02]) == (2, 0.0, 3.0)


class TestProbabilityOfBacktestOverfitting:
    def test_pure_noise_trials_are_not_selectable(self):
        """With no real edge anywhere, picking the in-sample winner is a coin
        flip out of sample, so PBO sits near 0.5.

        Averaged over several draws rather than asserted on one: a single
        20-trial panel is itself a small sample, and individual seeds land
        anywhere from 0.40 to 0.76.
        """
        pbos = []
        for seed in range(6):
            rng = np.random.default_rng(seed)
            trial_returns = rng.normal(0.0, 0.01, size=(1000, 20))
            result = probability_of_backtest_overfitting(trial_returns, n_splits=10)
            assert result["n_combinations"] == 252  # C(10, 5)
            pbos.append(result["pbo"])

        assert float(np.mean(pbos)) == pytest.approx(0.5, abs=0.1)

    def test_a_genuinely_superior_strategy_is_selectable(self):
        """When one column really is better, the selection procedure finds it
        and keeps finding it out of sample — PBO collapses toward zero."""
        rng = np.random.default_rng(9)
        trial_returns = rng.normal(0.0, 0.01, size=(1000, 20))
        trial_returns[:, 3] += 0.004  # a large, persistent edge

        result = probability_of_backtest_overfitting(trial_returns, n_splits=10)

        assert result["pbo"] < 0.05

    def test_rejects_an_odd_number_of_splits(self):
        with pytest.raises(ValueError):
            probability_of_backtest_overfitting(np.zeros((100, 4)), n_splits=7)

    def test_a_single_trial_has_nothing_to_select_between(self):
        result = probability_of_backtest_overfitting(np.zeros((100, 1)))
        assert result["pbo"] == 0.0


class TestRankInformationCoefficient:
    def _panel(self, n_dates=50, n_names=30, seed=0):
        rng = np.random.default_rng(seed)
        dates = np.repeat(pd.date_range("2024-01-01", periods=n_dates), n_names)
        realized = rng.normal(0, 0.05, size=n_dates * n_names)
        return dates, realized

    def test_perfect_ranking_scores_one(self):
        dates, realized = self._panel()
        result = rank_information_coefficient(
            pd.Series(realized), pd.Series(realized), dates=dates
        )
        assert result["mean_ic"] == pytest.approx(1.0)
        assert result["hit_rate"] == 1.0

    def test_inverted_ranking_scores_minus_one(self):
        dates, realized = self._panel()
        result = rank_information_coefficient(
            pd.Series(-realized), pd.Series(realized), dates=dates
        )
        assert result["mean_ic"] == pytest.approx(-1.0)

    def test_an_uninformative_prediction_scores_about_zero(self):
        dates, realized = self._panel(n_dates=200, seed=1)
        noise = np.random.default_rng(2).normal(0, 1, size=realized.size)

        result = rank_information_coefficient(
            pd.Series(noise), pd.Series(realized), dates=dates
        )

        assert result["mean_ic"] == pytest.approx(0.0, abs=0.05)
        assert result["n_dates"] == 200

    def test_icir_annualizes_by_the_label_horizon(self):
        """The annualization moved to its own key.

        `icir` used to be the annualized figure here while the evaluation layer
        reported the raw ratio under the same name — a factor of 16 apart at a
        daily horizon, on the number that decides which model ships. The raw
        ratio took the name because that is what the literature quotes; the
        annualized one is still computed, under `icir_annualized`.
        """
        dates, realized = self._panel(n_dates=200, seed=3)
        signal = realized + np.random.default_rng(4).normal(0, 0.05, size=realized.size)

        daily = rank_information_coefficient(
            pd.Series(signal), pd.Series(realized), dates=dates, horizon_days=1
        )
        weekly = rank_information_coefficient(
            pd.Series(signal), pd.Series(realized), dates=dates, horizon_days=5
        )

        assert daily["icir_annualized"] > weekly["icir_annualized"] > 0
        assert daily["icir_annualized"] / weekly["icir_annualized"] == pytest.approx(
            math.sqrt(5), rel=1e-6
        )

    def test_the_raw_icir_does_not_move_with_the_horizon(self):
        """It is mean/sd of the same IC series either way — the horizon is not in it."""
        dates, realized = self._panel(n_dates=200, seed=3)
        signal = realized + np.random.default_rng(4).normal(0, 0.05, size=realized.size)

        daily = rank_information_coefficient(
            pd.Series(signal), pd.Series(realized), dates=dates, horizon_days=1
        )
        weekly = rank_information_coefficient(
            pd.Series(signal), pd.Series(realized), dates=dates, horizon_days=5
        )

        assert daily["icir"] == pytest.approx(weekly["icir"])

    def test_dates_with_no_cross_section_are_skipped(self):
        result = rank_information_coefficient(
            pd.Series([0.1, 0.2]), pd.Series([0.3, 0.4]), dates=["a", "b"]
        )
        assert result["n_dates"] == 0.0

    def test_empty_input_is_not_an_error(self):
        result = rank_information_coefficient(pd.Series(dtype=float), pd.Series(dtype=float))
        assert result["mean_ic"] == 0.0


class TestTrialLog:
    def test_round_trips_through_the_log(self, tmp_path):
        path = tmp_path / "trials.jsonl"
        log_trial(path, Trial(label="momentum", sharpe=0.7, parameters={"stop": 6.0}))
        log_trial(path, Trial(label="lowvol", sharpe=-0.2, parameters={"stop": 1.5}))

        trials = read_trials(path)

        assert [t["label"] for t in trials] == ["momentum", "lowvol"]
        assert trials[0]["parameters"] == {"stop": 6.0}
        assert all(t["timestamp"] for t in trials)

    def test_a_corrupt_line_does_not_destroy_the_history(self, tmp_path):
        path = tmp_path / "trials.jsonl"
        log_trial(path, Trial(label="a", sharpe=0.1))
        path.write_text(path.read_text() + "{not json\n")
        log_trial(path, Trial(label="b", sharpe=0.2))

        assert [t["label"] for t in read_trials(path)] == ["a", "b"]

    def test_missing_log_reads_as_empty(self, tmp_path):
        assert read_trials(tmp_path / "absent.jsonl") == []

    def test_supplies_the_trial_count_and_variance_for_deflation(self, tmp_path):
        path = tmp_path / "trials.jsonl"
        for i, s in enumerate([0.1, 0.4, -0.2, 0.9]):
            log_trial(path, Trial(label=f"t{i}", sharpe=s))

        n_trials, variance = trial_sharpe_variance(read_trials(path))

        assert n_trials == 4
        assert variance == pytest.approx(np.var([0.1, 0.4, -0.2, 0.9], ddof=1))

    def test_deflation_uses_the_logged_history_end_to_end(self, tmp_path):
        """The point of the log: N is a recorded fact, not a recollection."""
        path = tmp_path / "trials.jsonl"
        rng = np.random.default_rng(10)
        for i in range(40):
            log_trial(path, Trial(label=f"cfg{i}", sharpe=float(rng.normal(0.1, 0.5))))

        n_trials, variance = trial_sharpe_variance(read_trials(path))
        returns = rng.normal(0.0008, 0.012, size=1250)

        undeflated = evaluate_sharpe(returns, n_trials=1)
        deflated = evaluate_sharpe(returns, n_trials=n_trials, sharpe_variance=variance)

        assert deflated["dsr"] < undeflated["dsr"]
        assert deflated["deflation_threshold_sharpe"] > 0


class TestConfigHashAndDistinctTrials:
    """Task 2.1's remaining piece: identify a trial by its whole config.

    The trial log already records an explicit `parameters` dict, but that dict
    is hand-enumerated in agents/backtester.py. Any knob nobody thought to add
    to it — a new simulation option, a strategy YAML weight, the drift-prior
    switch — leaves two genuinely different configurations indistinguishable
    in the log. N is then wrong, and N is the whole input to the deflation.
    """

    def test_hash_is_stable_across_calls(self):
        from portfolio_agent.src.performance_stats import config_hash

        payload = {"b": 2, "a": {"nested": [1, 2, 3]}}
        assert config_hash(payload) == config_hash(payload)

    def test_hash_does_not_depend_on_key_order(self):
        """Two spellings of the same configuration are the same trial."""
        from portfolio_agent.src.performance_stats import config_hash

        assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})

    def test_hash_is_stable_across_processes(self):
        """Not Python's hash().

        str.__hash__ is salted per process (PYTHONHASHSEED), so a trial log
        keyed on it would count the same configuration as a fresh trial on
        every run — silently inflating N and over-deflating every Sharpe the
        platform reports.
        """
        import subprocess
        import sys

        from portfolio_agent.src.performance_stats import config_hash

        expected = config_hash({"strategy": "rule_based", "stop": 2.5})
        program = (
            "import sys; sys.path.insert(0, 'portfolio_agent');"
            "from portfolio_agent.src.performance_stats import config_hash;"
            "print(config_hash({'strategy': 'rule_based', 'stop': 2.5}))"
        )
        out = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True, text=True, check=True, cwd=".",
        )
        assert out.stdout.strip() == expected

    def test_any_field_changes_the_hash_not_just_enumerated_ones(self):
        """The defect the hash exists to close.

        `use_empirical_drift_prior` is a real knob that changes every
        probability-of-profit in the run, and it is not in backtester.py's
        hand-written parameter list. Under the old scheme those two runs
        recorded identical trials.
        """
        from portfolio_agent.src.performance_stats import config_hash

        base = {"simulation": {"method": "gaussian", "use_empirical_drift_prior": True}}
        changed = {"simulation": {"method": "gaussian", "use_empirical_drift_prior": False}}

        assert config_hash(base) != config_hash(changed)

    def test_trial_round_trips_its_config_hash(self, tmp_path):
        from portfolio_agent.src.performance_stats import Trial, log_trial, read_trials

        path = tmp_path / "trials.jsonl"
        log_trial(path, Trial(label="a", sharpe=0.5, config_hash="deadbeef"))

        assert read_trials(path)[0]["config_hash"] == "deadbeef"

    def test_distinct_trials_collapses_repeat_runs_of_one_config(self):
        """Re-running a configuration is not a new trial.

        Determinism is enforced, so the same config produces the same Sharpe;
        counting it twice inflates N and deflates the reported Sharpe against a
        search that never happened.
        """
        from portfolio_agent.src.performance_stats import distinct_trials

        trials = [
            {"label": "a", "sharpe": 0.5, "config_hash": "aaa"},
            {"label": "a", "sharpe": 0.5, "config_hash": "aaa"},
            {"label": "b", "sharpe": 0.9, "config_hash": "bbb"},
        ]
        distinct = distinct_trials(trials)

        assert [t["config_hash"] for t in distinct] == ["aaa", "bbb"]

    def test_distinct_trials_keeps_unhashed_history(self):
        """Trials written before the hash existed still count.

        Dropping them would silently shrink N on any log with history in it,
        which is the opposite of what the deflation is for. They fall back to
        their parameters dict, and entries with neither are kept as distinct.
        """
        from portfolio_agent.src.performance_stats import distinct_trials

        trials = [
            {"label": "old", "sharpe": 0.4, "parameters": {"stop": 2.0}},
            {"label": "old", "sharpe": 0.4, "parameters": {"stop": 2.0}},
            {"label": "old", "sharpe": 0.6, "parameters": {"stop": 3.0}},
            {"label": "ancient", "sharpe": 0.1},
        ]
        assert len(distinct_trials(trials)) == 3

    def test_deflation_uses_the_distinct_count(self, tmp_path):
        """End to end: a log with repeats must not out-deflate a log without.

        Five recordings of two configurations is a two-trial search, and has to
        deflate exactly as hard as two recordings of those same two.
        """
        from portfolio_agent.src.performance_stats import (
            Trial, distinct_trials, log_trial, read_trials, trial_sharpe_variance,
        )

        path = tmp_path / "trials.jsonl"
        for _ in range(3):
            log_trial(path, Trial(label="a", sharpe=0.5, config_hash="aaa"))
        for _ in range(2):
            log_trial(path, Trial(label="b", sharpe=1.1, config_hash="bbb"))

        raw_n, _ = trial_sharpe_variance(read_trials(path))
        distinct_n, distinct_var = trial_sharpe_variance(
            distinct_trials(read_trials(path))
        )

        assert raw_n == 5
        assert distinct_n == 2
        assert distinct_var == pytest.approx(np.var([0.5, 1.1], ddof=1))

    def test_backtester_fingerprints_the_whole_resolved_config(self):
        """The wiring, not just the hash function.

        Pins the three properties the deduplication depends on: the same
        configuration fingerprints the same way twice, the backtest window
        (which arrives by CLI, not config) participates, and a knob absent from
        backtester.py's hand-written parameter list still changes the result.
        """
        from portfolio_agent.agents.backtester import BacktesterAgent
        from portfolio_agent.config.loader import load_config

        config = load_config("config.yaml")
        agent = BacktesterAgent(config)

        stable = agent._config_fingerprint("2020-01-01", "2024-12-31")
        assert stable == agent._config_fingerprint("2020-01-01", "2024-12-31")
        assert stable != agent._config_fingerprint("2021-01-01", "2024-12-31")

        other = load_config("config.yaml")
        other.simulation.use_empirical_drift_prior = (
            not other.simulation.use_empirical_drift_prior
        )
        assert stable != BacktesterAgent(other)._config_fingerprint(
            "2020-01-01", "2024-12-31"
        )

    def test_report_path_alone_is_not_a_new_trial(self):
        """Paths are excluded deliberately.

        Writing the same backtest to a different filename is not a search step,
        and timestamped output paths would otherwise make every single run
        unique — which would defeat the deduplication entirely.
        """
        from portfolio_agent.agents.backtester import BacktesterAgent
        from portfolio_agent.config.loader import load_config

        config = load_config("config.yaml")
        baseline = BacktesterAgent(config)._config_fingerprint("2020-01-01", "2024-12-31")

        renamed = load_config("config.yaml")
        renamed.paths.trial_log = "output/somewhere-else.jsonl"
        assert BacktesterAgent(renamed)._config_fingerprint(
            "2020-01-01", "2024-12-31"
        ) == baseline
