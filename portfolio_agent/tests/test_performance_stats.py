"""Tests for the selection-bias-aware performance statistics.

These are the ruler, so they are checked against closed forms and known
limiting behaviour rather than against golden numbers from the code itself.
"""

import math

import numpy as np
import pytest
from scipy import stats

from portfolio_agent.src.performance_stats import (
    EULER_MASCHERONI,
    TRADING_DAYS_PER_YEAR,
    annual_rate_to_period,
    deflated_sharpe_ratio,
    expected_maximum_sharpe,
    information_coefficient,
    information_ratio_of_ic,
    newey_west_variance,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
    sharpe_ratio_overlapping,
)


class TestSharpeRatio:
    def test_matches_the_closed_form(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(0.001, 0.01, 500)
        expected = (returns.mean() - 0.0002) / returns.std(ddof=1) * math.sqrt(252)

        assert sharpe_ratio(returns, 0.0002, 252) == pytest.approx(expected)

    def test_is_not_biased_low_by_half_sigma(self):
        """The defect this replaced: (CAGR - rf)/sigma is approximately
        (mu - rf)/sigma - sigma/2, because CAGR ~= mu - sigma^2/2."""
        rng = np.random.default_rng(1)
        returns = rng.normal(0.0008, 0.02, 3000)  # ~32% annualized volatility

        arithmetic = sharpe_ratio(returns, 0.0, 252)
        annual_sigma = returns.std(ddof=1) * math.sqrt(252)
        cagr = np.prod(1 + returns) ** (252 / len(returns)) - 1
        geometric_style = cagr / annual_sigma

        # The old expression sits roughly annual_sigma/2 below the correct one.
        assert arithmetic - geometric_style == pytest.approx(annual_sigma / 2, rel=0.25)
        assert arithmetic > geometric_style

    def test_degenerate_inputs(self):
        assert sharpe_ratio([], 0.0, 252) == 0.0
        assert sharpe_ratio([0.01], 0.0, 252) == 0.0
        assert sharpe_ratio([0.01] * 50, 0.0, 252) == 0.0  # no dispersion

    def test_annual_rate_conversion_compounds(self):
        daily = annual_rate_to_period(0.065, 252)
        assert (1 + daily) ** 252 == pytest.approx(1.065)


class TestNeweyWest:
    def test_matches_sample_variance_at_zero_lags(self):
        rng = np.random.default_rng(2)
        values = rng.normal(0, 1, 500)
        assert newey_west_variance(values, lags=0) == pytest.approx(
            float(np.var(values, ddof=0))
        )

    def test_inflates_the_variance_of_an_overlapping_series(self):
        """Daily-sampled 5-day sums share 4 days with each neighbour, so the
        i.i.d. variance of the mean is far too small."""
        rng = np.random.default_rng(3)
        daily = rng.normal(0, 0.01, 4000)
        overlapping = np.convolve(daily, np.ones(5), mode="valid")  # rolling 5-day sums

        iid = float(np.var(overlapping, ddof=0))
        corrected = newey_west_variance(overlapping, lags=4)

        # The long-run variance should be several times the i.i.d. one; the
        # theoretical ratio for a perfect 5-day overlap is near 5.
        assert corrected > 2.5 * iid

    def test_overlap_adjusted_sharpe_is_lower_on_overlapping_data(self):
        rng = np.random.default_rng(4)
        daily = rng.normal(0.0004, 0.01, 3000)
        overlapping = np.convolve(daily, np.ones(5), mode="valid")

        naive = sharpe_ratio(overlapping, 0.0, TRADING_DAYS_PER_YEAR // 5)
        adjusted = sharpe_ratio_overlapping(overlapping, horizon_days=5)

        assert 0 < adjusted < naive


class TestProbabilisticSharpeRatio:
    def test_reduces_to_the_lo_2002_normal_case(self):
        """With zero skew and normal kurtosis the estimator variance collapses
        to Lo's 1 + SR^2/2, so PSR is Phi(SR*sqrt(n-1) / sqrt(1 + SR^2/2))."""
        sr, n = 0.05, 1000
        psr = probabilistic_sharpe_ratio(sr, n, skewness=0.0, kurtosis=3.0)
        expected = float(
            stats.norm.cdf(sr * math.sqrt(n - 1) / math.sqrt(1 + sr ** 2 / 2))
        )
        assert psr == pytest.approx(expected)

    def test_a_zero_sharpe_is_a_coin_flip(self):
        assert probabilistic_sharpe_ratio(0.0, 500, 0.0, 3.0) == pytest.approx(0.5)

    def test_negative_skew_and_fat_tails_reduce_confidence(self):
        base = probabilistic_sharpe_ratio(0.06, 1000, skewness=0.0, kurtosis=3.0)
        skewed = probabilistic_sharpe_ratio(0.06, 1000, skewness=-1.5, kurtosis=3.0)
        fat = probabilistic_sharpe_ratio(0.06, 1000, skewness=0.0, kurtosis=9.0)

        assert skewed < base
        assert fat < base

    def test_rises_with_sample_length(self):
        short = probabilistic_sharpe_ratio(0.05, 100, 0.0, 3.0)
        long = probabilistic_sharpe_ratio(0.05, 2000, 0.0, 3.0)
        assert short < long

    def test_too_short_a_sample_returns_zero(self):
        assert probabilistic_sharpe_ratio(2.0, 1, 0.0, 3.0) == 0.0


class TestDeflatedSharpeRatio:
    def test_expected_maximum_grows_with_the_number_of_trials(self):
        variance = 0.0004
        assert expected_maximum_sharpe(2, variance) < expected_maximum_sharpe(
            100, variance
        ) < expected_maximum_sharpe(10_000, variance)

    def test_expected_maximum_matches_the_closed_form(self):
        n, variance = 50, 0.0009
        expected = math.sqrt(variance) * (
            (1 - EULER_MASCHERONI) * stats.norm.ppf(1 - 1 / n)
            + EULER_MASCHERONI * stats.norm.ppf(1 - 1 / (n * math.e))
        )
        assert expected_maximum_sharpe(n, variance) == pytest.approx(expected)

    def test_no_trials_or_no_dispersion_means_no_deflation(self):
        assert expected_maximum_sharpe(1, 0.01) == 0.0
        assert expected_maximum_sharpe(100, 0.0) == 0.0

    def test_deflation_strictly_reduces_the_probability(self):
        """This is the point of the statistic: a Sharpe found by searching a
        large space is less impressive than the same Sharpe found once."""
        common = dict(observed_sharpe=0.08, n_observations=1250, skewness=0.0, kurtosis=3.0)
        undeflated = probabilistic_sharpe_ratio(**common)
        few = deflated_sharpe_ratio(**common, n_trials=5, sharpe_variance=0.0004)
        many = deflated_sharpe_ratio(**common, n_trials=5000, sharpe_variance=0.0004)

        assert many < few < undeflated


class TestProbabilityOfBacktestOverfitting:
    def test_genuine_skill_is_not_flagged(self):
        """One column has a real, persistent edge; the rest are noise. The
        selection procedure should pick it in and out of sample alike."""
        rng = np.random.default_rng(10)
        n_obs, n_noise = 800, 9
        noise = rng.normal(0.0, 0.01, (n_obs, n_noise))
        skilled = rng.normal(0.0015, 0.01, (n_obs, 1))
        matrix = np.hstack([skilled, noise])

        result = probability_of_backtest_overfitting(matrix, n_blocks=10)

        assert result is not None
        assert result.n_strategies == 10
        assert result.pbo < 0.2
        assert not result.is_overfit
        assert result.median_logit > 0  # winner sits above the median OOS

    def test_pure_noise_selection_is_uninformative(self):
        """With no skill anywhere, the in-sample winner is a coin flip out of
        sample and PBO sits near 0.5.

        Averaged over several draws: a single realization of ten noise columns
        has one that genuinely leads over the whole window and therefore wins
        both halves, so per-seed PBO ranges roughly 0.2-0.9. That dispersion is
        a property of the estimator at ten strategies, not a defect.
        """
        pbos = []
        for seed in range(8):
            rng = np.random.default_rng(100 + seed)
            result = probability_of_backtest_overfitting(
                rng.normal(0.0, 0.01, (800, 10)), n_blocks=10
            )
            assert result is not None
            pbos.append(result.pbo)

        assert 0.35 < float(np.mean(pbos)) < 0.65

    def test_returns_none_when_the_input_cannot_support_the_procedure(self):
        rng = np.random.default_rng(12)
        assert probability_of_backtest_overfitting(rng.normal(size=(500, 1))) is None
        assert probability_of_backtest_overfitting(rng.normal(size=(10, 5)), n_blocks=16) is None

    def test_odd_block_count_is_rejected(self):
        rng = np.random.default_rng(13)
        with pytest.raises(ValueError, match="even"):
            probability_of_backtest_overfitting(rng.normal(size=(500, 4)), n_blocks=7)

    def test_split_count_is_the_binomial_coefficient(self):
        rng = np.random.default_rng(14)
        result = probability_of_backtest_overfitting(rng.normal(size=(400, 3)), n_blocks=8)
        assert result is not None
        assert result.n_splits == math.comb(8, 4)


class TestInformationCoefficient:
    def test_perfect_ranking_scores_one(self):
        predictions = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert information_coefficient(predictions, [10, 20, 30, 40, 50]) == pytest.approx(1.0)
        assert information_coefficient(predictions, [50, 40, 30, 20, 10]) == pytest.approx(-1.0)

    def test_is_invariant_to_monotone_transforms(self):
        """The reason for the rank form: a single circuit-locked +20% print
        must not dominate the number, and it does under Pearson."""
        rng = np.random.default_rng(20)
        predictions = rng.normal(size=200)
        realized = 0.3 * predictions + rng.normal(size=200) * 0.9

        base = information_coefficient(predictions, realized)
        with_outlier = np.array(realized)
        with_outlier[0] = 50.0  # a limit-up print

        assert information_coefficient(predictions, with_outlier) == pytest.approx(
            base, abs=0.02
        )

    def test_degenerate_inputs_score_zero(self):
        assert information_coefficient([], []) == 0.0
        assert information_coefficient([1.0, 2.0], [1.0, 2.0]) == 0.0  # too few
        assert information_coefficient([1.0] * 10, list(range(10))) == 0.0  # no dispersion
        assert information_coefficient([1.0, 2.0, 3.0], [1.0, 2.0]) == 0.0  # mismatched

    def test_icir_scales_the_mean_by_its_own_dispersion(self):
        ics = [0.03, 0.05, -0.01, 0.04, 0.02, 0.06, -0.02, 0.03]
        expected = np.mean(ics) / np.std(ics, ddof=1) * math.sqrt(252 / 5)
        assert information_ratio_of_ic(ics, horizon_days=5) == pytest.approx(expected)

    def test_icir_of_a_constant_series_is_zero(self):
        assert information_ratio_of_ic([0.03] * 10) == 0.0
        assert information_ratio_of_ic([0.03]) == 0.0
