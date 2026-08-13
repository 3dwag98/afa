"""Tests for portfolio-level covariance estimation and allocation."""

import math

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.src.portfolio import (
    AllocationResult,
    correlation_risk_multiple,
    diversification_ratio,
    exponentially_weighted_covariance,
    hierarchical_risk_parity,
    independent_portfolio_volatility,
    ledoit_wolf_covariance,
    optimize_long_only,
    portfolio_volatility,
    project_onto_capped_simplex,
    risk_contributions,
    sample_covariance,
    single_factor_covariance,
    summarize_book_risk,
)


def _equicorrelated(n: int, correlation: float, volatility: float) -> np.ndarray:
    """Covariance for n assets sharing one pairwise correlation."""
    corr = np.full((n, n), correlation, dtype=float)
    np.fill_diagonal(corr, 1.0)
    return corr * volatility**2


def _correlated_returns(
    n_periods: int, n_assets: int, correlation: float, volatility: float, seed: int
) -> np.ndarray:
    """Draw returns with a known equicorrelated structure."""
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 1, size=(n_periods, 1))
    idiosyncratic = rng.normal(0, 1, size=(n_periods, n_assets))
    loading = math.sqrt(max(0.0, correlation))
    shocks = loading * common + math.sqrt(max(0.0, 1 - correlation)) * idiosyncratic
    return shocks * volatility


def _two_sector_returns(n_periods: int, n_assets: int, seed: int) -> np.ndarray:
    """Two uncorrelated sectors with heterogeneous volatilities.

    Deliberately *not* equicorrelated: a constant-correlation target is exactly
    right for an equicorrelated panel, so shrinkage saturates at 1 there and
    every comparison between sample sizes reads the same. Cross-sector pairs
    are genuinely uncorrelated here, which is the realistic case where the
    target is a useful approximation rather than the truth.
    """
    rng = np.random.default_rng(seed)
    half = n_assets // 2
    loadings = np.zeros((n_assets, 2))
    loadings[:half, 0] = 0.8
    loadings[half:, 1] = 0.8

    residual_std = np.sqrt(1.0 - np.sum(loadings**2, axis=1))
    factors = rng.normal(0, 1, size=(n_periods, 2))
    idiosyncratic = rng.normal(0, 1, size=(n_periods, n_assets))
    volatilities = np.linspace(0.01, 0.04, n_assets)

    return (factors @ loadings.T + residual_std * idiosyncratic) * volatilities


class TestPortfolioVolatility:
    """The measurement the platform never made."""

    def test_reproduces_the_correlation_cost_of_independent_sizing(self):
        """The defect this module exists to fix, at the review's own numbers.

        Twenty 3% positions at 30% single-name annualized volatility. Sizing
        them independently claims 4.0% portfolio volatility; at the pairwise
        correlations Indian equities actually show, the truth is 2.8x to 4.1x
        that — and the gap is widest in the crash states the drawdown breaker
        exists to survive.
        """
        weights = np.full(20, 0.03)
        expected = {0.00: 4.02, 0.35: 11.13, 0.60: 14.17, 0.85: 16.67}

        for correlation, expected_vol_pct in expected.items():
            cov = _equicorrelated(20, correlation, 0.30)
            assert portfolio_volatility(weights, cov) * 100 == pytest.approx(
                expected_vol_pct, abs=0.02
            )

        independent = independent_portfolio_volatility(weights, _equicorrelated(20, 0.35, 0.30))
        assert independent * 100 == pytest.approx(4.02, abs=0.02)
        assert correlation_risk_multiple(weights, _equicorrelated(20, 0.35, 0.30)) == pytest.approx(
            2.77, abs=0.02
        )
        assert correlation_risk_multiple(weights, _equicorrelated(20, 0.85, 0.30)) == pytest.approx(
            4.14, abs=0.02
        )

    def test_independent_and_true_volatility_agree_when_uncorrelated(self):
        weights = np.full(10, 0.05)
        cov = _equicorrelated(10, 0.0, 0.25)

        assert portfolio_volatility(weights, cov) == pytest.approx(
            independent_portfolio_volatility(weights, cov)
        )
        assert correlation_risk_multiple(weights, cov) == pytest.approx(1.0)

    def test_rejects_a_mismatched_covariance(self):
        with pytest.raises(ValueError):
            portfolio_volatility(np.ones(3), np.eye(4))

    def test_empty_book_has_no_volatility(self):
        assert portfolio_volatility([], np.zeros((0, 0))) == 0.0


class TestRiskContributions:
    def test_contributions_sum_to_one(self):
        weights = np.array([0.05, 0.03, 0.02])
        cov = _equicorrelated(3, 0.4, 0.3)

        assert risk_contributions(weights, cov).sum() == pytest.approx(1.0)

    def test_equal_weights_on_a_symmetric_book_contribute_equally(self):
        weights = np.full(5, 0.04)
        contributions = risk_contributions(weights, _equicorrelated(5, 0.5, 0.2))

        assert contributions == pytest.approx(np.full(5, 0.2))

    def test_a_correlated_name_carries_more_risk_than_its_weight(self):
        """Why a weight cap is not a risk cap: three names that move together
        are one bet, and each carries more risk than its 5% suggests."""
        cov = np.array(
            [
                [0.04, 0.038, 0.038, 0.000],
                [0.038, 0.04, 0.038, 0.000],
                [0.038, 0.038, 0.04, 0.000],
                [0.000, 0.000, 0.000, 0.04],
            ]
        )
        weights = np.full(4, 0.05)
        contributions = risk_contributions(weights, cov)

        assert contributions[0] > 0.25 > contributions[3]

    def test_diversification_ratio_is_one_for_a_single_bet(self):
        cov = _equicorrelated(6, 1.0, 0.3)
        assert diversification_ratio(np.full(6, 0.05), cov) == pytest.approx(1.0)

    def test_diversification_ratio_rises_as_correlation_falls(self):
        weights = np.full(6, 0.05)
        high = diversification_ratio(weights, _equicorrelated(6, 0.8, 0.3))
        low = diversification_ratio(weights, _equicorrelated(6, 0.1, 0.3))
        assert low > high > 1.0


class TestCovarianceEstimators:
    def test_ledoit_wolf_recovers_a_known_correlation_structure(self):
        returns = _correlated_returns(2000, 12, correlation=0.4, volatility=0.02, seed=1)
        cov, intensity = ledoit_wolf_covariance(returns)

        std = np.sqrt(np.diag(cov))
        correlation = cov / np.outer(std, std)
        off_diagonal = correlation[~np.eye(12, dtype=bool)]

        assert float(np.mean(off_diagonal)) == pytest.approx(0.4, abs=0.05)
        assert float(np.mean(std)) == pytest.approx(0.02, abs=0.002)
        assert 0.0 <= intensity <= 1.0

    def test_shrinkage_is_stronger_when_there_is_less_history(self):
        """The estimator has to know when it is guessing: with T close to N the
        sample matrix is mostly noise and the target should carry more weight."""
        short = _two_sector_returns(40, 25, seed=2)
        medium = _two_sector_returns(250, 25, seed=2)
        long = _two_sector_returns(2000, 25, seed=2)

        intensities = [
            ledoit_wolf_covariance(r)[1] for r in (short, medium, long)
        ]

        assert intensities == sorted(intensities, reverse=True)
        assert intensities[0] > 0.15
        assert intensities[-1] < 0.02

    def test_shrinkage_beats_the_sample_covariance_when_the_target_fits(self):
        """What the shrinkage is for, measured against the true matrix.

        On an equicorrelated panel — one market factor, which is close to what
        a long-only Indian book actually faces — the constant-correlation
        target is well specified, and shrinking toward it cuts estimation error
        substantially at the sample sizes this platform has.
        """
        n_assets, n_periods = 60, 40
        true_cov = _equicorrelated(n_assets, 0.49, 0.02)

        sample_error, shrunk_error = [], []
        for seed in range(20):
            returns = _correlated_returns(n_periods, n_assets, 0.49, 0.02, seed=seed)
            shrunk, _ = ledoit_wolf_covariance(returns)
            sample_error.append(
                np.linalg.norm(np.asarray(sample_covariance(returns)) - true_cov)
            )
            shrunk_error.append(np.linalg.norm(np.asarray(shrunk) - true_cov))

        assert np.mean(shrunk_error) < 0.8 * np.mean(sample_error)

    def test_shrinkage_makes_a_singular_sample_matrix_invertible(self):
        """T < N leaves the sample covariance rank-deficient, so any optimizer
        that inverts it is inverting noise. The shrunk matrix is full rank."""
        returns = _correlated_returns(30, 50, 0.3, 0.02, seed=3)

        sample = np.asarray(sample_covariance(returns))
        shrunk, intensity = ledoit_wolf_covariance(returns)

        assert np.linalg.matrix_rank(sample) < 50
        assert intensity > 0
        assert np.min(np.linalg.eigvalsh(np.asarray(shrunk))) > 0

    def test_explicit_shrinkage_endpoints_are_honoured(self):
        returns = _correlated_returns(500, 8, 0.3, 0.02, seed=4)
        none, _ = ledoit_wolf_covariance(returns, shrinkage=0.0)
        full, _ = ledoit_wolf_covariance(returns, shrinkage=1.0)

        # Zero shrinkage is the MLE sample covariance (1/T rather than 1/(T-1)).
        matrix = np.asarray(returns)
        demeaned = matrix - matrix.mean(axis=0)
        assert np.asarray(none) == pytest.approx(demeaned.T @ demeaned / len(matrix))
        # Full shrinkage leaves the variances alone and equalizes correlations.
        full_matrix = np.asarray(full)
        std = np.sqrt(np.diag(full_matrix))
        correlation = full_matrix / np.outer(std, std)
        off_diagonal = correlation[~np.eye(8, dtype=bool)]
        assert float(np.std(off_diagonal)) == pytest.approx(0.0, abs=1e-12)

    def test_preserves_dataframe_labels(self):
        frame = pd.DataFrame(
            _correlated_returns(300, 3, 0.3, 0.02, seed=5), columns=["A", "B", "C"]
        )
        cov, _ = ledoit_wolf_covariance(frame)

        assert isinstance(cov, pd.DataFrame)
        assert list(cov.columns) == ["A", "B", "C"]
        assert list(cov.index) == ["A", "B", "C"]

    def test_exponential_weighting_tracks_a_correlation_regime_shift(self):
        """An equally-weighted window averages the calm and the crisis; a
        half-life responds to the regime the book is actually in."""
        calm = _correlated_returns(500, 6, correlation=0.1, volatility=0.01, seed=6)
        crisis = _correlated_returns(60, 6, correlation=0.9, volatility=0.03, seed=7)
        returns = np.vstack([calm, crisis])

        equal = np.asarray(sample_covariance(returns))
        weighted = np.asarray(exponentially_weighted_covariance(returns, half_life_days=20))

        def mean_correlation(cov):
            std = np.sqrt(np.diag(cov))
            corr = cov / np.outer(std, std)
            return float(np.mean(corr[~np.eye(6, dtype=bool)]))

        assert mean_correlation(weighted) > mean_correlation(equal)
        assert mean_correlation(weighted) > 0.6

    def test_single_factor_covariance_is_positive_definite_on_a_wide_universe(self):
        returns = _correlated_returns(60, 120, 0.35, 0.02, seed=8)
        cov = np.asarray(single_factor_covariance(returns))

        assert cov.shape == (120, 120)
        assert np.min(np.linalg.eigvalsh(cov)) > 0

    def test_single_factor_covariance_rejects_a_mismatched_factor(self):
        with pytest.raises(ValueError):
            single_factor_covariance(np.zeros((100, 5)), factor_returns=np.zeros(99))

    def test_annualization_scales_by_the_trading_year(self):
        returns = _correlated_returns(500, 4, 0.3, 0.02, seed=9)
        daily, _ = ledoit_wolf_covariance(returns)
        annual, _ = ledoit_wolf_covariance(returns, annualize=True)

        assert np.asarray(annual) == pytest.approx(np.asarray(daily) * 252)


class TestCappedSimplexProjection:
    def test_respects_the_box_and_the_budget(self):
        projected = project_onto_capped_simplex([0.5, 0.4, 0.3, 0.2], upper_bounds=0.1, budget=1.0)

        assert np.all(projected >= -1e-12)
        assert np.all(projected <= 0.1 + 1e-12)
        assert projected.sum() <= 1.0 + 1e-9

    def test_binds_the_budget_when_the_point_is_too_large(self):
        projected = project_onto_capped_simplex([0.8, 0.8, 0.8], upper_bounds=1.0, budget=1.0)

        assert projected.sum() == pytest.approx(1.0)
        assert projected == pytest.approx(np.full(3, 1 / 3))

    def test_leaves_a_feasible_point_untouched(self):
        point = np.array([0.02, 0.03, 0.01])
        assert project_onto_capped_simplex(point, 0.05, 1.0) == pytest.approx(point)

    def test_clips_negatives_to_zero(self):
        projected = project_onto_capped_simplex([-0.5, 0.2], upper_bounds=1.0, budget=1.0)
        assert projected[0] == 0.0


class TestOptimizeLongOnly:
    def test_prefers_the_higher_expected_return_at_equal_risk(self):
        cov = _equicorrelated(3, 0.2, 0.2)
        result = optimize_long_only(
            expected_returns=[0.10, 0.05, 0.01],
            covariance=cov,
            risk_aversion=5.0,
            max_weight=0.4,
        )

        assert result.weights[0] > result.weights[1] > result.weights[2]

    def test_every_constraint_holds_by_construction(self):
        rng = np.random.default_rng(11)
        n = 30
        cov, _ = ledoit_wolf_covariance(_correlated_returns(500, n, 0.35, 0.02, seed=12))
        result = optimize_long_only(
            expected_returns=rng.normal(0.001, 0.002, size=n),
            covariance=cov,
            max_weight=0.05,
            budget=1.0,
        )

        assert np.all(result.weights >= -1e-9)
        assert np.all(result.weights <= 0.05 + 1e-9)
        assert result.weights.sum() <= 1.0 + 1e-6

    def test_avoids_the_correlated_pair_in_favour_of_the_independent_name(self):
        """The decision independent sizing cannot make. Three names with
        identical expected returns, two of them nearly the same bet."""
        cov = np.array(
            [
                [0.04, 0.039, 0.0],
                [0.039, 0.04, 0.0],
                [0.0, 0.0, 0.04],
            ]
        )
        result = optimize_long_only(
            expected_returns=[0.02, 0.02, 0.02],
            covariance=cov,
            risk_aversion=8.0,
            max_weight=1.0,
        )

        assert result.weights[2] > result.weights[0]
        assert result.weights[2] > result.weights[1]

    def test_higher_risk_aversion_shrinks_the_book(self):
        cov = _equicorrelated(5, 0.3, 0.2)
        mu = np.full(5, 0.02)

        timid = optimize_long_only(mu, cov, risk_aversion=50.0, max_weight=1.0)
        bold = optimize_long_only(mu, cov, risk_aversion=2.0, max_weight=1.0)

        assert timid.weights.sum() < bold.weights.sum()
        assert timid.volatility < bold.volatility

    def test_turnover_penalty_holds_the_existing_book(self):
        """Without this the optimizer re-solves to a different portfolio daily
        and pays the whole Indian friction stack to chase estimation noise."""
        rng = np.random.default_rng(13)
        n = 15
        cov, _ = ledoit_wolf_covariance(_correlated_returns(400, n, 0.3, 0.02, seed=14))
        mu = rng.normal(0.001, 0.002, size=n)
        previous = np.full(n, 1.0 / n * 0.9)

        free = optimize_long_only(mu, cov, max_weight=0.2, previous_weights=previous)
        penalized = optimize_long_only(
            mu, cov, max_weight=0.2, previous_weights=previous, turnover_cost=0.05
        )

        assert penalized.turnover < free.turnover

    def test_volatility_ceiling_is_enforced(self):
        cov = _equicorrelated(10, 0.6, 0.30)
        mu = np.full(10, 0.05)

        result = optimize_long_only(
            mu, cov, risk_aversion=1.0, max_weight=0.5, max_volatility=0.10
        )

        assert result.volatility <= 0.10 + 1e-9
        assert np.all(result.weights >= 0)

    def test_a_book_with_no_edge_stays_in_cash(self):
        cov = _equicorrelated(6, 0.4, 0.2)
        result = optimize_long_only(np.zeros(6), cov, risk_aversion=5.0, max_weight=0.2)

        assert result.weights.sum() == pytest.approx(0.0, abs=1e-6)

    def test_carries_labels_through_from_a_dataframe_covariance(self):
        frame = pd.DataFrame(
            _correlated_returns(300, 3, 0.3, 0.02, seed=15), columns=["A", "B", "C"]
        )
        cov, _ = ledoit_wolf_covariance(frame)
        result = optimize_long_only([0.01, 0.02, 0.03], cov, max_weight=0.5)

        assert isinstance(result, AllocationResult)
        assert list(result.as_series().index) == ["A", "B", "C"]

    def test_rejects_shape_mismatches(self):
        with pytest.raises(ValueError):
            optimize_long_only([0.01, 0.02], np.eye(3))
        with pytest.raises(ValueError):
            optimize_long_only([0.01, 0.02], np.eye(2), previous_weights=[0.1])


class TestHierarchicalRiskParity:
    def test_weights_are_positive_and_sum_to_one(self):
        cov, _ = ledoit_wolf_covariance(_correlated_returns(500, 12, 0.3, 0.02, seed=16))
        weights = np.asarray(hierarchical_risk_parity(cov))

        assert weights.sum() == pytest.approx(1.0)
        assert np.all(weights > 0)

    def test_allocates_less_to_the_noisier_asset(self):
        cov = np.diag([0.01, 0.04])
        weights = np.asarray(hierarchical_risk_parity(cov))

        assert weights[0] > weights[1]
        assert weights[0] / weights[1] == pytest.approx(4.0, rel=1e-6)

    def test_splits_capital_across_clusters_not_across_tickers(self):
        """Three names that are one bet, plus one that is not: HRP should give
        the independent name far more than a naive 1/N would."""
        cov = np.array(
            [
                [0.04, 0.0396, 0.0396, 0.0],
                [0.0396, 0.04, 0.0396, 0.0],
                [0.0396, 0.0396, 0.04, 0.0],
                [0.0, 0.0, 0.0, 0.04],
            ]
        )
        weights = np.asarray(hierarchical_risk_parity(cov))

        assert weights[3] > 0.25
        assert weights[3] > weights[0]

    def test_respects_a_maximum_weight(self):
        cov = np.diag([0.0001, 0.04, 0.04, 0.04])
        weights = np.asarray(hierarchical_risk_parity(cov, max_weight=0.4))

        assert np.max(weights) <= 0.4 + 1e-9
        assert weights.sum() == pytest.approx(1.0)

    def test_handles_degenerate_sizes(self):
        assert np.asarray(hierarchical_risk_parity(np.zeros((0, 0)))).size == 0
        assert np.asarray(hierarchical_risk_parity(np.array([[0.04]]))) == pytest.approx([1.0])

    def test_preserves_labels(self):
        cov = pd.DataFrame(np.diag([0.01, 0.04]), index=["A", "B"], columns=["A", "B"])
        weights = hierarchical_risk_parity(cov)

        assert isinstance(weights, pd.Series)
        assert list(weights.index) == ["A", "B"]


class TestSummarizeBookRisk:
    def test_reports_the_gap_between_true_and_independent_risk(self):
        weights = np.full(20, 0.03)
        summary = summarize_book_risk(weights, _equicorrelated(20, 0.6, 0.30))

        assert summary["portfolio_volatility"] > summary["independent_volatility"]
        assert summary["correlation_risk_multiple"] == pytest.approx(3.52, abs=0.02)
        assert summary["n_positions"] == 20
        assert summary["max_weight"] == pytest.approx(0.03)

    def test_largest_risk_contribution_exceeds_the_largest_weight(self):
        """The reason a weight cap is not a risk cap."""
        cov = np.array(
            [
                [0.09, 0.085, 0.085],
                [0.085, 0.09, 0.085],
                [0.085, 0.085, 0.01],
            ]
        )
        summary = summarize_book_risk([0.03, 0.03, 0.03], cov)

        assert summary["max_risk_contribution"] > summary["max_weight"]


class TestShrunkEwmaCovariance:
    """Task 3.1: EWMA weighting and Ledoit-Wolf shrinkage, composed.

    Both halves already existed but nothing ran them together, and the two
    properties that make the result usable by an optimizer — positive
    semi-definiteness and a bounded condition number — were not asserted
    anywhere. They matter for a concrete reason: a mean-variance optimizer
    inverts this matrix, and a wide universe estimated over a short window
    gives a *singular* sample covariance, whose inverse is where crash-state
    volatility blowups come from.
    """

    @staticmethod
    def _wide_panel(n_assets=60, n_periods=40, seed=11):
        """More assets than observations — the realistic case, and the one the
        raw sample covariance cannot survive."""
        rng = np.random.default_rng(seed)
        market = rng.normal(0.0, 0.012, size=(n_periods, 1))
        betas = rng.uniform(0.6, 1.4, size=(1, n_assets))
        idiosyncratic = rng.normal(0.0, 0.008, size=(n_periods, n_assets))
        return pd.DataFrame(
            market @ betas + idiosyncratic,
            columns=[f"T{i:03d}" for i in range(n_assets)],
        )

    def test_result_is_positive_semi_definite(self):
        from portfolio_agent.src.portfolio import shrunk_ewma_covariance

        returns = self._wide_panel()
        cov, _ = shrunk_ewma_covariance(returns)
        eigenvalues = np.linalg.eigvalsh(np.asarray(cov))

        # Symmetric to machine precision, and no negative eigenvalue beyond it.
        assert np.allclose(np.asarray(cov), np.asarray(cov).T, atol=1e-15)
        assert eigenvalues.min() >= -1e-12

    def test_result_is_strictly_positive_definite_where_the_sample_is_singular(self):
        """N > T makes the sample covariance rank-deficient, so it has exact
        zero eigenvalues and no inverse. Shrinkage is what buys invertibility."""
        from portfolio_agent.src.portfolio import shrunk_ewma_covariance

        returns = self._wide_panel(n_assets=60, n_periods=40)
        raw = np.asarray(sample_covariance(returns))
        shrunk, intensity = shrunk_ewma_covariance(returns)

        assert np.linalg.eigvalsh(raw).min() < 1e-10  # singular, as expected
        assert intensity > 0.0
        assert np.linalg.eigvalsh(np.asarray(shrunk)).min() > 1e-12

    def test_condition_number_is_far_lower_than_the_sample_matrix(self):
        """The acceptance criterion.

        Condition number is the amplification factor from input error to
        output error when the matrix is inverted. The sample matrix's is
        effectively infinite here; what matters is that the shrunk one is
        small enough that optimizer weights are a function of the data rather
        than of the noise in the smallest eigenvalue.
        """
        from portfolio_agent.src.portfolio import shrunk_ewma_covariance

        returns = self._wide_panel()
        raw_condition = np.linalg.cond(np.asarray(sample_covariance(returns)))
        shrunk_condition = np.linalg.cond(
            np.asarray(shrunk_ewma_covariance(returns)[0])
        )

        assert shrunk_condition < raw_condition / 1e6
        assert shrunk_condition < 1e4

    def test_uniform_weighting_reproduces_the_unweighted_estimator_exactly(self):
        """The EWMA path must be a strict generalization.

        With half_life=None the weights are uniform and every expression has
        to collapse to the existing, separately-tested Ledoit-Wolf result —
        otherwise this silently changes every number the platform already
        reports.
        """
        from portfolio_agent.src.portfolio import shrunk_ewma_covariance

        returns = self._wide_panel(n_assets=12, n_periods=250)
        composed, composed_intensity = shrunk_ewma_covariance(
            returns, half_life_days=None
        )
        baseline, baseline_intensity = ledoit_wolf_covariance(returns)

        assert composed_intensity == pytest.approx(baseline_intensity, rel=1e-12)
        np.testing.assert_allclose(
            np.asarray(composed), np.asarray(baseline), rtol=1e-12, atol=1e-18
        )

    def test_recent_observations_dominate_after_a_regime_shift(self):
        """Why the half-life is there at all.

        A book that was uncorrelated for a year and then moved as one block
        must be estimated as correlated *now*. An equally-weighted window
        averages the two regimes and is wrong about both.
        """
        from portfolio_agent.src.portfolio import shrunk_ewma_covariance

        # Three half-lives of shock, so it carries 1 - 2^-3 = 87.5% of the
        # weight. The threshold below follows from that rather than being
        # picked to pass: at one half-life the answer would be ~0.5, and
        # asserting 0.8 there would be asserting something untrue.
        rng = np.random.default_rng(5)
        calm = rng.normal(0.0, 0.01, size=(400, 2))
        shock = rng.normal(0.0, 0.01, size=(180, 1)) @ np.ones((1, 2))
        returns = pd.DataFrame(np.vstack([calm, shock]), columns=["A", "B"])

        def correlation(cov):
            cov = np.asarray(cov)
            return cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1])

        equal_weighted, _ = shrunk_ewma_covariance(returns, half_life_days=None)
        recent, _ = shrunk_ewma_covariance(returns, half_life_days=60.0)

        assert correlation(recent) > correlation(equal_weighted)
        assert correlation(recent) > 0.8

    def test_preserves_asset_labels(self):
        from portfolio_agent.src.portfolio import shrunk_ewma_covariance

        returns = self._wide_panel(n_assets=5, n_periods=80)
        cov, _ = shrunk_ewma_covariance(returns)

        assert isinstance(cov, pd.DataFrame)
        assert list(cov.columns) == list(returns.columns)
        assert list(cov.index) == list(returns.columns)

    def test_is_deterministic(self):
        from portfolio_agent.src.portfolio import shrunk_ewma_covariance

        returns = self._wide_panel()
        first, first_intensity = shrunk_ewma_covariance(returns)
        second, second_intensity = shrunk_ewma_covariance(returns)

        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
        assert first_intensity == second_intensity
