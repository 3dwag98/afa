"""Tests for the constrained mean-variance QP (Task 3.2).

The projected-subgradient optimizer in src/portfolio.py already handles the
box, the budget and the turnover penalty. What it cannot express is a *group*
constraint: its feasible set is a capped simplex with an exact projection, and
adding arbitrary linear sector limits destroys that projection. Sector caps are
the constraint that actually binds in an Indian equity book — the momentum
signal concentrates in whatever sector has been running, and a book with no
group limit ends up expressing one macro bet through twenty tickers.
"""

import numpy as np
import pandas as pd
import pytest

cvxpy = pytest.importorskip("cvxpy", reason="the QP optimizer is an optional extra")

from src.portfolio_optimizer import optimize_mean_variance_qp, sector_constraint_matrix


def _universe(n_per_sector=5, sectors=("TECH", "BANK", "PHARM", "AUTO")):
    names, labels = [], {}
    for sector in sectors:
        for i in range(n_per_sector):
            ticker = f"{sector}{i}"
            names.append(ticker)
            labels[ticker] = sector
    return names, labels


def _diagonal_covariance(n, variance=0.04):
    return np.eye(n) * variance


class TestSectorConstraint:
    """The acceptance criterion: a sector cap that actually binds."""

    def test_sector_cap_binds_against_a_sector_with_massive_alpha(self):
        names, labels = _universe()
        n = len(names)

        # TECH has an overwhelming edge; every other name has none. Without a
        # group constraint the optimizer puts everything it can into TECH.
        mu = np.array([0.20 if labels[t] == "TECH" else 0.0 for t in names])
        cov = _diagonal_covariance(n)

        sectors, matrix = sector_constraint_matrix(names, labels)
        result = optimize_mean_variance_qp(
            expected_returns=mu,
            covariance=cov,
            risk_aversion=1.0,
            max_weight=1.0,
            budget=1.0,
            sector_matrix=matrix,
            max_sector_weight=0.25,
            names=names,
        )
        weights = result.as_series()
        tech = float(weights[[t for t in names if labels[t] == "TECH"]].sum())

        # The cap binds: it is tight, not merely respected by luck.
        assert tech == pytest.approx(0.25, abs=1e-6)
        assert tech <= 0.25 + 1e-9

    def test_without_the_cap_the_same_book_concentrates(self):
        """Guards the premise of the test above.

        A sector cap that "prevents >25%" means nothing unless the
        unconstrained solution would have exceeded it.
        """
        names, labels = _universe()
        n = len(names)
        mu = np.array([0.20 if labels[t] == "TECH" else 0.0 for t in names])

        result = optimize_mean_variance_qp(
            expected_returns=mu,
            covariance=_diagonal_covariance(n),
            risk_aversion=1.0,
            max_weight=1.0,
            budget=1.0,
            names=names,
        )
        weights = result.as_series()
        tech = float(weights[[t for t in names if labels[t] == "TECH"]].sum())

        assert tech > 0.9

    def test_every_sector_cap_holds_simultaneously(self):
        names, labels = _universe()
        n = len(names)
        rng = np.random.default_rng(4)
        mu = rng.uniform(0.0, 0.15, size=n)

        sectors, matrix = sector_constraint_matrix(names, labels)
        result = optimize_mean_variance_qp(
            expected_returns=mu,
            covariance=_diagonal_covariance(n),
            risk_aversion=0.5,
            max_weight=1.0,
            budget=1.0,
            sector_matrix=matrix,
            max_sector_weight=0.30,
            names=names,
        )
        exposures = matrix @ result.weights

        assert np.all(exposures <= 0.30 + 1e-6)

    def test_per_sector_caps_may_differ(self):
        """A regulated or illiquid sector can carry a tighter limit than the
        rest without needing a second optimizer."""
        names, labels = _universe()
        n = len(names)
        mu = np.full(n, 0.10)

        sectors, matrix = sector_constraint_matrix(names, labels)
        caps = np.array([0.10 if s == "TECH" else 0.40 for s in sectors])
        result = optimize_mean_variance_qp(
            expected_returns=mu,
            covariance=_diagonal_covariance(n),
            risk_aversion=1.0,
            max_weight=1.0,
            budget=1.0,
            sector_matrix=matrix,
            max_sector_weight=caps,
            names=names,
        )
        exposures = pd.Series(matrix @ result.weights, index=sectors)

        assert exposures["TECH"] <= 0.10 + 1e-6
        assert np.all(exposures.to_numpy() <= caps + 1e-6)

    def test_builds_the_matrix_from_a_ticker_to_sector_map(self):
        names, labels = _universe(n_per_sector=2, sectors=("TECH", "BANK"))
        sectors, matrix = sector_constraint_matrix(names, labels)

        assert sectors == ["BANK", "TECH"]  # sorted, so the rows are stable
        np.testing.assert_array_equal(
            matrix,
            np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]]),
        )

    def test_unmapped_tickers_get_their_own_bucket(self):
        """An unmapped name must not silently escape every group limit."""
        names = ["A", "B", "C"]
        sectors, matrix = sector_constraint_matrix(names, {"A": "TECH"})

        assert "UNKNOWN" in sectors
        unknown = matrix[sectors.index("UNKNOWN")]
        np.testing.assert_array_equal(unknown, np.array([0.0, 1.0, 1.0]))


class TestTurnoverLinearization:
    """The L1 penalty, linearized with auxiliary variables.

    |w - w_prev| is not differentiable, so it cannot go into a QP directly.
    The standard reformulation adds u with u >= w - w_prev and u >= -(w -
    w_prev); at the optimum u is driven down to exactly |w - w_prev| because
    it only ever appears with a positive cost.
    """

    def test_auxiliary_variable_equals_the_absolute_deviation(self):
        n = 6
        rng = np.random.default_rng(9)
        mu = rng.uniform(0.0, 0.1, size=n)
        previous = np.full(n, 1.0 / n)

        result = optimize_mean_variance_qp(
            expected_returns=mu,
            covariance=_diagonal_covariance(n),
            risk_aversion=1.0,
            max_weight=1.0,
            budget=1.0,
            previous_weights=previous,
            turnover_cost=0.01,
        )

        assert result.turnover == pytest.approx(
            float(np.sum(np.abs(result.weights - previous))), abs=1e-6
        )

    def test_a_larger_cost_trades_less(self):
        n = 8
        rng = np.random.default_rng(2)
        mu = rng.uniform(0.0, 0.12, size=n)
        previous = np.zeros(n)
        previous[0] = 1.0

        def turnover(cost):
            return optimize_mean_variance_qp(
                expected_returns=mu,
                covariance=_diagonal_covariance(n),
                risk_aversion=1.0,
                max_weight=1.0,
                budget=1.0,
                previous_weights=previous,
                turnover_cost=cost,
            ).turnover

        assert turnover(0.50) < turnover(0.001)

    def test_a_prohibitive_cost_holds_the_existing_book(self):
        n = 5
        previous = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        result = optimize_mean_variance_qp(
            expected_returns=np.array([0.0, 0.0, 0.0, 0.0, 0.05]),
            covariance=_diagonal_covariance(n),
            risk_aversion=1.0,
            max_weight=1.0,
            budget=1.0,
            previous_weights=previous,
            turnover_cost=10.0,
        )

        np.testing.assert_allclose(result.weights, previous, atol=1e-5)


class TestBasicConstraints:
    def test_weights_are_long_only_and_within_budget(self):
        n = 10
        rng = np.random.default_rng(6)
        mu = rng.normal(0.05, 0.05, size=n)

        result = optimize_mean_variance_qp(
            expected_returns=mu,
            covariance=_diagonal_covariance(n),
            risk_aversion=2.0,
            max_weight=0.15,
            budget=0.9,
        )

        assert np.all(result.weights >= -1e-9)
        assert np.all(result.weights <= 0.15 + 1e-6)
        assert result.weights.sum() <= 0.9 + 1e-6

    def test_negative_alpha_names_are_left_out(self):
        result = optimize_mean_variance_qp(
            expected_returns=np.array([0.10, -0.10]),
            covariance=_diagonal_covariance(2),
            risk_aversion=1.0,
            max_weight=1.0,
            budget=1.0,
        )

        assert result.weights[1] == pytest.approx(0.0, abs=1e-7)

    def test_accepts_a_covariance_with_tiny_negative_eigenvalues(self):
        """A shrunk covariance is PSD in exact arithmetic but can carry
        eigenvalues around -1e-18 after floating point. cvxpy rejects those
        outright, so the matrix is symmetrized and clipped before it is used —
        otherwise the optimizer fails on a matrix that is mathematically fine.
        """
        cov = np.array([[0.04, 0.04], [0.04, 0.04]])
        cov[0, 1] += 1e-17  # break symmetry too

        result = optimize_mean_variance_qp(
            expected_returns=np.array([0.05, 0.02]),
            covariance=cov,
            risk_aversion=1.0,
            max_weight=1.0,
            budget=1.0,
        )

        assert np.all(np.isfinite(result.weights))

    def test_is_deterministic(self):
        n = 12
        rng = np.random.default_rng(21)
        mu = rng.normal(0.04, 0.03, size=n)
        cov = _diagonal_covariance(n)

        first = optimize_mean_variance_qp(
            expected_returns=mu, covariance=cov, risk_aversion=1.5,
            max_weight=0.2, budget=1.0,
        )
        second = optimize_mean_variance_qp(
            expected_returns=mu, covariance=cov, risk_aversion=1.5,
            max_weight=0.2, budget=1.0,
        )

        np.testing.assert_array_equal(first.weights, second.weights)

    def test_empty_universe_returns_an_empty_allocation(self):
        result = optimize_mean_variance_qp(
            expected_returns=np.zeros(0),
            covariance=np.zeros((0, 0)),
        )

        assert result.weights.size == 0

    def test_infeasible_sector_caps_raise_rather_than_return_nonsense(self):
        """Caps summing below the budget cannot all hold at once. Returning a
        silently-infeasible book would put the violation into production."""
        names, labels = _universe(n_per_sector=2, sectors=("TECH", "BANK"))
        sectors, matrix = sector_constraint_matrix(names, labels)

        with pytest.raises(ValueError, match="infeasible|feasible"):
            optimize_mean_variance_qp(
                expected_returns=np.full(4, 0.1),
                covariance=_diagonal_covariance(4),
                budget=1.0,
                max_weight=1.0,
                sector_matrix=matrix,
                max_sector_weight=0.1,  # 2 sectors * 0.1 = 0.2 < budget 1.0
                names=names,
                require_full_investment=True,
            )
