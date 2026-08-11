"""Constrained mean-variance allocation as a proper convex program.

This is the QP companion to ``src/portfolio.py::optimize_long_only``. That one
solves the same objective by projected subgradient ascent, which is exact about
its feasible set precisely because that set is a capped simplex with a
closed-form projection. The limitation is structural rather than a matter of
tuning: add a *group* constraint — no more than 25% in any one sector — and the
exact projection no longer exists, so the constraint can only be approximated
or applied as a post-hoc clip, and a clipped solution is no longer the optimum
of anything.

Sector limits are the constraint that actually binds in an Indian equity book.
A momentum signal concentrates in whatever sector has been running, and a book
with no group limit ends up expressing a single macro bet through twenty
tickers while reporting itself as diversified — which is the same failure the
covariance layer exists to measure, arriving through a different door.

    maximize    w'mu - (lambda/2) w'Sigma w - c'|w - w_prev|
    subject to  sum(w) <= budget
                0 <= w_i <= max_weight_i
                S w <= sector_caps

Why the module and not src/portfolio/optimizer.py: ``src/portfolio`` is already
a module, so a package of that name cannot exist beside it without breaking
every existing import.

cvxpy is an optional extra (``pip install -e '.[optimize]'``), matching how the
package treats torch and huggingface_hub. Import it lazily so that a install
without it keeps working — everything else in the platform runs fine, it just
cannot solve group-constrained books.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .portfolio import AllocationResult, portfolio_volatility
except ImportError:  # running from inside src/ as a flat package
    from portfolio import AllocationResult, portfolio_volatility

# Bucket for tickers absent from the sector map. They get a row of their own
# rather than being exempted: an unmapped name that escapes every group limit
# is the one most likely to be a concentrated position nobody is watching.
UNKNOWN_SECTOR = "UNKNOWN"

_SOLVER_PREFERENCE = ("CLARABEL", "OSQP", "SCS")


def _require_cvxpy():
    try:
        import cvxpy as cp
    except ImportError as error:  # pragma: no cover - exercised by the extra
        raise ImportError(
            "optimize_mean_variance_qp needs cvxpy, which is an optional extra. "
            "Install it with: pip install -e '.[optimize]'"
        ) from error
    return cp


def sector_constraint_matrix(
    names: Sequence[str],
    sector_by_ticker: Dict[str, str],
) -> Tuple[List[str], np.ndarray]:
    """Build the (S, N) group matrix whose rows sum a sector's weights.

    Rows are in sorted sector order so the matrix — and therefore any cap
    vector aligned to it — is stable across runs rather than depending on dict
    iteration order.

    Args:
        names: Asset labels, in the same order as the weight vector.
        sector_by_ticker: Ticker to sector. Missing tickers fall into
            UNKNOWN_SECTOR.

    Returns:
        (sector_names, matrix) with matrix[s, i] = 1 when asset i is in sector s.
    """
    labels = [str(sector_by_ticker.get(name, UNKNOWN_SECTOR)) for name in names]
    sectors = sorted(set(labels))
    index = {sector: row for row, sector in enumerate(sectors)}

    matrix = np.zeros((len(sectors), len(names)), dtype=float)
    for column, label in enumerate(labels):
        matrix[index[label], column] = 1.0
    return sectors, matrix


def _psd_projection(covariance: np.ndarray) -> np.ndarray:
    """Symmetrize and clip negative eigenvalues to zero.

    A shrunk covariance is PSD in exact arithmetic, but floating point leaves
    eigenvalues around -1e-18 and an asymmetry in the last bit. cvxpy checks
    for PSD-ness and refuses the problem outright, so without this the
    optimizer fails on matrices that are mathematically fine. Clipping is the
    honest repair: it changes the matrix by less than the error already in it.
    """
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if eigenvalues.min() >= 0.0:
        return symmetric
    clipped = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * clipped) @ eigenvectors.T


def optimize_mean_variance_qp(
    expected_returns: Sequence[float] | np.ndarray,
    covariance: pd.DataFrame | np.ndarray,
    risk_aversion: float = 1.0,
    max_weight: float | Sequence[float] = 1.0,
    budget: float = 1.0,
    previous_weights: Optional[Sequence[float]] = None,
    turnover_cost: float | Sequence[float] = 0.0,
    sector_matrix: Optional[np.ndarray] = None,
    max_sector_weight: Optional[float | Sequence[float]] = None,
    require_full_investment: bool = False,
    names: Optional[List[str]] = None,
    solver: Optional[str] = None,
) -> AllocationResult:
    """Solve the constrained mean-variance program exactly.

    **The turnover term is linearized, not approximated.** ``|w - w_prev|`` is
    not differentiable, so it cannot enter a quadratic program directly. The
    standard reformulation introduces u and replaces the penalty with c'u
    subject to u >= w - w_prev and u >= -(w - w_prev). Since u appears in the
    objective only with a positive cost, the solver drives it down until one of
    the two constraints is tight, at which point u_i is exactly |w_i -
    w_prev,i|. The reformulation is therefore equivalent, not a relaxation —
    and the test suite asserts the equality rather than trusting it.

    Args:
        expected_returns: mu, per-period expected excess returns. Feed it a
            standardized cross-sectional score (see the probit composite) or a
            shrunk drift, not a raw sample mean.
        covariance: Sigma, same periodicity as mu. Symmetrized and projected to
            PSD before use.
        risk_aversion: lambda. Higher means a smaller, safer book.
        max_weight: Per-name cap, scalar or per-asset.
        budget: Maximum total invested fraction; the remainder is cash.
        previous_weights: The book being rebalanced from. Defaults to cash.
        turnover_cost: c, per-unit cost of trading each name.
        sector_matrix: (S, N) group matrix from sector_constraint_matrix.
        max_sector_weight: Cap per sector, scalar or one per row of
            sector_matrix.
        require_full_investment: Force sum(w) == budget rather than <=. Off by
            default because an unlevered book holding cash is a legitimate
            answer; turning it on is what makes conflicting caps *infeasible*
            rather than merely binding.
        names: Asset labels for the result.
        solver: Override the solver. Defaults to the first available of
            Clarabel, OSQP, SCS.

    Returns:
        An AllocationResult satisfying every constraint.

    Raises:
        ValueError: When the program is infeasible or the solver fails — never
            a silently-violating book, which would put the violation into
            production.
    """
    cp = _require_cvxpy()

    mu = np.asarray(expected_returns, dtype=float).ravel()
    cov = np.asarray(covariance, dtype=float)
    n = mu.size

    if isinstance(covariance, pd.DataFrame) and names is None:
        names = [str(c) for c in covariance.columns]

    if n == 0:
        return AllocationResult(np.zeros(0), 0.0, 0.0, 0.0, 0.0, 0, names)
    if cov.shape != (n, n):
        raise ValueError(
            f"covariance shape {cov.shape} does not match {n} expected returns"
        )

    upper = np.asarray(max_weight, dtype=float)
    if upper.ndim == 0:
        upper = np.full(n, float(upper))
    upper = np.clip(upper, 0.0, budget)

    previous = (
        np.zeros(n) if previous_weights is None
        else np.asarray(previous_weights, dtype=float).ravel()
    )
    if previous.size != n:
        raise ValueError(
            f"previous_weights length {previous.size} does not match {n} assets"
        )

    cost = np.asarray(turnover_cost, dtype=float)
    if cost.ndim == 0:
        cost = np.full(n, float(cost))
    cost = np.maximum(cost, 0.0)

    sigma = _psd_projection(cov)

    weights = cp.Variable(n, nonneg=True)
    # The auxiliary variable that makes the L1 turnover term expressible.
    deviation = cp.Variable(n, nonneg=True)

    objective = cp.Maximize(
        mu @ weights
        - 0.5 * risk_aversion * cp.quad_form(weights, cp.psd_wrap(sigma))
        - cost @ deviation
    )

    constraints = [
        weights <= upper,
        deviation >= weights - previous,
        deviation >= -(weights - previous),
    ]
    constraints.append(
        cp.sum(weights) == budget if require_full_investment
        else cp.sum(weights) <= budget
    )

    if sector_matrix is not None and max_sector_weight is not None:
        groups = np.asarray(sector_matrix, dtype=float)
        if groups.ndim != 2 or groups.shape[1] != n:
            raise ValueError(
                f"sector_matrix shape {groups.shape} does not match {n} assets"
            )
        caps = np.asarray(max_sector_weight, dtype=float)
        if caps.ndim == 0:
            caps = np.full(groups.shape[0], float(caps))
        if caps.size != groups.shape[0]:
            raise ValueError(
                f"max_sector_weight has {caps.size} entries for "
                f"{groups.shape[0]} sectors"
            )
        constraints.append(groups @ weights <= caps)

    problem = cp.Problem(objective, constraints)
    chosen = solver or _first_available_solver(cp)
    try:
        problem.solve(solver=chosen)
    except cp.error.SolverError as error:
        raise ValueError(f"portfolio optimization failed in {chosen}: {error}") from error

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise ValueError(
            f"portfolio optimization is {problem.status} — the constraint set has "
            f"no feasible book. Check that the sector caps sum to at least the "
            f"budget when full investment is required."
        )

    solved = np.asarray(weights.value, dtype=float).ravel()
    # The solver works to a tolerance, so a weight can come back at -1e-11 or a
    # hair over its cap. Clip to the constraint set rather than reporting a
    # book that violates it in the twelfth decimal.
    solved = np.clip(solved, 0.0, upper)

    return AllocationResult(
        weights=solved,
        expected_return=float(solved @ mu),
        volatility=float(portfolio_volatility(solved, sigma)),
        turnover=float(np.sum(np.abs(solved - previous))),
        objective=float(problem.value),
        iterations=int(problem.solver_stats.num_iters or 0)
        if problem.solver_stats is not None else 0,
        names=names,
    )


def _first_available_solver(cp) -> str:
    """Prefer Clarabel, then OSQP, then SCS.

    Clarabel is the interior-point solver cvxpy ships as its default for this
    problem class and is the successor to ECOS, which recent cvxpy releases no
    longer install.
    """
    installed = set(cp.installed_solvers())
    for candidate in _SOLVER_PREFERENCE:
        if candidate in installed:
            return candidate
    raise ImportError(
        f"no supported QP solver installed; expected one of {_SOLVER_PREFERENCE}"
    )
