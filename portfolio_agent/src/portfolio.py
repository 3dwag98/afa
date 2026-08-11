"""Portfolio-level covariance estimation and constrained allocation.

Everything upstream of this module sizes positions one at a time. A signal is
scored, a stop is set, a quantity is derived from that trade's own risk, and
the position is opened without reference to what else is in the book. For a
long-only Indian equity portfolio of 20-40 names that is the single largest
misstatement of risk in the platform, because the dominant term in realized
portfolio volatility is not any position's own variance — it is the covariance
between them:

    sigma_p^2 = sum_i sum_j w_i w_j sigma_i sigma_j rho_ij

At 20 positions of 3% each and 30% single-name volatility, moving the average
pairwise correlation from 0 to 0.35 raises portfolio volatility from 4.0% to
11.1%, and to 16.7% at 0.85. Indian equity correlations sit around 0.3-0.4 in
calm markets and rise toward 0.6-0.85 in exactly the panic states a drawdown
breaker exists to survive, so a model that assumes independence is most wrong
precisely when being wrong is most expensive.

The same gap explains a result the platform already observed: a
meta-orchestrator that underperformed its own momentum sleeve. Four sleeves
ranking one universe on correlated signals produce a book far less diversified
than "four sleeves" suggests; the extra sleeves added correlated risk without
adding independent return, and nothing in the stack could see it.

What is here
------------
- ``ledoit_wolf_covariance`` — sample covariance shrunk toward a constant
  correlation target, with the analytically optimal intensity. At N names and
  T days with T not much larger than N the sample covariance is near-singular
  and its extreme eigenvalues are mostly noise; shrinkage is what makes it
  invertible and stable enough to optimize against.
- ``exponentially_weighted_covariance`` — the same matrix with a half-life, so
  it responds to a correlation regime shift instead of averaging across one.
- ``single_factor_covariance`` — B*var(f)*B' + D, O(N) parameters instead of
  O(N^2), and far better conditioned on a wide universe.
- ``portfolio_volatility`` / ``risk_contributions`` / ``diversification_ratio``
  — the measurements that were simply absent.
- ``optimize_long_only`` — constrained mean-variance with an explicit turnover
  penalty. The penalty is not optional decoration: without it a daily
  rebalanced optimizer trades the whole book on estimation noise and hands
  every rupee of edge to the friction stack.
- ``hierarchical_risk_parity`` — the allocation to use when expected returns
  are not trustworthy, which (see monte_carlo.shrink_drift) is most of the
  time. It never inverts the covariance matrix at all.

Everything takes and returns plain numpy/pandas and holds no configuration, so
it is testable in isolation and usable from both the backtest engine and the
live orchestrator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# Eigenvalues below this (relative to the largest) are treated as numerically
# zero when a matrix has to be inverted or square-rooted.
_EIGENVALUE_FLOOR = 1e-12


def _as_matrix(returns: pd.DataFrame | np.ndarray) -> Tuple[np.ndarray, Optional[List[str]]]:
    """(T, N) float matrix plus column labels, with non-finite rows dropped."""
    if isinstance(returns, pd.DataFrame):
        names = [str(c) for c in returns.columns]
        matrix = returns.to_numpy(dtype=float)
    else:
        names = None
        matrix = np.asarray(returns, dtype=float)

    if matrix.ndim != 2:
        raise ValueError(f"returns must be 2-dimensional (T, N), got shape {matrix.shape}")

    finite_rows = np.all(np.isfinite(matrix), axis=1)
    return matrix[finite_rows], names


def _wrap(matrix: np.ndarray, names: Optional[List[str]]) -> pd.DataFrame | np.ndarray:
    """Return a labelled frame when the input carried labels."""
    if names is None:
        return matrix
    return pd.DataFrame(matrix, index=names, columns=names)


def sample_covariance(
    returns: pd.DataFrame | np.ndarray,
    annualize: bool = False,
) -> pd.DataFrame | np.ndarray:
    """Plain sample covariance of return columns.

    Provided mostly as the thing to compare shrinkage against: with N names and
    T periods it needs T > N even to be invertible, and its largest eigenvalues
    are biased up and smallest biased down, which is exactly the distortion a
    mean-variance optimizer amplifies.
    """
    matrix, names = _as_matrix(returns)
    if matrix.shape[0] < 2:
        n = matrix.shape[1]
        return _wrap(np.zeros((n, n)), names)

    cov = np.cov(matrix, rowvar=False, ddof=1)
    cov = np.atleast_2d(cov)
    if annualize:
        cov = cov * TRADING_DAYS_PER_YEAR
    return _wrap(cov, names)


def ledoit_wolf_covariance(
    returns: pd.DataFrame | np.ndarray,
    annualize: bool = False,
    shrinkage: Optional[float] = None,
) -> Tuple[pd.DataFrame | np.ndarray, float]:
    """Covariance shrunk toward a constant-correlation target.

    The target F assumes every pair of assets shares one average correlation
    r_bar, keeping each asset's own variance:

        F_ij = r_bar * sqrt(s_ii * s_jj),   F_ii = s_ii

    That is a good target for equities specifically — it is wrong in detail but
    right about the thing that matters, that the names move together — and it
    is far better conditioned than the sample matrix. The shrinkage intensity
    is Ledoit & Wolf's analytically optimal one,

        delta = max(0, min(1, (pi - rho) / gamma / T))

    where pi is the summed asymptotic variance of the sample covariances, rho
    the covariance between the sample and target estimation errors, and gamma
    the squared Frobenius distance between them. Intuitively: shrink hard when
    the sample matrix is noisy (large pi) and the target is close (small gamma).

    Args:
        returns: (T, N) frame or array of period returns.
        annualize: Multiply by 252 to return an annualized covariance.
        shrinkage: Override the computed intensity, mainly for testing. 0
            recovers the sample covariance, 1 the pure target.

    Returns:
        (covariance, shrinkage_intensity). The covariance is a DataFrame when
        the input was one.

    Reference:
        Ledoit & Wolf (2003), "Honey, I Shrunk the Sample Covariance Matrix".
    """
    matrix, names = _as_matrix(returns)
    n_periods, n_assets = matrix.shape

    if n_periods < 2 or n_assets < 1:
        return _wrap(np.zeros((n_assets, n_assets)), names), 0.0

    demeaned = matrix - matrix.mean(axis=0)
    # MLE covariance (1/T), which is the convention the shrinkage constants
    # below are derived under.
    sample = demeaned.T @ demeaned / n_periods
    variances = np.diag(sample).copy()

    if n_assets == 1:
        cov = sample * (TRADING_DAYS_PER_YEAR if annualize else 1.0)
        return _wrap(cov, names), 0.0

    std = np.sqrt(np.maximum(variances, 0.0))
    outer_std = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.where(outer_std > 0, sample / np.where(outer_std > 0, outer_std, 1.0), 0.0)

    off_diagonal = ~np.eye(n_assets, dtype=bool)
    mean_correlation = float(np.mean(correlation[off_diagonal])) if n_assets > 1 else 0.0

    target = mean_correlation * outer_std
    np.fill_diagonal(target, variances)

    if shrinkage is None:
        # pi: summed asymptotic variances of the sample covariance entries.
        squared = demeaned**2
        pi_matrix = (squared.T @ squared) / n_periods - sample**2
        pi = float(np.sum(pi_matrix))

        # rho: the diagonal contributes pi_ii; the off-diagonal picks up how
        # the target's own estimation error (through r_bar and the variances)
        # co-moves with the sample covariances.
        term = (demeaned**3).T @ demeaned / n_periods - variances[:, None] * sample
        np.fill_diagonal(term, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(outer_std > 0, std[None, :] / np.where(std[:, None] > 0, std[:, None], 1.0), 0.0)
        rho = float(np.sum(np.diag(pi_matrix))) + mean_correlation * float(
            np.sum(ratio * term)
        )

        gamma = float(np.sum((target - sample) ** 2))
        if gamma <= 0:
            intensity = 0.0
        else:
            intensity = max(0.0, min(1.0, (pi - rho) / gamma / n_periods))
    else:
        intensity = float(min(1.0, max(0.0, shrinkage)))

    cov = intensity * target + (1.0 - intensity) * sample
    if annualize:
        cov = cov * TRADING_DAYS_PER_YEAR
    return _wrap(cov, names), intensity


def exponentially_weighted_covariance(
    returns: pd.DataFrame | np.ndarray,
    half_life_days: float = 60.0,
    annualize: bool = False,
) -> pd.DataFrame | np.ndarray:
    """Covariance with exponentially decaying observation weights.

    A correlation regime shift is a change in the matrix, not noise around a
    stable one. An equally-weighted five-year window averages the calm and the
    crisis together and is therefore wrong in both. A ~60-day half-life keeps
    enough observations to estimate anything at all while letting the matrix
    move when the market's correlation structure does.

    Args:
        returns: (T, N) frame or array, oldest row first.
        half_life_days: Periods over which an observation's weight halves.
        annualize: Multiply by 252.
    """
    matrix, names = _as_matrix(returns)
    n_periods, n_assets = matrix.shape
    if n_periods < 2:
        return _wrap(np.zeros((n_assets, n_assets)), names)

    decay = math.log(2.0) / max(1e-9, float(half_life_days))
    age = np.arange(n_periods - 1, -1, -1, dtype=float)  # newest row has age 0
    weights = np.exp(-decay * age)
    weights /= weights.sum()

    mean = weights @ matrix
    demeaned = matrix - mean
    cov = (demeaned * weights[:, None]).T @ demeaned

    # Weighted analogue of the (T-1) correction: without it the estimate is
    # biased low by the effective sample size implied by the weights.
    effective_n = 1.0 / np.sum(weights**2)
    if effective_n > 1:
        cov = cov * effective_n / (effective_n - 1.0)

    if annualize:
        cov = cov * TRADING_DAYS_PER_YEAR
    return _wrap(cov, names)


def single_factor_covariance(
    returns: pd.DataFrame | np.ndarray,
    factor_returns: Optional[Sequence[float]] = None,
    annualize: bool = False,
) -> pd.DataFrame | np.ndarray:
    """Covariance from a one-factor model: Sigma = b b' var(f) + diag(resid).

    Estimating N(N+1)/2 free parameters from a few hundred observations is
    hopeless for a wide universe; a factor structure estimates 2N instead and
    is positive-definite by construction. With the market as the single factor
    this captures the term that actually dominates a long-only Indian book,
    and it produces the betas a factor-neutral signal would need anyway.

    Args:
        returns: (T, N) frame or array.
        factor_returns: The factor series. Defaults to the equal-weighted mean
            of the columns, which is a serviceable market proxy when no index
            is cached.
        annualize: Multiply by 252.
    """
    matrix, names = _as_matrix(returns)
    n_periods, n_assets = matrix.shape
    if n_periods < 2:
        return _wrap(np.zeros((n_assets, n_assets)), names)

    if factor_returns is None:
        factor = matrix.mean(axis=1)
    else:
        factor = np.asarray(factor_returns, dtype=float).ravel()
        if factor.shape[0] != n_periods:
            raise ValueError(
                f"factor_returns length ({factor.shape[0]}) must match returns rows ({n_periods})"
            )

    factor_variance = float(np.var(factor, ddof=1))
    if factor_variance <= 0:
        return _wrap(np.diag(np.var(matrix, axis=0, ddof=1)), names)

    factor_demeaned = factor - factor.mean()
    demeaned = matrix - matrix.mean(axis=0)
    betas = (factor_demeaned @ demeaned) / (factor_demeaned @ factor_demeaned)

    residuals = demeaned - np.outer(factor_demeaned, betas)
    residual_variance = np.var(residuals, axis=0, ddof=1)

    cov = np.outer(betas, betas) * factor_variance + np.diag(residual_variance)
    if annualize:
        cov = cov * TRADING_DAYS_PER_YEAR
    return _wrap(cov, names)


def portfolio_volatility(
    weights: Sequence[float],
    covariance: pd.DataFrame | np.ndarray,
) -> float:
    """sqrt(w' Sigma w) — the number the platform never computed.

    Args:
        weights: Portfolio weights as fractions of portfolio value. They need
            not sum to 1; cash is simply the remainder.
        covariance: Covariance matrix in the same periodicity as the answer
            wanted (annualized in, annualized out).
    """
    w = np.asarray(weights, dtype=float).ravel()
    cov = np.asarray(covariance, dtype=float)
    if w.size == 0 or cov.size == 0:
        return 0.0
    if cov.shape != (w.size, w.size):
        raise ValueError(f"covariance shape {cov.shape} does not match {w.size} weights")

    variance = float(w @ cov @ w)
    return math.sqrt(max(0.0, variance))


def independent_portfolio_volatility(
    weights: Sequence[float],
    covariance: pd.DataFrame | np.ndarray,
) -> float:
    """What per-trade sizing implicitly assumes: sqrt(sum (w_i sigma_i)^2).

    Reporting this alongside portfolio_volatility is the cheapest way to make
    the gap visible, because the ratio between them is exactly the factor by
    which independent sizing understates portfolio risk.
    """
    w = np.asarray(weights, dtype=float).ravel()
    cov = np.asarray(covariance, dtype=float)
    if w.size == 0 or cov.size == 0:
        return 0.0
    variances = np.diag(cov)
    return math.sqrt(max(0.0, float(np.sum((w**2) * variances))))


def correlation_risk_multiple(
    weights: Sequence[float],
    covariance: pd.DataFrame | np.ndarray,
) -> float:
    """How many times larger true portfolio volatility is than the independent
    assumption implies. 1.0 means correlation costs nothing; Indian equities in
    a drawdown routinely put this between 2.5 and 4."""
    independent = independent_portfolio_volatility(weights, covariance)
    if independent <= 0:
        return 1.0
    return portfolio_volatility(weights, covariance) / independent


def risk_contributions(
    weights: Sequence[float],
    covariance: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Each position's share of total portfolio variance.

    RC_i = w_i * (Sigma w)_i / (w' Sigma w). These sum to 1 and are what a
    concentration limit should actually be written against — a 3% position in a
    name correlated 0.9 with eight others is not a 3% risk position.
    """
    w = np.asarray(weights, dtype=float).ravel()
    cov = np.asarray(covariance, dtype=float)
    if w.size == 0:
        return np.zeros(0)

    variance = float(w @ cov @ w)
    if variance <= 0:
        return np.zeros(w.size)
    return (w * (cov @ w)) / variance


def diversification_ratio(
    weights: Sequence[float],
    covariance: pd.DataFrame | np.ndarray,
) -> float:
    """Weighted average volatility divided by portfolio volatility.

    1.0 means the book is one bet wearing many tickers. Higher is more
    genuinely diversified.
    """
    w = np.asarray(weights, dtype=float).ravel()
    cov = np.asarray(covariance, dtype=float)
    if w.size == 0:
        return 1.0

    weighted_average = float(np.sum(np.abs(w) * np.sqrt(np.maximum(np.diag(cov), 0.0))))
    total = portfolio_volatility(w, cov)
    if total <= 0:
        return 1.0
    return weighted_average / total


def project_onto_capped_simplex(
    vector: Sequence[float],
    upper_bounds: Sequence[float] | float,
    budget: float = 1.0,
) -> np.ndarray:
    """Euclidean projection onto {w : 0 <= w <= ub, sum(w) <= budget}.

    The constraint set for an unlevered long-only cash book. Projection is by
    bisection on the single dual variable of the budget constraint: for a shift
    theta, clip(v - theta, 0, ub) is the projection under the box alone, and
    its sum is non-increasing in theta, so one bisection finds the theta that
    meets the budget exactly.

    Args:
        vector: The point to project.
        upper_bounds: Per-asset cap, or one cap for all.
        budget: Maximum total allocation. 1.0 is fully invested with no leverage.

    Returns:
        The projected weights.
    """
    v = np.asarray(vector, dtype=float).ravel()
    if v.size == 0:
        return v

    ub = np.asarray(upper_bounds, dtype=float)
    if ub.ndim == 0:
        ub = np.full(v.size, float(ub))
    ub = np.maximum(ub, 0.0)

    def clipped(theta: float) -> np.ndarray:
        return np.clip(v - theta, 0.0, ub)

    # Already feasible with theta = 0.
    if clipped(0.0).sum() <= budget:
        return clipped(0.0)

    low, high = 0.0, float(np.max(v)) + float(np.max(ub)) + 1.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if clipped(mid).sum() > budget:
            low = mid
        else:
            high = mid
    return clipped(high)


@dataclass
class AllocationResult:
    """The output of a portfolio optimization, with its diagnostics."""

    weights: np.ndarray
    expected_return: float
    volatility: float
    turnover: float
    objective: float
    iterations: int
    names: Optional[List[str]] = None

    def as_series(self) -> pd.Series:
        """Weights as a labelled Series when names are available."""
        if self.names is None:
            return pd.Series(self.weights)
        return pd.Series(self.weights, index=self.names)


def optimize_long_only(
    expected_returns: Sequence[float],
    covariance: pd.DataFrame | np.ndarray,
    risk_aversion: float = 5.0,
    max_weight: Sequence[float] | float = 0.03,
    budget: float = 1.0,
    previous_weights: Optional[Sequence[float]] = None,
    turnover_cost: Sequence[float] | float = 0.0,
    max_volatility: Optional[float] = None,
    iterations: int = 2000,
    names: Optional[List[str]] = None,
) -> AllocationResult:
    """Constrained mean-variance allocation for an unlevered long-only book.

        maximize    w'mu - (lambda/2) w'Sigma w - c'|w - w_prev|
        subject to  sum(w) <= budget,  0 <= w_i <= max_weight_i,
                    sqrt(w'Sigma w) <= max_volatility

    Solved by projected subgradient ascent: the smooth part is differentiable,
    the turnover term contributes a bounded subgradient, and the feasible set
    is a capped simplex with an exact projection (see
    project_onto_capped_simplex), so every iterate is feasible and the final
    answer satisfies the constraints by construction rather than by tolerance.

    **The turnover penalty is load-bearing.** Expected returns are estimated
    with enormous error (see monte_carlo.shrink_drift), so an unpenalized
    optimizer re-solves to a materially different book every day and pays the
    full Indian friction stack — brokerage, STT on both legs, exchange and SEBI
    charges, GST, stamp duty and slippage — for the privilege of chasing noise.
    Pass the per-name round-trip cost the execution simulator already computes.

    Args:
        expected_returns: mu, per-period expected excess returns.
        covariance: Sigma, in the same periodicity as mu.
        risk_aversion: lambda. Higher means a smaller, safer book.
        max_weight: Per-name concentration cap, scalar or per-asset.
        budget: Maximum total invested fraction; the remainder is cash.
        previous_weights: The book being rebalanced from. Defaults to cash.
        turnover_cost: c, per-unit cost of trading each name.
        max_volatility: Optional hard ceiling on portfolio volatility. Applied
            by scaling the whole book down, which preserves its composition.
        iterations: Subgradient iterations.
        names: Asset labels for the result.

    Returns:
        An AllocationResult whose weights satisfy every constraint.
    """
    mu = np.asarray(expected_returns, dtype=float).ravel()
    cov = np.asarray(covariance, dtype=float)
    n = mu.size

    if isinstance(covariance, pd.DataFrame) and names is None:
        names = [str(c) for c in covariance.columns]

    if n == 0:
        return AllocationResult(np.zeros(0), 0.0, 0.0, 0.0, 0.0, 0, names)
    if cov.shape != (n, n):
        raise ValueError(f"covariance shape {cov.shape} does not match {n} expected returns")

    ub = np.asarray(max_weight, dtype=float)
    if ub.ndim == 0:
        ub = np.full(n, float(ub))
    ub = np.clip(ub, 0.0, budget)

    previous = (
        np.zeros(n) if previous_weights is None
        else np.asarray(previous_weights, dtype=float).ravel()
    )
    if previous.size != n:
        raise ValueError(f"previous_weights length {previous.size} does not match {n} assets")

    cost = np.asarray(turnover_cost, dtype=float)
    if cost.ndim == 0:
        cost = np.full(n, float(cost))
    cost = np.maximum(cost, 0.0)

    def objective(w: np.ndarray) -> float:
        return float(
            w @ mu
            - 0.5 * risk_aversion * (w @ cov @ w)
            - np.sum(cost * np.abs(w - previous))
        )

    # Step size from the curvature of the quadratic term: 1/L for L the
    # Lipschitz constant of the smooth gradient keeps the ascent stable without
    # any line search.
    eigenvalues = np.linalg.eigvalsh(cov)
    lipschitz = max(risk_aversion * float(np.max(eigenvalues)), 1e-12)
    step = 1.0 / lipschitz

    weights = project_onto_capped_simplex(previous, ub, budget)
    best_weights, best_objective = weights.copy(), objective(weights)

    for iteration in range(1, int(iterations) + 1):
        gradient = mu - risk_aversion * (cov @ weights) - cost * np.sign(weights - previous)
        # Decaying step on the non-smooth part: a constant step cannot settle
        # on the kink at w == w_prev, which is exactly where a turnover-averse
        # solution wants to sit.
        weights = project_onto_capped_simplex(
            weights + step * gradient / (1.0 + iteration / 200.0), ub, budget
        )
        value = objective(weights)
        if value > best_objective:
            best_weights, best_objective = weights.copy(), value

    weights = best_weights

    if max_volatility is not None and max_volatility > 0:
        realized = portfolio_volatility(weights, cov)
        if realized > max_volatility:
            # Scale the whole book toward cash rather than re-solving: it holds
            # the relative composition fixed and cannot violate any box or
            # budget constraint, both of which are scale-monotone here.
            weights = weights * (max_volatility / realized)

    return AllocationResult(
        weights=weights,
        expected_return=float(weights @ mu),
        volatility=portfolio_volatility(weights, cov),
        turnover=float(np.sum(np.abs(weights - previous))),
        objective=objective(weights),
        iterations=int(iterations),
        names=names,
    )


def _correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    """Correlation matrix, with zero-variance assets left uncorrelated."""
    std = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    outer = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.where(outer > 0, covariance / np.where(outer > 0, outer, 1.0), 0.0)
    np.fill_diagonal(correlation, 1.0)
    return np.clip(correlation, -1.0, 1.0)


def _quasi_diagonal_order(distance: np.ndarray) -> List[int]:
    """Order assets so similar ones sit adjacent, by single-linkage clustering.

    Written out rather than pulled from scipy so this module keeps the same
    dependency footprint as the rest of the package. Single linkage on the
    correlation distance is what HRP uses to decide which names belong in the
    same branch of the tree.
    """
    n = distance.shape[0]
    if n <= 1:
        return list(range(n))

    clusters: List[List[int]] = [[i] for i in range(n)]
    while len(clusters) > 1:
        best = (float("inf"), 0, 1)
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                linkage = min(
                    distance[i, j] for i in clusters[a] for j in clusters[b]
                )
                if linkage < best[0]:
                    best = (linkage, a, b)
        _, a, b = best
        merged = clusters[a] + clusters[b]
        clusters = [c for k, c in enumerate(clusters) if k not in (a, b)] + [merged]

    return clusters[0]


def hierarchical_risk_parity(
    covariance: pd.DataFrame | np.ndarray,
    max_weight: Optional[float] = None,
) -> pd.Series | np.ndarray:
    """Lopez de Prado's HRP allocation — no matrix inversion, no expected returns.

    Mean-variance needs mu, and mu is the input this platform has least right:
    the drift estimate is mostly noise (see monte_carlo.shrink_drift), and an
    optimizer handed a noisy mu will happily concentrate the entire book in
    whichever name got the luckiest sample. HRP sidesteps both problems. It
    clusters the correlation matrix, then recursively bisects the tree and
    splits capital between the two halves in inverse proportion to their
    variance, so it uses the covariance structure without ever inverting it.

    This is the right default for the "no confident view" state, and out of
    sample it is consistently more stable than mean-variance at the sample
    sizes a retail-scale platform actually has.

    Args:
        covariance: Asset covariance matrix.
        max_weight: Optional per-name cap, applied by clipping and
            renormalizing so the weights still sum to 1.

    Returns:
        Weights summing to 1, as a Series when the input was a DataFrame.
    """
    names = None
    if isinstance(covariance, pd.DataFrame):
        names = [str(c) for c in covariance.columns]
    cov = np.asarray(covariance, dtype=float)
    n = cov.shape[0]

    if n == 0:
        return pd.Series(dtype=float) if names is not None else np.zeros(0)
    if n == 1:
        weights = np.ones(1)
        return pd.Series(weights, index=names) if names is not None else weights

    correlation = _correlation_from_covariance(cov)
    # The standard correlation distance: sqrt((1 - rho) / 2), which is 0 for
    # perfectly correlated names and 1 for perfectly anti-correlated ones.
    distance = np.sqrt(np.maximum(0.0, (1.0 - correlation) / 2.0))
    order = _quasi_diagonal_order(distance)

    def cluster_variance(indices: List[int]) -> float:
        """Variance of an inverse-variance-weighted sub-portfolio."""
        sub = cov[np.ix_(indices, indices)]
        diagonal = np.maximum(np.diag(sub), _EIGENVALUE_FLOOR)
        w = 1.0 / diagonal
        w = w / w.sum()
        return float(w @ sub @ w)

    weights = np.ones(n)
    stack: List[List[int]] = [list(order)]
    while stack:
        cluster = stack.pop()
        if len(cluster) <= 1:
            continue
        split = len(cluster) // 2
        left, right = cluster[:split], cluster[split:]

        var_left = cluster_variance(left)
        var_right = cluster_variance(right)
        total = var_left + var_right
        # Inverse-variance split: the quieter half gets more capital.
        alpha = 1.0 - var_left / total if total > 0 else 0.5

        for i in left:
            weights[i] *= alpha
        for i in right:
            weights[i] *= 1.0 - alpha

        stack.extend([left, right])

    total = weights.sum()
    if total > 0:
        weights = weights / total

    if max_weight is not None and 0 < max_weight < 1:
        for _ in range(100):
            excess = weights - max_weight
            if not np.any(excess > 1e-12):
                break
            weights = np.minimum(weights, max_weight)
            shortfall = 1.0 - weights.sum()
            headroom = max_weight - weights
            if shortfall <= 0 or headroom.sum() <= 0:
                break
            weights = weights + shortfall * headroom / headroom.sum()

    return pd.Series(weights, index=names) if names is not None else weights


def summarize_book_risk(
    weights: Sequence[float],
    covariance: pd.DataFrame | np.ndarray,
) -> Dict[str, float]:
    """One call producing the portfolio risk figures reports should carry.

    Returns portfolio volatility, the volatility independent sizing implies,
    the ratio between them, the diversification ratio, and the largest single
    risk contribution — which is usually a good deal larger than the largest
    single weight, and is the number a concentration limit is really about.
    """
    w = np.asarray(weights, dtype=float).ravel()
    contributions = risk_contributions(w, covariance)
    return {
        "portfolio_volatility": portfolio_volatility(w, covariance),
        "independent_volatility": independent_portfolio_volatility(w, covariance),
        "correlation_risk_multiple": correlation_risk_multiple(w, covariance),
        "diversification_ratio": diversification_ratio(w, covariance),
        "max_weight": float(np.max(w)) if w.size else 0.0,
        "max_risk_contribution": float(np.max(contributions)) if contributions.size else 0.0,
        "n_positions": float(np.sum(w > 0)),
    }
