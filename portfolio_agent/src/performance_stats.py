"""Selection-bias-aware performance statistics.

A raw Sharpe ratio answers "how did this configuration do?". It does not
answer the question a research platform actually needs answered, which is
"given how many configurations I tried, how surprised should I be?" — and this
platform searches a large space: stop multipliers, five strategies, three UMA
combination methods, three simulation methods, two model architectures, and a
regime map with four hand-set thresholds. Every one of those is a trial, and
the maximum Sharpe over N trials of a *worthless* strategy grows without bound
in N.

Three statistics close that gap, all from Bailey & Lopez de Prado:

- **PSR** — the probability that the true Sharpe exceeds a benchmark, given
  the observed Sharpe, the sample length, and the return series' skewness and
  kurtosis. It is the single-trial question asked properly.
- **DSR** — PSR evaluated against the Sharpe you would *expect* to see as the
  maximum of N independent trials of zero-skill strategies. This is the
  multiple-testing correction, and it is what makes a reported Sharpe
  comparable across research programmes of different sizes.
- **PBO** — the probability that the *selection procedure* is anti-predictive:
  the fraction of combinatorially-symmetric train/test splits in which the
  in-sample winner lands below the median out-of-sample. Above 0.5 the search
  is worse than picking at random.

Plus two things the platform needs before any of the above is meaningful:

- **Rank information coefficient**, the correct primary metric for a system
  whose output is a cross-sectional ranking. It is far less noisy than
  backtested P&L and it separates signal quality from portfolio construction —
  a separation this platform currently cannot make.
- **Newey-West** standard errors, because daily-sampled H-day forward returns
  overlap on H-1 days and the naive standard error is too small by roughly
  sqrt(H).

See docs/QUANT_RESEARCH.md and Bailey & Lopez de Prado (2014), "The Deflated
Sharpe Ratio", Journal of Portfolio Management 40(5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Sequence

import numpy as np
from scipy import stats

TRADING_DAYS_PER_YEAR = 252

# Euler-Mascheroni constant, from the expected value of the maximum of N
# independent standard normals.
EULER_MASCHERONI = 0.5772156649015329


def _clean(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def _has_dispersion(sigma: float, mean: float) -> bool:
    """Is this standard deviation real, or floating-point residue?

    A constant series does not necessarily produce sigma == 0.0 exactly:
    [0.03] * 10 leaves ~1e-18 after mean subtraction, and dividing a mean of
    0.03 by that yields a Sharpe of 1e16. The guard is therefore relative to
    the mean's own scale rather than a test against exact zero.
    """
    if not math.isfinite(sigma) or sigma <= 0.0:
        return False
    return sigma > 1e-12 * max(1.0, abs(mean))


def sharpe_ratio(
    returns: Sequence[float],
    risk_free_per_period: float = 0.0,
    periods_per_year: Optional[int] = None,
) -> float:
    """Arithmetic Sharpe ratio.

    Both moments are measured in the same space, which is the whole point:
    dividing a *geometric* mean (CAGR) by an *arithmetic* standard deviation
    is biased low by roughly sigma/2, because CAGR ~= mu - sigma^2/2. At 20%
    volatility that is a constant -0.10 of Sharpe, and it grows with
    volatility, so it penalizes volatile strategies twice.

    Args:
        returns: Per-period returns (not excess).
        risk_free_per_period: Risk-free rate over the same period. Pass a
            per-period figure, not an annual one.
        periods_per_year: Annualization factor; None returns the
            per-period ratio.

    Returns:
        Sharpe ratio; 0.0 when the sample is too short or has no dispersion.
    """
    excess = _clean(returns) - risk_free_per_period
    if excess.size < 2:
        return 0.0
    mean = float(np.mean(excess))
    sigma = float(np.std(excess, ddof=1))
    if not _has_dispersion(sigma, mean):
        return 0.0
    ratio = mean / sigma
    if periods_per_year:
        ratio *= math.sqrt(periods_per_year)
    return ratio


def annual_rate_to_period(annual_rate: float, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Convert an annual rate to the equivalent compounded per-period rate."""
    if periods_per_year <= 0:
        return 0.0
    return (1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0


def newey_west_variance(values: Sequence[float], lags: int) -> float:
    """Long-run variance of the mean under serial correlation (Bartlett kernel).

    Daily-sampled H-day forward returns share H-1 days with their neighbours,
    so successive observations are strongly positively autocorrelated by
    construction. The i.i.d. variance of the mean therefore understates the
    true sampling variance, typically by a factor near H, and every t-statistic
    and Sharpe built on it reads high.

        LRV = gamma_0 + 2 * sum_{k=1..L} (1 - k/(L+1)) * gamma_k

    The Bartlett weights taper to zero at the truncation lag, which guarantees
    the estimate is non-negative — an unweighted sum does not.

    Args:
        values: The series whose mean is being tested.
        lags: Truncation lag L. For an H-day overlapping label, H-1.

    Returns:
        Long-run variance of a single observation (divide by n for the
        variance of the mean). Falls back to the sample variance when the
        kernel estimate is degenerate.
    """
    arr = _clean(values)
    n = arr.size
    if n < 2:
        return 0.0

    demeaned = arr - arr.mean()
    gamma_0 = float(np.dot(demeaned, demeaned) / n)
    total = gamma_0

    for k in range(1, min(int(lags), n - 1) + 1):
        gamma_k = float(np.dot(demeaned[k:], demeaned[:-k]) / n)
        weight = 1.0 - k / (lags + 1.0)
        total += 2.0 * weight * gamma_k

    if not math.isfinite(total) or total <= 0:
        return gamma_0
    return total


def sharpe_ratio_overlapping(
    returns: Sequence[float],
    horizon_days: int,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Sharpe of an overlapping H-day return series, corrected for the overlap.

    The mean is unaffected by overlap; the standard error is not. The
    denominator therefore uses the Newey-West long-run standard deviation at
    lag H-1 rather than the i.i.d. one, and annualization is by
    sqrt(periods_per_year / H) because there are only that many independent
    H-day windows in a year.
    """
    arr = _clean(returns)
    if arr.size < 2:
        return 0.0
    long_run_variance = newey_west_variance(arr, lags=max(0, int(horizon_days) - 1))
    if long_run_variance <= 0:
        return 0.0
    ratio = float(np.mean(arr)) / math.sqrt(long_run_variance)
    return ratio * math.sqrt(periods_per_year / max(1, int(horizon_days)))


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    n_observations: int,
    skewness: float,
    kurtosis: float,
    benchmark_sharpe: float = 0.0,
) -> float:
    """P(true Sharpe > benchmark_sharpe), given the observed sample.

        PSR = Phi( (SR_hat - SR*) * sqrt(n-1)
                   / sqrt(1 - g3*SR_hat + (g4-1)/4 * SR_hat^2) )

    The denominator is the standard error of the Sharpe estimator under
    non-normal returns. Negative skew and fat tails both *inflate* it, which is
    exactly right: a strategy that makes small gains and occasional large
    losses has a less trustworthy Sharpe than a symmetric one with the same
    mean and variance, and the naive sqrt(n) standard error cannot see that.

    Args:
        observed_sharpe: Sharpe in the SAME frequency as n_observations —
            i.e. non-annualized. Passing an annualized figure alongside a
            daily n silently overstates significance.
        n_observations: Number of return observations.
        skewness: Sample skewness (g3) of the return series.
        kurtosis: Sample kurtosis (g4), NOT excess — 3.0 for a normal.
        benchmark_sharpe: The Sharpe to beat, in the same frequency.

    Returns:
        Probability in [0, 1]; 0.0 when the sample is too short.
    """
    if n_observations < 2:
        return 0.0

    variance = 1.0 - skewness * observed_sharpe + (kurtosis - 1.0) / 4.0 * observed_sharpe ** 2
    if variance <= 0:
        return 0.0

    statistic = (observed_sharpe - benchmark_sharpe) * math.sqrt(n_observations - 1)
    return float(stats.norm.cdf(statistic / math.sqrt(variance)))


def expected_maximum_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Sharpe you would expect from the LUCKIEST of N zero-skill trials.

        SR*_0 = sqrt(V[SR]) * [ (1-g)*Phi^-1(1 - 1/N) + g*Phi^-1(1 - 1/(N*e)) ]

    with g the Euler-Mascheroni constant. This is the extreme-value benchmark
    the Deflated Sharpe Ratio deflates against: it is the Sharpe a research
    programme of N trials produces *by construction*, with no skill involved.
    It grows without bound in N, which is why an unrecorded trial count makes
    any reported Sharpe uninterpretable.

    Args:
        n_trials: Number of independent configurations tried.
        sharpe_variance: Variance of the Sharpe estimates ACROSS trials.

    Returns:
        The expected maximum Sharpe, in the same frequency as the inputs.
    """
    if n_trials < 2 or sharpe_variance <= 0:
        return 0.0

    quantile_1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    quantile_2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sharpe_variance) * (
        (1.0 - EULER_MASCHERONI) * quantile_1 + EULER_MASCHERONI * quantile_2
    )


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_observations: int,
    skewness: float,
    kurtosis: float,
    n_trials: int,
    sharpe_variance: float,
) -> float:
    """PSR evaluated against the expected maximum Sharpe of N trials.

    A DSR below 0.95 means the reported Sharpe is not distinguishable from
    what the search itself would have produced on noise. Note that this is
    strictly weaker than the raw Sharpe looks: deflating is meant to hurt.

    Args:
        observed_sharpe: Non-annualized Sharpe of the selected strategy.
        n_observations: Number of return observations.
        skewness: Sample skewness of the selected strategy's returns.
        kurtosis: Sample kurtosis (not excess) of the same.
        n_trials: Number of configurations tried (see src/trial_log.py — this
            is the number the platform must record, not guess).
        sharpe_variance: Variance of the Sharpe estimates across those trials.

    Returns:
        Probability in [0, 1].
    """
    benchmark = expected_maximum_sharpe(n_trials, sharpe_variance)
    return probabilistic_sharpe_ratio(
        observed_sharpe=observed_sharpe,
        n_observations=n_observations,
        skewness=skewness,
        kurtosis=kurtosis,
        benchmark_sharpe=benchmark,
    )


@dataclass(frozen=True)
class PBOResult:
    """Outcome of a combinatorially-symmetric cross-validation."""

    pbo: float
    n_splits: int
    n_strategies: int
    median_logit: float

    @property
    def is_overfit(self) -> bool:
        """PBO above 0.5 means the selection procedure is anti-predictive."""
        return self.pbo > 0.5


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    n_blocks: int = 16,
) -> Optional[PBOResult]:
    """Probability of Backtest Overfitting via CSCV (Bailey et al., 2016).

    The question is not "is this strategy overfit" but "is my *selection
    procedure* overfit" — does picking the in-sample best tend to produce an
    out-of-sample loser? CSCV answers it without any assumption about the
    return distribution:

    1. Split the T observations into S contiguous blocks.
    2. For every one of C(S, S/2) ways to choose half the blocks as training,
       reassemble train and test sets in chronological order.
    3. Find the strategy with the highest in-sample Sharpe.
    4. Record its *relative rank* out of sample.

    PBO is the fraction of splits where that rank falls below the median. At
    0.5 the procedure carries no information; above it, the in-sample winner
    is systematically an out-of-sample loser, which is the signature of fitting
    noise.

    Args:
        returns_matrix: Shape (n_observations, n_strategies). One column per
            configuration tried, all over the same period.
        n_blocks: S, which must be even. C(16, 8) = 12,870 splits, which is
            the usual choice and runs in well under a second.

    Returns:
        A PBOResult, or None when the input cannot support the procedure
        (fewer than two strategies, or too few observations to block).
    """
    matrix = np.asarray(returns_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        return None

    n_blocks = int(n_blocks)
    if n_blocks < 2 or n_blocks % 2 != 0:
        raise ValueError(f"n_blocks must be a positive even integer, got {n_blocks}")
    if matrix.shape[0] < n_blocks * 2:
        return None

    # Trim to a whole number of blocks from the *start*, so the most recent
    # observations are the ones kept.
    block_length = matrix.shape[0] // n_blocks
    matrix = matrix[matrix.shape[0] - block_length * n_blocks:]
    blocks = np.split(matrix, n_blocks, axis=0)

    n_strategies = matrix.shape[1]
    logits: list[float] = []
    below_median = 0

    for train_ids in combinations(range(n_blocks), n_blocks // 2):
        train_set = set(train_ids)
        train = np.concatenate([blocks[i] for i in range(n_blocks) if i in train_set])
        test = np.concatenate([blocks[i] for i in range(n_blocks) if i not in train_set])

        best = int(np.argmax(_column_sharpes(train)))
        test_sharpes = _column_sharpes(test)

        # Relative rank of the in-sample winner among the out-of-sample
        # results, in (0, 1). Ties are broken by average rank so a degenerate
        # column cannot make the procedure look better than it is.
        rank = float(stats.rankdata(test_sharpes)[best])
        relative_rank = rank / (n_strategies + 1)

        if relative_rank <= 0.5:
            below_median += 1
        logits.append(math.log(relative_rank / (1.0 - relative_rank)))

    n_splits = len(logits)
    if n_splits == 0:
        return None

    return PBOResult(
        pbo=below_median / n_splits,
        n_splits=n_splits,
        n_strategies=n_strategies,
        median_logit=float(np.median(logits)),
    )


def _column_sharpes(matrix: np.ndarray) -> np.ndarray:
    """Per-column Sharpe, with zero-dispersion columns scored at zero."""
    means = matrix.mean(axis=0)
    sigmas = matrix.std(axis=0, ddof=1)
    return np.divide(means, sigmas, out=np.zeros_like(means), where=sigmas > 0)


def information_coefficient(
    predictions: Sequence[float],
    realized: Sequence[float],
) -> float:
    """Spearman rank correlation between a cross-section's forecast and outcome.

    The rank form rather than Pearson, because Indian daily cross-sections are
    dominated by circuit-limit outliers: a single +20% locked print moves a
    Pearson IC far more than it moves the decision the number is supposed to
    describe. Rank IC is invariant to any monotone transform of either side,
    which is the right invariance for a system that only ever acts on the
    ordering.

    Returns:
        IC in [-1, 1]; 0.0 when there is nothing rankable.
    """
    pred = np.asarray(predictions, dtype=float).ravel()
    real = np.asarray(realized, dtype=float).ravel()
    if pred.size != real.size or pred.size < 3:
        return 0.0

    finite = np.isfinite(pred) & np.isfinite(real)
    pred, real = pred[finite], real[finite]
    if pred.size < 3:
        return 0.0
    if np.all(pred == pred[0]) or np.all(real == real[0]):
        return 0.0

    correlation = stats.spearmanr(pred, real).statistic
    return 0.0 if not np.isfinite(correlation) else float(correlation)


def information_ratio_of_ic(
    ic_series: Sequence[float],
    horizon_days: int = 1,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """ICIR — mean IC over its standard deviation, annualized.

    The signal-quality analogue of a Sharpe ratio, and the number to judge a
    ranking model on: it is measurable from predictions alone, so it does not
    confound the model with the portfolio construction downstream of it. A
    daily rank IC of 0.03 with an ICIR near 0.5 is a real, ordinary equity
    signal; a backtested Sharpe of 1.5 on the same predictions is a statement
    about the sizing rules.

    Args:
        ic_series: One IC per cross-section (typically one per date).
        horizon_days: Forecast horizon, for the annualization factor.
        periods_per_year: Trading periods in a year.
    """
    arr = _clean(ic_series)
    if arr.size < 2:
        return 0.0
    mean = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    if not _has_dispersion(sigma, mean):
        return 0.0
    return mean / sigma * math.sqrt(periods_per_year / max(1, int(horizon_days)))
