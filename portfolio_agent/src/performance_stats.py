"""Selection-bias-aware performance statistics.

A backtest Sharpe ratio is a point estimate drawn from a search. This platform
searches: stop multipliers, five strategies, three UMA combination methods,
three simulation methods, two model architectures, and a regime map with four
hand-set thresholds. Report the best number that search produces and you are
reporting the maximum of many draws, not the quality of a strategy — and the
maximum of many draws from a zero-edge process is comfortably positive.

This module implements the statistics that price that in:

- ``sharpe_ratio`` — the arithmetic-excess-return Sharpe, in place of the
  geometric-over-arithmetic hybrid that was biased low by roughly sigma/2.
- ``probabilistic_sharpe_ratio`` (PSR) — the probability the true Sharpe
  exceeds a benchmark, given the sample's length, skewness and kurtosis.
- ``deflated_sharpe_ratio`` (DSR) — PSR evaluated against the Sharpe you would
  *expect* the best of N trials to produce by luck alone.
- ``probability_of_backtest_overfitting`` (PBO) — how often the configuration
  that looks best in sample lands below the median out of sample.
- ``rank_information_coefficient`` — the cross-sectional metric a ranking
  system should actually be judged on, which separates signal quality from
  portfolio construction.
- ``newey_west_standard_error`` — the correction for overlapping labels, which
  daily-sampled multi-day targets always have.

Nothing here needs market data, a model or a config: they are pure functions
over return series, so they can be unit-tested against known answers.

References
----------
Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio", Journal of
Portfolio Management 40(5). Bailey, Borwein, Lopez de Prado & Zhu (2016), "The
Probability of Backtest Overfitting", Journal of Computational Finance.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# Euler-Mascheroni constant, which appears in the expected maximum of N draws
# from a Gaussian (the DSR's deflation threshold).
EULER_MASCHERONI = 0.5772156649015329

_NORMAL = NormalDist()

# Below this, a return series has no dispersion worth dividing by. The test is
# absolute rather than `== 0` on purpose: np.std of a genuinely constant series
# lands around 1e-19 rather than exactly zero, and dividing a real mean by that
# reports a Sharpe of 1e16 instead of "this series does not move".
_MIN_STANDARD_DEVIATION = 1e-12


def _standard_normal_cdf(x: float) -> float:
    """Phi(x), via the stdlib rather than a new dependency."""
    if math.isnan(x):
        return float("nan")
    return _NORMAL.cdf(x)


def _standard_normal_ppf(p: float) -> float:
    """Phi^-1(p), clamped away from the singularities at 0 and 1."""
    return _NORMAL.inv_cdf(min(1.0 - 1e-15, max(1e-15, p)))


def _clean(values: Sequence[float]) -> np.ndarray:
    """Finite float array; NaN and inf dropped."""
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def to_daily_risk_free(annual_rate: float) -> float:
    """Daily equivalent of an annually-compounded risk-free rate.

    Compounded, not divided: (1+r)^(1/252) - 1. Over a 6.5% policy rate the
    difference from r/252 is small, but it is free to be right and the error
    would sit directly in the Sharpe numerator.
    """
    return (1.0 + annual_rate) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0


def excess_returns(
    returns: Sequence[float],
    risk_free_rate: float | Sequence[float] = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> np.ndarray:
    """Period returns net of the risk-free rate.

    Args:
        returns: Per-period (typically daily) portfolio returns.
        risk_free_rate: Either an annual scalar, or a per-period series already
            aligned to `returns`. A *series* is the correct input over any
            window in which the policy rate moved — India's did materially over
            2021-2025 — and a constant is a guess about the whole window.
        periods_per_year: Periods per year, for converting a scalar annual rate.

    Returns:
        Array of excess returns, same length as the finite part of `returns`.
    """
    r = np.asarray(returns, dtype=float).ravel()
    if np.isscalar(risk_free_rate) or np.ndim(risk_free_rate) == 0:
        per_period = (1.0 + float(risk_free_rate)) ** (1.0 / periods_per_year) - 1.0
        excess = r - per_period
    else:
        rf = np.asarray(risk_free_rate, dtype=float).ravel()
        if rf.shape != r.shape:
            raise ValueError(
                f"risk_free_rate series length ({rf.shape[0]}) must match returns ({r.shape[0]})"
            )
        excess = r - rf
    return excess[np.isfinite(excess)]


def sharpe_ratio(
    returns: Sequence[float],
    risk_free_rate: float | Sequence[float] = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio, arithmetic numerator over arithmetic denominator.

    The definition matters because the platform previously divided a
    *geometric* mean (CAGR) by an *arithmetic* standard deviation. Since
    CAGR ~= mu - sigma^2/2, that hybrid returns approximately

        (mu - rf)/sigma - sigma/2

    i.e. it is biased low by sigma/2 — a constant -0.10 of Sharpe at 20%
    volatility, and larger for more volatile strategies, so it penalizes
    volatility twice and is not comparable to a conventional 1.2 target.

    Returns:
        Annualized Sharpe; 0.0 when there is no usable dispersion.
    """
    excess = excess_returns(returns, risk_free_rate, periods_per_year)
    if excess.size < 2:
        return 0.0
    sigma = float(np.std(excess, ddof=1))
    if sigma <= _MIN_STANDARD_DEVIATION:
        return 0.0
    return float(np.mean(excess) / sigma * math.sqrt(periods_per_year))


def newey_west_standard_error(values: Sequence[float], lags: int) -> float:
    """Standard error of the mean under serial correlation (Bartlett kernel).

    Daily-sampled H-day forward returns share H-1 days with their neighbours.
    Treating them as independent understates the standard error by roughly
    sqrt(H), which makes every t-statistic and Sharpe confidence interval built
    on them too narrow — in the direction that manufactures significance.

        se^2 = (1/n) * [ gamma_0 + 2 * sum_{k=1..q} (1 - k/(q+1)) * gamma_k ]

    The Bartlett weights (1 - k/(q+1)) are what keep the estimate positive.

    Args:
        values: The overlapping observations.
        lags: q, the truncation lag. For an H-day overlapping label use H-1.

    Returns:
        Newey-West standard error of the sample mean; 0.0 for degenerate input.
    """
    arr = _clean(values)
    n = arr.size
    if n < 2:
        return 0.0

    demeaned = arr - arr.mean()
    variance = float(np.dot(demeaned, demeaned) / n)

    q = max(0, min(int(lags), n - 1))
    for k in range(1, q + 1):
        gamma_k = float(np.dot(demeaned[k:], demeaned[:-k]) / n)
        variance += 2.0 * (1.0 - k / (q + 1.0)) * gamma_k

    if variance <= 0:
        return 0.0
    return math.sqrt(variance / n)


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    benchmark_sharpe: float = 0.0,
    periods_per_year: Optional[int] = TRADING_DAYS_PER_YEAR,
) -> float:
    """Probability that the true Sharpe exceeds `benchmark_sharpe`.

        PSR = Phi( (SR - SR*) * sqrt(n-1) / sqrt(1 - g3*SR + (g4-1)/4 * SR^2) )

    where g3 is skewness and g4 is (non-excess) kurtosis. The denominator is
    the point of the statistic: negative skew and fat tails — both endemic to
    equity strategies, and to Indian small caps in particular — inflate the
    standard error of a Sharpe estimate, so the same headline number is weaker
    evidence than a Gaussian assumption would suggest.

    Args:
        observed_sharpe: The estimated Sharpe.
        n_observations: Number of return observations it was computed from.
        skewness: Sample skewness of the return series.
        kurtosis: Sample kurtosis, *not* excess (3.0 is Gaussian).
        benchmark_sharpe: The threshold SR* to beat, in the same annualization
            as `observed_sharpe`.
        periods_per_year: Annualization factor to strip before applying the
            formula, which is defined on per-period Sharpes. Pass None if the
            inputs are already per-period.

    Returns:
        Probability in [0, 1]; 0.0 when the sample is too short to say anything.
    """
    if n_observations < 2:
        return 0.0

    scale = 1.0 if periods_per_year in (None, 0) else math.sqrt(periods_per_year)
    sr = observed_sharpe / scale
    sr_benchmark = benchmark_sharpe / scale

    variance_term = 1.0 - skewness * sr + (kurtosis - 1.0) / 4.0 * sr * sr
    if variance_term <= 0:
        # The moment estimates are self-inconsistent (possible on tiny
        # samples); refusing to answer beats reporting a complex number.
        return 0.0

    z = (sr - sr_benchmark) * math.sqrt(n_observations - 1) / math.sqrt(variance_term)
    return float(_standard_normal_cdf(z))


def expected_maximum_sharpe(
    n_trials: int,
    sharpe_variance: float,
    periods_per_year: Optional[int] = TRADING_DAYS_PER_YEAR,
) -> float:
    """The Sharpe the *best* of N independent zero-skill trials would show.

        SR*_0 = sqrt(V[SR]) * [ (1-g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N*e)) ]

    with g the Euler-Mascheroni constant. This is the null a reported Sharpe
    has to beat once you admit how many configurations were tried: search 50
    variants of a strategy with no edge and the winner still prints a Sharpe
    well above zero.

    Args:
        n_trials: Number of configurations tried. Under-report it and the
            deflation is too weak; this is what the trial log exists to supply.
        sharpe_variance: Variance of the Sharpe estimates *across* trials, in
            the same annualization as the reported Sharpe.
        periods_per_year: Annualization of the inputs, or None if per-period.

    Returns:
        The deflation threshold, in the same units as `sharpe_variance`'s
        square root. 0.0 for a single trial or degenerate variance.
    """
    if n_trials <= 1 or sharpe_variance <= 0:
        return 0.0

    scale = 1.0 if periods_per_year in (None, 0) else math.sqrt(periods_per_year)
    sigma = math.sqrt(sharpe_variance) / scale

    n = float(n_trials)
    threshold = sigma * (
        (1.0 - EULER_MASCHERONI) * _standard_normal_ppf(1.0 - 1.0 / n)
        + EULER_MASCHERONI * _standard_normal_ppf(1.0 - 1.0 / (n * math.e))
    )
    return float(threshold * scale)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_observations: int,
    n_trials: int,
    sharpe_variance: float,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    periods_per_year: Optional[int] = TRADING_DAYS_PER_YEAR,
) -> float:
    """PSR measured against the expected maximum of N trials.

    DSR is the number to publish. A DSR below 0.95 means the reported Sharpe is
    not distinguishable from what the search itself would have produced on a
    strategy with no edge.

    Returns:
        Probability in [0, 1].
    """
    threshold = expected_maximum_sharpe(n_trials, sharpe_variance, periods_per_year)
    return probabilistic_sharpe_ratio(
        observed_sharpe=observed_sharpe,
        n_observations=n_observations,
        skewness=skewness,
        kurtosis=kurtosis,
        benchmark_sharpe=threshold,
        periods_per_year=periods_per_year,
    )


def return_moments(returns: Sequence[float]) -> Tuple[int, float, float]:
    """(n, skewness, non-excess kurtosis) of a return series.

    Computed here rather than pulled from pandas so PSR and DSR always read the
    same moment conventions: kurtosis is raw (3.0 for a Gaussian), not excess.
    """
    arr = _clean(returns)
    n = arr.size
    if n < 3:
        return n, 0.0, 3.0

    sigma = float(np.std(arr, ddof=0))
    if sigma <= _MIN_STANDARD_DEVIATION:
        return n, 0.0, 3.0

    demeaned = arr - arr.mean()
    skewness = float(np.mean(demeaned**3) / sigma**3)
    kurtosis = float(np.mean(demeaned**4) / sigma**4)
    return n, skewness, kurtosis


def evaluate_sharpe(
    returns: Sequence[float],
    risk_free_rate: float | Sequence[float] = 0.0,
    n_trials: int = 1,
    sharpe_variance: Optional[float] = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    benchmark_sharpe: float = 0.0,
) -> Dict[str, float]:
    """The full Sharpe report: point estimate, PSR, DSR and the deflation used.

    Args:
        returns: Per-period portfolio returns.
        risk_free_rate: Annual scalar or per-period series.
        n_trials: Number of configurations tried to arrive at this result.
        sharpe_variance: Variance of Sharpe across those trials. When omitted
            and n_trials > 1, falls back to the sample's own Sharpe estimator
            variance, which is a *conservative* stand-in — it assumes the
            trials were as dispersed as sampling noise alone would make them.
        periods_per_year: Annualization factor.
        benchmark_sharpe: SR* for the (undeflated) PSR.

    Returns:
        Dict of sharpe_ratio, psr, dsr, deflation threshold and the moments.
    """
    excess = excess_returns(returns, risk_free_rate, periods_per_year)
    n, skewness, kurtosis = return_moments(excess)
    observed = sharpe_ratio(returns, risk_free_rate, periods_per_year)

    if sharpe_variance is None:
        # Var[SR_hat] ~= (1 + SR^2/2)/n for per-period Sharpes; annualized, the
        # leading term is periods_per_year/n.
        per_period_sr = observed / math.sqrt(periods_per_year)
        sharpe_variance = (
            (1.0 + 0.5 * per_period_sr**2) / n * periods_per_year if n > 0 else 0.0
        )

    threshold = expected_maximum_sharpe(n_trials, sharpe_variance, periods_per_year)
    return {
        "sharpe_ratio": observed,
        "n_observations": float(n),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "n_trials": float(n_trials),
        "psr": probabilistic_sharpe_ratio(
            observed, n, skewness, kurtosis, benchmark_sharpe, periods_per_year
        ),
        "deflation_threshold_sharpe": threshold,
        "dsr": deflated_sharpe_ratio(
            observed, n, n_trials, sharpe_variance, skewness, kurtosis, periods_per_year
        ),
    }


def probability_of_backtest_overfitting(
    trial_returns: np.ndarray,
    n_splits: int = 16,
) -> Dict[str, float]:
    """Combinatorially-symmetric cross-validation (CSCV) estimate of PBO.

    The question PBO answers is not "is this strategy good" but "is my
    *selection procedure* predictive". Split the return series into S blocks,
    take every way of choosing S/2 of them as in-sample, pick the trial that
    looks best there, and see where it lands out of sample. If the winner
    reliably falls below the OOS median, the search is fitting noise and the
    ranking it produces is worse than useless.

    PBO > 0.5 means the selection process is *anti*-predictive: the
    configuration that looked best in sample is more likely than not to be
    below average out of sample.

    Args:
        trial_returns: Array of shape (n_periods, n_trials) — one column per
            configuration tried, all over the same periods.
        n_splits: S, the number of blocks. Must be even. C(S, S/2) combinations
            are evaluated; S=16 gives 12,870, which runs in about a second.

    Returns:
        Dict with pbo, the mean logit, and the number of combinations used.
    """
    matrix = np.asarray(trial_returns, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        return {"pbo": 0.0, "mean_logit": 0.0, "n_combinations": 0.0, "n_trials": 0.0}

    if n_splits % 2 != 0:
        raise ValueError(f"n_splits must be even, got {n_splits}")

    n_periods, n_trials = matrix.shape
    n_splits = min(n_splits, n_periods)
    n_splits -= n_splits % 2
    if n_splits < 2:
        return {"pbo": 0.0, "mean_logit": 0.0, "n_combinations": 0.0, "n_trials": float(n_trials)}

    blocks = np.array_split(np.arange(n_periods), n_splits)

    def _block_sharpe(rows: np.ndarray) -> np.ndarray:
        """Per-trial Sharpe over a subset of periods, as a (n_trials,) array."""
        sub = matrix[rows]
        sigma = sub.std(axis=0, ddof=1)
        mean = sub.mean(axis=0)
        # A trial with no dispersion carries no information; score it at the
        # bottom rather than dividing by zero.
        usable = sigma > _MIN_STANDARD_DEVIATION
        return np.where(usable, mean / np.where(usable, sigma, 1.0), -np.inf)

    logits: List[float] = []
    for chosen in itertools.combinations(range(n_splits), n_splits // 2):
        in_sample = np.concatenate([blocks[i] for i in chosen])
        out_of_sample = np.concatenate(
            [blocks[i] for i in range(n_splits) if i not in chosen]
        )
        if in_sample.size < 2 or out_of_sample.size < 2:
            continue

        best = int(np.argmax(_block_sharpe(in_sample)))
        oos = _block_sharpe(out_of_sample)

        # Relative rank of the in-sample winner among all trials, out of sample.
        rank = float(np.sum(oos <= oos[best]))
        omega = rank / (n_trials + 1.0)
        omega = min(1.0 - 1e-12, max(1e-12, omega))
        logits.append(math.log(omega / (1.0 - omega)))

    if not logits:
        return {"pbo": 0.0, "mean_logit": 0.0, "n_combinations": 0.0, "n_trials": float(n_trials)}

    logit_array = np.asarray(logits, dtype=float)
    return {
        "pbo": float(np.mean(logit_array <= 0.0)),
        "mean_logit": float(np.mean(logit_array)),
        "n_combinations": float(logit_array.size),
        "n_trials": float(n_trials),
    }


def rank_information_coefficient(
    predictions: pd.Series,
    realized: pd.Series,
    dates: Optional[Sequence[Any]] = None,
    horizon_days: int = 1,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Dict[str, float]:
    """Spearman rank IC per date, and its information ratio.

    This is the right primary metric for a system whose output is a *ranking*
    over a cross-section, and it is far less noisy than backtested P&L because
    it does not confound signal quality with position sizing, costs, or the
    covariance structure of the book. A platform that cannot separate those two
    things cannot tell whether a disappointing backtest is a bad signal or bad
    portfolio construction.

        IC_t   = spearman( prediction_i,t , realized_i,t )
        ICIR   = mean(IC) / sd(IC) * sqrt(periods_per_year / horizon_days)

    Args:
        predictions: Predicted scores, indexed to match `realized`.
        realized: Realized forward returns over the same horizon.
        dates: Grouping key per observation. Defaults to the shared index,
            which is the common case for a date-indexed panel.
        horizon_days: Label horizon, used only to annualize the ICIR.
        periods_per_year: Trading periods per year.

    Returns:
        Dict of mean_ic, ic_std, icir, hit_rate (share of dates with IC > 0)
        and n_dates.
    """
    empty = {"mean_ic": 0.0, "ic_std": 0.0, "icir": 0.0, "hit_rate": 0.0, "n_dates": 0.0}

    frame = pd.DataFrame(
        {
            "prediction": np.asarray(predictions, dtype=float),
            "realized": np.asarray(realized, dtype=float),
        }
    )
    frame["date"] = list(dates) if dates is not None else list(pd.Series(predictions).index)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return empty

    ics: List[float] = []
    for _, group in frame.groupby("date", sort=True):
        # A single name has no cross-section to rank, and a constant column has
        # no rank variation — both are undefined rather than zero.
        if len(group) < 2:
            continue
        pred_rank = group["prediction"].rank()
        real_rank = group["realized"].rank()
        if pred_rank.nunique() < 2 or real_rank.nunique() < 2:
            continue
        ics.append(float(np.corrcoef(pred_rank.to_numpy(), real_rank.to_numpy())[0, 1]))

    if not ics:
        return empty

    ic_array = np.asarray(ics, dtype=float)
    ic_array = ic_array[np.isfinite(ic_array)]
    if ic_array.size == 0:
        return empty

    mean_ic = float(np.mean(ic_array))
    ic_std = float(np.std(ic_array, ddof=1)) if ic_array.size > 1 else 0.0
    annualization = math.sqrt(periods_per_year / max(1, horizon_days))
    icir = float(mean_ic / ic_std * annualization) if ic_std > 0 else 0.0

    return {
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "icir": icir,
        "hit_rate": float(np.mean(ic_array > 0)),
        "n_dates": float(ic_array.size),
    }


@dataclass
class Trial:
    """One backtest configuration and the result it produced."""

    label: str
    sharpe: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: str = ""
    # Fingerprint of the *whole* resolved config, not just the parameters
    # enumerated above — see config_hash() for why the enumeration is not
    # enough on its own.
    config_hash: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "label": self.label,
                "sharpe": self.sharpe,
                "parameters": self.parameters,
                "metrics": self.metrics,
                "config_hash": self.config_hash,
                "timestamp": self.timestamp
                or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            sort_keys=True,
            default=str,
        )


def config_hash(payload: Any) -> str:
    """Stable fingerprint of a resolved configuration.

    Two runs are the same trial when they ran the same configuration, and the
    only reliable way to say that is to hash all of it. The `parameters` dict
    on a Trial is hand-enumerated at the call site, so it catches exactly the
    knobs someone remembered to list: add a simulation option or change a
    weight inside a strategy YAML and two materially different runs record as
    the same trial, which corrupts N — and N is the entire input to the
    deflation.

    SHA-256 over canonical JSON rather than the builtin hash(). str.__hash__
    is randomized per process unless PYTHONHASHSEED is pinned, so a log keyed
    on it would treat every re-run as a fresh trial and over-deflate every
    Sharpe the platform reports. Determinism is a repo-wide requirement and
    this is one of the places it is easy to lose silently.

    Args:
        payload: Any JSON-serializable structure — typically an AppConfig
            dumped to a dict. Non-serializable leaves fall back to their repr,
            which is stable for the config primitives in use here.

    Returns:
        16 hex characters: 64 bits, which is far more than enough to keep a
        few thousand research trials collision-free, and short enough to read
        in a log.
    """
    canonical = json.dumps(payload, sort_keys=True, default=repr, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def distinct_trials(trials: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse repeat recordings of one configuration, oldest kept.

    Re-running a configuration is not a new trial. The platform enforces
    determinism, so a repeat produces the same Sharpe; counting it again
    inflates N and deflates the reported Sharpe against a search that never
    happened.

    Identity falls back through three levels so a log with history in it does
    not lose entries written before the hash existed: the config hash, then
    the parameters dict, then nothing — and an entry with neither is kept as
    its own trial, because the safe assumption about an unidentifiable run is
    that it was a real one.
    """
    seen: set[str] = set()
    unique: List[Dict[str, Any]] = []

    for trial in trials:
        fingerprint = str(trial.get("config_hash") or "")
        if not fingerprint:
            parameters = trial.get("parameters")
            if parameters:
                fingerprint = config_hash(parameters)
        if not fingerprint:
            unique.append(trial)
            continue
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(trial)

    return unique


def log_trial(path: str | Path, trial: Trial) -> None:
    """Append one trial to the append-only trial log.

    The Deflated Sharpe Ratio is undefined without N, the number of
    configurations tried, and N is exactly the quantity a research process
    forgets. Writing it down costs nothing and is the difference between a
    reported Sharpe that can be interpreted and one that cannot.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(trial.to_json() + "\n")


def read_trials(path: str | Path) -> List[Dict[str, Any]]:
    """Every trial recorded in the log, oldest first.

    Malformed lines are skipped rather than raising: a corrupt entry from an
    interrupted run must not make the whole research history unreadable.
    """
    source = Path(path)
    if not source.exists():
        return []

    trials: List[Dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            trials.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return trials


def trial_sharpe_variance(trials: Iterable[Dict[str, Any]]) -> Tuple[int, float]:
    """(n_trials, variance of Sharpe across them) for the DSR deflation."""
    sharpes = _clean([t.get("sharpe", float("nan")) for t in trials])
    if sharpes.size < 2:
        return int(sharpes.size), 0.0
    return int(sharpes.size), float(np.var(sharpes, ddof=1))
