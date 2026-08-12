"""Forecast skill metrics, computed on a tidy panel and nothing else.

Every function here takes `(date, symbol, score, forward_return)` rows and
returns a number or a small dataclass. Nothing in this module knows what a
strategy is, loads a file, or holds state — which is what makes the answers
checkable against hand-worked examples, and what lets the same code score a
rule-based screen, a trained network and a synthetic control.

Three decisions that are easy to get wrong and change the answer
----------------------------------------------------------------
**Correlate within a date, never across the pool.** Pooling every observation
into one correlation mostly measures whether the score tracks the market's
day-to-day level. A long-only book that picks between names cannot trade a
market view, so the quantity that matters is: on each date, did the predicted
ordering match the realized one? That is the information coefficient, and it is
computed one date at a time.

**Weight dates equally, not observations.** A date with 2,000 eligible names
would otherwise count ten times a date with 200, so a bucket's "average return"
would silently become an average over the widest days. Everything here averages
within a date first and across dates second.

**Assume the observations are autocorrelated, because they are.** Daily-sampled
5-day forward returns share four days of outcome with their neighbours. A
t-statistic built on the naive standard error is too narrow by roughly
sqrt(horizon) — in the direction that manufactures significance. The IC series
inherits that overlap, so its t-statistic goes through the Newey–West estimator
in `src/performance_stats.py`, at `horizon - 1` lags.

What "hit rate" means here
--------------------------
Directional accuracy of a *relative* forecast: did a name the model ranked
above the day's median actually beat the day's mean return? A score on the
0–100 canonical scale carries no absolute prediction, so "was the sign right"
has no meaning without recentring, and the recentred version is the one that
corresponds to what a ranking book does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

#: Below this many names a cross-section is not a cross-section — a rank
#: correlation over three stocks is noise with a decimal point. Kept equal to
#: `features/scaling.py::MIN_CROSS_SECTION_NAMES`, which drops the same thin
#: dates from the feature transform for the same reason.
MIN_CROSS_SECTION_NAMES = 5

#: Column names the panel must carry. Stated once so a caller that assembles a
#: panel by hand fails loudly rather than silently scoring an empty frame.
PANEL_COLUMNS = ("date", "symbol", "score", "forward_return")


def validate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Check the panel's shape and drop rows that cannot be scored.

    Raises:
        ValueError: If a required column is missing. A panel assembled with a
            differently-spelled column would otherwise score as empty, and an
            empty result looks like "no skill" rather than "wrong input".
    """
    missing = [column for column in PANEL_COLUMNS if column not in panel.columns]
    if missing:
        raise ValueError(
            f"panel is missing column(s) {missing}; expected {list(PANEL_COLUMNS)}"
        )
    clean = panel[list(PANEL_COLUMNS)].replace([np.inf, -np.inf], np.nan).dropna()
    return clean.sort_values(["date", "symbol"]).reset_index(drop=True)


def _ratio(numerator: float, denominator: float) -> float:
    """`numerator / denominator`, with the degenerate zero-denominator limits.

    A ratio whose denominator is a dispersion estimate has two distinct
    degenerate cases, and collapsing them to 0.0 gets one of them badly wrong:

    * dispersion zero, mean zero — nothing was observed; 0.0 is right.
    * dispersion zero, mean non-zero — the series is *constant and non-zero*,
      which is the strongest evidence available, not the weakest. The limit is
      infinite and reporting it as 0.0 inverts the conclusion.
    """
    if abs(denominator) > 1e-12:
        return numerator / denominator
    if abs(numerator) <= 1e-12:
        return 0.0
    return math.inf if numerator > 0 else -math.inf


def cross_sectional_percentile(values: Sequence[float]) -> np.ndarray:
    """Rank in (0, 1), average-ranked on ties.

    `rank / (n + 1)` rather than `(rank - 1) / (n - 1)`: the open interval keeps
    the top and bottom names off the boundary, so a percentile can be compared
    across dates whose cross-sections differ in size without the extremes
    being pinned to exactly 0 and 1.
    """
    series = pd.Series(np.asarray(values, dtype=float))
    return (series.rank(method="average") / (len(series) + 1.0)).to_numpy()


def rank_ic(scores: Sequence[float], labels: Sequence[float]) -> float:
    """Spearman correlation of one date's scores against its realized returns.

    Returns:
        The correlation, or NaN when it is undefined — fewer than two names, or
        either side constant. NaN rather than 0.0 on purpose: a constant score
        makes no ordering claim at all, and recording that as "zero skill"
        would average it in with dates where a real claim was made and missed.
    """
    x = pd.Series(np.asarray(scores, dtype=float))
    y = pd.Series(np.asarray(labels, dtype=float))
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(x.corr(y, method="spearman"))


def rank_ic_series(
    panel: pd.DataFrame, min_names: int = MIN_CROSS_SECTION_NAMES
) -> pd.Series:
    """Per-date rank IC, indexed by date and sorted, with undefined dates dropped."""
    panel = validate_panel(panel)
    scores: Dict[Any, float] = {}
    for date, group in panel.groupby("date", sort=True):
        if len(group) < min_names:
            continue
        value = rank_ic(group["score"], group["forward_return"])
        if not math.isnan(value):
            scores[date] = value
    return pd.Series(scores, dtype=float).sort_index()


@dataclass(frozen=True)
class ICSummary:
    """The IC series reduced to the numbers a decision gets made on."""

    mean: float
    std: float
    icir: float
    t_stat: float
    p_value: float
    positive_share: float
    n_dates: int
    newey_west_lags: int

    @property
    def significant(self) -> bool:
        """Two-sided rejection at 5%, on the autocorrelation-adjusted t."""
        return self.p_value < 0.05

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_ic": self.mean,
            "ic_std": self.std,
            "icir": self.icir,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "positive_share": self.positive_share,
            "n_dates": self.n_dates,
            "newey_west_lags": self.newey_west_lags,
        }


def overlap_lags(horizon: int, stride: int = 1) -> int:
    """Newey–West truncation lag for an IC series sampled every `stride` days.

    Two IC observations overlap when their label windows do. Sampled daily, a
    5-day label overlaps its next four neighbours, so the lag is `horizon - 1`.
    Sampled every 5th day it overlaps *none* of them, and using 4 lags there
    corrects for an overlap that is not present — conservative, but wrong, and
    wrong in a way that quietly costs power on exactly the runs someone strode
    to make fast.
    """
    if stride < 1:
        raise ValueError(f"stride must be at least 1, got {stride}")
    return max(0, math.ceil(int(horizon) / stride) - 1)


def summarize_ic(ic: pd.Series, horizon: int, stride: int = 1) -> ICSummary:
    """Reduce an IC series to mean, stability and a t-statistic that holds up.

    Args:
        ic: Per-date rank IC.
        horizon: Label horizon in sessions.
        stride: Sessions between consecutive IC observations. Together with the
            horizon this sets the Newey–West truncation lag — see
            `overlap_lags`.

    Returns:
        An `ICSummary`. An empty or single-date series yields zeros with a
        p-value of 1.0, which is the honest reading: nothing was measured.
    """
    from portfolio_agent.src.performance_stats import newey_west_standard_error

    values = pd.Series(ic, dtype=float).dropna().to_numpy()
    lags = overlap_lags(horizon, stride)
    if values.size < 2:
        return ICSummary(
            mean=float(values.mean()) if values.size else 0.0,
            std=0.0, icir=0.0, t_stat=0.0, p_value=1.0,
            positive_share=float((values > 0).mean()) if values.size else 0.0,
            n_dates=int(values.size), newey_west_lags=lags,
        )

    mean = float(values.mean())
    std = float(values.std(ddof=1))
    # ICIR is the plain mean/std — the conventional stability ratio, reported
    # unadjusted so it stays comparable with the literature. The overlap
    # adjustment belongs on the t-statistic, which is what significance is read
    # off, and applying it in both places would double-count it.
    icir = _ratio(mean, std)

    standard_error = newey_west_standard_error(values, lags)
    t_stat = _ratio(mean, standard_error)
    if math.isinf(t_stat):
        # Zero dispersion around a non-zero mean. The limit of the t-statistic
        # is infinite, so the p-value is zero — reporting the degenerate case
        # as t=0, p=1 would call the *strongest* possible evidence
        # insignificant, which is exactly backwards. It shows up the moment
        # anyone scores an oracle signal, whose IC is 1.0 on every date.
        p_value = 0.0
    elif t_stat == 0.0:
        p_value = 1.0
    else:
        # Normal rather than Student-t: the Newey–West estimator is itself a
        # large-sample result, so a t distribution's extra tail would be false
        # precision on top of an asymptotic approximation.
        from scipy.stats import norm

        p_value = float(2.0 * norm.sf(abs(t_stat)))

    return ICSummary(
        mean=mean, std=std, icir=icir, t_stat=float(t_stat), p_value=p_value,
        positive_share=float((values > 0).mean()),
        n_dates=int(values.size), newey_west_lags=lags,
    )


@dataclass(frozen=True)
class BucketAnalysis:
    """Mean forward return per score bucket, and whether it climbs.

    `monotonicity` is the metric that catches a signal driven entirely by its
    extreme tail — a very common shape, and an expensive one to mistake for
    breadth, because a book that holds the top two deciles gets none of the
    return that only exists in the top 2%.
    """

    n_buckets: int
    mean_returns: List[float]
    counts: List[int]
    spread: float
    monotonicity: float
    monotone_steps: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_buckets": self.n_buckets,
            "bucket_mean_returns": list(self.mean_returns),
            "bucket_counts": list(self.counts),
            "spread": self.spread,
            "monotonicity": self.monotonicity,
            "monotone_steps": self.monotone_steps,
        }


def assign_buckets(scores: Sequence[float], n_buckets: int) -> np.ndarray:
    """Bucket one date's names by score, 0 = lowest.

    Uses the percentile rank rather than `pd.qcut` on the raw values: a score
    with heavy ties (most rule-based screens emit one value for everything they
    reject) makes `qcut` raise on duplicate bin edges, where average-ranking
    spreads the tied block evenly and keeps the date usable.
    """
    percentile = cross_sectional_percentile(scores)
    bucket = np.floor(percentile * n_buckets).astype(int)
    return np.clip(bucket, 0, n_buckets - 1)


def bucket_analysis(
    panel: pd.DataFrame,
    n_buckets: int = 10,
    min_names: int = MIN_CROSS_SECTION_NAMES,
) -> BucketAnalysis:
    """Average forward return by score bucket, equal-weighting dates.

    Dates thinner than `n_buckets` names are skipped: bucketing eight names
    into ten deciles produces empty buckets whose "mean return" is an artifact
    of which decile happened to be occupied.
    """
    panel = validate_panel(panel)
    if n_buckets < 2:
        raise ValueError(f"n_buckets must be at least 2, got {n_buckets}")

    per_date: List[np.ndarray] = []
    counts = np.zeros(n_buckets, dtype=int)

    threshold = max(min_names, n_buckets)
    for _, group in panel.groupby("date", sort=True):
        if len(group) < threshold:
            continue
        bucket = assign_buckets(group["score"].to_numpy(), n_buckets)
        returns = group["forward_return"].to_numpy(dtype=float)
        means = np.full(n_buckets, np.nan)
        for index in range(n_buckets):
            mask = bucket == index
            if mask.any():
                means[index] = float(returns[mask].mean())
                counts[index] += int(mask.sum())
        per_date.append(means)

    if not per_date:
        return BucketAnalysis(
            n_buckets=n_buckets, mean_returns=[float("nan")] * n_buckets,
            counts=[0] * n_buckets, spread=0.0, monotonicity=0.0, monotone_steps=0.0,
        )

    stacked = np.vstack(per_date)
    # A bucket can be unoccupied on *every* date when the score is heavily
    # tied — a rule-based screen emitting one floor value for half the universe
    # pushes that whole block to an average rank in the middle, leaving the
    # bottom bucket empty. That is a real property of the signal, reported as
    # NaN with a count of zero rather than warned about and averaged away.
    occupied = np.isfinite(stacked).any(axis=0)
    mean_returns = np.full(n_buckets, np.nan)
    if occupied.any():
        mean_returns[occupied] = np.nanmean(stacked[:, occupied], axis=0)

    finite = np.isfinite(mean_returns)
    if finite.sum() < 2:
        return BucketAnalysis(
            n_buckets=n_buckets, mean_returns=[float(v) for v in mean_returns],
            counts=[int(c) for c in counts], spread=0.0,
            monotonicity=0.0, monotone_steps=0.0,
        )

    spread = float(mean_returns[finite][-1] - mean_returns[finite][0])
    monotonicity = float(
        pd.Series(np.flatnonzero(finite)).corr(
            pd.Series(mean_returns[finite]), method="spearman"
        )
    )
    steps = np.diff(mean_returns[finite])
    monotone_steps = float((steps > 0).mean()) if steps.size else 0.0

    return BucketAnalysis(
        n_buckets=n_buckets,
        mean_returns=[float(v) for v in mean_returns],
        counts=[int(c) for c in counts],
        spread=spread,
        monotonicity=0.0 if math.isnan(monotonicity) else monotonicity,
        monotone_steps=monotone_steps,
    )


def directional_hit_rate(
    panel: pd.DataFrame, min_names: int = MIN_CROSS_SECTION_NAMES
) -> float:
    """Share of observations where the relative call was directionally right.

    "Relative" is load-bearing. A 0–100 canonical score carries no absolute
    prediction, so the sign of the score says nothing; the sign of the score
    *minus the day's median* says the model expects this name to beat the
    field. That is checked against the return minus the day's mean.
    """
    panel = validate_panel(panel)
    hits = 0
    total = 0
    for _, group in panel.groupby("date", sort=True):
        if len(group) < min_names:
            continue
        scores = group["score"].to_numpy(dtype=float)
        returns = group["forward_return"].to_numpy(dtype=float)
        if np.ptp(scores) <= 0.0:
            continue
        predicted = np.sign(scores - np.median(scores))
        realized = np.sign(returns - returns.mean())
        graded = predicted != 0.0
        hits += int((predicted[graded] == realized[graded]).sum())
        total += int(graded.sum())
    return hits / total if total else 0.0


@dataclass(frozen=True)
class ErrorSummary:
    """Where the forecast is wrong, on a scale that compares across strategies.

    The error is `predicted percentile - realized percentile`, both taken
    within the date, so it lives in (-1, 1) and means the same thing for a
    0–100 screen score and a [-1, 1] network output. A raw `y - y_hat` would
    not: the two scores are not on the same scale and never were.
    """

    mean_abs_error: float
    quantiles: Dict[str, float]
    mean_abs_error_by_bucket: List[float]
    n_observations: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_abs_rank_error": self.mean_abs_error,
            "rank_error_quantiles": dict(self.quantiles),
            "mean_abs_rank_error_by_bucket": list(self.mean_abs_error_by_bucket),
            "n_observations": self.n_observations,
        }


def rank_error_summary(
    panel: pd.DataFrame,
    n_buckets: int = 10,
    min_names: int = MIN_CROSS_SECTION_NAMES,
) -> ErrorSummary:
    """Distribution of the rank error, overall and by predicted bucket.

    The per-bucket breakdown is the part worth reading: a model whose error is
    flat across buckets is uniformly mediocre, and one whose error concentrates
    in the top bucket is wrong exactly where a long-only book would act on it.
    """
    panel = validate_panel(panel)
    errors: List[np.ndarray] = []
    buckets: List[np.ndarray] = []

    for _, group in panel.groupby("date", sort=True):
        if len(group) < min_names:
            continue
        predicted = cross_sectional_percentile(group["score"].to_numpy())
        realized = cross_sectional_percentile(group["forward_return"].to_numpy())
        errors.append(predicted - realized)
        buckets.append(assign_buckets(group["score"].to_numpy(), n_buckets))

    if not errors:
        return ErrorSummary(
            mean_abs_error=0.0, quantiles={}, mean_abs_error_by_bucket=[], n_observations=0
        )

    error = np.concatenate(errors)
    bucket = np.concatenate(buckets)
    absolute = np.abs(error)

    quantiles = {
        f"p{int(q * 100):02d}": float(np.quantile(error, q))
        for q in (0.05, 0.25, 0.50, 0.75, 0.95)
    }
    by_bucket = [
        float(absolute[bucket == index].mean()) if (bucket == index).any() else float("nan")
        for index in range(n_buckets)
    ]

    return ErrorSummary(
        mean_abs_error=float(absolute.mean()),
        quantiles=quantiles,
        mean_abs_error_by_bucket=by_bucket,
        n_observations=int(error.size),
    )


def score_dispersion(
    panel: pd.DataFrame, min_names: int = MIN_CROSS_SECTION_NAMES
) -> float:
    """Average share of a date's scores that are distinct values.

    Reported because a near-zero IC has two very different causes and this
    tells them apart. A strategy that emits one floor value for everything it
    rejects has almost no cross-section left to rank, so its IC is low because
    it made few claims — not because the claims it made were wrong. Without
    this number the two are indistinguishable in the results table.
    """
    panel = validate_panel(panel)
    shares: List[float] = []
    for _, group in panel.groupby("date", sort=True):
        if len(group) < min_names:
            continue
        shares.append(group["score"].nunique() / len(group))
    return float(np.mean(shares)) if shares else 0.0


def signal_decay(
    panel_by_horizon: Dict[int, pd.DataFrame],
    min_names: int = MIN_CROSS_SECTION_NAMES,
) -> pd.Series:
    """Mean rank IC at each horizon, for reading how fast the edge dies.

    A signal whose IC is flat from 1 to 20 days is being carried by something
    slow-moving; one that halves by day 3 has to be traded at a cost the
    platform's friction assumptions may not support. Both are decisions the
    single-horizon number cannot inform.
    """
    return pd.Series(
        {
            horizon: float(rank_ic_series(panel, min_names).mean())
            for horizon, panel in sorted(panel_by_horizon.items())
        },
        dtype=float,
    )
