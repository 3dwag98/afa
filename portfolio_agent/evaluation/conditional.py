"""When a signal paid, not just whether it did.

A pooled IC of 0.03 made of +0.09 in falling markets and −0.02 in rising ones
is a different object from a steady 0.03, and only the split tells them apart.
The distinction is not academic for this platform: the low-risk anomaly is the
clearest case, where 2025 Asian work finds betting-against-beta concentrated in
downturns, and a single pooled number would describe neither state.

Two conditioners, and the difference between them matters
---------------------------------------------------------
**`realized`** splits on the market's return *over the label horizon* — the
same window the forward return is measured over. This answers "when did the
signal pay?", which is an attribution question, and the answer is not tradable:
on the decision date nobody knows which bucket the date will land in.

**`trailing`** splits on the market's state as of the decision date. This is
tradable — a strategy really can condition on it — and it is a weaker
conditioner, because the trailing state predicts the forward one only loosely.

The two are reported the same way and mean different things, so the conditioner
travels into every result and into the notes. Reading a `realized` split as
though it were `trailing` is the mistake this module is most likely to cause,
and it would turn an attribution into an imaginary timing rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import (
    MIN_CROSS_SECTION_NAMES,
    BucketAnalysis,
    ICSummary,
    bucket_analysis,
    rank_ic_series,
    summarize_ic,
    validate_panel,
)

#: How a date is assigned to a market state.
CONDITIONERS = ("realized", "trailing")

#: State labels. Two rather than three: a "flat" bucket sounds more careful but
#: needs a threshold nobody can justify, and it thins both real buckets to buy
#: a third whose interpretation is "the market did approximately nothing".
UP, DOWN = "up", "down"

#: Sessions in the trailing window that decides the `trailing` state. One
#: quarter — long enough not to flip on a single bad week, short enough to be
#: about the current regime rather than the year.
DEFAULT_TRAILING_WINDOW = 63


@dataclass(frozen=True)
class ConditionalIC:
    """One signal's IC, split by the state of the market."""

    conditioner: str
    by_state: Dict[str, ICSummary]
    n_dates: Dict[str, int]
    buckets: Dict[str, BucketAnalysis] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def gap(self) -> float:
        """Down-market IC minus up-market IC.

        Signed so that a *positive* gap means the signal works better when the
        market falls — the direction the low-risk anomaly is claimed to have,
        and the one a long-only book cares about most.
        """
        up = self.by_state.get(UP)
        down = self.by_state.get(DOWN)
        if up is None or down is None:
            return float("nan")
        return down.mean - up.mean

    @property
    def is_conditional(self) -> bool:
        """Whether the pooled IC would describe neither state.

        Two ways that happens, and the second is the one worth having:

        1. Significant in one state and not the other — the shape the low-risk
           anomaly is claimed to have.
        2. **Significant in both, with opposite signs.** This is the stronger
           case and the easier one to miss, because both halves look healthy in
           isolation. A signal at +0.32 in falling markets and -0.35 in rising
           ones pools to roughly zero, and "no skill" is the one description
           that is wrong about both halves.

        A crude test and deliberately so: comparing two Newey-West
        t-statistics is not a test of their difference, and pretending
        otherwise would put a p-value on a claim this cannot support. It is a
        flag, not a p-value.
        """
        up = self.by_state.get(UP)
        down = self.by_state.get(DOWN)
        if up is None or down is None:
            return False
        if up.significant != down.significant:
            return True
        return (
            up.significant
            and down.significant
            and np.sign(up.mean) != np.sign(down.mean)
        )

    @property
    def signs_disagree(self) -> bool:
        """Whether the signal points opposite ways in the two states."""
        up = self.by_state.get(UP)
        down = self.by_state.get(DOWN)
        if up is None or down is None:
            return False
        return bool(np.sign(up.mean) != np.sign(down.mean))

    def to_dict(self) -> Dict[str, Any]:
        document: Dict[str, Any] = {
            "conditioner": self.conditioner,
            "conditional_ic_gap": self.gap,
            "conditional_is_state_dependent": self.is_conditional,
        }
        for state, summary in self.by_state.items():
            document[f"mean_ic_{state}"] = summary.mean
            document[f"icir_{state}"] = summary.icir
            document[f"t_stat_{state}"] = summary.t_stat
            document[f"p_value_{state}"] = summary.p_value
            document[f"n_dates_{state}"] = self.n_dates.get(state, 0)
        for state, analysis in self.buckets.items():
            document[f"spread_{state}"] = analysis.spread
        return document


def market_return_by_date(panel: pd.DataFrame) -> pd.Series:
    """The cross-section's mean forward return on each date.

    The market proxy, defined the same way `features/market_relative.py`
    defines it — an equal-weighted mean of whatever is in the universe. Using
    the panel's own labels rather than a separate index series means the
    conditioner and the thing being conditioned are measured over exactly the
    same window and the same names, which is what makes the split clean.
    """
    clean = validate_panel(panel)
    return clean.groupby("date")["forward_return"].mean().sort_index()


def realized_states(panel: pd.DataFrame) -> pd.Series:
    """Label each date by the market's return over the label horizon.

    **Attribution, not timing.** On the decision date nobody knows which
    bucket the date will land in, so a result split this way says when the
    signal paid and never how to trade it.
    """
    return market_return_by_date(panel).apply(lambda r: UP if r >= 0 else DOWN)


def trailing_states(
    market: pd.Series,
    dates: Sequence[Any],
    window: int = DEFAULT_TRAILING_WINDOW,
) -> pd.Series:
    """Label each date by the market's trailing return, as of that date.

    Tradable, and weaker for it: the trailing state predicts the forward one
    only loosely, so a split on it will show a smaller gap than a `realized`
    split of the same data even when the underlying conditionality is real.

    Args:
        market: Daily market returns, indexed by date. Needs to cover the
            window *before* the first evaluated date, or early dates are
            unlabelled rather than labelled from a partial window.
        dates: The dates to label.
        window: Trailing sessions.

    Returns:
        A state per labellable date. Dates without a full trailing window are
        omitted, so a caller can see how many were dropped.
    """
    series = pd.Series(market).sort_index()
    trailing = series.rolling(window, min_periods=window).sum()

    labels: Dict[Any, str] = {}
    for date in sorted(dates):
        # `<= date` and not `< date`: the trailing window ends at the decision
        # date, matching T19's convention that a decision on D may read D.
        usable = trailing.loc[trailing.index <= date].dropna()
        if usable.empty:
            continue
        labels[date] = UP if float(usable.iloc[-1]) >= 0 else DOWN
    return pd.Series(labels).sort_index()


def conditional_ic(
    panel: pd.DataFrame,
    *,
    horizon: int,
    conditioner: str = "realized",
    market: Optional[pd.Series] = None,
    trailing_window: int = DEFAULT_TRAILING_WINDOW,
    min_names: int = MIN_CROSS_SECTION_NAMES,
    stride: int = 1,
    n_buckets: int = 10,
    min_dates_per_state: int = 20,
) -> ConditionalIC:
    """Split a signal's IC and decile spread by the state of the market.

    Args:
        panel: Tidy `(date, symbol, score, forward_return)` rows.
        horizon: Label horizon in sessions, for the Newey-West lag.
        conditioner: `"realized"` (attribution, the default) or `"trailing"`
            (tradable). See the module docstring — the two mean different
            things and reading one as the other turns an attribution into an
            imaginary timing rule.
        market: Daily market returns. Required by `"trailing"`; ignored by
            `"realized"`, which uses the panel's own cross-sectional mean.
        trailing_window: Sessions in the trailing state, when conditioning on it.
        min_names: Minimum cross-section width for a date to count.
        stride: Sessions between observations, for the Newey-West lag.
        n_buckets: Buckets for the per-state decile spread.
        min_dates_per_state: A state with fewer dates than this is reported but
            flagged, because a Newey-West t on a dozen dates is not evidence.

    Returns:
        A `ConditionalIC`.

    Raises:
        ValueError: On an unknown conditioner, or `"trailing"` without a market
            series. Falling back to `"realized"` would silently answer a
            different question from the one asked.
    """
    if conditioner not in CONDITIONERS:
        raise ValueError(
            f"Unknown conditioner {conditioner!r}. Available: {list(CONDITIONERS)}"
        )

    clean = validate_panel(panel)
    if clean.empty:
        return ConditionalIC(
            conditioner=conditioner, by_state={}, n_dates={},
            notes=["The panel was empty, so nothing was split."],
        )

    if conditioner == "realized":
        states = realized_states(clean)
    else:
        if market is None:
            raise ValueError(
                "conditioner='trailing' needs a `market` return series. "
                "Falling back to 'realized' would silently answer a different "
                "question: 'trailing' is tradable and 'realized' is attribution."
            )
        states = trailing_states(
            market, clean["date"].unique(), window=trailing_window
        )

    by_state: Dict[str, ICSummary] = {}
    counts: Dict[str, int] = {}
    buckets: Dict[str, BucketAnalysis] = {}
    notes: List[str] = []

    for state in (UP, DOWN):
        dates = set(states[states == state].index)
        if not dates:
            continue
        subset = clean[clean["date"].isin(dates)]
        if subset.empty:
            continue

        ic = rank_ic_series(subset, min_names)
        by_state[state] = summarize_ic(ic, horizon, stride)
        counts[state] = int(subset["date"].nunique())
        buckets[state] = bucket_analysis(subset, n_buckets, min_names)

        if counts[state] < min_dates_per_state:
            notes.append(
                f"Only {counts[state]} {state}-market date(s) — a Newey-West "
                f"t-statistic on that many is not evidence, and the "
                f"{state}-market figures below should be read as descriptive."
            )

    unlabelled = clean["date"].nunique() - sum(counts.values())
    if unlabelled > 0:
        notes.append(
            f"{unlabelled} date(s) could not be assigned a market state and "
            f"are in neither column."
        )

    if conditioner == "realized":
        notes.append(
            "Conditioned on the market's return *over the label horizon*. This "
            "is attribution — it says when the signal paid — and is not "
            "tradable: on the decision date nobody knows which bucket the date "
            "will land in."
        )
    else:
        notes.append(
            f"Conditioned on the market's trailing {trailing_window}-session "
            f"return as of the decision date. Tradable, and a weaker "
            f"conditioner than 'realized' for that reason."
        )

    return ConditionalIC(
        conditioner=conditioner, by_state=by_state, n_dates=counts,
        buckets=buckets, notes=notes,
    )


def conditional_notes(result: ConditionalIC) -> List[str]:
    """Sentences for the report that say what the split found.

    Written here rather than in a renderer because the caveats are properties
    of the calculation, and a caveat that lives in a template goes missing the
    first time someone writes a second template.
    """
    lines = list(result.notes)
    up = result.by_state.get(UP)
    down = result.by_state.get(DOWN)
    if up is None or down is None:
        return lines

    lines.append(
        f"IC in falling markets {down.mean:+.4f} "
        f"(t={down.t_stat:+.2f}, n={result.n_dates.get(DOWN, 0)}); "
        f"in rising markets {up.mean:+.4f} "
        f"(t={up.t_stat:+.2f}, n={result.n_dates.get(UP, 0)}); "
        f"gap {result.gap:+.4f}."
    )
    if result.signs_disagree and up.significant and down.significant:
        lines.append(
            "The signal points opposite ways in the two states and clears "
            "significance in both. A pooled IC averages them toward zero and "
            "reports 'no skill', which is the one description that is wrong "
            "about both halves."
        )
    elif result.is_conditional:
        stronger = DOWN if down.significant else UP
        lines.append(
            f"The signal clears significance in {stronger} markets and not in "
            f"the other, so a pooled IC describes neither state."
        )
    if result.is_conditional:
        lines.append(
            "This compares two t-statistics rather than testing their "
            "difference — a flag, not a p-value."
        )
    return lines
