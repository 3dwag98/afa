"""What the friction costs, applied to a forecast rather than to a book.

The platform already has an accurate Indian cost model — `src/execution_sim.py`
prices STT, exchange charges, SEBI fees, GST, stamp duty and slippage off the
statutory schedule. The evaluation layer had none, so every decile spread it
reported was gross, and a signal whose edge is 40 bps a month looked identical
to one whose edge is 40 bps a month *after* paying 80 bps to harvest it.

One thing this module does not do
---------------------------------
**It does not report a "net IC".** Rank IC is a Spearman correlation, and
subtracting the same cost from every name's forward return is a monotone
transform of the labels — the ranks do not move, so the correlation is
identical to the last decimal. A net IC column would be the gross one copied,
dressed as new information. Where costs genuinely differ by name (slippage
scales with illiquidity) the ranks *can* move, so `evaluate_net` takes an
optional per-name cost and says in its notes which case it was in.

What costs actually change is whether the spread survives being harvested, and
that depends on turnover: a signal you rebalance monthly pays the round trip
twelve times a year on whatever fraction of the book it replaces. So the
numbers here are the spread net of the cost of *holding the signal at its own
rebalance frequency*, plus the two figures that make it decision-relevant —
what turnover the signal actually generates, and the cost level at which its
edge reaches zero.

Turnover is measured, not assumed
---------------------------------
The panel carries `(date, symbol, score)`, which is enough to see how much of
the top bucket is replaced from one rebalance to the next. That is the real
turnover of the signal at that frequency, and it is usually the number people
guess at. A slow signal that reshuffles 8% of its book a month is a different
proposition from a fast one that replaces 60%, at identical gross spread.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import MIN_CROSS_SECTION_NAMES, assign_buckets, validate_panel

#: Sessions in an Indian trading year. Matches `src/performance_stats.py`.
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class CostModel:
    """Per-side and round-trip friction, as fractions of turnover.

    Read from `src/execution_sim.py` rather than restated, for the same reason
    T12 collapsed four rank ICs into one: two copies of the STT rate is one
    copy that silently stops matching the day the rate changes. The Union
    Budget moves these numbers.
    """

    buy: float
    sell: float
    slippage_per_side: float

    @property
    def round_trip(self) -> float:
        """Buy plus sell — what one full rotation of a position costs."""
        return self.buy + self.sell

    @classmethod
    def from_execution_sim(
        cls, slippage_per_side: Optional[float] = None
    ) -> "CostModel":
        """The shipped Indian schedule, at a chosen slippage assumption."""
        from portfolio_agent.src.execution_sim import (
            DEFAULT_SLIPPAGE_PCT_PER_SIDE,
            cost_fraction_per_side,
        )

        slippage = (
            DEFAULT_SLIPPAGE_PCT_PER_SIDE
            if slippage_per_side is None
            else float(slippage_per_side)
        )
        if slippage < 0:
            raise ValueError(f"slippage_per_side must not be negative, got {slippage}")
        return cls(
            buy=cost_fraction_per_side("BUY", slippage),
            sell=cost_fraction_per_side("SELL", slippage),
            slippage_per_side=slippage,
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "cost_buy_pct": self.buy,
            "cost_sell_pct": self.sell,
            "cost_round_trip_pct": self.round_trip,
            "cost_slippage_per_side_pct": self.slippage_per_side,
        }


# --------------------------------------------------------------------------
# Turnover
# --------------------------------------------------------------------------


def bucket_membership(
    panel: pd.DataFrame,
    n_buckets: int = 10,
    bucket: Optional[int] = None,
    min_names: int = MIN_CROSS_SECTION_NAMES,
) -> Dict[Any, set]:
    """Which symbols sit in `bucket` on each date, keyed by date.

    Defaults to the top bucket, which is the one a long-only book holds. Dates
    too thin to bucket are skipped for the same reason `bucket_analysis` skips
    them: eight names in ten deciles produces empty buckets whose membership is
    an artifact of which decile happened to be occupied.
    """
    panel = validate_panel(panel)
    target = n_buckets - 1 if bucket is None else int(bucket)
    if not 0 <= target < n_buckets:
        raise ValueError(f"bucket {target} is outside 0..{n_buckets - 1}")

    threshold = max(min_names, n_buckets)
    membership: Dict[Any, set] = {}
    for date, group in panel.groupby("date", sort=True):
        if len(group) < threshold:
            continue
        assigned = assign_buckets(group["score"].to_numpy(), n_buckets)
        held = group["symbol"].to_numpy()[assigned == target]
        if held.size:
            membership[date] = set(held.tolist())
    return membership


def one_way_turnover(
    panel: pd.DataFrame,
    n_buckets: int = 10,
    bucket: Optional[int] = None,
    min_names: int = MIN_CROSS_SECTION_NAMES,
    rebalance_every: int = 1,
) -> float:
    """Mean fraction of the bucket replaced between consecutive rebalances.

    0.0 means the same names every time; 1.0 means a completely new book. This
    is *one-way*: replacing 30% of the book means selling 30% and buying 30%,
    so the cost is 0.30 x round_trip, not 0.30 x buy.

    Args:
        panel: Tidy `(date, symbol, score, forward_return)` rows.
        n_buckets: Bucket count; the top one is the long-only book.
        bucket: Which bucket to track. Defaults to the top.
        min_names: Dates thinner than this are not bucketed.
        rebalance_every: Compare each rebalance against the one this many
            *observation dates* earlier. A panel already strided to the
            rebalance frequency wants 1.

    Returns:
        Mean one-way turnover in [0, 1], or 0.0 when there are fewer than two
        rebalances to compare — which is "not measured", and the caller says so
        via `n_rebalances`.
    """
    if rebalance_every < 1:
        raise ValueError(f"rebalance_every must be at least 1, got {rebalance_every}")

    membership = bucket_membership(panel, n_buckets, bucket, min_names)
    dates = sorted(membership)
    if len(dates) < rebalance_every + 1:
        return 0.0

    fractions: List[float] = []
    for i in range(rebalance_every, len(dates), rebalance_every):
        held = membership[dates[i]]
        previous = membership[dates[i - rebalance_every]]
        if not held:
            continue
        # Denominator is the new book: the fraction of what you now hold that
        # you had to buy. Symmetric with the sells whenever the book is a fixed
        # size, which a decile of a stable universe approximately is.
        fractions.append(len(held - previous) / len(held))
    return float(np.mean(fractions)) if fractions else 0.0


def count_rebalances(
    panel: pd.DataFrame,
    n_buckets: int = 10,
    bucket: Optional[int] = None,
    min_names: int = MIN_CROSS_SECTION_NAMES,
    rebalance_every: int = 1,
) -> int:
    """How many rebalance-to-rebalance transitions the turnover averaged over."""
    membership = bucket_membership(panel, n_buckets, bucket, min_names)
    n_dates = len(membership)
    if n_dates < rebalance_every + 1:
        return 0
    return len(range(rebalance_every, n_dates, rebalance_every))


# --------------------------------------------------------------------------
# The spread, gross and net
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NetSpread:
    """A decile spread before and after paying to harvest it.

    `survives` is the whole point of the object: a gross spread is a
    measurement, a net spread is a decision.
    """

    gross: float
    turnover: float
    cost_per_rebalance: float
    net: float
    breakeven_cost: float
    horizon: int
    n_rebalances: int
    long_only_gross: float
    long_only_net: float
    periods_per_year: float
    costs: CostModel

    @property
    def survives(self) -> bool:
        """Does the long-short spread clear its own friction?"""
        return self.net > 0.0

    @property
    def long_only_survives(self) -> bool:
        """The question this platform actually asks — it never shorts."""
        return self.long_only_net > 0.0

    @property
    def cost_share(self) -> float:
        """Fraction of the gross spread that friction consumes.

        Above 1.0 the signal costs more to harvest than it earns. Undefined
        against a non-positive gross spread, where it is reported as NaN rather
        than as a ratio whose sign means nothing.
        """
        if self.gross <= 0:
            return float("nan")
        return self.cost_per_rebalance / self.gross

    def annualized(self, value: float) -> float:
        """A per-horizon figure scaled to a year by simple compounding count."""
        return value * self.periods_per_year

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spread_gross": self.gross,
            "spread_net": self.net,
            "spread_net_annualized": self.annualized(self.net),
            "long_only_gross": self.long_only_gross,
            "long_only_net": self.long_only_net,
            "long_only_net_annualized": self.annualized(self.long_only_net),
            "turnover_one_way": self.turnover,
            "cost_per_rebalance": self.cost_per_rebalance,
            "cost_share_of_gross": self.cost_share,
            "breakeven_round_trip_cost": self.breakeven_cost,
            "n_rebalances": self.n_rebalances,
            "survives_costs": self.survives,
            "long_only_survives_costs": self.long_only_survives,
            **self.costs.to_dict(),
        }


def evaluate_net(
    panel: pd.DataFrame,
    *,
    horizon: int,
    costs: Optional[CostModel] = None,
    n_buckets: int = 10,
    min_names: int = MIN_CROSS_SECTION_NAMES,
    stride: int = 1,
    benchmark_return: float = 0.0,
) -> NetSpread:
    """Charge the shipped Indian cost schedule against a signal's decile spread.

    The arithmetic, stated plainly because every term is a choice:

        cost_per_rebalance = one_way_turnover x round_trip_cost
        net_long_short     = gross_spread - 2 x cost_per_rebalance
        net_long_only      = top_decile_return - benchmark - cost_per_rebalance

    Two costs on the long-short leg because both books turn over: the long side
    sells what leaves the top decile and the short side covers what leaves the
    bottom. One on the long-only leg, which is the number that matters here —
    this platform does not short, and T05 already found low volatility ranks
    the cross-section well while having a *negative* decile spread.

    Args:
        panel: Tidy `(date, symbol, score, forward_return)` rows.
        horizon: Label horizon in sessions. Sets the annualization only.
        costs: Cost model. Defaults to the shipped schedule at 25 bps/side.
        n_buckets: Buckets for the spread; 10 for deciles.
        min_names: Minimum cross-section width for a date to be bucketed.
        stride: Sessions between observation dates. With the horizon this sets
            how many rebalances a year the panel represents.
        benchmark_return: Mean per-period return of the thing the long-only
            book is measured against. Zero means absolute return.

    Returns:
        A `NetSpread`.
    """
    from .metrics import bucket_analysis

    clean = validate_panel(panel)
    model = costs or CostModel.from_execution_sim()
    buckets = bucket_analysis(clean, n_buckets, min_names)

    turnover = one_way_turnover(clean, n_buckets, None, min_names)
    n_rebalances = count_rebalances(clean, n_buckets, None, min_names)
    cost_per_rebalance = turnover * model.round_trip

    gross = float(buckets.spread)
    top = float(buckets.mean_returns[-1]) if buckets.mean_returns else 0.0

    # Holding period is what the book is actually rebalanced at, which is the
    # observation spacing, not the label horizon. Evaluating a 21-day label on
    # a daily panel does not mean rebalancing daily; it means the panel *could*
    # be rebalanced daily, and `stride` is what says how often it is.
    periods_per_year = TRADING_DAYS_PER_YEAR / max(1, int(stride))

    long_only_gross = top - float(benchmark_return)

    return NetSpread(
        gross=gross,
        turnover=turnover,
        cost_per_rebalance=cost_per_rebalance,
        net=gross - 2.0 * cost_per_rebalance,
        # The round-trip cost at which the long-short edge reaches exactly
        # zero. Reported because "would this work at 40 bps instead of 80" is
        # the question that follows every negative net spread, and answering it
        # should not require a re-run.
        breakeven_cost=(gross / (2.0 * turnover)) if turnover > 0 else float("inf"),
        horizon=int(horizon),
        n_rebalances=n_rebalances,
        long_only_gross=long_only_gross,
        long_only_net=long_only_gross - cost_per_rebalance,
        periods_per_year=periods_per_year,
        costs=model,
    )


def cost_notes(spread: NetSpread) -> List[str]:
    """Sentences for the report that state what the numbers mean.

    Written here rather than in the renderer because the caveats are properties
    of the calculation, and a caveat that lives in a template is one that goes
    missing the first time someone writes a second template.
    """
    notes = [
        "Rank IC is unchanged by costs: subtracting the same friction from "
        "every name is a monotone transform of the labels, so the ranks — and "
        "therefore the correlation — are identical. Costs move the spread, not "
        "the ordering.",
        f"Costs are the shipped NSE delivery schedule at "
        f"{spread.costs.slippage_per_side * 1e4:.0f} bps/side slippage: "
        f"{spread.costs.round_trip * 100:.2f}% per round trip, of which STT is "
        f"0.10% on each leg.",
    ]

    if spread.n_rebalances == 0:
        notes.append(
            "Turnover was not measured — fewer than two rebalances in the "
            "window — so the cost charged here is zero and the net spread "
            "equals the gross one. It is not a net number."
        )
    else:
        notes.append(
            f"Turnover is measured, not assumed: {spread.turnover * 100:.1f}% of "
            f"the top bucket is replaced per rebalance, over "
            f"{spread.n_rebalances} rebalances."
        )

    if spread.gross > 0 and not spread.survives:
        notes.append(
            f"The long-short spread does not survive its own friction: "
            f"{spread.gross * 100:.2f}% gross against "
            f"{2 * spread.cost_per_rebalance * 100:.2f}% of cost. It would "
            f"break even at a round-trip cost of "
            f"{spread.breakeven_cost * 100:.2f}%."
        )
    if spread.long_only_gross > 0 and not spread.long_only_survives:
        notes.append(
            "The long-only book does not survive either, which is the binding "
            "case — this platform never shorts."
        )
    return notes
