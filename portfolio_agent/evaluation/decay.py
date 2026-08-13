"""How long an edge lasts, measured rather than assumed.

IC as a function of horizon decides two things nothing else answers.

**Rebalancing frequency, and therefore whether the edge survives costs.** A
signal whose IC peaks at one day and is gone by five has to be traded daily to
capture anything, and daily turnover on Indian mid-caps costs far more than a
0.03 IC is worth. The single-horizon number cannot tell you that; it reports a
level and says nothing about the shape.

**Whether the signal is real.** A genuine slow signal — value, quality,
long-formation momentum — decays gradually, because the mispricing it reads
takes weeks to correct. A curve that spikes at one day and collapses is usually
microstructure: bid-ask bounce, a stale close, or a feature that is partly
tomorrow's information leaking through a timestamp. The shape distinguishes
them and the level does not.

Scoring happens once
--------------------
The score a strategy emits on a date does not depend on the horizon it will be
scored against — only the label does. So a six-horizon curve costs one pass
over the universe, not six, and every horizon is measured on *identical*
scores. That second part matters more than the speed: two horizons scored in
separate runs could differ by anything that moved between them.

Every horizon is scored on the widest set of dates it can support, rather than
on the intersection with the longest one. Restricting a 1-day IC to the dates
where a 21-day label also exists would drop the most recent month from every
point on the curve, for no reason other than the longest horizon's reach.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import MIN_CROSS_SECTION_NAMES, ICSummary, rank_ic_series, summarize_ic

logger = logging.getLogger(__name__)

#: Horizons the curve is evaluated at by default: daily out to a month. Chosen
#: to straddle the frequencies a decision actually gets made at — 1 to 3 days is
#: a costly rebalance, 5 to 10 is a normal one, 21 is monthly.
DEFAULT_HORIZONS = (1, 2, 3, 5, 10, 21)


@dataclass(frozen=True)
class DecayPoint:
    """One horizon's IC."""

    horizon: int
    ic: ICSummary
    n_observations: int

    def to_dict(self) -> Dict[str, Any]:
        return {"horizon": self.horizon, "n_observations": self.n_observations,
                **self.ic.to_dict()}


@dataclass(frozen=True)
class DecayCurve:
    """IC against horizon, and what its shape implies."""

    strategy: str
    points: List[DecayPoint] = field(default_factory=list)
    stride: int = 1

    def to_frame(self) -> pd.DataFrame:
        if not self.points:
            return pd.DataFrame()
        return pd.DataFrame([point.to_dict() for point in self.points])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "stride": self.stride,
            "points": [point.to_dict() for point in self.points],
            "peak_horizon": self.peak_horizon(),
            "half_life": self.half_life(),
        }

    def peak_horizon(self) -> Optional[int]:
        """Horizon with the largest mean IC, or None when nothing was measured."""
        if not self.points:
            return None
        return max(self.points, key=lambda p: p.ic.mean).horizon

    def half_life(self) -> Optional[float]:
        """Horizon at which IC first falls to half its peak, interpolated.

        None when it never does within the horizons measured — which is itself
        the answer for a slow signal, and is reported as such rather than as a
        number extrapolated past the data.
        """
        if len(self.points) < 2:
            return None
        peak = max(self.points, key=lambda p: p.ic.mean)
        if peak.ic.mean <= 0:
            return None
        target = peak.ic.mean / 2.0

        after = [p for p in self.points if p.horizon > peak.horizon]
        previous = peak
        for point in after:
            if point.ic.mean <= target:
                # Linear interpolation between the bracketing horizons. Crude,
                # and honest about being crude: the alternative is fitting an
                # exponential to six points, which reports a decay constant with
                # more confidence than six points support.
                span = point.horizon - previous.horizon
                drop = previous.ic.mean - point.ic.mean
                if drop <= 0:
                    return float(point.horizon)
                return float(previous.horizon + span * (previous.ic.mean - target) / drop)
            previous = point
        return None

    def shape(self) -> str:
        """A one-line reading of the curve, for someone who wants the verdict.

        Deliberately conservative. These are heuristics over six points, and
        they are phrased as what the curve looks like rather than as what the
        signal is.
        """
        if len(self.points) < 2:
            return "too few horizons to describe a shape"
        peak = max(self.points, key=lambda p: p.ic.mean)
        if peak.ic.mean <= 0:
            return "no positive IC at any horizon measured"

        first, last = self.points[0], self.points[-1]
        half_life = self.half_life()

        if peak.horizon == first.horizon and last.ic.mean < peak.ic.mean * 0.25:
            return (
                f"front-loaded: IC peaks at {peak.horizon}d and is mostly gone by "
                f"{last.horizon}d. Capturing it needs turnover at that frequency, "
                f"and a fast peak is as often microstructure as it is signal"
            )
        if half_life is None:
            return (
                f"slow: IC peaks at {peak.horizon}d and has not halved by "
                f"{last.horizon}d, so a low-turnover rebalance keeps most of it"
            )
        return (
            f"IC peaks at {peak.horizon}d with a half-life near {half_life:.0f}d; "
            f"rebalancing much slower than that gives up most of the edge"
        )

    def render(self) -> str:
        lines = [
            f"Signal decay — {self.strategy}",
            "=" * 62,
            f"  {'horizon':>8}{'mean IC':>12}{'ICIR':>9}{'t':>8}{'p':>10}{'obs':>10}",
        ]
        scale = max((abs(p.ic.mean) for p in self.points), default=0.0)
        for point in self.points:
            bar = ""
            if scale > 0 and np.isfinite(point.ic.mean):
                filled = int(round(abs(point.ic.mean) / scale * 20))
                bar = ("+" if point.ic.mean >= 0 else "-") * max(filled, 1)
            lines.append(
                f"  {point.horizon:>7}d{point.ic.mean:>+12.4f}{point.ic.icir:>9.2f}"
                f"{point.ic.t_stat:>8.2f}{point.ic.p_value:>10.4f}"
                f"{point.n_observations:>10,}  {bar}"
            )
        lines += ["", f"  {self.shape()}"]
        return "\n".join(lines)


def decay_from_panel(
    panel: pd.DataFrame,
    horizons: Sequence[int],
    *,
    strategy: str = "signal",
    primary_horizon: Optional[int] = None,
    stride: int = 1,
    min_names: int = MIN_CROSS_SECTION_NAMES,
) -> DecayCurve:
    """Build a decay curve from one multi-horizon panel.

    Args:
        panel: Panel carrying `score` and a `forward_return_<h>` column per
            horizon, plus `forward_return` for the primary one.
        horizons: Horizons to report, in order.
        strategy: Name for the header.
        primary_horizon: Which horizon the plain `forward_return` column holds.
        stride: Sessions between observations, for the Newey–West lag.
        min_names: Minimum cross-section width.

    Raises:
        ValueError: If a requested horizon has no column. Silently omitting one
            would produce a curve with a hole that reads as a decay.
    """
    points: List[DecayPoint] = []

    for horizon in horizons:
        if primary_horizon is not None and horizon == primary_horizon:
            column = "forward_return"
        else:
            column = f"forward_return_{horizon}"
        if column not in panel.columns:
            raise ValueError(
                f"panel has no column {column!r} for horizon {horizon}. Build it "
                f"with build_forecast_panel(..., extra_horizons={list(horizons)})"
            )

        block = panel[["date", "symbol", "score", column]].rename(
            columns={column: "forward_return"}
        )
        ic = rank_ic_series(block, min_names)
        points.append(DecayPoint(
            horizon=int(horizon),
            ic=summarize_ic(ic, horizon, stride),
            n_observations=int(block.dropna().shape[0]),
        ))

    return DecayCurve(strategy=strategy, points=points, stride=stride)


def decay_curve(
    app_config: Any,
    strategy: Any,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    universe: Optional[Sequence[str]] = None,
    universe_size: Optional[int] = None,
    snapshot: Optional[str] = None,
    stride: int = 1,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_history: int = 252,
    min_names: int = MIN_CROSS_SECTION_NAMES,
    max_dates: Optional[int] = None,
    use_benchmark: bool = True,
) -> DecayCurve:
    """Measure a registered strategy's IC at several horizons, in one pass."""
    from .harness import DEFAULT_MIN_HISTORY, _resolve_strategy, build_forecast_panel

    resolved, name = _resolve_strategy(app_config, strategy)
    horizons = sorted({int(h) for h in horizons})
    if not horizons:
        raise ValueError("decay_curve needs at least one horizon")

    if universe is None:
        from portfolio_agent.src.universe import MEASUREMENT_PURPOSE
        from portfolio_agent.training.universe import resolve_universe

        snap = resolve_universe(
            app_config, snapshot=snapshot, size=universe_size, name=f"decay:{name}",
            purpose=MEASUREMENT_PURPOSE,
        )
        universe = list(snap.tickers)

    primary = horizons[0]
    panel = build_forecast_panel(
        app_config, resolved, universe,
        horizon=primary, extra_horizons=horizons[1:], stride=stride,
        start_date=start_date, end_date=end_date, min_history=min_history,
        min_names=min_names, max_dates=max_dates, use_benchmark=use_benchmark,
    )
    return decay_from_panel(
        panel, horizons, strategy=name, primary_horizon=primary,
        stride=stride, min_names=min_names,
    )
