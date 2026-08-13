"""Separating stock selection from factor tilt.

Why this matters here specifically
----------------------------------
A cross-sectional signal can score a respectable IC while making no
stock-selection claim at all — it need only load on something that happened to
work. Indian momentum concentrates hard by sector: IT through 2020–21, PSU
banks through 2022–23. A momentum signal in those windows is substantially a
sector bet wearing a signal's clothes, and the raw IC cannot tell you which it
is. The gap between raw IC and IC-of-the-residual is the part that is genuinely
selection.

The method
----------
On each date, regress the score cross-sectionally on an intercept plus the
chosen exposures and keep the residual. The residual is by construction
uncorrelated with those exposures *within that date*, so any IC it retains
cannot be coming from them. Then score the residual exactly as the raw signal
was scored, and report both.

Both, not one. Replacing the raw number with the neutralized one throws away
the comparison that carries the information — a signal whose IC survives
neutralization intact and a signal whose IC vanishes are different objects, and
only the pair distinguishes them.

Exposures, and one honest gap
-----------------------------
* **Market beta**, from a rolling regression of the stock's return on the
  universe composite. Causal by construction: the window ends at the decision
  date.
* **Size**, which is where the gap is. Size means log market capitalization,
  which needs shares outstanding, which the platform does not have — and for
  Indian equities the *right* figure is free-float capitalization, since
  promoter holdings run 50–75% and the total is not what trades. Log traded
  value is used instead. It correlates with size and is emphatically not size;
  every result carrying it says so in the output, not only in this docstring.
* **Sector**, which needs a sector map that does not ship with the repository
  (finding A8). Supported when a caller supplies one, and reported as absent
  when they do not — because "sector-neutralized" silently meaning "not
  sector-neutralized" is the failure this whole module exists to prevent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import MIN_CROSS_SECTION_NAMES, ICSummary, rank_ic_series, summarize_ic

logger = logging.getLogger(__name__)

#: Sessions in the rolling beta window. One year: long enough for the estimate
#: to be stable, short enough to track a name whose beta genuinely moved.
DEFAULT_BETA_WINDOW = 252

#: Sessions in the traded-value window behind the size proxy. A single day's
#: turnover is dominated by whatever happened that day; a quarter is not.
DEFAULT_SIZE_WINDOW = 60

#: Fewest rows a rolling window may be reduced to. Below about twenty
#: observations a beta estimate is noise, so a heavily-strided panel gets a
#: floor rather than a two-row regression.
MIN_WINDOW_ROWS = 20


def _rows_for(sessions: int, stride: int) -> int:
    """Convert a window stated in sessions into panel rows at this stride."""
    return max(MIN_WINDOW_ROWS, int(round(sessions / max(1, stride))))


#: Stated in every result that uses it. The substitution is real and the output
#: has to carry it — an acceptance criterion, and the right one.
SIZE_PROXY_NOTE = (
    "size is a proxy: log rolling-median traded value, not log market cap. "
    "Market cap needs shares outstanding, which this platform does not have, "
    "and for Indian equities the correct figure is free float — promoter "
    "holdings run 50-75%, so total capitalisation is not what trades."
)


@dataclass(frozen=True)
class NeutralizationResult:
    """Raw and residual IC side by side, plus what was neutralized against."""

    raw: ICSummary
    neutralized: ICSummary
    exposures: List[str]
    n_dates_neutralized: int
    n_dates_skipped: int
    notes: List[str] = field(default_factory=list)

    @property
    def retained(self) -> float:
        """Share of the raw IC that survives neutralization.

        The headline number: 1.0 means the exposures explained none of it,
        0.0 means all of it. Undefined when the raw IC is ~0, which is reported
        as 0.0 rather than as a ratio of two noise terms.
        """
        if abs(self.raw.mean) <= 1e-12:
            return 0.0
        return self.neutralized.mean / self.raw.mean

    @property
    def explained(self) -> float:
        """Share of the raw IC the exposures account for."""
        return 1.0 - self.retained

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exposures": list(self.exposures),
            "raw_mean_ic": self.raw.mean,
            "raw_t_stat": self.raw.t_stat,
            "neutralized_mean_ic": self.neutralized.mean,
            "neutralized_t_stat": self.neutralized.t_stat,
            "neutralized_p_value": self.neutralized.p_value,
            "retained": self.retained,
            "explained": self.explained,
            "n_dates_neutralized": self.n_dates_neutralized,
            "n_dates_skipped": self.n_dates_skipped,
            "notes": list(self.notes),
        }

    def render(self) -> str:
        lines = [
            f"Neutralized against: {', '.join(self.exposures) or '(nothing)'}",
            "",
            f"{'':<16}{'mean IC':>10}{'t':>8}{'p':>10}",
            f"  {'raw':<14}{self.raw.mean:>+10.4f}{self.raw.t_stat:>8.2f}"
            f"{self.raw.p_value:>10.4f}",
            f"  {'neutralized':<14}{self.neutralized.mean:>+10.4f}"
            f"{self.neutralized.t_stat:>8.2f}{self.neutralized.p_value:>10.4f}",
            "",
            f"  {self.retained:.0%} of the raw IC is stock selection; "
            f"{self.explained:.0%} is the exposures.",
        ]
        if self.n_dates_neutralized and self.neutralized.n_dates == 0:
            lines.append(
                f"  The exposures explained the signal entirely on all "
                f"{self.n_dates_neutralized} date(s): nothing rankable was left "
                f"to correlate, so there is no residual IC to report."
            )
        if self.n_dates_skipped:
            lines.append(
                f"  {self.n_dates_skipped} date(s) skipped for want of usable exposures."
            )
        for note in self.notes:
            lines.append(f"  Note: {note}")
        return "\n".join(lines)


#: Below this fraction of the original dispersion, a residual is float noise
#: rather than signal, and is snapped to exactly zero. See `residualize`.
RESIDUAL_TOLERANCE = 1e-8


def residualize(
    values: np.ndarray, exposures: np.ndarray, rtol: float = RESIDUAL_TOLERANCE
) -> np.ndarray:
    """Least-squares residual of `values` on `[1, exposures]`.

    An intercept is always included, so neutralization removes the date's mean
    level as well as the exposure loadings. Without it a signal whose level
    drifts would appear to load on whichever exposure happened to have a
    non-zero mean.

    `lstsq` rather than a normal-equation solve: sector dummies are
    rank-deficient by construction (they sum to the intercept), and the
    minimum-norm solution handles that without anyone having to remember to
    drop a reference level.

    Why the tolerance is not fussiness
    ----------------------------------
    When the exposures span the signal exactly — a pure sector bet against
    sector dummies is the canonical case — the residual is float noise around
    zero. That noise is *not* random: the same arithmetic runs for every name
    in a sector, so every name in a sector gets the same 1e-17 value, and the
    residual is still a perfect sector ordering three hundred orders of
    magnitude below anything meaningful. Spearman does not care about scale, so
    it happily reports a substantial IC on it.

    Snapping a residual whose dispersion is negligible against the original's
    to exactly zero is what makes "the exposures explained all of it" report as
    no signal rather than as a rank correlation of rounding error. The check is
    on the vector, not element-wise: it is a statement about the date, and
    clipping individual elements would eat the small end of a real residual.
    """
    values = np.asarray(values, dtype=float)
    design = np.column_stack([np.ones(len(values)), np.asarray(exposures, dtype=float)])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    residual = values - design @ coefficients

    original_spread = float(np.sqrt(np.mean((values - values.mean()) ** 2)))
    residual_spread = float(np.sqrt(np.mean(residual ** 2)))
    if original_spread > 0.0 and residual_spread <= rtol * original_spread:
        return np.zeros_like(residual)
    return residual


def neutralize_panel(
    panel: pd.DataFrame,
    exposure_columns: Sequence[str],
    *,
    min_names: int = MIN_CROSS_SECTION_NAMES,
) -> tuple:
    """Replace each date's scores with their residual on the exposures.

    Args:
        panel: Tidy panel carrying `score` and every column in
            `exposure_columns`.
        exposure_columns: Exposure columns to neutralize against.
        min_names: Dates thinner than this are dropped rather than regressed —
            fitting `k` exposures across six names produces a residual that is
            mostly fitting noise.

    Returns:
        `(neutralized_panel, n_dates_used, n_dates_skipped)`.
    """
    exposure_columns = list(exposure_columns)
    missing = [c for c in exposure_columns if c not in panel.columns]
    if missing:
        raise ValueError(f"panel has no exposure column(s) {missing}")

    # A date needs more names than it has parameters, with room to spare, or
    # the residual is an artifact of the fit rather than a property of the data.
    required = max(min_names, len(exposure_columns) + 2)

    blocks: List[pd.DataFrame] = []
    used = 0
    skipped = 0

    for _, group in panel.groupby("date", sort=True):
        usable = group.dropna(subset=["score", *exposure_columns])
        if len(usable) < required:
            skipped += 1
            continue
        design = usable[exposure_columns].to_numpy(dtype=float)
        if not np.isfinite(design).all():
            skipped += 1
            continue
        block = usable.copy()
        block["score"] = residualize(usable["score"].to_numpy(dtype=float), design)
        blocks.append(block)
        used += 1

    if not blocks:
        return panel.iloc[0:0].copy(), 0, skipped
    return pd.concat(blocks, ignore_index=True), used, skipped


def neutralized_ic(
    panel: pd.DataFrame,
    exposure_columns: Sequence[str],
    *,
    horizon: int = 5,
    stride: int = 1,
    min_names: int = MIN_CROSS_SECTION_NAMES,
    notes: Optional[Sequence[str]] = None,
) -> NeutralizationResult:
    """Raw and residual IC for one panel, computed on the same dates.

    The dates are matched deliberately. A neutralized IC computed over fewer
    dates than the raw one is not a comparison — the difference would mix the
    effect of the exposures with the effect of a different sample.
    """
    neutralized, used, skipped = neutralize_panel(
        panel, exposure_columns, min_names=min_names
    )
    if neutralized.empty:
        empty = summarize_ic(pd.Series(dtype=float), horizon, stride)
        return NeutralizationResult(
            raw=empty, neutralized=empty, exposures=list(exposure_columns),
            n_dates_neutralized=0, n_dates_skipped=skipped, notes=list(notes or []),
        )

    kept_dates = set(neutralized["date"].unique())
    comparable = panel[panel["date"].isin(kept_dates)]

    return NeutralizationResult(
        raw=summarize_ic(rank_ic_series(comparable, min_names), horizon, stride),
        neutralized=summarize_ic(rank_ic_series(neutralized, min_names), horizon, stride),
        exposures=list(exposure_columns),
        n_dates_neutralized=used,
        n_dates_skipped=skipped,
        notes=list(notes or []),
    )


# --------------------------------------------------------------------------
# Building exposures from the panel itself
# --------------------------------------------------------------------------


def evaluate_neutralized(
    app_config: Any,
    strategy: Any,
    *,
    universe: Optional[Sequence[str]] = None,
    universe_size: Optional[int] = None,
    snapshot: Optional[str] = None,
    horizon: int = 5,
    stride: int = 1,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_history: int = 252,
    min_names: int = MIN_CROSS_SECTION_NAMES,
    max_dates: Optional[int] = None,
    use_benchmark: bool = True,
    beta_window: int = DEFAULT_BETA_WINDOW,
    size_window: int = DEFAULT_SIZE_WINDOW,
    sector_map: Optional[Mapping[str, str]] = None,
) -> NeutralizationResult:
    """Raw and neutralized IC for a registered strategy, end to end.

    Args:
        app_config: Loaded AppConfig.
        strategy: A registered strategy name or an instantiated strategy.
        sector_map: Symbol to sector. Absent by default because none ships with
            the repository; its absence is reported in the result rather than
            passed over, since "neutralized" quietly meaning "not neutralized
            against the thing that matters most" is the exact failure this
            module exists to prevent.

    Returns:
        A `NeutralizationResult` whose notes carry the size-proxy substitution
        and the sector-map gap.
    """
    from .harness import _resolve_strategy, build_forecast_panel

    resolved, name = _resolve_strategy(app_config, strategy)

    if universe is None:
        from portfolio_agent.training.universe import resolve_universe

        snap = resolve_universe(
            app_config, snapshot=snapshot, size=universe_size,
            name=f"neutralized:{name}",
        )
        universe = list(snap.tickers)

    panel = build_forecast_panel(
        app_config, resolved, universe,
        horizon=horizon, stride=stride, start_date=start_date, end_date=end_date,
        min_history=min_history, min_names=min_names, max_dates=max_dates,
        use_benchmark=use_benchmark, keep_prices=True,
    )

    panel, columns, notes = add_exposures(
        panel, beta_window=beta_window, size_window=size_window,
        sector_map=sector_map, stride=stride,
    )
    if not columns:
        notes.append(
            "no exposures could be built, so the neutralized column below is "
            "the raw signal and the comparison is vacuous"
        )
        from .metrics import rank_ic_series as _ic

        summary = summarize_ic(_ic(panel, min_names), horizon, stride)
        return NeutralizationResult(
            raw=summary, neutralized=summary, exposures=[],
            n_dates_neutralized=0, n_dates_skipped=0, notes=notes,
        )

    return neutralized_ic(
        panel, columns, horizon=horizon, stride=stride,
        min_names=min_names, notes=notes,
    )


def rolling_beta(
    returns: pd.DataFrame, window: int = DEFAULT_BETA_WINDOW
) -> pd.DataFrame:
    """Rolling market beta per symbol, against the equal-weighted composite.

    The composite stands in for the market because `^NSEI` is a *price* index
    and is frequently not in the cache at all; an equal-weighted mean of the
    universe's own returns is always available and is what the platform's
    regime filter already falls back to.

    Causal by construction: `rolling` windows end at the row they label, so the
    beta used on date `t` was estimable on date `t`.

    Args:
        returns: Wide (date x symbol) daily returns.
        window: Sessions in the estimation window.

    Returns:
        Wide (date x symbol) betas, NaN until the window fills.
    """
    from portfolio_agent.features.market_relative import market_composite

    market = market_composite(returns)
    market_variance = market.rolling(window, min_periods=window // 2).var()
    betas = {}
    for symbol in returns.columns:
        covariance = returns[symbol].rolling(window, min_periods=window // 2).cov(market)
        betas[symbol] = covariance / market_variance.replace(0.0, np.nan)
    return pd.DataFrame(betas, index=returns.index)


def add_exposures(
    panel: pd.DataFrame,
    *,
    beta_window: int = DEFAULT_BETA_WINDOW,
    size_window: int = DEFAULT_SIZE_WINDOW,
    sector_map: Optional[Mapping[str, str]] = None,
    stride: int = 1,
) -> tuple:
    """Attach beta, size-proxy and (optionally) sector exposures to a panel.

    Args:
        panel: Panel from `build_forecast_panel(..., keep_prices=True)`. The
            price and volume columns are what beta and the size proxy are built
            from; without them neither can be computed and both are skipped.
        beta_window: Sessions in the rolling beta window.
        size_window: Sessions in the traded-value window.
        stride: Sessions between the panel's rows. Both windows are stated in
            *sessions* and the panel is indexed in *rows*, so a strided panel
            needs them converted or a 252-session beta on a stride of 5 quietly
            becomes a five-year one — and then never fills, because the panel
            does not have 252 strided rows.
        sector_map: Symbol to sector. None means no sector neutralization is
            possible, which is reported rather than silently omitted.

    Returns:
        `(panel_with_exposures, exposure_columns, notes)`. The notes travel into
        the result and then into the printed report, which is where the size
        substitution has to appear.
    """
    notes: List[str] = []
    columns: List[str] = []
    panel = panel.copy()

    if "close" not in panel.columns:
        notes.append(
            "no price column in the panel, so beta and size could not be built "
            "(pass keep_prices=True to build_forecast_panel)"
        )
        return panel, columns, notes

    wide_close = panel.pivot_table(
        index="date", columns="symbol", values="close", aggfunc="last"
    ).sort_index()
    returns = wide_close.pct_change()

    beta_rows = _rows_for(beta_window, stride)
    size_rows = _rows_for(size_window, stride)
    if stride > 1:
        notes.append(
            f"stride={stride}, so the {beta_window}-session beta window is "
            f"{beta_rows} panel rows and the {size_window}-session size window "
            f"is {size_rows}"
        )

    betas = rolling_beta(returns, beta_rows)
    long_beta = betas.stack(future_stack=True).rename("beta").reset_index()
    long_beta.columns = ["date", "symbol", "beta"]
    panel = panel.merge(long_beta, on=["date", "symbol"], how="left")
    if panel["beta"].notna().any():
        columns.append("beta")
    else:
        notes.append(
            f"beta was NaN everywhere — the panel has fewer than {beta_rows} "
            f"rows, so it was not neutralized against"
        )
        panel = panel.drop(columns=["beta"])

    if "volume" in panel.columns and panel["volume"].notna().any():
        wide_volume = panel.pivot_table(
            index="date", columns="symbol", values="volume", aggfunc="last"
        ).sort_index()
        traded_value = (wide_close * wide_volume).replace(0.0, np.nan)
        size = np.log(
            traded_value.rolling(size_rows, min_periods=max(2, size_rows // 4)).median()
        )
        long_size = size.stack(future_stack=True).rename("size").reset_index()
        long_size.columns = ["date", "symbol", "size"]
        panel = panel.merge(long_size, on=["date", "symbol"], how="left")
        if panel["size"].notna().any():
            columns.append("size")
            notes.append(SIZE_PROXY_NOTE)
        else:
            panel = panel.drop(columns=["size"])
    else:
        notes.append("no volume in the panel, so the size proxy could not be built")

    if sector_map:
        sectors = panel["symbol"].map(sector_map)
        known = sectors.notna()
        if known.any():
            dummies = pd.get_dummies(sectors[known], prefix="sector", dtype=float)
            for column in dummies.columns:
                panel[column] = 0.0
                panel.loc[known, column] = dummies[column].to_numpy()
            columns.extend(list(dummies.columns))
            unmapped = int((~known).sum())
            if unmapped:
                notes.append(
                    f"{unmapped} observation(s) had no sector in the supplied map "
                    f"and were given an all-zero sector row"
                )
    else:
        notes.append(
            "no sector map supplied, so the result is NOT sector-neutral. Indian "
            "momentum concentrates hard by sector, which is exactly where an "
            "apparent alpha most often turns out to be a sector bet (finding A8)."
        )

    return panel, columns, notes
