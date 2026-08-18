"""Evaluate a forecast without simulating a book.

The platform could previously only judge a signal by running `BacktestEngine`
— 2,412 lines of order router, tax lots, circuit breakers and Kelly sizing, for
a book that will never trade. Two costs came with that. The obvious one is
speed. The one that actually matters is that every measurement arrived filtered
through portfolio-construction choices — position caps, stop placement,
regime gating — that have nothing to do with whether the forecast was any good,
so a change in the equity curve never told you which of the two had moved.

This routes around the engine rather than decomposing it. The engine stays
intact for the day a portfolio question is genuinely asked; research asks a
forecasting question and gets a forecasting answer.

The two seams
-------------
`build_forecast_panel` turns a strategy and a universe into tidy
`(date, symbol, score, forward_return)` rows. `evaluate_panel` turns those rows
into metrics and knows nothing about strategies. Splitting there is what makes
the harness testable: a panel whose scores *are* the future returns must score
near-perfect IC, and a panel of noise must not — neither of which requires a
strategy, a checkpoint or a cache to assert.

Features are built once, not once per date
------------------------------------------
`BacktestEngine` rebuilds every ticker's features from scratch on every date,
which is most of why a backtest is slow. Every registered feature is causal —
rolling and EWM windows, nothing that reads forward — so building once over the
full history and slicing to `.loc[:date]` gives bit-identical values at a
fraction of the cost. That is a property, not an assumption, and
`test_forecast_harness.py` asserts it directly for every feature the strategies
use. If a non-causal feature is ever registered, that test fails rather than
this module silently leaking.

Point-in-time discipline
------------------------
A strategy is handed `frame.loc[:date]` — inclusive of the decision date and
nothing after it. The label is the return from that date's close forward, so
the value being predicted is unknown at the moment of the prediction, by
construction. Two things that would break it and do not happen here: the
feature frames are sliced, never passed whole; and the forward return is
computed from the price series, never from a column that was already in the
feature block.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from portfolio_agent.data_quality.membership import (
    SURVIVORSHIP_NOTE,
    apply_membership,
    load_membership,
)

from .allocation import BookPerformance, build_book, evaluate_book
from .costs import CostModel, NetSpread, cost_notes, evaluate_net
from .metrics import (
    MIN_CROSS_SECTION_NAMES,
    BucketAnalysis,
    ErrorSummary,
    ICSummary,
    bucket_analysis,
    directional_hit_rate,
    rank_error_summary,
    rank_ic_series,
    score_dispersion,
    summarize_ic,
    validate_panel,
)

logger = logging.getLogger(__name__)

#: Rows a ticker needs before it can be scored at all — a floor, not the
#: answer. It reads as a judgement about sample adequacy (one trading year),
#: and `effective_min_history` raises it to the requested features' warm-up
#: whenever that is larger. It used to carry both meanings at once, and was
#: only ever correct because 252 happens to exceed the registry's longest
#: lookback of 211.
DEFAULT_MIN_HISTORY = 252


@dataclass(frozen=True)
class FoldEvaluation:
    """One walk-forward fold's IC, so stability is visible rather than assumed.

    A single pooled IC hides the shape that matters most: a mean of 0.04 made
    of one fold at 0.15 and four at 0.01 is a different object from five folds
    at 0.04, and only the per-fold numbers tell them apart.
    """

    index: int
    test_start: Any
    test_end: Any
    n_dates: int
    n_train_dates: int
    n_purged: int
    n_embargoed: int
    ic: ICSummary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold": self.index,
            "test_start": str(self.test_start),
            "test_end": str(self.test_end),
            "n_dates": self.n_dates,
            "n_train_dates": self.n_train_dates,
            "n_purged": self.n_purged,
            "n_embargoed": self.n_embargoed,
            **self.ic.to_dict(),
        }


@dataclass(frozen=True)
class ForecastEvaluation:
    """Everything measured about one signal, in one renderable object."""

    strategy: str
    horizon: int
    n_dates: int
    n_symbols: int
    n_observations: int
    ic: ICSummary
    buckets: BucketAnalysis
    hit_rate: float
    errors: ErrorSummary
    dispersion: float
    ic_series: pd.Series = field(repr=False, default_factory=lambda: pd.Series(dtype=float))
    folds: List[FoldEvaluation] = field(default_factory=list)
    #: The spread after paying the shipped Indian cost schedule to harvest it.
    #: Optional because a caller can switch it off, and None reads differently
    #: from a zero cost — one means "not charged", the other "charged nothing".
    costs: Optional[NetSpread] = None
    universe_fingerprint: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    #: Set when a manifest was written. Frozen dataclass, so the writer uses
    #: object.__setattr__ — the alternative is threading provenance through
    #: every construction site for two fields nobody passes by hand.
    run_id: Optional[str] = None
    manifest_path: Optional[str] = None
    #: The weighted book, when one was constructed. The decile spread already
    #: implies a book — an equal-weighted basket of the top bucket — so this is
    #: not a new claim so much as the same claim with its allocation rule made
    #: explicit and, optionally, changed.
    book: Optional[BookPerformance] = None
    #: What the point-in-time membership filter removed, when one was applied.
    #: Empty means no membership data was supplied, which `notes` states
    #: explicitly — the two are different claims and the report keeps them so.
    membership: Dict[str, Any] = field(default_factory=dict)

    def worst_dates(self, n: int = 5) -> pd.Series:
        """The dates the ordering was most wrong, for looking at what happened."""
        return self.ic_series.nsmallest(n)

    def best_dates(self, n: int = 5) -> pd.Series:
        return self.ic_series.nlargest(n)

    def to_dict(self) -> Dict[str, Any]:
        """Flat enough for a JSON manifest, nested only where nesting is real."""
        document: Dict[str, Any] = {
            "strategy": self.strategy,
            "horizon": self.horizon,
            "n_dates": self.n_dates,
            "n_symbols": self.n_symbols,
            "n_observations": self.n_observations,
            "hit_rate": self.hit_rate,
            "score_dispersion": self.dispersion,
            "universe_fingerprint": self.universe_fingerprint,
            "run_id": self.run_id,
            **self.ic.to_dict(),
            **self.buckets.to_dict(),
            **self.errors.to_dict(),
        }
        if self.costs is not None:
            document.update(self.costs.to_dict())
        if self.book is not None:
            document.update(self.book.to_dict())
        if self.membership:
            document.update(self.membership)
        if self.folds:
            document["folds"] = [fold.to_dict() for fold in self.folds]
        if self.notes:
            document["notes"] = list(self.notes)
        return document

    def to_frame(self) -> pd.DataFrame:
        """One row, for stacking several evaluations into a comparison table."""
        row = {
            key: value
            for key, value in self.to_dict().items()
            if not isinstance(value, (list, dict))
        }
        return pd.DataFrame([row])

    def render(self) -> str:
        """A fixed-width report, which is what gets pasted into a PR."""
        significance = "significant" if self.ic.significant else "not significant"
        lines = [
            f"Forecast evaluation — {self.strategy} @ {self.horizon}d",
            "=" * 62,
            f"  dates {self.n_dates}   symbols {self.n_symbols}   "
            f"observations {self.n_observations}",
        ]
        if self.universe_fingerprint:
            lines.append(f"  universe {self.universe_fingerprint}")
        lines += [
            "",
            "  Information coefficient",
            f"    mean rank IC     {self.ic.mean:+.4f}",
            f"    ICIR             {self.ic.icir:+.3f}   (std {self.ic.std:.4f})",
            f"    Newey-West t     {self.ic.t_stat:+.2f}   p={self.ic.p_value:.4f}"
            f"  [{significance}, {self.ic.newey_west_lags} lags]",
            f"    dates positive   {self.ic.positive_share:.1%}",
            "",
            "  Cross-section",
            f"    decile spread    {self.buckets.spread:+.4%}",
            f"    monotonicity     {self.buckets.monotonicity:+.3f}   "
            f"({self.buckets.monotone_steps:.0%} of steps rise)",
            f"    hit rate         {self.hit_rate:.1%}",
            f"    score dispersion {self.dispersion:.1%} of names distinctly scored",
        ]

        if self.costs is not None:
            net = self.costs
            verdict = "clears costs" if net.survives else "does not clear costs"
            long_only = (
                "clears costs" if net.long_only_survives else "does not clear costs"
            )
            share = (
                "     n/a" if not np.isfinite(net.cost_share)
                else f"{net.cost_share:7.1%}"
            )
            breakeven = (
                "      n/a" if not np.isfinite(net.breakeven_cost)
                else f"{net.breakeven_cost:+.4%}"
            )
            lines += [
                "",
                "  Net of costs",
                f"    turnover         {net.turnover:.1%} one-way per rebalance "
                f"({net.n_rebalances} rebalances)",
                f"    cost charged     {net.cost_per_rebalance:.4%} per rebalance "
                f"({net.costs.round_trip:.4%} round trip)",
                f"    spread net       {net.net:+.4%}   [{verdict}]",
                f"    long-only net    {net.long_only_net:+.4%}   [{long_only}]",
                f"    cost share       {share} of the gross spread",
                f"    breakeven cost   {breakeven} round trip",
            ]

        lines += [
            "",
            "  Bucket mean forward return (low score -> high)",
        ]
        for index, value in enumerate(self.buckets.mean_returns):
            bar = "" if not np.isfinite(value) else _bar(value, self.buckets.mean_returns)
            shown = "     n/a" if not np.isfinite(value) else f"{value:+8.4%}"
            lines.append(f"    {index:>2}  {shown}  {bar}")

        lines += [
            "",
            "  Rank error (predicted percentile - realized percentile)",
            f"    mean |error|     {self.errors.mean_abs_error:.4f}",
        ]
        if self.errors.quantiles:
            quantiles = "  ".join(
                f"{name} {value:+.3f}" for name, value in self.errors.quantiles.items()
            )
            lines.append(f"    quantiles        {quantiles}")

        if self.folds:
            lines += ["", "  Walk-forward folds"]
            for fold in self.folds:
                lines.append(
                    f"    {fold.index:>2}  {fold.test_start} .. {fold.test_end}  "
                    f"IC {fold.ic.mean:+.4f}  t {fold.ic.t_stat:+.2f}  "
                    f"({fold.n_dates} dates, {fold.n_purged} purged, "
                    f"{fold.n_embargoed} embargoed)"
                )
        for note in self.notes:
            lines += ["", f"  Note: {note}"]
        return "\n".join(lines)


def _bar(value: float, values: Sequence[float], width: int = 24) -> str:
    """A crude magnitude bar, so the shape of the bucket profile is visible."""
    finite = [v for v in values if np.isfinite(v)]
    scale = max(abs(v) for v in finite) if finite else 0.0
    if scale <= 0.0:
        return ""
    filled = int(round(abs(value) / scale * width))
    return ("+" if value >= 0 else "-") * max(filled, 1)


# --------------------------------------------------------------------------
# Scoring a panel
# --------------------------------------------------------------------------


def fundamentals_notes(path: Optional[str]) -> List[str]:
    """What a run should say about the fundamentals it did or did not have.

    Same contract as T15's survivorship note: absence is a property of the
    number sitting next to it, not an outstanding chore, so it is stated in the
    result rather than tracked somewhere else.

    A file that *was* supplied gets its validation warnings surfaced too — a
    fundamentals set whose report lags are all identical is worse than none,
    and a run on it should say that where the result can be seen.
    """
    from portfolio_agent.data_quality.fundamentals import (
        FUNDAMENTALS_NOTE,
        load_fundamentals,
    )

    if path is None:
        return [FUNDAMENTALS_NOTE]

    try:
        store = load_fundamentals(path)
    except (FileNotFoundError, ValueError) as exc:
        # Not fatal: the evaluation itself is price-based and still means
        # something. But it must not read as though fundamentals were applied.
        return [
            f"Fundamentals file {path!r} could not be used ({exc}). This "
            f"result controls for no accounting characteristic."
        ]

    lines = [
        f"Fundamentals: {len(store.symbols)} symbol(s), fields "
        f"{store.available_facts}, median report lag "
        f"{store.validation.median_lag_days:.0f} days."
        if store.validation else f"Fundamentals: {len(store.symbols)} symbol(s)."
    ]
    if store.validation:
        lines.extend(store.validation.warnings)
    return lines


def evaluate_panel(
    panel: pd.DataFrame,
    *,
    horizon: int,
    strategy: str = "signal",
    n_buckets: int = 10,
    min_names: int = MIN_CROSS_SECTION_NAMES,
    stride: int = 1,
    splitter: Any = None,
    universe_fingerprint: Optional[str] = None,
    charge_costs: bool = True,
    slippage_per_side: Optional[float] = None,
    weighting: Optional[str] = None,
    returns: Optional[pd.DataFrame] = None,
    **book_kwargs: Any,
) -> ForecastEvaluation:
    """Score a tidy `(date, symbol, score, forward_return)` panel.

    Args:
        panel: The observations. Rows with a non-finite score or label are
            dropped; a missing column raises.
        horizon: Label horizon in sessions. Sets the Newey–West lag.
        strategy: Name for the report header.
        n_buckets: Buckets for the spread and monotonicity checks.
        min_names: Minimum cross-section width for a date to count.
        stride: Sessions between consecutive observations. Sets the Newey-West
            lag together with the horizon; leave at 1 for a daily panel.
        splitter: Optional `validation.purged.PurgedWalkForward`. When given,
            per-fold IC is reported on each fold's *test* dates — which is the
            only place an out-of-sample number can come from — and the fold's
            purge and embargo counts travel into the report so the exclusions
            stay visible.
        universe_fingerprint: Provenance, carried into the result.
        charge_costs: Report the spread net of the shipped Indian cost
            schedule. On by default: a gross spread is the number that looks
            best and means least.
        slippage_per_side: Slippage assumption. Defaults to the 25 bps/side in
            `execution_sim`, deliberately conservative for mid-caps.
        weighting: Build a weighted book alongside the spread — one of
            `allocation.WEIGHTING_SCHEMES`. None reports the spread only, which
            is what every prior release did; "equal" reproduces the spread's own
            implicit allocation as an explicit equity curve.
        returns: Wide `(date x symbol)` trailing daily returns, required by
            every weighting scheme except "equal".
        **book_kwargs: Forwarded to `allocation.build_book` (max_weight,
            covariance_window, risk_aversion, ...).

    Returns:
        A `ForecastEvaluation`.
    """
    clean = validate_panel(panel)
    ic = rank_ic_series(clean, min_names)

    folds: List[FoldEvaluation] = []
    notes: List[str] = []

    net: Optional[NetSpread] = None
    if charge_costs and not clean.empty:
        net = evaluate_net(
            clean,
            horizon=horizon,
            costs=CostModel.from_execution_sim(slippage_per_side),
            n_buckets=n_buckets,
            min_names=min_names,
            stride=stride,
        )
        notes.extend(cost_notes(net))
    book: Optional[BookPerformance] = None
    if weighting is not None and not clean.empty:
        weights = build_book(
            clean, scheme=weighting, returns=returns,
            n_buckets=n_buckets, min_names=min_names, **book_kwargs,
        )
        book = evaluate_book(
            weights, clean,
            costs=CostModel.from_execution_sim(slippage_per_side), stride=stride,
        )
        notes.extend(book.notes)

    if splitter is not None and not clean.empty:
        folds = _evaluate_folds(clean, splitter, horizon, min_names, stride)
        if getattr(splitter, "embargo", 0) and not any(f.n_embargoed for f in folds):
            # Not a bug, and worth saying so before someone concludes it is. An
            # expanding walk-forward only ever trains on dates *before* its test
            # block, and the embargo excludes dates *after* it — so there is
            # nothing for it to remove. It bites on a scheme that trains on both
            # sides of a fold; here the purge is doing all the work.
            notes.append(
                f"embargo={splitter.embargo} removed nothing: an expanding "
                "walk-forward has no training dates after its test block for an "
                "embargo to exclude. The purge counts above are the real "
                "exclusions."
            )

    return ForecastEvaluation(
        strategy=strategy,
        horizon=int(horizon),
        n_dates=int(clean["date"].nunique()),
        n_symbols=int(clean["symbol"].nunique()),
        n_observations=int(len(clean)),
        ic=summarize_ic(ic, horizon, stride),
        buckets=bucket_analysis(clean, n_buckets, min_names),
        hit_rate=directional_hit_rate(clean, min_names),
        book=book,
        errors=rank_error_summary(clean, n_buckets, min_names),
        dispersion=score_dispersion(clean, min_names),
        ic_series=ic,
        folds=folds,
        universe_fingerprint=universe_fingerprint,
        notes=notes,
        costs=net,
    )


def _evaluate_folds(
    panel: pd.DataFrame, splitter: Any, horizon: int, min_names: int, stride: int = 1
) -> List[FoldEvaluation]:
    """Per-fold IC on the test blocks of a purged walk-forward split.

    The split is asserted leak-free before it is used. That assertion is cheap
    and the alternative is a number that looks like out-of-sample skill and is
    not — the single most expensive mistake available in this codebase.
    """
    from portfolio_agent.validation.purged import assert_no_leakage

    dates = pd.DatetimeIndex(pd.to_datetime(panel["date"]).unique()).sort_values()
    evaluations: List[FoldEvaluation] = []

    for index, fold in enumerate(splitter.split(dates)):
        assert_no_leakage(fold, dates, horizon)
        block = panel[pd.to_datetime(panel["date"]).isin(fold.test)]
        if block.empty:
            continue
        fold_ic = rank_ic_series(block, min_names)
        evaluations.append(
            FoldEvaluation(
                index=index,
                test_start=fold.test.min().date(),
                test_end=fold.test.max().date(),
                n_dates=int(len(fold_ic)),
                n_train_dates=fold.n_train,
                n_purged=len(fold.purged),
                n_embargoed=len(fold.embargoed),
                ic=summarize_ic(fold_ic, horizon, stride),
            )
        )
    return evaluations


# --------------------------------------------------------------------------
# Producing a panel from a registered strategy
# --------------------------------------------------------------------------


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Realized return from `t` to `t + horizon`, dated at `t`.

    Dated at the decision point on purpose: the row's features are what was
    known then, and the value is what happened after. The final `horizon` rows
    are NaN because their outcome has not occurred, and they are dropped rather
    than filled.
    """
    close = pd.Series(close, dtype=float)
    return close.shift(-horizon) / close - 1.0


def build_forecast_panel(
    app_config: Any,
    strategy: Any,
    universe: Sequence[str],
    *,
    horizon: int = 5,
    stride: int = 1,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_history: int = DEFAULT_MIN_HISTORY,
    min_names: int = MIN_CROSS_SECTION_NAMES,
    max_dates: Optional[int] = None,
    use_benchmark: bool = True,
    extra_horizons: Optional[Sequence[int]] = None,
    keep_prices: bool = False,
) -> pd.DataFrame:
    """Run a strategy across a universe and collect what it predicted.

    Args:
        app_config: Loaded AppConfig, supplying risk parameters and the
            benchmark symbol.
        strategy: An instantiated `BaseStrategy`.
        universe: Tickers to score. Pinned by the caller so two evaluations
            compare.
        horizon: Forward-return horizon in sessions.
        stride: Score every `stride`-th date. 1 evaluates daily, which is what
            the Newey–West adjustment is there to handle; a larger stride
            trades statistical power for wall-clock time and is honest about it
            because the reduced date count travels into the result.
        start_date: ISO lower bound on the evaluation window.
        end_date: ISO upper bound.
        min_history: Rows a ticker needs before it is eligible on a date.
        min_names: Dates with a thinner cross-section are skipped entirely.
        max_dates: Hard cap, applied to the most recent dates. A guard for
            interactive use; leave None for a full run.
        use_benchmark: Pass the cached benchmark index into the strategy
            context. The crash and regime filters read it, so turning it off
            changes what several strategies emit.
        extra_horizons: Additional forward-return horizons to attach as
            `forward_return_<h>` columns. Scoring is by far the expensive part
            and does not depend on the horizon at all, so a decay curve over
            six horizons costs one pass rather than six.
        keep_prices: Also emit `close` and `volume` at the decision date. Needed
            to build size and beta exposures without re-reading the cache.

    Returns:
        A tidy `(date, symbol, score, forward_return)` frame, plus a `signal`
        column carrying the strategy's own verdict for callers that want to
        score only the names it would actually have bought.

    Raises:
        ValueError: If no ticker yielded usable history, or no date had a wide
            enough cross-section. Both are configuration problems, and an empty
            panel scores as "no skill" rather than announcing itself.
    """
    from portfolio_agent.features.pipeline import build_features, effective_min_history
    from portfolio_agent.src.data_store import load_ticker_data
    from portfolio_agent.strategies.types import RiskParams, StrategyContext

    from portfolio_agent.features.cross_section import (
        warmup_rows as cross_sectional_warmup,
    )

    feature_names = list(strategy.required_features())
    # Never below the warm-up: under it the feature is NaN, and a strategy
    # handed NaNs ranks them somewhere arbitrary rather than refusing.
    #
    # Both registries. A strategy can rank on a cross-sectional feature while
    # requesting only cheap per-ticker ones — `residual_momentum` needs 242
    # rows for its formation window and nothing per-ticker beyond 62 — so
    # consulting one registry would score dates whose ranking key is still NaN.
    requested_min_history = min_history
    min_history = max(
        effective_min_history(feature_names, min_history),
        cross_sectional_warmup(strategy.required_cross_sectional_features()),
    )
    if min_history != requested_min_history:
        logger.info(
            "Raised min_history from %d to %d — %s needs that much history "
            "before every feature it ranks on is defined",
            requested_min_history, min_history, getattr(strategy, "name", "strategy"),
        )
    horizons = [int(h) for h in (extra_horizons or []) if int(h) != int(horizon)]
    features_by_ticker: Dict[str, pd.DataFrame] = {}
    labels_by_ticker: Dict[str, pd.Series] = {}
    prices_by_ticker: Dict[str, pd.DataFrame] = {}
    extra_labels: Dict[int, Dict[str, pd.Series]] = {h: {} for h in horizons}

    for ticker in universe:
        try:
            raw = load_ticker_data(ticker, start_date=start_date, end_date=end_date)
        except Exception as exc:  # pragma: no cover - depends on cache state
            logger.debug("Failed to load %s: %s", ticker, exc)
            continue
        if raw is None or len(raw) < min_history:
            continue

        raw = raw.copy()
        raw.columns = [str(column).lower() for column in raw.columns]
        try:
            # Built once over the whole history. Every registered feature is
            # causal, so slicing this is identical to rebuilding from the
            # truncated history — asserted in the tests, not assumed here.
            built = build_features(
                raw,
                feature_names,
                normalize=app_config.features.normalize,
                normalize_window=app_config.features.normalize_window,
            )
        except (KeyError, ValueError) as exc:
            logger.warning("Feature build failed for %s: %s", ticker, exc)
            continue

        features_by_ticker[ticker] = built
        labels_by_ticker[ticker] = forward_return(raw["close"], horizon)
        prices_by_ticker[ticker] = raw
        for extra in horizons:
            extra_labels[extra][ticker] = forward_return(raw["close"], extra)

    if not features_by_ticker:
        raise ValueError(
            f"No ticker in a universe of {len(universe)} produced usable history. "
            "Run `portfolio-agent download-data` if the parquet cache is empty."
        )

    dates = _evaluation_dates(
        features_by_ticker, labels_by_ticker, min_history, stride, max_dates
    )
    if not len(dates):
        raise ValueError(
            "No date had both enough history and a realized forward return. "
            f"Check horizon={horizon} and min_history={min_history} against the "
            "cached span."
        )

    risk = RiskParams.from_app_config(app_config)
    benchmark = _load_benchmark(app_config) if use_benchmark else None

    rows: List[Dict[str, Any]] = []
    skipped_thin = 0

    for date in dates:
        eligible: Dict[str, pd.DataFrame] = {}
        for ticker, frame in features_by_ticker.items():
            if date not in frame.index:
                continue
            window = frame.loc[:date]
            if len(window) < min_history:
                continue
            label = labels_by_ticker[ticker].get(date, np.nan)
            if not np.isfinite(label):
                continue
            eligible[ticker] = window

        if len(eligible) < min_names:
            skipped_thin += 1
            continue

        context = StrategyContext(
            risk=risk,
            benchmark_close=(
                benchmark["close"].loc[:date] if benchmark is not None else None
            ),
            benchmark_ohlcv=benchmark.loc[:date] if benchmark is not None else None,
        )
        signals = strategy.score_batch(eligible, context)

        for ticker, signal in signals.items():
            row = {
                "date": date,
                "symbol": ticker,
                "score": float(signal.score),
                "forward_return": float(labels_by_ticker[ticker][date]),
                "signal": signal.signal,
            }
            for extra in horizons:
                # NaN where the outcome has not occurred yet. Left in rather
                # than dropped, so every horizon is scored on the widest set of
                # dates it can support instead of on the intersection with the
                # longest one — which would silently shorten the whole curve to
                # whatever the 21-day horizon could reach.
                row[f"forward_return_{extra}"] = float(
                    extra_labels[extra][ticker].get(date, np.nan)
                )
            if keep_prices:
                bars = prices_by_ticker[ticker]
                row["close"] = float(bars["close"].get(date, np.nan))
                row["volume"] = float(bars["volume"].get(date, np.nan)) if "volume" in bars else np.nan
            rows.append(row)

    if not rows:
        raise ValueError(
            f"No date produced a cross-section of at least {min_names} names "
            f"({skipped_thin} dates were too thin). Widen the universe."
        )

    logger.info(
        "Scored %d observations across %d dates and %d tickers (%d thin dates skipped)",
        len(rows), len({row["date"] for row in rows}), len(features_by_ticker), skipped_thin,
    )
    return pd.DataFrame(rows)


def _evaluation_dates(
    features_by_ticker: Dict[str, pd.DataFrame],
    labels_by_ticker: Dict[str, pd.Series],
    min_history: int,
    stride: int,
    max_dates: Optional[int],
) -> pd.DatetimeIndex:
    """Dates on which at least one ticker is both mature and has an outcome.

    The per-date eligibility check runs again inside the loop, per ticker; this
    is only to avoid walking thousands of dates on which nothing could possibly
    qualify.
    """
    if stride < 1:
        raise ValueError(f"stride must be at least 1, got {stride}")

    candidates: set = set()
    for ticker, frame in features_by_ticker.items():
        if len(frame) < min_history:
            continue
        labelled = labels_by_ticker[ticker].dropna().index
        candidates.update(frame.index[min_history - 1:].intersection(labelled))

    dates = pd.DatetimeIndex(sorted(candidates))
    if stride > 1:
        dates = dates[::stride]
    if max_dates is not None and len(dates) > max_dates:
        # Keep the most recent window: a truncated evaluation should describe
        # the regime the model would be deployed into, not the oldest data.
        dates = dates[-max_dates:]
    return dates


def _load_benchmark(app_config: Any) -> Optional[pd.DataFrame]:
    """The cached benchmark index, or None when it was never downloaded."""
    from portfolio_agent.src.data_store import load_ticker_data

    symbol = getattr(app_config.data, "benchmark_symbol", None)
    if not symbol:
        return None
    try:
        frame = load_ticker_data(symbol)
    except Exception as exc:  # pragma: no cover - depends on cache state
        logger.debug("Benchmark %s unavailable: %s", symbol, exc)
        return None
    if frame is None or frame.empty:
        logger.info(
            "Benchmark %s is not cached; regime-aware strategies will fall back "
            "to their composite proxy.", symbol,
        )
        return None
    frame = frame.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    return frame


def evaluate_forecast(
    app_config: Any,
    strategy: Any,
    *,
    universe: Optional[Sequence[str]] = None,
    universe_size: Optional[int] = None,
    snapshot: Optional[str] = None,
    horizon: int = 5,
    n_buckets: int = 10,
    stride: int = 1,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_history: int = DEFAULT_MIN_HISTORY,
    min_names: int = MIN_CROSS_SECTION_NAMES,
    max_dates: Optional[int] = None,
    use_benchmark: bool = True,
    splitter: Any = None,
    buys_only: bool = False,
    manifest: bool = True,
    runs_dir: Optional[str] = None,
    charge_costs: bool = True,
    slippage_per_side: Optional[float] = None,
    membership: Optional[str] = None,
    index_name: Optional[str] = None,
    fundamentals: Optional[str] = None,
) -> ForecastEvaluation:
    """Measure one strategy's forecast skill, end to end.

    Args:
        app_config: Loaded AppConfig.
        strategy: A registered strategy name, or an instantiated strategy.
        universe: Explicit tickers. Otherwise a pinned universe is resolved the
            same way training resolves one, so a model and its evaluation can
            be run on identical names.
        universe_size: Size for a fresh draw.
        snapshot: Path to a saved universe snapshot.
        horizon: Forward-return horizon in sessions.
        n_buckets: Buckets for spread and monotonicity.
        stride: Evaluate every `stride`-th date.
        start_date / end_date: ISO bounds on the window.
        min_history / min_names / max_dates: See `build_forecast_panel`.
        use_benchmark: Pass the cached index into the strategy context.
        splitter: Optional `PurgedWalkForward` for per-fold reporting.
        buys_only: Score only the names the strategy would have bought. Off by
            default, and the default is the meaningful one: restricting to buys
            measures the screen *and* the forecast together, and a screen that
            emits four names a day has no cross-section left to rank.
        manifest: Record what produced this evaluation under `runs/`. On by
            default — a metric whose universe, config and commit are unrecorded
            is one nobody can check later, including the person who ran it.
        runs_dir: Where the manifest goes.
        charge_costs: Report the decile spread net of the shipped Indian cost
            schedule, with the signal's measured turnover. On by default.
        slippage_per_side: Slippage assumption per leg, as a fraction of
            turnover. Defaults to the 25 bps in `execution_sim`.
        membership: Path to a point-in-time index membership CSV. Without one
            every date is ranked against the names that survived to be
            downloaded, and the result says so in its notes.
        index_name: Narrow a multi-index membership file to one index.
        fundamentals: Path to a point-in-time fundamentals CSV. Without one the
            result controls for no accounting characteristic, and says so in
            its notes — the same contract `membership` has.

    Returns:
        A `ForecastEvaluation`, carrying `run_id` when a manifest was written.
    """
    import time

    started = time.monotonic()
    resolved, name = _resolve_strategy(app_config, strategy)

    snap = None
    if universe is None:
        from portfolio_agent.src.universe import MEASUREMENT_PURPOSE
        from portfolio_agent.training.universe import resolve_universe

        snap = resolve_universe(
            app_config, snapshot=snapshot, size=universe_size, name=f"eval:{name}",
            purpose=MEASUREMENT_PURPOSE,
        )
        universe = list(snap.tickers)

    panel = build_forecast_panel(
        app_config, resolved, universe,
        horizon=horizon, stride=stride, start_date=start_date, end_date=end_date,
        min_history=min_history, min_names=min_names, max_dates=max_dates,
        use_benchmark=use_benchmark,
    )

    notes: List[str] = []
    membership_metrics: Dict[str, Any] = {}

    # Applied before every other filter: which names existed in the index is a
    # fact about the date, not a preference about the sample, and restricting
    # after a bucket has been formed would rank against a universe the book
    # could not have held.
    resolved_membership = load_membership(membership, index_name)
    if resolved_membership is None:
        notes.append(SURVIVORSHIP_NOTE)
    else:
        outcome = apply_membership(panel, resolved_membership)
        panel = outcome.panel
        membership_metrics = outcome.to_dict()
        notes.append(outcome.note())
        if panel.empty:
            raise ValueError(
                "point-in-time membership left no observations to score — the "
                f"membership file {resolved_membership.source} and the "
                "evaluation universe may not overlap"
            )

    # Same contract as the membership note above, for the same reason: a run
    # without fundamentals has controlled for no accounting characteristic, and
    # that is a property of the number rather than an outstanding chore.
    notes.extend(fundamentals_notes(fundamentals))

    if buys_only:
        before = len(panel)
        panel = panel[panel["signal"] == "BUY"]
        notes.append(
            f"Restricted to BUY signals: {len(panel)}/{before} observations. "
            "The IC below describes the screen and the forecast together."
        )
        if panel.empty:
            raise ValueError("buys_only left no observations to score")

    evaluation = evaluate_panel(
        panel, horizon=horizon, strategy=name, n_buckets=n_buckets,
        min_names=min_names, stride=stride, splitter=splitter,
        universe_fingerprint=snap.fingerprint if snap is not None else None,
        charge_costs=charge_costs, slippage_per_side=slippage_per_side,
    )
    if notes:
        evaluation.notes.extend(notes)
    if membership_metrics:
        # Same frozen-dataclass compromise as run_id above, and for the same
        # reason: the filter runs before `evaluate_panel` sees the panel, so
        # the alternative is a membership-shaped parameter on a function whose
        # job is scoring a panel it is handed.
        object.__setattr__(evaluation, "membership", membership_metrics)

    if manifest:
        _record_manifest(
            app_config, evaluation, name, list(universe), snap,
            settings={
                "horizon": horizon, "stride": stride, "n_buckets": n_buckets,
                "min_history": min_history, "min_names": min_names,
                "max_dates": max_dates, "use_benchmark": use_benchmark,
                "buys_only": buys_only, "start_date": start_date, "end_date": end_date,
                # Provenance, not a metric: which membership file a number was
                # computed against is the first thing to check when two runs
                # of the "same" strategy disagree.
                "membership": (
                    resolved_membership.source if resolved_membership else None
                ),
                "index_name": index_name,
            },
            splitter=splitter,
            seconds=time.monotonic() - started,
            runs_dir=runs_dir,
        )
    return evaluation


def _record_manifest(
    app_config: Any,
    evaluation: "ForecastEvaluation",
    name: str,
    universe: List[str],
    snap: Any,
    *,
    settings: Dict[str, Any],
    splitter: Any,
    seconds: float,
    runs_dir: Optional[str],
) -> None:
    """Write this evaluation's manifest, and never let doing so break it.

    Provenance is worth a file and is not worth losing a result over. A failure
    here is logged and swallowed — the evaluation is already computed, and
    raising would throw it away to protect a record of it.
    """
    try:
        from portfolio_agent.provenance import DEFAULT_RUNS_DIR, build_manifest

        split: Dict[str, Any] = {"horizon": evaluation.horizon}
        if splitter is not None:
            for field_name in ("n_splits", "horizon", "embargo", "min_train_fraction"):
                if hasattr(splitter, field_name):
                    split[field_name] = getattr(splitter, field_name)
            split["scheme"] = type(splitter).__name__
        else:
            split["scheme"] = "single window (no walk-forward split)"

        metrics = {
            key: value for key, value in evaluation.to_dict().items()
            if isinstance(value, (int, float, bool))
        }
        extras: Dict[str, Any] = {}
        if evaluation.folds:
            extras["folds"] = [fold.to_dict() for fold in evaluation.folds]

        record = build_manifest(
            "evaluate",
            app_config=app_config,
            strategy=name,
            universe=universe,
            universe_fingerprint=(
                snap.fingerprint if snap is not None else evaluation.universe_fingerprint
            ),
            universe_name=snap.name if snap is not None else None,
            settings=settings,
            split=split,
            metrics=metrics,
            timings={"total": seconds},
            notes=list(evaluation.notes),
            extras=extras,
        )
        path = record.save(runs_dir if runs_dir is not None else DEFAULT_RUNS_DIR)
        object.__setattr__(evaluation, "run_id", record.run_id)
        object.__setattr__(evaluation, "manifest_path", str(path))
    except Exception:  # pragma: no cover - provenance must never break a result
        logger.exception("Could not write a run manifest; the evaluation is unaffected")


def _resolve_strategy(app_config: Any, strategy: Any) -> tuple:
    """Accept a registered name or an already-built strategy object."""
    if not isinstance(strategy, str):
        return strategy, getattr(strategy, "name", type(strategy).__name__)

    from portfolio_agent.strategies.registry import load_strategy

    for candidate in getattr(app_config, "strategies", []) or []:
        if getattr(candidate, "type", None) == strategy or getattr(candidate, "name", None) == strategy:
            return load_strategy(candidate), strategy

    # Not configured in this AppConfig: build it from a minimal StrategyConfig
    # so an unconfigured strategy can still be measured. Its own defaults apply,
    # which is what someone asking "how good is momentum" means.
    from portfolio_agent.config.schema import StrategyConfig

    return load_strategy(StrategyConfig(name=strategy, type=strategy)), strategy


def compare_forecasts(evaluations: Iterable[ForecastEvaluation]) -> pd.DataFrame:
    """Stack several evaluations into one table, best mean IC first."""
    frames = [evaluation.to_frame() for evaluation in evaluations]
    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True)
    return table.sort_values("mean_ic", ascending=False).reset_index(drop=True)
