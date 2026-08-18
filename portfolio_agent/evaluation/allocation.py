"""From a ranking to a book — the step the evaluation layer was missing.

`evaluate` measured a signal and stopped at the decile spread. A decile spread
is the return of an **equal-weighted** basket of the top bucket, and nothing
said so: the choice was implicit in `bucket_analysis` taking a mean, so every
number the platform reported about a strategy silently assumed the one
allocation rule that needs no covariance and no view.

Meanwhile `src/portfolio.py` is 821 lines of Ledoit-Wolf shrinkage, risk
contributions, a capped-simplex projection, `optimize_long_only` and
Hierarchical Risk Parity. Its non-test callers, before this module:

| function | callers |
| --- | --- |
| `ledoit_wolf_covariance`, `summarize_book_risk` | `backtest_engine.py` |
| `optimize_long_only` | `portfolio_optimizer.py` — which nothing imports |
| `hierarchical_risk_parity`, `shrunk_ewma_covariance` | **none** |

So the platform had a portfolio-construction library and a forecast evaluator,
and no path between them. This is that path.

What a scheme is answering
--------------------------
Given today's scores, how much of the book goes in each name? Four answers ship,
and the differences between them are the point rather than an implementation
detail:

- **equal** — what the decile spread was already doing. No covariance, no view,
  and a real baseline: it is hard to beat out of sample and every other scheme
  here has to earn its complexity against it.
- **inverse_vol** — size down what moves. Uses only the diagonal, so it needs no
  correlation estimate and cannot be wrecked by one.
- **hrp** — Lopez de Prado's allocation. Uses the correlation structure without
  inverting it and without expected returns, which is the right default when the
  view is a *ranking* rather than a return forecast.
- **mean_variance** — `optimize_long_only`, the only scheme that consumes the
  score as an expected return. It is included because it is the textbook answer
  and because seeing it lose to HRP on this platform's sample sizes is more
  convincing than being told it would.

The score is a rank percentile, not a return
--------------------------------------------
Every strategy here emits a 0-100 *goodness* score. Only `mean_variance` needs
that to be a return, and it is not one — so the score is mapped to a
cross-sectional expected return by centering the percentile and scaling it to a
caller-supplied dispersion (`expected_return_spread`), which is an assumption
stated in one place rather than smuggled in as "the score, divided by 100".
Getting that assumption wrong changes the book; that it is *visible* is the
difference between a modelling choice and a bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .costs import CostModel
from .metrics import MIN_CROSS_SECTION_NAMES, assign_buckets, validate_panel

#: Sessions in a trading year. Matches `evaluation/costs.py`.
TRADING_DAYS_PER_YEAR = 252

#: Allocation rules, in increasing order of what they assume about the inputs.
WEIGHTING_SCHEMES = ("equal", "inverse_vol", "hrp", "mean_variance")

#: Schemes that cannot be built from scores alone.
_NEEDS_RETURNS = ("inverse_vol", "hrp", "mean_variance")

#: Per-name cap. **None by default, and that default is load-bearing.**
#:
#: A cap of `c` on an `n`-name book is unsatisfiable when `c * n < 1` — `n`
#: names cannot each hold less than `1/n`. A decile book is small: a 10-bucket
#: split of a 60-name universe holds six names, so any cap under 0.167 is
#: unsatisfiable and the only feasible answer is equal weights.
#:
#: A 0.10 default therefore made every scheme return *identical* weights on a
#: realistic universe, and `compare_schemes` printed four indistinguishable
#: rows from which the obvious conclusion — "the weighting rule doesn't matter"
#: — would have been entirely an artifact of the cap. Uncapped is the honest
#: default for a book whose concentration is already set by the selection; a
#: caller who wants a cap passes one and is told when it cannot bind.
DEFAULT_MAX_WEIGHT: Optional[float] = None

#: Trailing sessions used to estimate covariance. One quarter — long enough to
#: estimate a 10-30 name covariance without shrinkage doing all the work, short
#: enough that the estimate is about the current regime.
DEFAULT_COVARIANCE_WINDOW = 63

#: Minimum trailing observations before a covariance is worth estimating.
DEFAULT_MIN_OBSERVATIONS = 20

#: Subgradient iterations per date for `mean_variance`. Far below
#: `optimize_long_only`'s own default of 2000, because a rebalancing book
#: warm-starts from the weights it already holds rather than from cash — the
#: solve is a correction, not a fresh optimization. At 2000 a 190-date
#: evaluation does not finish in two minutes.
DEFAULT_OPTIMIZER_ITERATIONS = 300

#: Cross-sectional spread of expected returns, per period, that a full-range
#: score is taken to imply for `mean_variance`. 2% between the best- and
#: worst-ranked name over the label horizon is deliberately modest: the drift
#: estimate is the input this platform has least right, and an optimizer handed
#: an overconfident mu concentrates the book in whichever name sampled luckiest.
DEFAULT_EXPECTED_RETURN_SPREAD = 0.02


@dataclass(frozen=True)
class BookWeights:
    """A weight per name per rebalance date, and what produced it."""

    weights: pd.DataFrame
    scheme: str
    n_dates: int
    n_skipped: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def mean_names_held(self) -> float:
        """Average count of non-zero positions — the book's realized breadth."""
        if self.weights.empty:
            return 0.0
        return float((self.weights > 0).sum(axis=1).mean())

    @property
    def mean_max_weight(self) -> float:
        """Average largest single position, for seeing a cap bind."""
        if self.weights.empty:
            return 0.0
        return float(self.weights.max(axis=1).mean())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weighting_scheme": self.scheme,
            "n_rebalance_dates": self.n_dates,
            "n_dates_skipped": self.n_skipped,
            "mean_names_held": self.mean_names_held,
            "mean_max_weight": self.mean_max_weight,
        }


@dataclass(frozen=True)
class BookPerformance:
    """What the book earned, before and after paying to run it."""

    scheme: str
    gross: pd.Series = field(repr=False)
    net: pd.Series = field(repr=False)
    turnover: pd.Series = field(repr=False)
    equity_curve: pd.Series = field(repr=False)
    costs: CostModel = field(repr=False, default_factory=CostModel.from_execution_sim)
    periods_per_year: float = float(TRADING_DAYS_PER_YEAR)
    #: Breadth, carried from the weights. Without it the report says what the
    #: book earned and not what it held, and the difference between an
    #: optimizer that spread across the decile and one that put 91% in a single
    #: name is invisible in the return alone.
    mean_names_held: float = 0.0
    mean_max_weight: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def total_return(self) -> float:
        """Compounded net return over the whole evaluation."""
        if self.equity_curve.empty:
            return 0.0
        return float(self.equity_curve.iloc[-1]) - 1.0

    @property
    def mean_gross(self) -> float:
        return float(self.gross.mean()) if len(self.gross) else 0.0

    @property
    def mean_net(self) -> float:
        return float(self.net.mean()) if len(self.net) else 0.0

    @property
    def mean_turnover(self) -> float:
        return float(self.turnover.mean()) if len(self.turnover) else 0.0

    @property
    def cost_drag(self) -> float:
        """Mean per-period return consumed by friction."""
        return self.mean_gross - self.mean_net

    @property
    def annualized_return(self) -> float:
        """Net return scaled to a year by compounding count, not by the curve.

        Deliberately not CAGR from the equity curve: over a short evaluation
        CAGR is dominated by where the window happened to start and stop, and
        the honest statement is "this per-period mean, that many times a year".
        """
        return self.mean_net * self.periods_per_year

    @property
    def annualized_volatility(self) -> float:
        if len(self.net) < 2:
            return float("nan")
        return float(self.net.std(ddof=1)) * float(np.sqrt(self.periods_per_year))

    @property
    def sharpe(self) -> float:
        """Net return over net volatility, annualized. Excess of zero.

        Not excess of the risk-free rate: a long-only book funded at the Indian
        repo would look materially worse, and the number this is compared
        against — the decile spread — is also an absolute return. Two absolute
        numbers can be compared; one absolute and one excess cannot.
        """
        volatility = self.annualized_volatility
        if not np.isfinite(volatility) or volatility == 0.0:
            return float("nan")
        return self.annualized_return / volatility

    @property
    def max_drawdown(self) -> float:
        """Largest peak-to-trough fall of the net equity curve, as a fraction."""
        if self.equity_curve.empty:
            return 0.0
        peak = self.equity_curve.cummax()
        return float((self.equity_curve / peak - 1.0).min())

    @property
    def survives_costs(self) -> bool:
        """Does the book still make money after paying to run it?"""
        return self.mean_net > 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "book_scheme": self.scheme,
            "book_total_return": self.total_return,
            "book_mean_gross": self.mean_gross,
            "book_mean_net": self.mean_net,
            "book_cost_drag": self.cost_drag,
            "book_mean_turnover": self.mean_turnover,
            "book_annualized_return": self.annualized_return,
            "book_annualized_volatility": self.annualized_volatility,
            "book_sharpe": self.sharpe,
            "book_max_drawdown": self.max_drawdown,
            "book_survives_costs": self.survives_costs,
            "book_n_periods": int(len(self.net)),
            "book_mean_names_held": self.mean_names_held,
            "book_mean_max_weight": self.mean_max_weight,
        }


# --------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------


def _selected_symbols(
    group: pd.DataFrame, n_buckets: int, min_names: int
) -> Sequence[str]:
    """The names in the top bucket on one date, or nothing if too thin.

    Same threshold `bucket_analysis` and `bucket_membership` use, so the book
    holds exactly the names the decile spread measured — the comparison between
    "the spread" and "the book" is only meaningful if they are the same
    selection differently weighted.
    """
    threshold = max(min_names, n_buckets)
    if len(group) < threshold:
        return ()
    assigned = assign_buckets(group["score"].to_numpy(), n_buckets)
    return group["symbol"].to_numpy()[assigned == n_buckets - 1].tolist()


def _covariance_for(
    returns: pd.DataFrame,
    symbols: Sequence[str],
    date: Any,
    window: int,
    min_observations: int,
) -> Optional[pd.DataFrame]:
    """Ledoit-Wolf covariance from the trailing window ending at `date`.

    Strictly `<= date`, matching T19's decision-date convention: the covariance
    used to build a book on date D is estimated from returns observable on D.
    Returns None when the window is too thin, and the caller falls back to equal
    weights rather than optimizing a covariance made of four observations.
    """
    from portfolio_agent.src.portfolio import ledoit_wolf_covariance

    available = [s for s in symbols if s in returns.columns]
    if len(available) < 2:
        return None

    history = returns.loc[returns.index <= date, available].tail(window).dropna(how="all")
    history = history.dropna(axis=1, how="any")
    if len(history) < min_observations or history.shape[1] < 2:
        return None

    # `ledoit_wolf_covariance` returns (covariance, shrinkage_intensity) and
    # already labels the frame when handed one.
    covariance, _intensity = ledoit_wolf_covariance(history)
    return covariance


def _equal_weights(symbols: Sequence[str]) -> pd.Series:
    if not len(symbols):
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(symbols), index=list(symbols))


def _inverse_vol_weights(covariance: pd.DataFrame) -> pd.Series:
    """Weight inversely to standard deviation, using only the diagonal.

    Inverse *volatility* rather than inverse variance: variance-weighting is a
    far more aggressive tilt toward the quietest names, and on Indian small caps
    the quietest names are frequently the least liquid ones — which is the
    failure T14's tradability screen exists for, and there is no reason to walk
    back into it through the weighting rule.
    """
    sigma = np.sqrt(np.maximum(np.diag(covariance.to_numpy()), 0.0))
    usable = sigma > 0
    if not usable.any():
        return _equal_weights(list(covariance.columns))
    inverse = np.where(usable, 1.0 / np.where(usable, sigma, 1.0), 0.0)
    return pd.Series(inverse / inverse.sum(), index=list(covariance.columns))


def _expected_returns_from_scores(
    scores: pd.Series, spread: float
) -> pd.Series:
    """Map 0-100 goodness scores onto a centered cross-sectional mu.

    The score is a rank percentile. Treating it as a return directly would
    assert that a score of 80 means an 80% expected gain, so instead the
    cross-section is centered and scaled: the best-ranked name is
    `+spread/2` and the worst `-spread/2`, linear in between.

    Centering matters as much as scaling. An all-positive mu makes a long-only
    optimizer want the whole budget invested regardless of the ranking's
    strength; a centered one lets the risk term push back.
    """
    if scores.empty:
        return scores
    low, high = float(scores.min()), float(scores.max())
    if high <= low:
        return pd.Series(0.0, index=scores.index)
    unit = (scores - low) / (high - low)  # 0..1
    return (unit - 0.5) * float(spread)


def cap_is_binding(max_weight: Optional[float], n_names: int) -> bool:
    """Whether a per-name cap can be satisfied by an `n_names` book.

    `c * n < 1` means it cannot: `n` names cannot each hold less than `1/n` of
    a fully invested book. Public because the answer decides whether a reported
    comparison between weighting schemes means anything — an unsatisfiable cap
    forces every scheme to equal weights, and four identical rows look like
    evidence that the weighting rule does not matter.
    """
    if max_weight is None or n_names <= 0:
        return False
    return float(max_weight) * n_names > 1.0


def _cap_and_normalize(
    weights: pd.Series, max_weight: Optional[float]
) -> tuple[pd.Series, bool]:
    """Normalize, and clip to the cap when the cap can actually be met.

    One clip-and-rescale does not converge: rescaling after a clip pushes other
    names back over the cap. Iterating is cheap at these sizes and means the
    reported book satisfies the constraint it claims to.

    Returns:
        `(weights, cap_was_unsatisfiable)`. The flag travels back to the caller
        rather than being swallowed, because falling back to equal weights is a
        different book from the one that was asked for and the report has to be
        able to say so.
    """
    if weights.empty:
        return weights, False
    capped = weights.clip(lower=0.0)
    total = capped.sum()
    if total <= 0:
        return _equal_weights(list(weights.index)), False
    capped = capped / total

    if max_weight is None:
        return capped, False
    if not cap_is_binding(max_weight, len(capped)):
        return _equal_weights(list(capped.index)), True

    for _ in range(100):
        over = capped > max_weight
        if not over.any():
            break
        excess = float((capped[over] - max_weight).sum())
        capped[over] = max_weight
        room = ~over
        if not room.any():
            break
        under = capped[room]
        capped[room] = under + excess * (under / under.sum() if under.sum() > 0
                                         else 1.0 / len(under))
    return capped / capped.sum(), False


def build_book(
    panel: pd.DataFrame,
    *,
    scheme: str = "equal",
    returns: Optional[pd.DataFrame] = None,
    n_buckets: int = 10,
    min_names: int = MIN_CROSS_SECTION_NAMES,
    max_weight: Optional[float] = DEFAULT_MAX_WEIGHT,
    covariance_window: int = DEFAULT_COVARIANCE_WINDOW,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
    expected_return_spread: float = DEFAULT_EXPECTED_RETURN_SPREAD,
    risk_aversion: float = 5.0,
    optimizer_iterations: int = DEFAULT_OPTIMIZER_ITERATIONS,
    costs: Optional[CostModel] = None,
) -> BookWeights:
    """Turn a scored panel into a long-only weight per name per date.

    Args:
        panel: Tidy `(date, symbol, score, forward_return)` rows — what
            `build_forecast_panel` produces.
        scheme: One of `WEIGHTING_SCHEMES`.
        returns: Wide `(date x symbol)` **trailing** daily returns, for the
            covariance. Required by every scheme except `equal`.
        n_buckets: Buckets for selection; the top one is the book.
        min_names: Dates with a thinner cross-section are skipped, matching
            `bucket_analysis` — so the book holds the names the spread measured.
        max_weight: Per-name cap, enforced by iterated clip-and-rescale.
        covariance_window: Trailing sessions for the covariance estimate.
        min_observations: Below this the covariance is not estimated and the
            date falls back to equal weights, which is recorded in `notes`.
        expected_return_spread: `mean_variance` only. Cross-sectional spread of
            per-period expected return implied by the full score range.
        risk_aversion: `mean_variance` only. Higher means a smaller, safer book.
        optimizer_iterations: `mean_variance` only. Subgradient iterations per
            date. Lower than `optimize_long_only`'s own default because each
            date warm-starts from the previous book rather than from cash.
        costs: Supplies the per-name round-trip cost `mean_variance` charges as
            its turnover penalty. Defaults to the shipped schedule.

    Returns:
        A `BookWeights` whose rows sum to 1 on every date that produced a book.

    Raises:
        ValueError: On an unknown scheme, or a scheme that needs returns
            without them. Both are caller mistakes that would otherwise degrade
            to an equal-weighted book reported under another scheme's name.
    """
    if scheme not in WEIGHTING_SCHEMES:
        raise ValueError(
            f"Unknown weighting scheme {scheme!r}. Available: {list(WEIGHTING_SCHEMES)}"
        )
    if scheme in _NEEDS_RETURNS and returns is None:
        raise ValueError(
            f"Scheme {scheme!r} needs a trailing returns panel to estimate "
            f"covariance; pass `returns=`. Only 'equal' can be built from "
            f"scores alone, and silently falling back to it would report an "
            f"equal-weighted book under another scheme's name."
        )

    clean = validate_panel(panel)
    rows: Dict[Any, pd.Series] = {}
    previous: Optional[pd.Series] = None
    turnover_cost = (costs or CostModel.from_execution_sim()).round_trip
    n_skipped = 0
    n_fell_back = 0
    n_uncappable = 0

    for date, group in clean.groupby("date", sort=True):
        symbols = _selected_symbols(group, n_buckets, min_names)
        if not len(symbols):
            n_skipped += 1
            continue

        if scheme == "equal":
            rows[date] = _equal_weights(symbols)
            previous = rows[date]
            continue

        covariance = _covariance_for(
            returns, symbols, date, covariance_window, min_observations
        )
        if covariance is None:
            n_fell_back += 1
            rows[date] = _equal_weights(symbols)
            previous = rows[date]
            continue

        weights = _weights_for_scheme(
            scheme, covariance, group, max_weight,
            expected_return_spread, risk_aversion,
            previous_weights=previous, turnover_cost=turnover_cost,
            iterations=optimizer_iterations,
        )
        # Names dropped by the covariance's own NaN screen get zero rather than
        # disappearing, so every row of the frame spans the same selection and
        # turnover between dates is measured against a stable book.
        rows[date], unsatisfiable = _cap_and_normalize(
            weights.reindex(list(symbols)).fillna(0.0), max_weight
        )
        n_uncappable += int(unsatisfiable)
        previous = rows[date]

    frame = (
        pd.DataFrame(rows).T.fillna(0.0).sort_index()
        if rows
        else pd.DataFrame(dtype=float)
    )

    notes: List[str] = []
    if n_skipped:
        notes.append(
            f"{n_skipped} date(s) had too thin a cross-section to bucket and "
            f"hold no book — the same threshold the decile spread applies."
        )
    if n_fell_back:
        notes.append(
            f"{n_fell_back} date(s) fell back to equal weights because fewer "
            f"than {min_observations} trailing observations were available to "
            f"estimate a covariance."
        )
    if n_uncappable:
        notes.append(
            f"max_weight={max_weight:g} is unsatisfiable on {n_uncappable} "
            f"date(s) — that many names cannot each hold less than 1/n of a "
            f"fully invested book — so those dates are equal-weighted. A "
            f"comparison between schemes over those dates compares nothing."
        )
    return BookWeights(
        weights=frame, scheme=scheme, n_dates=len(frame),
        n_skipped=n_skipped, notes=notes,
    )


def _weights_for_scheme(
    scheme: str,
    covariance: pd.DataFrame,
    group: pd.DataFrame,
    max_weight: Optional[float],
    expected_return_spread: float,
    risk_aversion: float,
    previous_weights: Optional[pd.Series] = None,
    turnover_cost: float = 0.0,
    iterations: int = DEFAULT_OPTIMIZER_ITERATIONS,
) -> pd.Series:
    """Dispatch to the allocation rule. Every branch returns unnormalized."""
    if scheme == "inverse_vol":
        return _inverse_vol_weights(covariance)

    if scheme == "hrp":
        from portfolio_agent.src.portfolio import hierarchical_risk_parity

        return pd.Series(hierarchical_risk_parity(covariance))

    if scheme == "mean_variance":
        from portfolio_agent.src.portfolio import optimize_long_only

        scores = group.set_index("symbol")["score"].reindex(covariance.columns)
        mu = _expected_returns_from_scores(scores.dropna(), expected_return_spread)
        mu = mu.reindex(covariance.columns).fillna(0.0)
        # Rebalancing from the book actually held, not from cash. This is what
        # `optimize_long_only`'s turnover penalty is for — its docstring calls
        # it load-bearing, because expected returns are estimated with enormous
        # error and an unpenalized optimizer re-solves to a materially
        # different book every day and pays the full Indian friction stack to
        # chase noise. It also warm-starts the subgradient ascent, which is
        # what makes a per-date optimization affordable at all.
        previous = (
            None if previous_weights is None
            else previous_weights.reindex(covariance.columns).fillna(0.0).to_numpy()
        )
        result = optimize_long_only(
            mu.to_numpy(),
            covariance,
            risk_aversion=risk_aversion,
            # The optimizer needs a numeric cap; 1.0 is "uncapped" for a book
            # whose weights sum to at most 1 anyway.
            max_weight=1.0 if max_weight is None else max_weight,
            previous_weights=previous,
            turnover_cost=turnover_cost,
            iterations=iterations,
            names=list(covariance.columns),
        )
        return result.as_series()

    raise ValueError(f"Unhandled scheme {scheme!r}")  # pragma: no cover


# --------------------------------------------------------------------------
# Performance
# --------------------------------------------------------------------------


def weight_turnover(weights: pd.DataFrame) -> pd.Series:
    """One-way turnover between consecutive weight vectors.

    `sum(|w_t - w_{t-1}|) / 2` — the fraction of the book that changed hands.
    The halving is what makes it *one-way*: selling 30% and buying 30% moves the
    absolute-difference sum by 0.60 and is 30% turnover, which is the convention
    `costs.one_way_turnover` already uses so the two numbers are comparable.

    **The first date is establishment, and the halving happens to be right for
    it too.** Going from cash to a fully invested book moves the sum by 1.0 and
    so reports 0.5, which looks like an understatement: 100% of the book was
    bought. But the caller charges `turnover x round_trip`, and establishing
    pays only the buy leg — so 0.5 round trips is one leg, which is exactly the
    cost incurred. Recorded because it is a coincidence of two conventions
    rather than a derivation, and the day either changes it stops holding.
    """
    if weights.empty:
        return pd.Series(dtype=float)
    previous = weights.shift(1).fillna(0.0)
    return (weights - previous).abs().sum(axis=1) / 2.0


def evaluate_book(
    book: BookWeights,
    panel: pd.DataFrame,
    *,
    costs: Optional[CostModel] = None,
    stride: int = 1,
) -> BookPerformance:
    """Charge the Indian cost schedule against a weighted book's return.

    The arithmetic:

        gross_t = sum_i w_{t,i} * forward_return_{t,i}
        net_t   = gross_t - turnover_t * round_trip_cost

    One round-trip charge rather than two, because this book is long-only: the
    turnover figure already counts both the sells and the buys that make up a
    single rebalance.

    Args:
        book: Weights from `build_book`.
        panel: The same tidy panel the book was built from, for the labels.
        costs: Cost model. Defaults to the shipped schedule at 25 bps/side.
        stride: Sessions between rebalances, for annualization.

    Returns:
        A `BookPerformance`.
    """
    model = costs or CostModel.from_execution_sim()
    if book.weights.empty:
        empty = pd.Series(dtype=float)
        return BookPerformance(
            scheme=book.scheme, gross=empty, net=empty, turnover=empty,
            equity_curve=empty, costs=model,
            periods_per_year=TRADING_DAYS_PER_YEAR / max(1, int(stride)),
            notes=["No date produced a book, so nothing was evaluated."],
        )

    clean = validate_panel(panel)
    labels = clean.pivot_table(
        index="date", columns="symbol", values="forward_return", aggfunc="first"
    )

    aligned_labels = labels.reindex(index=book.weights.index, columns=book.weights.columns)
    # A name held but unlabelled contributes nothing rather than propagating
    # NaN through the whole date's return. The weights already sum to 1, so
    # this is implicitly assuming the missing name returned zero — stated here
    # because it is a choice, and it is the conservative one.
    gross = (book.weights * aligned_labels.fillna(0.0)).sum(axis=1)

    turnover = weight_turnover(book.weights)
    net = gross - turnover * model.round_trip
    equity_curve = (1.0 + net).cumprod()

    notes = list(book.notes)
    unlabelled = int(aligned_labels.isna().sum().sum())
    if unlabelled:
        notes.append(
            f"{unlabelled} held-name-date(s) had no forward return and were "
            f"treated as flat."
        )
    notes.append(
        f"Turnover is charged one round trip per rebalance "
        f"({model.round_trip * 1e4:.0f} bps), not two: the book is long-only "
        f"and the turnover figure already spans its sells and its buys."
    )

    return BookPerformance(
        scheme=book.scheme,
        gross=gross,
        net=net,
        turnover=turnover,
        equity_curve=equity_curve,
        costs=model,
        mean_names_held=book.mean_names_held,
        mean_max_weight=book.mean_max_weight,
        periods_per_year=TRADING_DAYS_PER_YEAR / max(1, int(stride)),
        notes=notes,
    )


def compare_schemes(
    panel: pd.DataFrame,
    *,
    returns: Optional[pd.DataFrame] = None,
    schemes: Sequence[str] = WEIGHTING_SCHEMES,
    costs: Optional[CostModel] = None,
    stride: int = 1,
    **build_kwargs: Any,
) -> pd.DataFrame:
    """One row per weighting scheme, for the question the whole module exists for.

    Whether the complexity buys anything. `equal` is in the default list on
    purpose: a scheme that cannot beat it on this sample does not deserve to be
    the default, and the honest way to find that out is to put them in the same
    table.

    Schemes that cannot be built (no returns panel supplied) are omitted with a
    reason in the `note` column rather than dropped silently.
    """
    rows: List[Dict[str, Any]] = []
    for scheme in schemes:
        try:
            book = build_book(panel, scheme=scheme, returns=returns, **build_kwargs)
        except ValueError as exc:
            rows.append({"book_scheme": scheme, "note": str(exc)})
            continue
        performance = evaluate_book(book, panel, costs=costs, stride=stride)
        rows.append({**book.to_dict(), **performance.to_dict()})
    return pd.DataFrame(rows)
