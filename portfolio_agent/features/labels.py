"""Forward-return labels and the cross-sectional target transform.

Extracted from `agents/trainer.py` so the *definition of what a model is asked
to predict* has one home, below any particular training procedure. The
supervised pipeline re-exports these names, so nothing that imported them from
there had to change; what changed is that a trainer which is not the supervised
one can now predict the identical label without importing PyTorch to get at it.

That matters more than it sounds. Two trainers whose labels differ by a
horizon, a sign, or a rank normalization produce metrics that cannot be
compared, and nothing in a results table would show it.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd


def target_column_name(target: str) -> str:
    """Column name for the training target, namespaced away from features.

    The target must never share a name with a registered feature. It used to:
    config.training.target defaults to "return_5d", which is also in
    TRAINING_FEATURE_NAMES, so the "create target if it doesn't exist" branch in
    `agents/trainer.py::prepare_features` never fired and the model was trained
    to reproduce the *trailing* 5-day return — a quantity fully determined by
    the price history it was already being shown. That trains and validates
    beautifully and forecasts nothing. Prefixing the target guarantees the
    collision cannot recur.
    """
    return f"target_{target}"


def build_forward_return(close: pd.Series, target: str) -> pd.Series:
    """Realized *forward* return over the horizon encoded in `target`.

    For target="return_5d": (close[t+5] - close[t]) / close[t] — what the
    model is supposed to predict, dated at the decision point t. The value is
    unknown at t by construction, which is the point; rows near the end of the
    series are NaN and get dropped.
    """
    periods = 1
    if 'return' in target:
        digits = "".join(ch for ch in target if ch.isdigit())
        if digits:
            periods = max(1, int(digits))
    return close.shift(-periods).pct_change(periods)


# Below this many names on a date there is no cross-section to rank against, so
# a relative target would mostly encode which handful of tickers happened to
# have history that day. Those rows are dropped rather than mixed in on the
# absolute scale, which would give the model two different label definitions.
MIN_CROSS_SECTION_NAMES = 5


def apply_cross_sectional_target(
    panel_by_ticker: Dict[str, pd.DataFrame],
    target_column: str,
    method: str = "cross_sectional_rank",
    min_names: int = MIN_CROSS_SECTION_NAMES,
) -> Dict[str, pd.DataFrame]:
    """Restate each ticker's label relative to the cross-section on its date.

    The model was trained to predict the *absolute* forward return of a stock.
    In an equity panel most of the variance of a 5-day return is the common
    market factor — a typical Indian mid-cap runs an R^2 against the Nifty of
    0.35-0.55 daily, and higher over a week — so the network spent most of its
    capacity forecasting the market. That is both nearly unforecastable and
    unusable: this platform is long-only with no index hedge, so it cannot act
    on a market view at all. The only component it can monetize is the
    idiosyncratic part, which is what choosing *between* stocks expresses, and
    that was a minority of the training signal.

    Two transforms, both the standard fix from the empirical asset pricing
    literature (Gu, Kelly & Xiu):

    - ``cross_sectional_demean``: y - mean(y over names on that date). Keeps
      return units, so the label stays interpretable as an excess return.
    - ``cross_sectional_rank``: 2*rank/(N+1) - 1, mapped to [-1, 1]. The more
      robust of the two on Indian data, because a circuit-limited +20% print
      dominates the cross-sectional mean but moves a rank by one place.

    **This introduces no look-ahead.** The transform mixes only labels dated at
    the same t, each of which is already realized at t+H; it adds no
    information that the raw forward return did not already contain. It is a
    label transform only, so nothing about inference changes: the model still
    scores one ticker at a time and its output is a relative score, which is
    what the ranking downstream of it always wanted.

    Args:
        panel_by_ticker: Featurized, date-indexed frames keyed by ticker. The
            target must be the last column (the convention prepare_features
            establishes).
        target_column: Name of the target column.
        method: "cross_sectional_rank", "cross_sectional_demean", or
            "absolute" (returns the panel unchanged).
        min_names: Minimum names on a date for its cross-section to be usable.

    Returns:
        A new dict of frames with the target restated. Dates with too thin a
        cross-section are dropped. Falls back to the unchanged panel when there
        are too few tickers for any relative target to mean anything.
    """
    if method == "absolute" or len(panel_by_ticker) < 2:
        return panel_by_ticker
    if method not in ("cross_sectional_rank", "cross_sectional_demean"):
        raise ValueError(
            f"unknown target transform {method!r}; expected 'absolute', "
            f"'cross_sectional_demean' or 'cross_sectional_rank'"
        )

    # Wide (date x ticker) view of the labels only. Every ticker's frame keeps
    # its own rows; this is purely a lookup for what the rest of the universe
    # did on the same date.
    wide = pd.DataFrame(
        {ticker: frame[target_column] for ticker, frame in panel_by_ticker.items()}
    )

    usable = wide.notna().sum(axis=1) >= max(2, int(min_names))
    wide = wide[usable]
    if wide.empty:
        return panel_by_ticker

    if method == "cross_sectional_demean":
        restated = wide.sub(wide.mean(axis=1), axis=0)
    else:
        ranks = wide.rank(axis=1)
        counts = wide.notna().sum(axis=1)
        restated = ranks.mul(2.0).div(counts + 1.0, axis=0).sub(1.0)

    transformed: Dict[str, pd.DataFrame] = {}
    for ticker, frame in panel_by_ticker.items():
        if ticker not in restated.columns:
            continue
        labels = restated[ticker].dropna()
        kept = frame.index.intersection(labels.index)
        if kept.empty:
            continue
        updated = frame.loc[kept].copy()
        updated[target_column] = labels.loc[kept].to_numpy()
        transformed[ticker] = updated

    return transformed or panel_by_ticker
