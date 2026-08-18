"""Panel assembly shared by every trainer.

One place that turns "a list of tickers" into "standardized feature frames plus
aligned prices", so a new trainer does not re-derive the data path — and does
not re-derive its bugs. Three of them are easy to write and hard to see:

- **Fitting the scaler on everything.** `FeatureScaler.fit` is documented as
  taking *training rows only*. Fitting on the whole history lets the validation
  segment's mean and variance into the transform, and the resulting validation
  score is optimistic by an amount nobody can bound afterwards. The split
  happens here, before the fit, and the fit sees only the training block.

- **Assuming the frame is contiguous.** Rows with a non-finite feature are
  dropped (a ratio feature divided by a zero-volume week, say), which leaves
  gaps in the index. Prices are re-aligned to the surviving feature index and
  returns are computed on *that*, so a gap yields the true realized return
  across it rather than a one-day return computed over a splice.

- **Losing the column order.** The scaler and the network are both positional.
  Feature frames are built and kept in the caller's `feature_names` order, and
  that order is what travels into the checkpoint.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from portfolio_agent.features.pipeline import build_features
from portfolio_agent.features.scaling import FeatureScaler

from .base import TrainingData

logger = logging.getLogger(__name__)

#: Rows required before a ticker is kept — a floor, not the answer. It reads
#: as a judgement about sample adequacy (one trading year), and
#: `effective_min_history` raises it to the requested features' warm-up
#: whenever that is larger. It used to carry both meanings at once, and was
#: only ever correct because 252 happens to exceed the registry's longest
#: lookback of 211.
DEFAULT_MIN_HISTORY = 252


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are not finite — NaN *or* infinite.

    `dropna()` alone leaves infinities behind, and every one of them becomes a
    NaN loss the moment it reaches a network. They are not hypothetical on this
    data: several features are ratios whose denominator can legitimately be
    zero in the cache.
    """
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def prepare_panel(
    app_config: Any,
    universe: Sequence[str],
    feature_names: Sequence[str],
    *,
    train_fraction: float = 0.8,
    min_history: int = DEFAULT_MIN_HISTORY,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fit_scaler: bool = True,
) -> TrainingData:
    """Load, featurize, split and standardize a universe.

    Args:
        app_config: Loaded AppConfig, supplying the feature-normalization
            settings.
        universe: Exact tickers to load. Pinned by the caller so runs compare.
        feature_names: Feature registry names, in the order the model expects.
        train_fraction: Chronological share of each ticker used for fitting.
        min_history: Rows required *after* cleaning for a ticker to be kept.
        start_date: Optional ISO lower bound on the price history.
        end_date: Optional ISO upper bound.
        fit_scaler: Whether to standardize. False leaves features raw, which is
            recorded in the artifact as `feature_scaler: null` so inference can
            tell that apart from a scaler someone forgot to save.

    Returns:
        A `TrainingData` holding only the tickers that survived.

    Raises:
        ValueError: If no ticker yielded usable history. Callers must not
            proceed on an empty panel.
    """
    from portfolio_agent.features.pipeline import effective_min_history
    from portfolio_agent.src.data_store import load_ticker_data

    feature_names = list(feature_names)
    # Never below the warm-up: under it the feature is NaN, and `_clean_frame`
    # would drop every such row anyway — so a too-low setting does not train on
    # NaN here so much as quietly train on a shorter sample than asked for.
    requested_min_history = min_history
    min_history = effective_min_history(feature_names, min_history)
    if min_history != requested_min_history:
        logger.info(
            "Raised min_history from %d to %d — the requested features are not "
            "defined below that",
            requested_min_history, min_history,
        )
    features_cfg = getattr(app_config, "features", None)
    normalize = bool(getattr(features_cfg, "normalize", False))
    normalize_window = int(getattr(features_cfg, "normalize_window", 252))

    features_by_ticker: Dict[str, pd.DataFrame] = {}
    prices_by_ticker: Dict[str, pd.DataFrame] = {}
    split_index: Dict[str, int] = {}

    skipped: Dict[str, int] = {"no_data": 0, "short": 0, "missing_features": 0, "error": 0}

    for ticker in universe:
        try:
            raw = load_ticker_data(ticker, start_date=start_date, end_date=end_date)
        except Exception as exc:  # pragma: no cover - depends on cache state
            logger.debug("Failed to load %s: %s", ticker, exc)
            skipped["error"] += 1
            continue

        if raw is None or len(raw) < min_history:
            skipped["no_data" if raw is None else "short"] += 1
            continue

        raw = raw.copy()
        raw.columns = [str(c).lower() for c in raw.columns]

        try:
            built = build_features(
                raw,
                feature_names,
                normalize=normalize,
                normalize_window=normalize_window,
            )
        except (KeyError, ValueError) as exc:
            # A feature that is not in the registry is a configuration error
            # worth surfacing loudly; it will hit every ticker identically.
            logger.warning("Feature build failed for %s: %s", ticker, exc)
            skipped["missing_features"] += 1
            continue

        missing = [name for name in feature_names if name not in built.columns]
        if missing:
            logger.warning("%s is missing features %s", ticker, missing)
            skipped["missing_features"] += 1
            continue

        frame = _clean_frame(built[feature_names])
        if len(frame) < min_history:
            skipped["short"] += 1
            continue

        cut = int(len(frame) * train_fraction)
        if cut <= 0 or cut >= len(frame):
            skipped["short"] += 1
            continue

        features_by_ticker[ticker] = frame
        # Re-align prices to the surviving feature rows so a return computed
        # downstream spans the same interval the two feature rows do.
        prices_by_ticker[ticker] = raw.reindex(frame.index)
        split_index[ticker] = cut

    if not features_by_ticker:
        raise ValueError(
            "No ticker produced usable history. Checked "
            f"{len(universe)} names: {skipped}. If 'no_data' dominates, the "
            "parquet cache is empty — run `portfolio-agent download-data`. If "
            "'missing_features' dominates, a requested feature is not in the "
            "registry (features/registry.py)."
        )

    logger.info(
        "Prepared %d/%d tickers (%d features); skipped %s",
        len(features_by_ticker), len(universe), len(feature_names), skipped,
    )

    scaler: Optional[FeatureScaler] = None
    if fit_scaler:
        scaler = _fit_scaler_on_training_rows(features_by_ticker, split_index)
        for ticker, frame in features_by_ticker.items():
            scaled = scaler.transform(frame.to_numpy(dtype=np.float64))
            features_by_ticker[ticker] = pd.DataFrame(
                scaled, index=frame.index, columns=frame.columns
            )

    return TrainingData(
        features_by_ticker=features_by_ticker,
        prices_by_ticker=prices_by_ticker,
        tickers=sorted(features_by_ticker),
        feature_names=feature_names,
        scaler=scaler,
        split_index_by_ticker=split_index,
    )


def _fit_scaler_on_training_rows(
    features_by_ticker: Dict[str, pd.DataFrame], split_index: Dict[str, int]
) -> FeatureScaler:
    """Fit one standardizer on the pooled *training* rows of every ticker.

    Pooling across tickers rather than fitting per-ticker is deliberate: the
    network sees one name at a time but is asked to compare them, so a feature
    has to mean the same thing in every column it appears in.
    """
    blocks: List[np.ndarray] = []
    for ticker, frame in features_by_ticker.items():
        cut = split_index.get(ticker, len(frame))
        block = frame.iloc[:cut].to_numpy(dtype=np.float64)
        if block.size:
            blocks.append(block)

    if not blocks:
        raise ValueError("no training rows to fit a scaler on")

    return FeatureScaler.fit(np.vstack(blocks))
