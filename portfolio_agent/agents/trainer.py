"""GPU-accelerated training loop for portfolio forecasting models."""

from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from portfolio_agent.config.schema import AppConfig, TrainingConfig
from portfolio_agent.data.dataset import TimeSeriesDataset, create_dataloaders
from portfolio_agent.features.pipeline import build_features
# Re-exported, not merely used: the label definition moved to features/labels.py
# so a trainer that is not this one can predict the identical target without
# importing PyTorch to reach it. Everything that imported these four names from
# `agents.trainer` — tests included — keeps working unchanged.
from portfolio_agent.features.labels import (  # noqa: F401
    MIN_CROSS_SECTION_NAMES,
    apply_cross_sectional_target,
    build_forward_return,
    target_column_name,
)
from portfolio_agent.features.scaling import FeatureScaler, apply_cross_sectional_scaling
from portfolio_agent.models.pytorch_models import PointLoss, QuantileLoss, sorted_quantiles
from portfolio_agent.models.registry import get_model
from portfolio_agent.src.calibration import IsotonicCalibrator, calibration_error
from portfolio_agent.src.data_store import load_ticker_data
from portfolio_agent.src.performance_stats import newey_west_standard_error
from portfolio_agent.src.universe import resolve_backtest_universe
from portfolio_agent.utils.device import get_device, mixed_precision_support
from portfolio_agent.utils.workers import (
    describe_worker_plan, resolve_dataloader_workers, resolve_process_workers,
)

TRAINING_FEATURE_NAMES = [
    'sma_20', 'sma_50', 'rsi_14', 'macd',
    'bollinger_pct_b', 'atr_14', 'return_1d', 'return_5d'
]

TRADING_DAYS_PER_YEAR = 252

# Global gradient-norm cap. Recurrent models on financial data produce
# occasional very large gradients (a single volatile window is enough), and one
# unclipped step is all it takes to move the weights somewhere every subsequent
# loss evaluates to NaN from. 1.0 is the standard choice for LSTMs.
GRAD_CLIP_NORM = 1.0


def _non_finite_loss_advice(use_mixed_precision: bool) -> str:
    """What to actually do about NaN/inf losses, given the current setup."""
    if use_mixed_precision:
        return (
            "Mixed precision is on: fp16 overflows at 65504, which any price-level "
            "feature can exceed. Set training.use_mixed_precision=false and re-run "
            "if this persists."
        )
    return (
        "Check the cached bars for the affected tickers — zero or missing prices "
        "make the ratio features (return_1d, bollinger_pct_b) non-finite."
    )


def head_width(training: TrainingConfig) -> int:
    """How many numbers the model's output layer emits.

    One for a point forecast; one per quantile when fitting pinball loss.
    Single-sourced here because the checkpoint, the walk-forward folds and the
    inference-time reconstruction in MLStrategy must all agree, and a mismatch
    surfaces as an inscrutable state-dict shape error.
    """
    if training.loss == "quantile":
        return max(1, len(training.quantiles))
    return 1


def median_output_index(training: TrainingConfig) -> Optional[int]:
    """Index of the median quantile in the model's output, or None for a point head.

    The median is the point forecast every scalar metric is computed against —
    directional accuracy, MSE, the strategy Sharpe. Picking the closest level
    to 0.5 rather than assuming the middle position keeps an asymmetric
    quantile set (say [0.05, 0.5, 0.7]) from being scored on the wrong column.
    """
    if training.loss != "quantile" or not training.quantiles:
        return None
    return min(range(len(training.quantiles)), key=lambda i: abs(training.quantiles[i] - 0.5))


def build_loss(training: TrainingConfig) -> nn.Module:
    """The training objective for this configuration.

    See models/pytorch_models.py for why quantile loss is the default: squared
    error on a 5-day equity return is minimized by a near-constant prediction,
    which validates well and forecasts nothing.
    """
    if training.loss == "quantile":
        return QuantileLoss(training.quantiles)
    return PointLoss()


def _generate_synthetic_ohlcv(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate a single synthetic random-walk OHLCV series.

    Used only when config.training.use_synthetic_data is set (offline/CI
    testing where no cached market data is available) — real training runs
    load actual cached tickers via load_data() below.
    """
    import numpy as np

    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')

    np.random.seed(seed)
    close_prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)

    return pd.DataFrame({
        'open': close_prices + np.random.randn(n_samples) * 0.1,
        'high': close_prices + np.abs(np.random.randn(n_samples)) * 0.3,
        'low': close_prices - np.abs(np.random.randn(n_samples)) * 0.3,
        'close': close_prices,
        'volume': np.random.randint(1000000, 10000000, n_samples),
    }, index=dates)


def _load_ticker_features(ticker: str, config: AppConfig) -> Optional[pd.DataFrame]:
    """Load and featurize one ticker, keeping its DatetimeIndex.

    Module-level (not a nested closure) so it can be dispatched across a
    ProcessPoolExecutor. The index is preserved deliberately: walk-forward
    validation splits by *date*, which is impossible once rows are stacked and
    re-indexed. Returns None if the ticker has insufficient cached history.
    """
    df = load_ticker_data(ticker)
    if df is None or len(df) < config.data.min_history_days:
        return None
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    try:
        feature_df = prepare_features(df, config, verbose=False)
    except Exception:
        return None
    if len(feature_df) < config.training.sequence_length * 2:
        return None
    return feature_df


def _load_and_split_ticker(
    ticker: str, config: AppConfig
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Load, featurize, and chronologically split (70/15/15) one ticker."""
    feature_df = _load_ticker_features(ticker, config)
    if feature_df is None:
        return None

    n = len(feature_df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return (
        feature_df.iloc[:train_end],
        feature_df.iloc[train_end:val_end],
        feature_df.iloc[val_end:],
    )


def load_panel_by_ticker(
    config: AppConfig, universe: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """Featurized, date-indexed frames keyed by ticker.

    The shared source for both panel constructions: load_data() stacks these
    into the single training panel, and run_walk_forward_validation() splits
    each one by date. Keeping them separate is what lets walk-forward respect
    chronology — see that function for why a row-index split of the stacked
    panel does not.

    Args:
        config: Application configuration.
        universe: Exact tickers to load, bypassing the cache draw below. Passed
            by the pluggable training layer (training/universe.py) when several
            runs must be compared on identical names — see the note on
            purpose="train" for what pinning gives up.
    """
    if config.training.use_synthetic_data:
        return {"SYNTHETIC": prepare_features(_generate_synthetic_ohlcv(), config, verbose=False)}

    # purpose="train" offsets the sampling seed, so the training universe is a
    # different draw from the cache than the backtest universe. Evaluating a
    # model on the very names it was fitted on is not out-of-sample in the
    # cross-sectional dimension, however carefully the dates are split.
    #
    # A caller that pins `universe` takes that separation on itself: it is the
    # right trade when the point of the run is to compare two models, which is
    # only meaningful when both saw the same names.
    tickers = list(universe) if universe else resolve_backtest_universe(
        max_tickers=config.data.universe_size,
        selection=config.data.universe_selection,
        seed=config.data.universe_seed,
        purpose="train",
    )
    if not tickers:
        raise RuntimeError(
            "No cached tickers found to build a training panel. Run "
            "`portfolio-agent download-data` first, or set "
            "training.use_synthetic_data=true for offline testing."
        )

    # Keyed by ticker rather than appended in completion order: workers finish
    # nondeterministically, and appending in that order would make the panel's
    # row order (and therefore training) differ run to run for identical
    # inputs. Reassembling in `tickers` order below makes the parallel path
    # produce byte-identical output to the serial path.
    by_ticker: Dict[str, pd.DataFrame] = {}
    show_progress = len(tickers) > 20
    failures = 0

    # Capped for the platform: on Windows each worker is a spawned interpreter
    # that re-imports torch and pandas, so "one per CPU" is how a 16 GB machine
    # ends up in the page file with a run that looks hung rather than failed.
    process_workers = resolve_process_workers(config.training.data_load_workers)
    if config.training.parallel_data_loading and len(tickers) > 1:
        print(describe_worker_plan(
            process_workers, resolve_dataloader_workers(config.training.num_workers)
        ))

    if config.training.parallel_data_loading and len(tickers) > 1:
        with ProcessPoolExecutor(max_workers=process_workers) as executor:
            futures = {executor.submit(_load_ticker_features, t, config): t for t in tickers}
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc="Loading training data", unit="ticker")
            for future in iterator:
                ticker = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    # One unloadable ticker must not abort a multi-hour run.
                    failures += 1
                    print(f"Warning: failed to load training data for {ticker}: {e}")
                    continue
                if result is not None:
                    by_ticker[ticker] = result
    else:
        ticker_iter = tqdm(tickers, desc="Loading training data", unit="ticker") if show_progress else tickers
        for ticker in ticker_iter:
            try:
                result = _load_ticker_features(ticker, config)
            except Exception as e:
                failures += 1
                print(f"Warning: failed to load training data for {ticker}: {e}")
                continue
            if result is not None:
                by_ticker[ticker] = result

    if failures:
        print(f"Warning: {failures}/{len(tickers)} tickers failed to load and were skipped")

    ordered = {t: by_ticker[t] for t in tickers if t in by_ticker}
    if not ordered:
        raise RuntimeError(
            "None of the resolved tickers had enough cached history to build a "
            "training panel. Run `portfolio-agent download-data` first, or set "
            "training.use_synthetic_data=true for offline testing."
        )

    transform = config.training.target_transform
    if transform != "absolute" and len(ordered) >= 2:
        target_name = target_column_name(config.training.target)
        before = sum(len(f) for f in ordered.values())
        ordered = apply_cross_sectional_target(ordered, target_name, transform)
        after = sum(len(f) for f in ordered.values())
        print(
            f"Target restated as {transform} across {len(ordered)} names "
            f"({before - after} rows dropped for too thin a cross-section)"
        )
    elif transform != "absolute":
        print(
            f"Target transform {transform!r} needs at least 2 tickers; "
            f"training on the absolute forward return instead"
        )

    # Inputs get the same treatment as the label, and for the same reason: a
    # feature measured against a pooled five-year mean carries the market
    # factor the label transform just removed. Applied here, while the panel is
    # still keyed by ticker and date — downstream it is stacked into a flat
    # matrix and the cross-section is no longer recoverable.
    if config.training.feature_normalization == "cross_sectional" and len(ordered) >= 2:
        target_name = target_column_name(config.training.target)
        feature_columns = [
            c for c in next(iter(ordered.values())).columns if c != target_name
        ]
        before = sum(len(f) for f in ordered.values())
        ordered = apply_cross_sectional_scaling(ordered, feature_columns)
        after = sum(len(f) for f in ordered.values())
        print(
            f"Features standardized cross-sectionally per date across "
            f"{len(ordered)} names ({before - after} rows dropped for too thin "
            f"a cross-section)"
        )

    return ordered


def load_data(config: AppConfig, universe: Optional[List[str]] = None) -> pd.DataFrame:
    """Load and featurize training data into a single stacked panel.

    Each ticker is featurized and split 70/15/15 chronologically
    *individually*, then all tickers' train portions are concatenated,
    followed by all val portions, then all test portions. That ordering lets
    create_dataloaders()'s single top-level 70/15/15 index split land exactly
    on those boundaries, so validation/test proportionally represent every
    ticker rather than only the last one in the panel.

    Note what this ordering is *not* suitable for: because it groups by split
    and then by ticker, a row-index split of the result is not chronological
    across the panel. Walk-forward validation therefore works from
    load_panel_by_ticker() and splits by date — see
    run_walk_forward_validation().

    Sequence windows that straddle two concatenated tickers' boundaries mix
    data from different instruments; this is a bounded, documented limitation
    of pooling multiple series through a single-series windowing dataset
    (TimeSeriesDataset), not a look-ahead bias — it affects at most
    sequence_length * (n_tickers - 1) windows out of the full panel.

    Args:
        config: Application configuration.
        universe: Exact tickers to load, bypassing the cache draw.

    Returns:
        DataFrame with computed features and target column (already featurized).
    """
    by_ticker = load_panel_by_ticker(config, universe)

    train_parts, val_parts, test_parts = [], [], []
    for frame in by_ticker.values():
        n = len(frame)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        train_parts.append(frame.iloc[:train_end])
        val_parts.append(frame.iloc[train_end:val_end])
        test_parts.append(frame.iloc[val_end:])

    combined = pd.concat(train_parts + val_parts + test_parts, ignore_index=True)
    print(f"Built training panel from {len(by_ticker)} tickers: {len(combined)} total rows")
    return combined


def prepare_features(df: pd.DataFrame, config: AppConfig, verbose: bool = True) -> pd.DataFrame:
    """Build feature matrix and forward-return target from raw OHLCV data.

    The returned frame's last column is always the target; every other column
    is an input feature. Downstream code (create_dataloaders, the walk-forward
    splitter, and the metadata written for MLStrategy) relies on that ordering.

    Args:
        df: Raw OHLCV DataFrame.
        config: Application configuration.
        verbose: Whether to print a summary (disabled when called per-ticker
            from load_data()'s multi-ticker panel construction).

    Returns:
        DataFrame with computed features and the forward-return target.
    """
    feature_df = build_features(
        df,
        TRAINING_FEATURE_NAMES,
        normalize=config.features.normalize,
        normalize_window=config.features.normalize_window
    )

    # Target is always recomputed as a forward return under a namespaced
    # column, so a feature of the same name (e.g. the trailing return_5d)
    # stays an input and never silently becomes the label.
    target_name = target_column_name(config.training.target)
    feature_df[target_name] = build_forward_return(df['close'], config.training.target)

    # Drop rows that are not finite — NaN *or* infinite. dropna() alone leaves
    # the infinities behind, and every one of them turns into a NaN loss the
    # moment it reaches the network. They are not hypothetical on this data:
    # several features are ratios (`return_1d` divides by the previous close,
    # `volume_ratio_20` by a 20-day average volume, `bollinger_pct_b` by the
    # band width), and a cached bar with a zero price or a zero-volume week
    # makes the denominator zero. So does the forward-return target itself.
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).dropna()

    # Finite is not the same as usable. Input features are standardized and
    # clipped to +/-10 sigma before training, but the *target* was passed
    # through untouched, and a single bad cached bar is enough to poison a run:
    # one close printed at 0.001 turns a 5-day forward return into 111,300
    # (eleven million percent), and one gradient step against a loss that size
    # moves the weights somewhere every subsequent batch evaluates to NaN.
    #
    # This is why NaN losses survived the mixed-precision fix and still appear
    # on CPU — the cause was never fp16, it was the label.
    #
    # Rows are dropped rather than clipped: a clip would pile a spike of
    # samples at the bound and teach the model that the bound is a common
    # outcome. The default admits any genuinely reachable move — five
    # consecutive 20% upper circuits compound to +149% — and rejects only
    # arithmetic that cannot be a price.
    target_values = feature_df[target_name]
    absurd = target_values.abs() > config.training.max_abs_target
    if absurd.any():
        if verbose:
            print(
                f"Dropped {int(absurd.sum())} row(s) whose |{target_name}| exceeded "
                f"{config.training.max_abs_target:g} "
                f"(max was {target_values.abs().max():.4g}) — almost certainly bad cached bars"
            )
        feature_df = feature_df[~absurd]

    if verbose:
        print(f"Built feature matrix with {len(feature_df)} samples and {len(feature_df.columns)} columns")
        print(f"Features: {list(feature_df.columns[:-1])}")
        print(f"Target: {feature_df.columns[-1]}")

    return feature_df


def _target_horizon_days(target_name: str) -> int:
    """Forecast horizon implied by a target name like 'return_5d'.

    Used only to annualize the walk-forward Sharpe ratios; defaults to 1 day
    for target names that don't encode a horizon.
    """
    digits = "".join(ch for ch in str(target_name) if ch.isdigit())
    try:
        return max(1, int(digits))
    except ValueError:
        return 1


def evaluate_predictions(
    predictions: np.ndarray,
    actuals: np.ndarray,
    horizon_days: int = 5,
    relative_target: bool = False,
) -> Dict[str, float]:
    """Score out-of-sample predictions on the terms a trader cares about.

    A regression loss says how close the predicted number is; it does not say
    whether acting on the prediction makes money. Both are reported, plus the
    benchmark that any forecasting model has to beat before it earns a place
    in the stack:

    - ``mse`` — the loss the model was actually trained on.
    - ``directional_accuracy`` — fraction of predictions whose sign matches
      the realized return. On noisy daily equity data anything much above 0.5
      is the whole edge.
    - ``strategy_sharpe`` — annualized Sharpe of a long-only rule that takes
      the position when the prediction is positive and sits in cash otherwise
      (long-only because this platform never shorts). Gross of costs; see
      src/execution_sim.py for what friction would remove.
    - ``benchmark_sharpe`` — annualized Sharpe of always being long. A model
      whose strategy Sharpe does not clear this is adding turnover, not alpha.
    - ``excess_sharpe`` — strategy minus benchmark, the number to judge on.

    Sharpe is annualized by sqrt(252 / horizon_days). With daily-sampled
    multi-day targets the observations overlap, which understates the standard
    error, so these ratios read slightly high in absolute terms — they are
    meaningful as a comparison between strategy and benchmark, both of which
    are computed on the identical overlapping sample.

    - ``strategy_t_stat`` — t-statistic of the strategy's mean return with a
      Newey-West standard error at lag H-1. Daily-sampled H-day targets overlap
      by H-1 days, and treating them as independent understates the standard
      error by roughly sqrt(H) — in the direction that manufactures
      significance. The Sharpe is still the right point estimate; what the
      naive error gets wrong is how much to believe it.
    - ``rank_ic`` — Spearman correlation between the predicted and realized
      ordering. This is the metric a ranking system should be judged on: it is
      far less noisy than a backtested P&L and it does not confound signal
      quality with position sizing, costs or the covariance of the book.

    Args:
        predictions: Model outputs, shape (n,).
        actuals: Realized target values, shape (n,).
        horizon_days: Forecast horizon in trading days, for annualization.
        relative_target: True when the label is a cross-sectional excess or
            rank rather than a return (see apply_cross_sectional_target). The
            Sharpe figures assume the label is in return units — a rank of
            +0.4 is not 40% — so they are reported as zero rather than as a
            confident number about the wrong quantity, and rank IC carries the
            evaluation instead.

    Returns:
        Dictionary of metrics; zeros when there is nothing to score.
    """
    predictions = np.asarray(predictions, dtype=float).ravel()
    actuals = np.asarray(actuals, dtype=float).ravel()

    empty = {
        "n_samples": 0, "mse": 0.0, "directional_accuracy": 0.0,
        "strategy_sharpe": 0.0, "benchmark_sharpe": 0.0, "excess_sharpe": 0.0,
        "strategy_t_stat": 0.0, "rank_ic": 0.0,
        "relative_target": float(relative_target),
    }
    if predictions.size == 0 or predictions.size != actuals.size:
        return empty

    finite = np.isfinite(predictions) & np.isfinite(actuals)
    predictions, actuals = predictions[finite], actuals[finite]
    if predictions.size == 0:
        return empty

    mse = float(np.mean((predictions - actuals) ** 2))
    directional_accuracy = float(np.mean(np.sign(predictions) == np.sign(actuals)))

    rank_ic = 0.0
    if predictions.size > 1:
        predicted_rank = pd.Series(predictions).rank().to_numpy()
        realized_rank = pd.Series(actuals).rank().to_numpy()
        if np.std(predicted_rank) > 0 and np.std(realized_rank) > 0:
            rank_ic = float(np.corrcoef(predicted_rank, realized_rank)[0, 1])

    annualization = math.sqrt(TRADING_DAYS_PER_YEAR / max(1, horizon_days))

    def _sharpe(returns: np.ndarray) -> float:
        if returns.size < 2:
            return 0.0
        sigma = float(np.std(returns, ddof=1))
        if sigma <= 0:
            return 0.0
        return float(np.mean(returns) / sigma * annualization)

    strategy_t_stat = 0.0
    if relative_target:
        strategy_sharpe = benchmark_sharpe = 0.0
    else:
        strategy_returns = np.where(predictions > 0, actuals, 0.0)
        strategy_sharpe = _sharpe(strategy_returns)
        benchmark_sharpe = _sharpe(actuals)

        # Overlapping labels: a daily-sampled H-day return shares H-1 days with
        # its neighbour, so treating the observations as independent understates
        # the standard error by roughly sqrt(H) — in the direction that
        # manufactures significance. The Sharpe above is still the right point
        # estimate; what the naive standard error gets wrong is how much to
        # believe it, so the correction is reported as a t-statistic rather than
        # applied to the ratio.
        standard_error = newey_west_standard_error(
            strategy_returns, lags=max(0, horizon_days - 1)
        )
        if standard_error > 0:
            strategy_t_stat = float(np.mean(strategy_returns) / standard_error)

    return {
        "n_samples": int(predictions.size),
        "mse": mse,
        "directional_accuracy": directional_accuracy,
        "strategy_sharpe": strategy_sharpe,
        "benchmark_sharpe": benchmark_sharpe,
        "excess_sharpe": strategy_sharpe - benchmark_sharpe,
        # Newey-West corrected, so it can be read against a conventional
        # hurdle. Harvey, Liu & Zhu argue for t > 3.0 rather than 2.0 on
        # anything drawn from a wide search, which this is.
        "strategy_t_stat": strategy_t_stat,
        "rank_ic": rank_ic,
        "relative_target": float(relative_target),
    }


def _make_loader(
    features: np.ndarray,
    targets: np.ndarray,
    config: TrainingConfig,
    shuffle: bool = False,
    drop_last: bool = False,
) -> Optional[DataLoader]:
    """Wrap a contiguous feature/target slice in a sequence DataLoader.

    Returns None when the slice is shorter than one sequence, so a fold that
    lands on too little data is skipped rather than raising.
    """
    if len(features) <= config.sequence_length:
        return None
    dataset = TimeSeriesDataset(features, targets, config.sequence_length)
    if len(dataset) == 0:
        return None
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,           # never True here: temporal order is the point
        num_workers=0,             # folds are short; worker startup dominates
        drop_last=drop_last,
    )


def _train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    learning_rate: float,
    use_mixed_precision: bool,
    patience: int = 10,
    min_delta: float = 1e-4,
    progress_label: str = "",
    on_improve=None,
    loss_fn: Optional[nn.Module] = None,
    grad_clip_norm: float = GRAD_CLIP_NORM,
    verbose: bool = True,
) -> Tuple[List[float], List[float], float]:
    """Train `model` with early stopping on validation loss.

    Shared by the final fit and by every walk-forward fold, so a fold is
    trained exactly the way the shipped model is — a validation procedure that
    trains differently from production is measuring the wrong thing.

    Args:
        model: Model to train in place.
        train_loader: Training batches.
        val_loader: Validation batches; without one, early stopping is
            disabled and the model trains for the full epoch budget.
        device: Device to train on.
        epochs: Maximum epochs.
        learning_rate: AdamW learning rate.
        use_mixed_precision: Whether to use CUDA autocast + GradScaler.
        patience: Epochs without improvement before stopping.
        min_delta: Minimum validation-loss improvement that counts.
        progress_label: Prefix for the progress bar description.
        on_improve: Optional callback(epoch, val_loss) invoked whenever
            validation loss improves — used to checkpoint the best weights.
        loss_fn: Training objective. Defaults to squared error only so a bare
            call still works; real runs pass build_loss(config.training), and
            folds must be given the *same* objective as the final fit or the
            generalization estimate describes a different model.
        grad_clip_norm: Global gradient-norm cap. 0 disables clipping.
        verbose: Print a one-line summary after every epoch.

    Returns:
        (train_losses, val_losses, best_val_loss). A returned loss is NaN only
        when every batch in that epoch was non-finite, which is reported
        loudly rather than being left to surface as a silent `nan`.
    """
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = (loss_fn or PointLoss()).to(device)
    scaler = torch.amp.GradScaler('cuda') if use_mixed_precision else None

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val_loss = float('inf')
    patience_counter = 0
    # Counted across the whole run, not per epoch: one bad batch is a data
    # artifact, thousands mean the run is not training and the operator needs
    # to be told why rather than reading `nan` off a progress bar.
    skipped_batches = 0

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"{progress_label}Epoch {epoch + 1}/{epochs} [Train]",
            leave=False,
            mininterval=0.5,
        )
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_mixed_precision:
                with torch.autocast('cuda'):
                    outputs = model(batch_x)
                    loss = loss_fn(outputs, batch_y)
            else:
                outputs = model(batch_x)
                loss = loss_fn(outputs, batch_y)

            # A non-finite loss cannot produce a usable gradient, and stepping
            # on it writes NaN into every weight — after which *every*
            # subsequent batch is NaN and the whole run is dead while still
            # printing epochs. Skipping the batch keeps one bad window (or one
            # fp16 overflow) from destroying the model.
            if not torch.isfinite(loss):
                skipped_batches += 1
                continue

            if use_mixed_precision:
                scaler.scale(loss).backward()
                if grad_clip_norm > 0:
                    # Gradients are still multiplied by the scale factor here,
                    # so they have to be unscaled before a norm computed in
                    # real units means anything.
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            epoch_train_loss += loss.item()
            num_train_batches += 1
            pbar.set_postfix(loss=f"{epoch_train_loss / num_train_batches:.6f}", refresh=False)

        # NaN rather than 0.0 when nothing was usable: a zero would read as a
        # perfectly fitted epoch and would win every early-stopping comparison.
        avg_train_loss = (
            epoch_train_loss / num_train_batches if num_train_batches else float('nan')
        )
        train_losses.append(avg_train_loss)

        if val_loader is None:
            if verbose:
                print(f"  {progress_label}Epoch {epoch + 1}/{epochs}: train_loss={avg_train_loss:.6f}")
            continue

        model.eval()
        epoch_val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                if use_mixed_precision:
                    with torch.autocast('cuda'):
                        outputs = model(batch_x)
                        loss = loss_fn(outputs, batch_y)
                else:
                    outputs = model(batch_x)
                    loss = loss_fn(outputs, batch_y)
                if not torch.isfinite(loss):
                    skipped_batches += 1
                    continue
                epoch_val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = (
            epoch_val_loss / num_val_batches if num_val_batches else float('nan')
        )
        val_losses.append(avg_val_loss)

        if verbose:
            print(
                f"  {progress_label}Epoch {epoch + 1}/{epochs}: "
                f"train_loss={avg_train_loss:.6f} val_loss={avg_val_loss:.6f}"
            )

        # NaN fails this comparison, so a dead epoch can never be checkpointed
        # as the best model.
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            if on_improve is not None:
                on_improve(epoch + 1, avg_val_loss, optimizer)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered (patience={patience} exceeded)")
                break

    if skipped_batches:
        print(
            f"  WARNING: skipped {skipped_batches} batch(es) whose loss was not finite. "
            f"{_non_finite_loss_advice(use_mixed_precision)}"
        )

    return train_losses, val_losses, best_val_loss


def _predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    median_index: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect (point predictions, actuals) over a loader, in order.

    With a quantile head the point prediction is the median column, taken after
    sorting so a crossed set of quantiles cannot put the 90th percentile where
    the median belongs.

    Args:
        model: Model to run.
        loader: Batches to predict over.
        device: Device to run on.
        median_index: Column holding the median quantile, or None for a
            single-output point head.

    Returns:
        (predictions, actuals), both shape (n,).
    """
    model.eval()
    predictions: List[np.ndarray] = []
    actuals: List[np.ndarray] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            outputs = model(batch_x)
            if median_index is not None and outputs.dim() > 1 and outputs.shape[-1] > 1:
                outputs = sorted_quantiles(outputs)[..., median_index]
            else:
                outputs = outputs.squeeze(-1)
            predictions.append(outputs.detach().cpu().numpy().ravel())
            actuals.append(batch_y.detach().cpu().numpy().ravel())
    if not predictions:
        return np.array([]), np.array([])
    return np.concatenate(predictions), np.concatenate(actuals)


def _stack_blocks(blocks: List[pd.DataFrame]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Concatenate per-ticker blocks into (features, targets) arrays."""
    usable = [b for b in blocks if b is not None and not b.empty]
    if not usable:
        return None
    panel = pd.concat(usable)
    return panel.iloc[:, :-1].values, panel.iloc[:, -1].values


def _fit_confidence_calibration(
    predictions: np.ndarray,
    actuals: np.ndarray,
) -> Dict[str, Any]:
    """Fit and score an isotonic map from model score to realized win rate.

    The outcome being calibrated against is "did the forward return come out
    positive" — the question a long-only platform actually acts on, and the p
    that Kelly sizing and the trigger engine's expected-value hurdle both
    consume. Reporting the expected calibration error before and after makes
    the correction auditable instead of a black box: if the raw model was
    already calibrated, the improvement is zero and the map is close to the
    identity.

    Args:
        predictions: Pooled out-of-sample point forecasts.
        actuals: Realized forward returns aligned to them.

    Returns:
        A JSON-serializable dict with the fitted map and its scores, or a
        `skipped` marker when there is too little evidence to fit one.
    """
    wins = (np.asarray(actuals, dtype=float) > 0).astype(float)
    calibrator = IsotonicCalibrator.fit(predictions, wins)
    if calibrator is None:
        return {"skipped": "not enough out-of-sample folds to calibrate confidence"}

    # A raw regression output is not a probability, so the "before" baseline is
    # the same squashing MLStrategy applies when no calibrator is available.
    raw_probabilities = np.clip(0.5 + np.asarray(predictions, dtype=float), 0.0, 1.0)
    calibrated = calibrator.predict(predictions)

    result = calibrator.to_dict()
    result.update({
        "expected_calibration_error_raw": calibration_error(raw_probabilities, wins),
        "expected_calibration_error_calibrated": calibration_error(calibrated, wins),
    })
    print(
        f"Confidence calibration: ECE {result['expected_calibration_error_raw']:.3f} -> "
        f"{result['expected_calibration_error_calibrated']:.3f} "
        f"over {calibrator.n_samples} out-of-sample predictions "
        f"(base rate {calibrator.base_rate:.1%})"
    )
    return result


def run_walk_forward_validation(
    panel_by_ticker: Dict[str, pd.DataFrame],
    config: AppConfig,
    device: torch.device,
    use_mixed_precision: bool = False,
) -> Dict[str, Any]:
    """Expanding-window walk-forward validation, split by calendar date.

    A single chronological 70/15/15 split answers "how did this model do on
    2023?" — one regime, one initialization, one number. That is not enough
    evidence to deploy a 5-day return forecaster on a market with this
    signal-to-noise ratio. Walk-forward re-fits on an expanding history and
    tests on the *next* contiguous period, repeatedly:

        fold 1: train dates < T1        test dates in [T1, T2)
        fold 2: train dates < T2        test dates in [T2, T3)
        ...

    **Splitting by date, not by row index.** The stacked training panel is
    ordered [every ticker's train block][every val block][every test block],
    so an index-based split of it would put 2019 rows for ticker B in the
    "future" test block while 2019 rows for ticker A sit in the training
    block — an out-of-sample measurement covering the same calendar dates it
    trained on. Each ticker is therefore split by date individually and the
    blocks concatenated afterwards, which keeps every ticker's rows contiguous
    (as the sequence windows require) while guaranteeing no training row is
    dated at or after its fold's test period.

    **Embargo.** The target is a *forward* return over `horizon_days`, so the
    last few training rows before a boundary carry labels computed from prices
    inside the test period. Those rows are dropped, or the model would be
    fitted against labels that already encode the moves it is about to be
    scored on.

    Each fold holds back the tail of its own training window for early
    stopping, so the test period is never seen during fitting. Folds train
    with a reduced epoch budget (`walk_forward_epochs`, default
    min(20, epochs)): their job is to estimate generalization, not to produce
    the shipped weights.

    Args:
        panel_by_ticker: Featurized, date-indexed frames keyed by ticker (see
            load_panel_by_ticker); the last column of each is the target.
        config: Application configuration.
        device: Device to train on.
        use_mixed_precision: Whether to use CUDA autocast.

    Returns:
        Dictionary with per-fold metrics and their averages, or a `skipped`
        marker when the history is too short to split.
    """
    training = config.training
    n_splits = training.walk_forward_splits
    if n_splits <= 0:
        return {"skipped": "walk_forward_splits <= 0"}
    if not panel_by_ticker:
        return {"skipped": "no tickers in the training panel"}

    sample = next(iter(panel_by_ticker.values()))
    feature_cols = sample.columns[:-1].tolist()
    target_col = sample.columns[-1]
    horizon_days = _target_horizon_days(target_col)

    # A cross-sectional label is not in return units, so the Sharpe-style
    # metrics below do not apply to it and rank IC carries the evaluation
    # instead (see evaluate_predictions). The panel is only actually
    # transformed when there were enough names to rank against, so this reads
    # the same condition load_panel_by_ticker applied.
    relative_target = (
        training.target_transform != "absolute" and len(panel_by_ticker) >= 2
    )

    # Fold boundaries come from the pooled distribution of dates, so each fold
    # holds a comparable number of observations even with ragged histories.
    all_dates = np.sort(np.concatenate([
        frame.index.values for frame in panel_by_ticker.values() if len(frame)
    ])) if panel_by_ticker else np.array([])
    if len(all_dates) == 0:
        return {"skipped": "training panel has no dated rows"}

    first_fraction = training.walk_forward_min_train_fraction
    boundaries = [
        pd.Timestamp(all_dates[min(len(all_dates) - 1, int(len(all_dates) * q))])
        for q in np.linspace(first_fraction, 1.0, n_splits + 1)
    ]

    epochs = training.walk_forward_epochs or min(20, training.epochs)
    model_class = get_model(training.model)
    n_outputs = head_width(training)
    median_index = median_output_index(training)
    loss_fn = build_loss(training)

    # Every fold's test-period predictions, pooled. These are the only
    # genuinely out-of-sample scores the run produces, which makes them the
    # only defensible sample to calibrate confidence on: fitting the map on
    # training predictions would measure memorization and hand back a mapping
    # that makes an overfitted model look perfectly calibrated.
    oos_predictions: List[np.ndarray] = []
    oos_actuals: List[np.ndarray] = []

    print("\n" + "=" * 60)
    print(f"Walk-Forward Validation ({n_splits} expanding-window folds, split by date)")
    print("=" * 60)

    fold_metrics: List[Dict[str, Any]] = []
    for fold in range(n_splits):
        train_end_date = boundaries[fold]
        test_end_date = boundaries[fold + 1]
        if test_end_date <= train_end_date:
            continue

        train_blocks: List[pd.DataFrame] = []
        val_blocks: List[pd.DataFrame] = []
        test_blocks: List[pd.DataFrame] = []

        for frame in panel_by_ticker.values():
            history = frame[frame.index < train_end_date]
            # Embargo: labels of the final `horizon_days` training rows are
            # computed from prices inside the test period.
            if horizon_days > 0:
                history = history.iloc[:-horizon_days] if len(history) > horizon_days else history.iloc[:0]
            if len(history) <= training.sequence_length:
                continue

            inner_val_size = max(training.sequence_length + 1, int(len(history) * 0.15))
            inner_train_end = len(history) - inner_val_size
            if inner_train_end <= training.sequence_length:
                continue

            train_blocks.append(history.iloc[:inner_train_end])
            val_blocks.append(history.iloc[inner_train_end:])

            test = frame[(frame.index >= train_end_date) & (frame.index < test_end_date)]
            if len(test) > training.sequence_length:
                test_blocks.append(test)

        train_data = _stack_blocks(train_blocks)
        test_data = _stack_blocks(test_blocks)
        val_data = _stack_blocks(val_blocks)
        if train_data is None or test_data is None:
            print(f"Fold {fold + 1}/{n_splits}: skipped (not enough rows)")
            continue

        # Standardize with statistics from this fold's training block only.
        # Re-fitting per fold is not an optimization — a scaler fitted on the
        # whole panel would carry the mean and spread of the test period into
        # the training inputs, which is exactly the leakage the date split
        # exists to prevent.
        fold_scaler = FeatureScaler.fit(train_data[0])
        train_data = (fold_scaler.transform(train_data[0]), train_data[1])
        test_data = (fold_scaler.transform(test_data[0]), test_data[1])
        if val_data is not None:
            val_data = (fold_scaler.transform(val_data[0]), val_data[1])

        train_loader = _make_loader(*train_data, training, drop_last=True)
        val_loader = _make_loader(*val_data, training) if val_data is not None else None
        test_loader = _make_loader(*test_data, training)
        if train_loader is None or test_loader is None:
            print(f"Fold {fold + 1}/{n_splits}: skipped (not enough rows)")
            continue

        model = model_class(
            n_features=len(feature_cols),
            hidden_size=64,
            n_layers=2,
            sequence_length=training.sequence_length,
            dropout=0.2,
            n_outputs=n_outputs,
        ).to(device)

        _train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            learning_rate=training.learning_rate,
            use_mixed_precision=use_mixed_precision,
            progress_label=f"[fold {fold + 1}/{n_splits}] ",
            loss_fn=loss_fn,
            # Folds print one summary line each (below); a per-epoch line per
            # fold would bury it.
            verbose=False,
        )

        predictions, actuals = _predict(model, test_loader, device, median_index)
        oos_predictions.append(predictions)
        oos_actuals.append(actuals)
        metrics = evaluate_predictions(
            predictions, actuals, horizon_days, relative_target=relative_target
        )
        metrics.update({
            "fold": fold + 1,
            "train_end": train_end_date.strftime("%Y-%m-%d"),
            "test_end": test_end_date.strftime("%Y-%m-%d"),
            "train_rows": int(len(train_data[0])),
            "test_rows": int(len(test_data[0])),
        })
        fold_metrics.append(metrics)

        headline = (
            f"rank_IC={metrics['rank_ic']:+.4f}" if relative_target
            else (
                f"Sharpe={metrics['strategy_sharpe']:.2f} vs benchmark "
                f"{metrics['benchmark_sharpe']:.2f} (excess {metrics['excess_sharpe']:+.2f})"
            )
        )
        print(
            f"Fold {fold + 1}/{n_splits}: train<{metrics['train_end']} "
            f"test<{metrics['test_end']} ({metrics['test_rows']} rows) | "
            f"MSE={metrics['mse']:.6f} | dir_acc={metrics['directional_accuracy']:.3f} | "
            f"{headline}"
        )

    if not fold_metrics:
        return {"skipped": "no fold had enough rows to evaluate"}

    def _mean(key: str) -> float:
        return float(np.mean([m[key] for m in fold_metrics]))

    summary = {
        "n_folds": len(fold_metrics),
        "epochs_per_fold": epochs,
        "horizon_days": horizon_days,
        "embargo_days": horizon_days,
        "folds": fold_metrics,
        "target_transform": training.target_transform,
        "mean_mse": _mean("mse"),
        "mean_directional_accuracy": _mean("directional_accuracy"),
        "mean_rank_ic": _mean("rank_ic"),
        # Information ratio of the fold ICs: a mean IC is only as good as its
        # consistency across folds, and this is the ratio that says so.
        "rank_icir": (
            float(_mean("rank_ic") / np.std([m["rank_ic"] for m in fold_metrics], ddof=1))
            if len(fold_metrics) > 1
            and np.std([m["rank_ic"] for m in fold_metrics], ddof=1) > 0
            else 0.0
        ),
        "folds_with_positive_ic": sum(1 for m in fold_metrics if m["rank_ic"] > 0),
    }

    if not relative_target:
        # Only meaningful when the label is in return units; a rank of +0.4 is
        # not a 40% return, so a Sharpe computed on ranks is a confident number
        # about the wrong quantity.
        summary.update({
            "mean_strategy_sharpe": _mean("strategy_sharpe"),
            "mean_benchmark_sharpe": _mean("benchmark_sharpe"),
            "mean_excess_sharpe": _mean("excess_sharpe"),
            "folds_beating_benchmark": sum(
                1 for m in fold_metrics if m["excess_sharpe"] > 0
            ),
        })

    if training.calibrate_confidence and oos_predictions:
        summary["calibration"] = _fit_confidence_calibration(
            np.concatenate(oos_predictions), np.concatenate(oos_actuals)
        )

    print("-" * 60)
    if relative_target:
        print(
            f"Mean across {summary['n_folds']} folds: "
            f"dir_acc={summary['mean_directional_accuracy']:.3f} | "
            f"rank_IC={summary['mean_rank_ic']:+.4f} | "
            f"ICIR={summary['rank_icir']:+.2f} | "
            f"positive IC in {summary['folds_with_positive_ic']}/{summary['n_folds']} folds"
        )
        if summary["mean_rank_ic"] <= 0:
            print(
                "  WARNING: out-of-sample rank IC is not positive. The model orders the "
                "cross-section no better than chance — do not trade it without changing "
                "something."
            )
    else:
        print(
            f"Mean across {summary['n_folds']} folds: "
            f"dir_acc={summary['mean_directional_accuracy']:.3f} | "
            f"Sharpe={summary['mean_strategy_sharpe']:.2f} vs benchmark "
            f"{summary['mean_benchmark_sharpe']:.2f} | "
            f"excess={summary['mean_excess_sharpe']:+.2f} | "
            f"beat benchmark in {summary['folds_beating_benchmark']}/{summary['n_folds']} folds"
        )
        if summary["mean_excess_sharpe"] <= 0:
            print(
                "  WARNING: out-of-sample Sharpe does not beat always-long. The model is "
                "adding turnover, not alpha — do not trade it without changing something."
            )
    print("=" * 60)

    return summary


def run_training(
    config: AppConfig, universe: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Run GPU-accelerated training loop for portfolio forecasting model.
    
    This function implements:
    - Device resolution using get_device()
    - Data loading and feature engineering
    - Model initialization from registry
    - Mixed precision training (when enabled and on CUDA)
    - Training loop with progress bar
    - Validation and early stopping
    - Checkpointing best model weights
    
    Args:
        config: Application configuration containing training parameters.
        universe: Exact tickers to train on, bypassing the cache draw. Passed by
            the pluggable training layer so several runs can be compared on
            identical names.

    Returns:
        Dictionary with training metadata including:
            - feature_names: List of input feature names
            - target: Target variable name
            - sequence_length: Input sequence length
            - device: Device used for training
            - metrics: Training and validation loss history
            - best_val_loss: Best validation loss achieved
            - epochs_trained: Number of epochs completed
    """
    # =========================================================================
    # 1. Resolve device
    # =========================================================================
    # get_device() has already downgraded an unavailable accelerator, so
    # device.type is what this process will really run on.
    device = get_device(config.training.device)
    config.training.device = device.type

    # Mixed precision is only ever enabled on hardware where fp16 is both fast
    # and numerically safe. On a card without tensor cores — the GeForce GTX
    # 16-series in particular — autocast buys nothing and reliably turns the
    # loss into NaN, so it is refused here with the reason rather than left to
    # produce a run whose every epoch prints `nan`.
    amp_supported, amp_reason = mixed_precision_support(device)
    use_mixed_precision = config.training.use_mixed_precision and amp_supported

    if use_mixed_precision:
        print(f"Mixed Precision: Enabled ({amp_reason})")
    elif config.training.use_mixed_precision:
        print(f"Mixed Precision: Disabled — {amp_reason}")
    else:
        print("Mixed Precision: Disabled (training.use_mixed_precision=false)")

    # =========================================================================
    # 2. Load and featurize training data (real cached tickers by default)
    # =========================================================================
    print("\nLoading and featurizing training data...")
    panel_by_ticker = load_panel_by_ticker(config, universe)

    # =========================================================================
    # 3. Walk-forward validation (before the final fit)
    # =========================================================================
    # Run first, so the generalization estimate is available even if the final
    # long fit is interrupted — and so a model that cannot beat always-long is
    # flagged before anyone reads its training loss as evidence of anything.
    # Takes the per-ticker frames, not the stacked panel: it splits by date.
    walk_forward = run_walk_forward_validation(
        panel_by_ticker, config, device, use_mixed_precision
    )

    train_parts, val_parts, test_parts = [], [], []
    for frame in panel_by_ticker.values():
        n = len(frame)
        train_parts.append(frame.iloc[:int(n * 0.70)])
        val_parts.append(frame.iloc[int(n * 0.70):int(n * 0.85)])
        test_parts.append(frame.iloc[int(n * 0.85):])
    feature_df = pd.concat(train_parts + val_parts + test_parts, ignore_index=True)
    print(f"Built training panel from {len(panel_by_ticker)} tickers: {len(feature_df)} total rows")

    # Extract feature and target names for metadata
    feature_names = list(feature_df.columns[:-1])
    target_name = feature_df.columns[-1]

    # =========================================================================
    # 3b. Standardize the inputs
    # =========================================================================
    # Half of these features are price levels: across a 4000-name Indian
    # universe `sma_20` spans roughly ₹5 to ₹1,50,000 while `return_1d` sits
    # in ±0.2. Unscaled, that overflows fp16 outright and diverges in fp32, and
    # both failure modes present identically — a NaN loss on every epoch. The
    # scaler is fitted on the training rows only and shipped in the checkpoint
    # so inference applies the identical transform (see features/scaling.py).
    scaler = FeatureScaler.fit(feature_df.iloc[:int(len(feature_df) * 0.70), :-1].values)
    feature_df = feature_df.copy()
    feature_df[feature_names] = scaler.transform(feature_df[feature_names].values)
    print(
        f"Standardized {len(feature_names)} input features on the training split "
        f"(clipped to ±{scaler.clip:g} sigma)"
    )

    # =========================================================================
    # 4. Create DataLoaders
    # =========================================================================
    print("\nCreating DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        feature_df, config.training
    )

    # =========================================================================
    # 5. Initialize model from registry
    # =========================================================================
    print(f"\nInitializing model: {config.training.model}")
    
    # Get model class from registry
    model_class = get_model(config.training.model)
    
    # Determine number of features from dataloader
    n_features = len(feature_names)
    
    # Create model instance. The head is one node per quantile when fitting
    # pinball loss, so the checkpoint's shape is a function of the config —
    # which is why both are written into the metadata below.
    n_outputs = head_width(config.training)
    median_index = median_output_index(config.training)
    loss_fn = build_loss(config.training)

    model = model_class(
        n_features=n_features,
        hidden_size=64,
        n_layers=2,
        sequence_length=config.training.sequence_length,
        dropout=0.2,
        n_outputs=n_outputs,
    )
    
    # Move model to device
    model = model.to(device)
    print(f"Model moved to device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if config.training.use_torch_compile:
        try:
            model = torch.compile(model)
            print("torch.compile: Enabled")
        except Exception as e:
            print(f"torch.compile unavailable, continuing uncompiled: {e}")
    
    # =========================================================================
    # 6. Train (AdamW + MSE, early stopping on validation loss)
    # =========================================================================
    print(f"Optimizer: AdamW (lr={config.training.learning_rate})")
    if config.training.loss == "quantile":
        print(f"Loss Function: pinball over quantiles {config.training.quantiles}")
    else:
        print("Loss Function: MSE")
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / f"{config.training.model}_best.pt"

    # Enable cudnn benchmarking for faster training on fixed-size inputs
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmarking: Enabled (optimized for fixed input sizes)")

    def _checkpoint(epoch: int, val_loss: float, optimizer) -> None:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'feature_names': feature_names,
            'target': target_name,
            'sequence_length': config.training.sequence_length,
            'device': str(device),
            'quantiles': list(config.training.quantiles) if n_outputs > 1 else None,
            'n_outputs': n_outputs,
            'feature_scaler': scaler.to_dict(),
            # The pipeline, not just its last step: inference must redo the
            # cross-sectional pass when training used one, or the near-identity
            # scaler above is applied to raw features and saturates the clip.
            'feature_normalization': config.training.feature_normalization,
        }, model_path)
        print(f"  ✓ Saved best model to {model_path}")

    train_losses, val_losses, best_val_loss = _train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        use_mixed_precision=use_mixed_precision,
        on_improve=_checkpoint,
        loss_fn=loss_fn,
    )

    # =========================================================================
    # 7. Held-out test evaluation
    # =========================================================================
    # Score the weights that actually ship, not the last epoch's. Early
    # stopping leaves `model` up to `patience` epochs past the validation
    # optimum, so evaluating it in place would report metrics for weights that
    # exist nowhere — MLStrategy loads the checkpoint.
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Restored best checkpoint (epoch {checkpoint.get('epoch', '?')}) for test scoring")

    # The 15% test tail was never touched by training or early stopping, so
    # this is the single-split counterpart to the walk-forward numbers above.
    test_metrics = evaluate_predictions(
        *_predict(model, test_loader, device, median_index),
        horizon_days=_target_horizon_days(target_name),
    )

    # =========================================================================
    # 8. Save Metadata
    # =========================================================================
    metadata = {
        'feature_names': feature_names,
        'target': target_name,
        'sequence_length': config.training.sequence_length,
        'device': str(device),
        # Inference has to rebuild the same head before loading the state dict,
        # and has to know which column is the median. None means a scalar head.
        'quantiles': list(config.training.quantiles) if n_outputs > 1 else None,
        'n_outputs': n_outputs,
        'median_quantile_index': median_index,
        'loss': config.training.loss,
        # Inference MUST apply this same transform: the network was fitted on
        # standardized inputs, so feeding it raw price levels would be feeding
        # it values tens of thousands of sigma from anything it ever saw.
        'feature_scaler': scaler.to_dict(),
        # The pipeline, not just its last step: inference must redo the
        # cross-sectional pass when training used one, or the near-identity
        # scaler above is applied to raw features and saturates the clip.
        'feature_normalization': config.training.feature_normalization,
        # The fitted score -> probability map, produced from the walk-forward
        # test folds. MLStrategy applies it so the confidence it publishes is a
        # frequency that held up out of sample, not a raw network output.
        'confidence_calibration': (
            walk_forward.get('calibration') if isinstance(walk_forward, dict) else None
        ),
        'metrics': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
        },
        'test_metrics': test_metrics,
        'walk_forward': walk_forward,
        'epochs_trained': len(train_losses),
        'mixed_precision_enabled': use_mixed_precision,
        'model_architecture': config.training.model,
    }

    metadata_path = models_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata to {metadata_path}")

    # =========================================================================
    # 9. Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"  Epochs trained: {len(train_losses)}")
    if not math.isfinite(best_val_loss):
        print(
            "  WARNING: no epoch produced a finite validation loss, so no checkpoint "
            "was written. This run trained nothing.\n"
            f"  {_non_finite_loss_advice(use_mixed_precision)}"
        )
    print(f"  Best validation loss: {best_val_loss:.6f}")
    if train_losses:
        print(f"  Final train loss: {train_losses[-1]:.6f}")
    if val_losses:
        print(f"  Final val loss: {val_losses[-1]:.6f}")
    print(
        f"  Held-out test: dir_acc={test_metrics['directional_accuracy']:.3f} | "
        f"Sharpe={test_metrics['strategy_sharpe']:.2f} vs always-long "
        f"{test_metrics['benchmark_sharpe']:.2f}"
    )
    if 'mean_excess_sharpe' in walk_forward:
        print(
            f"  Walk-forward mean excess Sharpe: "
            f"{walk_forward['mean_excess_sharpe']:+.2f} over "
            f"{walk_forward['n_folds']} folds"
        )

    return metadata
