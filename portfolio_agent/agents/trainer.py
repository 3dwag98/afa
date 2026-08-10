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
from portfolio_agent.models.registry import get_model
from portfolio_agent.src.data_store import load_ticker_data
from portfolio_agent.src.universe import resolve_backtest_universe
from portfolio_agent.utils.device import get_device

TRAINING_FEATURE_NAMES = [
    'sma_20', 'sma_50', 'rsi_14', 'macd',
    'bollinger_pct_b', 'atr_14', 'return_1d', 'return_5d'
]

TRADING_DAYS_PER_YEAR = 252


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


def _load_and_split_ticker(
    ticker: str, config: AppConfig
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Load, featurize, and chronologically split (70/15/15) one ticker.

    Module-level (not a nested closure) so it can be dispatched across a
    ProcessPoolExecutor for parallel training-panel construction. Returns
    None if the ticker has insufficient cached history.
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

    n = len(feature_df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return (
        feature_df.iloc[:train_end],
        feature_df.iloc[train_end:val_end],
        feature_df.iloc[val_end:],
    )


def load_data(config: AppConfig) -> pd.DataFrame:
    """Load and featurize training data.

    By default, loads real cached multi-ticker OHLCV data — the full 5-year
    cached universe (via data_store), not a synthetic stand-in — and builds a
    concatenated training panel: each ticker is featurized and split 70/15/15
    chronologically *individually*, then all tickers' train portions are
    concatenated, followed by all val portions, then all test portions. This
    ordering lets create_dataloaders()'s single top-level 70/15/15 index split
    land exactly on those boundaries, so validation/test proportionally
    represent every ticker rather than only the last one in the panel.

    Per-ticker loading + feature computation is CPU-bound (parquet decode +
    indicator math), so when config.training.parallel_data_loading is set
    (the default), tickers are dispatched across a ProcessPoolExecutor sized
    by config.training.data_load_workers (default: CPU count) — this is what
    makes training on the full ~2,400-ticker cached universe practical. The
    parallel path reassembles results in resolved-universe order, so it builds
    exactly the same panel as the serial path (workers completing out of order
    must not change what the model trains on).

    Sequence windows that straddle two concatenated tickers' boundaries mix
    data from different instruments; this is a bounded, documented limitation
    of pooling multiple series through a single-series windowing dataset
    (TimeSeriesDataset), not a look-ahead bias — it affects at most
    sequence_length * (n_tickers - 1) windows out of the full panel.

    Falls back to synthetic random-walk data only when
    config.training.use_synthetic_data is set.

    Args:
        config: Application configuration.

    Returns:
        DataFrame with computed features and target column (already featurized).
    """
    if config.training.use_synthetic_data:
        return prepare_features(_generate_synthetic_ohlcv(), config)

    tickers = resolve_backtest_universe(max_tickers=config.data.universe_size)
    if not tickers:
        raise RuntimeError(
            "No cached tickers found to build a training panel. Run "
            "`portfolio-agent download-data` first, or set "
            "training.use_synthetic_data=true for offline testing."
        )

    # Keyed by ticker rather than appended in completion order: workers finish
    # in a nondeterministic order, and appending in that order would make the
    # concatenated panel's row order (and therefore training) differ run to
    # run for identical inputs. Reassembling in `tickers` order below makes the
    # parallel path produce byte-identical output to the serial path.
    by_ticker: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    show_progress = len(tickers) > 20
    failures = 0

    if config.training.parallel_data_loading and len(tickers) > 1:
        with ProcessPoolExecutor(max_workers=config.training.data_load_workers) as executor:
            futures = {executor.submit(_load_and_split_ticker, t, config): t for t in tickers}
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
                result = _load_and_split_ticker(ticker, config)
            except Exception as e:
                failures += 1
                print(f"Warning: failed to load training data for {ticker}: {e}")
                continue
            if result is not None:
                by_ticker[ticker] = result

    results = [by_ticker[t] for t in tickers if t in by_ticker]

    if failures:
        print(f"Warning: {failures}/{len(tickers)} tickers failed to load and were skipped")

    if not results:
        raise RuntimeError(
            "None of the resolved tickers had enough cached history to build a "
            "training panel. Run `portfolio-agent download-data` first, or set "
            "training.use_synthetic_data=true for offline testing."
        )

    train_parts = [r[0] for r in results]
    val_parts = [r[1] for r in results]
    test_parts = [r[2] for r in results]

    combined = pd.concat(train_parts + val_parts + test_parts, ignore_index=True)
    print(f"Built training panel from {len(results)}/{len(tickers)} tickers: {len(combined)} total rows")
    return combined


def target_column_name(target: str) -> str:
    """Column name for the training target, namespaced away from features.

    The target must never share a name with a registered feature. It used to:
    config.training.target defaults to "return_5d", which is also in
    TRAINING_FEATURE_NAMES, so the "create target if it doesn't exist" branch
    below never fired and the model was trained to reproduce the *trailing*
    5-day return — a quantity fully determined by the price history it was
    already being shown. That trains and validates beautifully and forecasts
    nothing. Prefixing the target guarantees the collision cannot recur.
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

    # Drop NaN values
    feature_df = feature_df.dropna()

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

    Args:
        predictions: Model outputs, shape (n,).
        actuals: Realized target values, shape (n,).
        horizon_days: Forecast horizon in trading days, for annualization.

    Returns:
        Dictionary of metrics; zeros when there is nothing to score.
    """
    predictions = np.asarray(predictions, dtype=float).ravel()
    actuals = np.asarray(actuals, dtype=float).ravel()

    empty = {
        "n_samples": 0, "mse": 0.0, "directional_accuracy": 0.0,
        "strategy_sharpe": 0.0, "benchmark_sharpe": 0.0, "excess_sharpe": 0.0,
    }
    if predictions.size == 0 or predictions.size != actuals.size:
        return empty

    finite = np.isfinite(predictions) & np.isfinite(actuals)
    predictions, actuals = predictions[finite], actuals[finite]
    if predictions.size == 0:
        return empty

    mse = float(np.mean((predictions - actuals) ** 2))
    directional_accuracy = float(np.mean(np.sign(predictions) == np.sign(actuals)))

    annualization = math.sqrt(TRADING_DAYS_PER_YEAR / max(1, horizon_days))

    def _sharpe(returns: np.ndarray) -> float:
        if returns.size < 2:
            return 0.0
        sigma = float(np.std(returns, ddof=1))
        if sigma <= 0:
            return 0.0
        return float(np.mean(returns) / sigma * annualization)

    strategy_returns = np.where(predictions > 0, actuals, 0.0)

    strategy_sharpe = _sharpe(strategy_returns)
    benchmark_sharpe = _sharpe(actuals)

    return {
        "n_samples": int(predictions.size),
        "mse": mse,
        "directional_accuracy": directional_accuracy,
        "strategy_sharpe": strategy_sharpe,
        "benchmark_sharpe": benchmark_sharpe,
        "excess_sharpe": strategy_sharpe - benchmark_sharpe,
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

    Returns:
        (train_losses, val_losses, best_val_loss).
    """
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda') if use_mixed_precision else None

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val_loss = float('inf')
    patience_counter = 0

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

            optimizer.zero_grad()

            if use_mixed_precision:
                with torch.autocast('cuda'):
                    outputs = model(batch_x)
                    loss = loss_fn(outputs.squeeze(), batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_x)
                loss = loss_fn(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()

            epoch_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = epoch_train_loss / max(num_train_batches, 1)
        train_losses.append(avg_train_loss)

        if val_loader is None:
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
                        loss = loss_fn(outputs.squeeze(), batch_y)
                else:
                    outputs = model(batch_x)
                    loss = loss_fn(outputs.squeeze(), batch_y)
                epoch_val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = epoch_val_loss / max(num_val_batches, 1)
        val_losses.append(avg_val_loss)

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

    return train_losses, val_losses, best_val_loss


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Collect (predictions, actuals) over a loader, in order."""
    model.eval()
    predictions: List[np.ndarray] = []
    actuals: List[np.ndarray] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            outputs = model(batch_x).squeeze(-1)
            predictions.append(outputs.detach().cpu().numpy().ravel())
            actuals.append(batch_y.detach().cpu().numpy().ravel())
    if not predictions:
        return np.array([]), np.array([])
    return np.concatenate(predictions), np.concatenate(actuals)


def run_walk_forward_validation(
    feature_df: pd.DataFrame,
    config: AppConfig,
    device: torch.device,
    use_mixed_precision: bool = False,
) -> Dict[str, Any]:
    """Expanding-window walk-forward validation.

    A single chronological 70/15/15 split answers "how did this model do on
    2023?" — one regime, one initialization, one number. That is not enough
    evidence to deploy a 5-day return forecaster on a market with this
    signal-to-noise ratio. Walk-forward re-fits the model on an expanding
    history and tests it on the *next* contiguous block, repeatedly:

        fold 1: train [0, 40%)  test [40%, 52%)
        fold 2: train [0, 52%)  test [52%, 64%)
        ...

    Training data never includes anything at or after the test block, so each
    fold is a genuine out-of-sample measurement, and averaging across folds
    spans several market regimes instead of whichever one happened to land at
    the end of the panel. Each fold holds back the tail of its own training
    window for early stopping, so the test block is never seen during fitting.

    Folds are trained with a reduced epoch budget (walk_forward_epochs,
    default min(20, epochs)): their job is to estimate generalization, not to
    produce the shipped weights.

    Args:
        feature_df: Featurized panel; last column is the target.
        config: Application configuration.
        device: Device to train on.
        use_mixed_precision: Whether to use CUDA autocast.

    Returns:
        Dictionary with per-fold metrics and their averages, or a `skipped`
        marker when the panel is too short to split.
    """
    training = config.training
    n_splits = training.walk_forward_splits
    if n_splits <= 0:
        return {"skipped": "walk_forward_splits <= 0"}

    feature_cols = feature_df.columns[:-1].tolist()
    target_col = feature_df.columns[-1]
    features = feature_df[feature_cols].values
    targets = feature_df[target_col].values
    n_samples = len(feature_df)

    initial_train_end = int(n_samples * training.walk_forward_min_train_fraction)
    test_size = (n_samples - initial_train_end) // n_splits

    # Every fold needs at least one full sequence on each side of the split.
    if test_size <= training.sequence_length or initial_train_end <= training.sequence_length * 2:
        return {
            "skipped": (
                f"panel of {n_samples} rows is too short for {n_splits} folds at "
                f"sequence_length={training.sequence_length}"
            )
        }

    epochs = training.walk_forward_epochs or min(20, training.epochs)
    horizon_days = _target_horizon_days(target_col)
    model_class = get_model(training.model)

    print("\n" + "=" * 60)
    print(f"Walk-Forward Validation ({n_splits} expanding-window folds)")
    print("=" * 60)

    fold_metrics: List[Dict[str, Any]] = []
    for fold in range(n_splits):
        train_end = initial_train_end + fold * test_size
        test_end = train_end + test_size if fold < n_splits - 1 else n_samples

        # Hold back the tail of the training window for early stopping, so the
        # test block stays untouched during fitting.
        inner_val_size = max(training.sequence_length + 1, int(train_end * 0.15))
        inner_train_end = train_end - inner_val_size

        train_loader = _make_loader(
            features[:inner_train_end], targets[:inner_train_end], training, drop_last=True
        )
        val_loader = _make_loader(
            features[inner_train_end:train_end], targets[inner_train_end:train_end], training
        )
        test_loader = _make_loader(
            features[train_end:test_end], targets[train_end:test_end], training
        )
        if train_loader is None or test_loader is None:
            print(f"Fold {fold + 1}/{n_splits}: skipped (not enough rows)")
            continue

        model = model_class(
            n_features=len(feature_cols),
            hidden_size=64,
            n_layers=2,
            sequence_length=training.sequence_length,
            dropout=0.2,
            n_outputs=1,
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
        )

        predictions, actuals = _predict(model, test_loader, device)
        metrics = evaluate_predictions(predictions, actuals, horizon_days)
        metrics.update({
            "fold": fold + 1,
            "train_rows": int(train_end),
            "test_rows": int(test_end - train_end),
        })
        fold_metrics.append(metrics)

        print(
            f"Fold {fold + 1}/{n_splits}: train={train_end} test={test_end - train_end} | "
            f"MSE={metrics['mse']:.6f} | dir_acc={metrics['directional_accuracy']:.3f} | "
            f"Sharpe={metrics['strategy_sharpe']:.2f} vs benchmark "
            f"{metrics['benchmark_sharpe']:.2f} (excess {metrics['excess_sharpe']:+.2f})"
        )

    if not fold_metrics:
        return {"skipped": "no fold had enough rows to evaluate"}

    def _mean(key: str) -> float:
        return float(np.mean([m[key] for m in fold_metrics]))

    summary = {
        "n_folds": len(fold_metrics),
        "epochs_per_fold": epochs,
        "horizon_days": horizon_days,
        "folds": fold_metrics,
        "mean_mse": _mean("mse"),
        "mean_directional_accuracy": _mean("directional_accuracy"),
        "mean_strategy_sharpe": _mean("strategy_sharpe"),
        "mean_benchmark_sharpe": _mean("benchmark_sharpe"),
        "mean_excess_sharpe": _mean("excess_sharpe"),
        "folds_beating_benchmark": sum(1 for m in fold_metrics if m["excess_sharpe"] > 0),
    }

    print("-" * 60)
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


def run_training(config: AppConfig) -> Dict[str, Any]:
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

    # Mixed precision is a CUDA-only win here; torch.amp's CPU path would add
    # overhead without the tensor-core speedup.
    use_mixed_precision = config.training.use_mixed_precision and device.type == "cuda"

    if use_mixed_precision:
        print("Mixed Precision: Enabled")
    else:
        if config.training.use_mixed_precision and device.type != "cuda":
            print(f"Mixed Precision: Disabled (requires CUDA; running on {device.type})")
        else:
            print("Mixed Precision: Disabled")

    # =========================================================================
    # 2. Load and featurize training data (real cached tickers by default)
    # =========================================================================
    print("\nLoading and featurizing training data...")
    feature_df = load_data(config)

    # =========================================================================
    # 3. Walk-forward validation (before the final fit)
    # =========================================================================
    # Run first, so the generalization estimate is available even if the final
    # long fit is interrupted — and so a model that cannot beat always-long is
    # flagged before anyone reads its training loss as evidence of anything.
    walk_forward = run_walk_forward_validation(
        feature_df, config, device, use_mixed_precision
    )

    # =========================================================================
    # 4. Create DataLoaders
    # =========================================================================
    print("\nCreating DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        feature_df, config.training
    )

    # Extract feature and target names for metadata
    feature_names = list(feature_df.columns[:-1])
    target_name = feature_df.columns[-1]

    # =========================================================================
    # 5. Initialize model from registry
    # =========================================================================
    print(f"\nInitializing model: {config.training.model}")
    
    # Get model class from registry
    model_class = get_model(config.training.model)
    
    # Determine number of features from dataloader
    n_features = len(feature_names)
    
    # Create model instance
    model = model_class(
        n_features=n_features,
        hidden_size=64,
        n_layers=2,
        sequence_length=config.training.sequence_length,
        dropout=0.2,
        n_outputs=1,
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
    print(f"Loss Function: MSE")
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
    )

    # =========================================================================
    # 7. Held-out test evaluation
    # =========================================================================
    # The 15% test tail was never touched by training or early stopping, so
    # this is the single-split counterpart to the walk-forward numbers above.
    test_metrics = evaluate_predictions(
        *_predict(model, test_loader, device),
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
