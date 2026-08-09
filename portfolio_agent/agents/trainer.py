"""GPU-accelerated training loop for portfolio forecasting models."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    makes training on the full ~2,400-ticker cached universe practical.

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

    results: List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    show_progress = len(tickers) > 20

    if config.training.parallel_data_loading and len(tickers) > 1:
        with ProcessPoolExecutor(max_workers=config.training.data_load_workers) as executor:
            futures = {executor.submit(_load_and_split_ticker, t, config): t for t in tickers}
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc="Loading training data", unit="ticker")
            for future in iterator:
                result = future.result()
                if result is not None:
                    results.append(result)
    else:
        ticker_iter = tqdm(tickers, desc="Loading training data", unit="ticker") if show_progress else tickers
        for ticker in ticker_iter:
            result = _load_and_split_ticker(ticker, config)
            if result is not None:
                results.append(result)

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


def prepare_features(df: pd.DataFrame, config: AppConfig, verbose: bool = True) -> pd.DataFrame:
    """Build feature matrix from raw OHLCV data.

    Args:
        df: Raw OHLCV DataFrame.
        config: Application configuration.
        verbose: Whether to print a summary (disabled when called per-ticker
            from load_data()'s multi-ticker panel construction).

    Returns:
        DataFrame with computed features and target.
    """
    feature_df = build_features(
        df,
        TRAINING_FEATURE_NAMES,
        normalize=config.features.normalize,
        normalize_window=config.features.normalize_window
    )

    # Add target (next period return as example)
    target_name = config.training.target
    if target_name not in feature_df.columns:
        # Create target if it doesn't exist
        if 'return' in target_name:
            try:
                periods = int(target_name.replace('return_', ''))
                feature_df[target_name] = df['close'].shift(-periods).pct_change(periods)
            except ValueError:
                feature_df[target_name] = df['close'].shift(-1).pct_change()
        else:
            feature_df[target_name] = df['close'].shift(-1).pct_change()

    # Drop NaN values
    feature_df = feature_df.dropna()

    if verbose:
        print(f"Built feature matrix with {len(feature_df)} samples and {len(feature_df.columns)} columns")
        print(f"Features: {list(feature_df.columns[:-1])}")
        print(f"Target: {feature_df.columns[-1]}")

    return feature_df


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
    device = get_device(config.training.device)
    
    # Check mixed precision availability
    use_mixed_precision = (
        config.training.use_mixed_precision 
        and device.type == "cuda" 
        and torch.cuda.is_available()
    )
    
    if use_mixed_precision:
        print("Mixed Precision: Enabled")
        scaler = torch.amp.GradScaler('cuda')
    else:
        print("Mixed Precision: Disabled")
        scaler = None
    
    # =========================================================================
    # 2. Load and featurize training data (real cached tickers by default)
    # =========================================================================
    print("\nLoading and featurizing training data...")
    feature_df = load_data(config)


    # =========================================================================
    # 3. Create DataLoaders
    # =========================================================================
    print("\nCreating DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        feature_df, config.training
    )
    
    # Extract feature and target names for metadata
    feature_names = list(feature_df.columns[:-1])
    target_name = feature_df.columns[-1]
    
    # =========================================================================
    # 4. Initialize model from registry
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
    # 5. Initialize Optimizer and Loss Function
    # =========================================================================
    optimizer = optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    
    # Select loss function based on config or default to MSE
    loss_fn = nn.MSELoss()
    print(f"Optimizer: AdamW (lr={config.training.learning_rate})")
    print(f"Loss Function: MSE")
    
    # =========================================================================
    # 6. Training Loop Setup
    # =========================================================================
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    # Early stopping parameters
    patience = 10
    min_delta = 1e-4
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop = False
    
    # Track metrics
    train_losses: List[float] = []
    val_losses: List[float] = []
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # =========================================================================
    # 7. Epoch Loop
    # =========================================================================
    # Enable cudnn benchmarking for faster training on fixed-size inputs
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmarking: Enabled (optimized for fixed input sizes)")
    
    for epoch in range(config.training.epochs):
        if early_stop:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
        
        # ----- Training Phase -----
        model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0
        
        # Use autocast for mixed precision
        # Set progress bar to update less frequently for speed
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.training.epochs} [Train]", leave=False, mininterval=0.5)
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if use_mixed_precision:
                with torch.autocast('cuda' if device.type == 'cuda' else 'cpu'):
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
        
        # ----- Validation Phase -----
        model.eval()
        epoch_val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                
                if use_mixed_precision:
                    with torch.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                        outputs = model(batch_x)
                        loss = loss_fn(outputs.squeeze(), batch_y)
                else:
                    outputs = model(batch_x)
                    loss = loss_fn(outputs.squeeze(), batch_y)
                
                epoch_val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = epoch_val_loss / max(num_val_batches, 1)
        val_losses.append(avg_val_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1}/{config.training.epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss:   {avg_val_loss:.6f}")
        
        # ----- Early Stopping Check -----
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model weights
            model_path = models_dir / f"{config.training.model_name if hasattr(config.training, 'model_name') else config.training.model}_best.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'feature_names': feature_names,
                'target': target_name,
                'sequence_length': config.training.sequence_length,
                'device': str(device),
            }, model_path)
            print(f"  ✓ Saved best model to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                early_stop = True
                print(f"  Early stopping triggered (patience={patience} exceeded)")
    
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
    print(f"  Final train loss: {train_losses[-1]:.6f}")
    print(f"  Final val loss: {val_losses[-1]:.6f}")
    
    return metadata
