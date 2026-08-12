"""Generic trainer for custom strategies implementing TrainableStrategy.

This module provides a unified training loop for any strategy that implements
the TrainableStrategy interface. It handles:
- Data preparation (features, prices)
- Training loop execution
- Checkpoint saving with metadata
- Progress logging

Usage:
    portfolio-agent train-custom --strategy my_strategy --epochs 100
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch

from portfolio_agent.config.loader import load_config
from portfolio_agent.features.pipeline import build_features
from portfolio_agent.features.scaling import FeatureScaler
from portfolio_agent.src.data_store import load_ticker_data
from portfolio_agent.src.universe import resolve_backtest_universe
from portfolio_agent.strategies.base import TrainableStrategy
from portfolio_agent.strategies.registry import get_strategy
from portfolio_agent.utils.device import get_device

logger = logging.getLogger(__name__)


def prepare_training_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    required_features: list[str],
    min_history: int = 252,
) -> dict:
    """Prepare training data for a strategy.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for data loading (YYYY-MM-DD)
        end_date: End date for data loading (YYYY-MM-DD)
        required_features: List of feature names needed by the strategy
        min_history: Minimum history required for feature calculation
        
    Returns:
        dict containing:
            - 'features': Dict[str, pd.DataFrame] features by symbol
            - 'prices': Dict[str, pd.DataFrame] price data by symbol
            - 'tickers': List of valid tickers with sufficient data
            - 'feature_names': List of feature column names
            - 'scaler': Fitted FeatureScaler
    """
    logger.info(f"Loading data for {len(tickers)} tickers from {start_date} to {end_date}")
    
    # Load price data
    prices_by_symbol: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            df = load_ticker_data(ticker, start_date=start_date, end_date=end_date)
            if df is not None and len(df) >= min_history:
                prices_by_symbol[ticker] = df
        except Exception as e:
            logger.warning(f"Failed to load {ticker}: {e}")
    
    if not prices_by_symbol:
        raise ValueError("No valid price data loaded")
    
    valid_tickers = list(prices_by_symbol.keys())
    logger.info(f"Loaded data for {len(valid_tickers)} tickers")
    
    # Build features
    logger.info(f"Building features: {required_features}")
    features_dict = build_features(
        prices_by_symbol,
        features=required_features,
        check_min_history=min_history,
    )
    
    # Filter to tickers with valid features
    valid_features = {k: v for k, v in features_dict.items() if v is not None and not v.empty}
    
    if not valid_features:
        raise ValueError("No valid features computed")
    
    logger.info(f"Computed features for {len(valid_features)} tickers")
    
    # Fit scaler on all data
    all_features = pd.concat(valid_features.values(), axis=0)
    scaler = FeatureScaler()
    scaler.fit(all_features)
    
    # Scale features
    scaled_features = {}
    for ticker, feat_df in valid_features.items():
        scaled_features[ticker] = scaler.transform(feat_df)
    
    return {
        'features': scaled_features,
        'prices': prices_by_symbol,
        'tickers': list(scaled_features.keys()),
        'feature_names': required_features,
        'scaler': scaler,
    }


def save_checkpoint(
    model_state: dict,
    metadata: dict,
    filepath: Path,
) -> None:
    """Save training checkpoint.
    
    Args:
        model_state: Model state dict
        metadata: Training metadata (config, metrics, feature info)
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'model_state_dict': model_state,
        'metadata': metadata,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, filepath)
    logger.info(f"Saved checkpoint to {filepath}")


def run_custom_training(
    strategy_name: str,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: str = "auto",
    models_dir: str = "models",
    model_name: Optional[str] = None,
    years: int = 3,
    config_path: Optional[str] = None,
) -> dict:
    """Run training for a custom strategy.
    
    Args:
        strategy_name: Name of registered strategy implementing TrainableStrategy
        epochs: Number of training epochs
        batch_size: Mini-batch size
        lr: Learning rate
        device: Device for training ('cpu', 'cuda', 'mps', 'auto')
        models_dir: Directory to save checkpoints
        model_name: Name for saved model (default: strategy_name)
        years: Years of historical data to use
        config_path: Optional path to strategy-specific config file
        
    Returns:
        dict: Training metadata including final metrics
        
    Raises:
        ValueError: If strategy doesn't implement TrainableStrategy
        FileNotFoundError: If no data available for training
    """
    # Resolve device
    resolved_device = get_device(device)
    logger.info(f"Using device: {resolved_device}")
    
    # Load strategy
    StrategyClass = get_strategy(strategy_name)
    
    # Check if strategy supports training
    if not issubclass(StrategyClass, TrainableStrategy):
        raise ValueError(
            f"Strategy '{strategy_name}' does not implement TrainableStrategy. "
            "Only strategies with trainable models can use train-custom."
        )
    
    # Initialize strategy instance (may need config for some strategies)
    config = load_config()
    strategy_config = config.strategy.model_copy(deep=True)
    strategy_config.type = strategy_name
    if config_path:
        strategy_config.config_path = config_path
    
    try:
        strategy = StrategyClass(strategy_config)
    except Exception:
        # Some strategies might not need config at initialization
        strategy = StrategyClass.__new__(StrategyClass)
    
    # Get model name
    if model_name is None:
        model_name = strategy_name
    
    # Prepare training config
    training_config = {
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'device': resolved_device.type,
    }
    
    # Merge with strategy defaults
    default_config = strategy.get_default_training_config()
    training_config.update({k: v for k, v in default_config.items() if k not in training_config})
    
    logger.info(f"Training config: {training_config}")
    
    # Determine date range
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=years * 365)
    
    # Resolve universe
    tickers = resolve_backtest_universe(
        force_full_download=False,
        max_tickers=config.data.universe_size,
        selection=config.data.universe_selection,
        seed=config.data.universe_seed,
        purpose="training",
    )
    
    if not tickers:
        raise ValueError("No tickers found for training")
    
    logger.info(f"Training universe: {len(tickers)} tickers")
    
    # Prepare data
    required_features = strategy.required_features()
    data = prepare_training_data(
        tickers=tickers,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        required_features=required_features,
    )
    
    logger.info(f"Prepared data: {len(data['tickers'])} tickers, {len(data['feature_names'])} features")
    
    # Run training
    logger.info(f"Starting training for {epochs} epochs...")
    metrics = strategy.train(data=data, config=training_config)
    
    # Save checkpoint
    models_path = Path(models_dir)
    model_filename = f"{model_name}_best.pt"
    checkpoint_path = models_path / model_filename
    
    # Get model state dict (strategy should have a model attribute after training)
    if hasattr(strategy, 'actor') or hasattr(strategy, 'model') or hasattr(strategy, 'network'):
        # Try common attribute names
        model_obj = getattr(strategy, 'actor', None) or \
                    getattr(strategy, 'model', None) or \
                    getattr(strategy, 'network', None)
        
        if model_obj is not None:
            model_state = model_obj.state_dict()
        else:
            raise ValueError("Strategy training did not produce a model")
    else:
        raise ValueError("Strategy has no model attribute after training")
    
    # Prepare metadata
    metadata = {
        'strategy_name': strategy_name,
        'model_name': model_name,
        'training_config': training_config,
        'metrics': metrics,
        'feature_names': data['feature_names'],
        'num_tickers': len(data['tickers']),
        'date_range': {
            'start': start_date.strftime('%Y-%m-%d'),
            'end': end_date.strftime('%Y-%m-%d'),
        },
        'scaler_params': {
            'mean': data['scaler'].mean_.tolist() if hasattr(data['scaler'], 'mean_') else None,
            'scale': data['scaler'].scale_.tolist() if hasattr(data['scaler'], 'scale_') else None,
        },
    }
    
    save_checkpoint(model_state, metadata, checkpoint_path)
    
    logger.info(f"Training complete! Final metrics: {metrics}")
    logger.info(f"Model saved to: {checkpoint_path}")
    
    return metadata


def run_custom_training_cli(args) -> int:
    """CLI entry point for train-custom command."""
    try:
        metadata = run_custom_training(
            strategy_name=args.strategy,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            models_dir=args.models_dir,
            model_name=args.model_name,
            years=args.years,
            config_path=args.strategy_config,
        )
        
        print("\nTraining complete!")
        print(f"  Strategy: {metadata['strategy_name']}")
        print(f"  Epochs trained: {metadata['metrics'].get('epochs_trained', args.epochs)}")
        
        # Print primary metric
        if 'final_loss' in metadata['metrics']:
            print(f"  Final loss: {metadata['metrics']['final_loss']:.6f}")
        elif 'final_reward' in metadata['metrics']:
            print(f"  Final reward: {metadata['metrics']['final_reward']:.6f}")
        
        print(f"  Model saved to: models/{metadata['model_name']}_best.pt")
        
        return 0
        
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
