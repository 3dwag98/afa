# Custom Strategy Training Guide

This guide explains how to train and backtest custom strategies using the portfolio-agent framework's generic training infrastructure.

## Overview

The framework now supports **generic training** for any strategy that implements the `TrainableStrategy` interface. This provides a unified approach to:

1. **Training** ML/RL models with standardized data preparation
2. **Saving** checkpoints with metadata (features, scaler params, config)
3. **Backtesting** trained models automatically
4. **Extending** with your own custom strategies

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   TrainableStrategy                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ - model_artifact_name: str                           │   │
│  │ - train(data, config) -> dict                        │   │
│  │ - load_model(artifact_path) -> None                  │   │
│  │ - get_default_training_config() -> dict              │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │ implements
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────┴────────┐  ┌───────┴────────┐  ┌──────┴────────┐
│ IndiaSAC       │  │ MLStrategy     │  │ YourCustom    │
│ Strategy       │  │ (LSTM)         │  │ Strategy      │
└────────────────┘  └────────────────┘  └───────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                ┌───────────┴───────────┐
                │   custom_trainer.py   │
                │  (Generic Trainer)    │
                └───────────┬───────────┘
                            │
                ┌───────────┴───────────┐
                │   CLI: train-custom   │
                └───────────────────────┘
```

## Creating a Custom Trainable Strategy

### Step 1: Create Your Strategy Class

Create a new file in `portfolio_agent/strategies/`, e.g., `my_strategy.py`:

```python
from typing import List, Dict, Optional
import pandas as pd
import torch
import torch.nn as nn

from portfolio_agent.strategies.base import TrainableStrategy
from portfolio_agent.strategies.types import StrategyContext, StrategySignal
from portfolio_agent.config.schema import StrategyConfig


class MyMLStrategy(TrainableStrategy):
    """Example custom ML strategy."""
    
    def __init__(self, config: StrategyConfig):
        self._config = config
        params = config.params or {}
        
        # Model configuration
        self._hidden_dim = params.get("hidden_dim", 128)
        self._feature_names = params.get("features", ["close", "volume", "rsi_14"])
        
        # Model will be initialized during training
        self._model: Optional[nn.Module] = None
        
    @property
    def name(self) -> str:
        return "my_ml_strategy"
    
    @property
    def model_artifact_name(self) -> str:
        """Filename for saved checkpoint."""
        return "my_ml_strategy_best.pt"
    
    def required_features(self) -> List[str]:
        """Features needed by this strategy."""
        return self._feature_names
    
    def score(self, symbol: str, features: pd.DataFrame, 
              context: StrategyContext) -> StrategySignal:
        """Score a single ticker."""
        # Implement scoring logic using self._model
        pass
    
    @classmethod
    def get_default_training_config(cls) -> dict:
        """Default hyperparameters."""
        return {
            "epochs": 50,
            "batch_size": 128,
            "lr": 1e-3,
            "device": "auto",
        }
    
    def train(self, data: dict, config: dict) -> dict:
        """
        Training loop implementation.
        
        Args:
            data: Dict with keys:
                - 'features': Dict[str, pd.DataFrame] features by ticker
                - 'prices': Dict[str, pd.DataFrame] price data by ticker
                - 'tickers': List[str] valid tickers
                - 'scaler': Fitted FeatureScaler
            config: Training hyperparameters
            
        Returns:
            dict with training metrics (loss, reward, etc.)
        """
        # Your training logic here
        # Example:
        # 1. Build DataLoader from data['features']
        # 2. Initialize self._model
        # 3. Run training loop
        # 4. Return metrics
        
        return {"final_loss": 0.5, "epochs_trained": config["epochs"]}
    
    def load_model(self, artifact_path: str) -> None:
        """Load trained weights from disk."""
        import torch
        from pathlib import Path
        
        checkpoint_path = Path(artifact_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {artifact_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Reconstruct model architecture from metadata
        metadata = checkpoint.get('metadata', {})
        self._model = self._build_model(metadata)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.eval()
    
    def _build_model(self, metadata: dict) -> nn.Module:
        """Build model architecture (called during load_model)."""
        # Implement based on your model type
        pass
```

### Step 2: Register Your Strategy

Add your strategy to `portfolio_agent/strategies/registry.py`:

```python
from .my_strategy import MyMLStrategy

register_strategy("my_ml", MyMLStrategy)
```

### Step 3: Train Your Strategy

```bash
# Basic training with defaults
portfolio-agent train-custom --strategy my_ml

# Advanced training with custom parameters
portfolio-agent train-custom \
  --strategy my_ml \
  --epochs 200 \
  --batch-size 512 \
  --lr 1e-4 \
  --device cuda \
  --years 5

# With strategy-specific config file
portfolio-agent train-custom \
  --strategy my_ml \
  --strategy-config config/strategies/my_ml.yaml
```

### Step 4: Backtest Your Trained Strategy

```bash
# Basic backtest
portfolio-agent backtest --strategy my_ml --years 2

# Advanced backtest
portfolio-agent backtest \
  --strategy my_ml \
  --start-date 2022-01-01 \
  --end-date 2024-01-01 \
  --transaction-cost 0.0005 \
  --initial-capital 1000000
```

## Using the Generic Trainer

The `custom_trainer.py` module handles:

### Data Preparation

```python
from portfolio_agent.agents.custom_trainer import prepare_training_data

data = prepare_training_data(
    tickers=["RELIANCE.NS", "TCS.NS", "INFY.NS"],
    start_date="2020-01-01",
    end_date="2024-01-01",
    required_features=["close", "rsi_14", "macd"],
    min_history=252,
)

# Returns:
# {
#   'features': Dict[str, pd.DataFrame],  # Scaled features
#   'prices': Dict[str, pd.DataFrame],
#   'tickers': List[str],
#   'feature_names': List[str],
#   'scaler': FeatureScaler,
# }
```

### Checkpoint Management

Checkpoints include:
- `model_state_dict`: PyTorch model weights
- `metadata`: Training config, feature names, scaler params
- `timestamp`: When the checkpoint was saved

```python
from portfolio_agent.agents.custom_trainer import save_checkpoint

save_checkpoint(
    model_state=model.state_dict(),
    metadata={
        'feature_names': ['close', 'rsi_14'],
        'training_config': {'epochs': 100, 'lr': 1e-3},
        'scaler_params': {'mean': [...], 'scale': [...]},
    },
    filepath=Path("models/my_strategy_best.pt"),
)
```

## Built-in Trainable Strategies

### IndiaSAC (Reinforcement Learning)

Soft Actor-Critic for Indian equity allocation:

```bash
# Train SAC model
portfolio-agent train-sac --epochs 100 --batch-size 256

# OR using generic trainer
portfolio-agent train-custom --strategy india_sac

# Backtest
portfolio-agent backtest --strategy india_sac --years 2
```

### LSTM (Supervised Learning)

Standard LSTM for return prediction:

```bash
# Train LSTM
portfolio-agent train --epochs 50

# Backtest
portfolio-agent backtest --use-trained-model --years 2
```

## Best Practices

### 1. Feature Engineering

```python
def required_features(self) -> List[str]:
    # Use only features from features/registry.py
    return [
        "close", "volume_ratio_20", "return_1d", "return_5d",
        "mom_9m_skip1m", "rsi_14", "macd", "atr_14",
    ]
```

### 2. Checkpoint Metadata

Always save:
- Feature names (for consistent inference)
- Scaler parameters (mean, scale)
- Training config (hyperparameters)
- Date range of training data

### 3. Device Handling

```python
from portfolio_agent.utils.device import get_device

device = get_device(config.get("device", "auto"))
model = model.to(device)
```

### 4. Error Handling

```python
def load_model(self, artifact_path: str) -> None:
    if not Path(artifact_path).exists():
        raise FileNotFoundError(f"Model not found: {artifact_path}")
    
    try:
        checkpoint = torch.load(artifact_path, map_location='cpu')
        # ... load weights
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
```

## Troubleshooting

### "Strategy does not implement TrainableStrategy"

Your strategy class must inherit from `TrainableStrategy` and implement:
- `model_artifact_name` property
- `train()` method
- `load_model()` method

### "No valid data for training"

Check:
- Data is downloaded: `portfolio-agent download-data --years 3`
- Tickers have sufficient history (min_history parameter)
- Features can be computed (no NaN issues)

### "Checkpoint not found"

Ensure:
- Training completed successfully
- Models directory exists (`mkdir -p models`)
- Model name matches between training and backtesting

### GPU Out of Memory

Reduce batch size or use CPU:
```bash
portfolio-agent train-custom --strategy my_ml --batch-size 64 --device cpu
```

## Example: Complete Custom Strategy

See `portfolio_agent/strategies/india_sac.py` for a complete example of:
- TrainableStrategy implementation
- Custom neural network architecture
- Training loop with replay buffer
- Batched inference for backtesting
- Checkpoint saving/loading

## Next Steps

1. **Start simple**: Clone an existing strategy and modify
2. **Test incrementally**: Verify each component (features, model, training)
3. **Use the CLI**: Leverage `train-custom` for standardized training
4. **Document**: Add docstrings explaining your strategy's logic

For questions or issues, check the docs or open a GitHub issue.
