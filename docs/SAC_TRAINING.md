# Training Soft Actor-Critic (SAC) for Indian Equities

This document explains how to train the `IndiaSAC` strategy using the new SAC training infrastructure.

## Overview

The `IndiaSAC` strategy uses a **Soft Actor-Critic** reinforcement learning approach to learn optimal allocation weights for Indian stocks. Unlike the standard `train` command (which trains an LSTM forecasting model), `train-sac` trains an RL policy that directly outputs allocation weights in [0, 1].

### What Gets Trained

- **Actor Network**: A neural network that maps technical features → allocation weight
- **Output**: Sigmoid activation producing weights in [0, 1] (long-only allocation)
- **Reward Signal**: Differential Sortino ratio (downside-aware returns net of friction)
- **Checkpoint**: Saved to `models/india_sac_best.pt` with metadata

## Quick Start

### Prerequisites

1. **Download Indian market data**:
   ```bash
   portfolio-agent download-data --source yfinance --years 3
   ```

2. **Ensure you have PyTorch installed**:
   ```bash
   uv sync --extra gpu  # For GPU training
   # or
   uv sync  # CPU-only
   ```

### Basic Training

```bash
portfolio-agent train-sac
```

This runs with default parameters:
- 100 epochs
- Batch size: 256
- Hidden dimension: 256
- Learning rate: 3e-4
- Device: auto (uses GPU if available)

### Advanced Training Options

```bash
portfolio-agent train-sac \
  --epochs 200 \
  --batch-size 512 \
  --hidden-dim 512 \
  --lr 1e-4 \
  --buffer-size 200000 \
  --entropy-coef 0.05 \
  --device cuda \
  --model-name india_sac_v2
```

#### Parameter Descriptions

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100 | Number of training iterations |
| `--batch-size` | 256 | Mini-batch size for gradient updates |
| `--hidden-dim` | 256 | Hidden layer dimension in actor network |
| `--lr` | 3e-4 | Learning rate for Adam optimizer |
| `--buffer-size` | 100000 | Replay buffer capacity |
| `--entropy-coef` | 0.1 | Entropy regularization (exploration bonus) |
| `--device` | auto | Training device (auto/cuda/mps/cpu) |
| `--models-dir` | models | Directory to save checkpoints |
| `--model-name` | india_sac | Name for saved model file |

## How It Works

### Architecture

```
Features (11-dim) → [Linear + ReLU] × 2 → Linear → Sigmoid → Allocation Weight [0, 1]
     ↓
  close, volume_ratio_20, return_1d, return_5d, mom_9m_skip1m,
  realized_vol_60, traded_value_60, rsi_14, macd, bollinger_pct_b, atr_14
```

### Training Loop

1. **Load Data**: Fetch historical OHLCV for configured universe
2. **Build Features**: Compute technical indicators for each ticker
3. **Initialize Actor**: Random weights for the SAC actor network
4. **Populate Replay Buffer**: Generate initial experience using current policy
5. **Train**:
   - Sample mini-batches from replay buffer
   - Compute policy gradient loss (maximize expected reward)
   - Add entropy regularization (encourage exploration)
   - Update actor weights via backpropagation
6. **Save Checkpoint**: Store trained weights + metadata

### Reward Design

The reward function is critical for RL success:

```python
reward = (allocation × return) - friction_cost
```

Where:
- `allocation`: Actor's output weight [0, 1]
- `return`: Forward return the stock actually delivered
- `friction_cost`: Transaction cost (~0.8% round-trip for Indian equities)

This rewards **risk-adjusted returns net of costs**, not raw returns.

## Using the Trained Model

After training completes:

```bash
# Backtest the trained SAC strategy
portfolio-agent backtest --strategy india_sac --years 2

# Or run the live orchestrator
portfolio-agent run-agent
```

The `IndiaSACStrategy.load()` method will automatically find and load `models/india_sac_best.pt`.

## Monitoring Training

### Training Output

```
Starting SAC training for IndiaSAC strategy...
Device: cuda
Epochs: 100
Batch size: 256
Model will be saved to: models/india_sac_best.pt

Loaded features for 45/50 tickers (5 failures)
Populating replay buffer...
Replay buffer size: 8742

Epoch 10/100, Avg Reward: 0.002341
Epoch 20/100, Avg Reward: 0.003127
...
Epoch 100/100, Avg Reward: 0.004892

============================================================
SAC Training Complete!
============================================================
Epochs trained: 100
Final average reward: 0.004892
Checkpoint saved to: models/india_sac_best.pt

To use this model in backtesting:
  portfolio-agent backtest --strategy india_sac
============================================================
```

### Interpreting Results

- **Avg Reward increasing**: Policy is learning (good)
- **Avg Reward flat/decreasing**: Try lower learning rate or more epochs
- **Many ticker failures**: Check data quality / increase history

## Tips for Better Results

### 1. More Data
```bash
portfolio-agent download-data --years 5  # More history
```

### 2. Larger Universe
Edit `config.yaml`:
```yaml
data:
  universe_size: 100  # More tickers
```

### 3. Longer Training
```bash
portfolio-agent train-sac --epochs 500
```

### 4. Tune Hyperparameters
```bash
# Lower LR for stability
portfolio-agent train-sac --lr 1e-4 --epochs 300

# More exploration
portfolio-agent train-sac --entropy-coef 0.2
```

### 5. Custom Features

Edit the feature list in your strategy config:
```yaml
strategy:
  type: india_sac
  params:
    features:
      - close
      - rsi_14
      - macd
      # Add custom features
```

Then retrain with matching feature names.

## Troubleshooting

### "No cached tickers found"
Run `portfolio-agent download-data` first.

### "CUDA out of memory"
Reduce batch size: `--batch-size 128`

### "NaN loss" or training diverges
- Lower learning rate: `--lr 1e-5`
- Reduce entropy coefficient: `--entropy-coef 0.01`
- Check data quality (zero prices, missing values)

### Model performs worse than random
- Increase training epochs
- Add more historical data
- Reduce universe size (focus on liquid stocks)
- Check reward signal (may need better friction modeling)

## Comparison: `train` vs `train-sac`

| Aspect | `train` (LSTM) | `train-sac` (SAC) |
|--------|----------------|-------------------|
| **What it learns** | Predict future returns | Direct allocation weights |
| **Output** | Return forecast | Weight in [0, 1] |
| **Strategy** | MLStrategy | IndiaSACStrategy |
| **Loss** | Quantile/MSE | Policy gradient + entropy |
| **Best for** | Directional prediction | Risk-aware allocation |

## Next Steps

1. Train baseline model: `portfolio-agent train-sac`
2. Backtest: `portfolio-agent backtest --strategy india_sac`
3. Compare to benchmarks (Nifty 50, rule-based strategies)
4. Iterate on hyperparameters
5. Deploy with `run-agent`

For advanced usage (custom rewards, multi-GPU training), see the source code in `portfolio_agent/agents/sac_trainer.py`.
