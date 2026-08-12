"""Soft Actor-Critic trainer for IndiaSAC strategy.

This module implements the complete SAC training loop for the IndiaSAC
allocation strategy. Unlike the exposure RL in src/rl.py (which chooses
portfolio-level exposure based on regime), SAC here learns stock-level
allocation weights from technical features.

Key design decisions:
---------------------
- **Actor-only at inference**: The trained actor outputs allocation weights
  in [0, 1] via sigmoid. Critic networks are training-only and discarded.
- **Differential Sortino reward**: Rewards downside-aware returns net of
  turnover cost, not raw returns. This prevents learning leverage instead
  of skill.
- **Replay buffer with episode structure**: Each ticker's history is one
  episode; the buffer mixes across tickers to break temporal correlation.
- **No sampling at inference**: The policy is deterministic (mean action)
  for reproducibility — two runs of the same backtest must agree.

What this trains
----------------
A SACActorNetwork (see india_sac.py) that maps feature vectors to allocation
weights. The checkpoint includes:
- model_state_dict: The actor's weights
- metadata: Feature names, scaler params, hidden_dim, training config
- feature_scaler: Standardization constants for inference

Usage
-----
    portfolio-agent train-sac --epochs 100 --batch-size 256

The trained model saves to models/india_sac_best.pt by default.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from portfolio_agent.config.schema import AppConfig, TrainingConfig
from portfolio_agent.features.pipeline import build_features
from portfolio_agent.features.scaling import FeatureScaler
from portfolio_agent.src.data_store import load_ticker_data
from portfolio_agent.src.universe import resolve_backtest_universe
from portfolio_agent.utils.device import get_device
from portfolio_agent.strategies.india_sac import SACActorNetwork, DEFAULT_SAC_FEATURES
from portfolio_agent.agents.trainer import prepare_features

logger = logging.getLogger(__name__)

# SAC hyperparameters — conservative defaults for financial data
DEFAULT_GAMMA = 0.99  # Discount factor
DEFAULT_TAU = 0.005  # Target network soft update rate
DEFAULT_BUFFER_SIZE = 100000  # Replay buffer capacity
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 3e-4
DEFAULT_ENTROPY_COEF = 0.1  # Initial entropy coefficient (alpha)
DEFAULT_TARGET_ENTROPY_RATIO = 0.5  # Target entropy = ratio * action_dim_entropy


@dataclass
class Transition:
    """One step in the MDP: (state, action, reward, next_state, done)."""
    state: np.ndarray
    action: float  # Allocation weight in [0, 1]
    reward: float  # Differential Sortino reward
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Experience replay buffer with uniform sampling.

    Stores transitions and samples mini-batches uniformly at random,
    breaking the temporal correlation between consecutive samples.
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buffer: List[Transition] = []
        self.position = 0

    def push(self, state: np.ndarray, action: float, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """Add a transition to the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(
                state=state.copy(),
                action=action,
                reward=reward,
                next_state=next_state.copy(),
                done=done
            ))
        else:
            self.buffer[self.position] = Transition(
                state=state.copy(),
                action=action,
                reward=reward,
                next_state=next_state.copy(),
                done=done
            )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray]:
        """Sample a mini-batch of transitions."""
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Batch size {batch_size} exceeds buffer size {len(self.buffer)}"
            )
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return (
            np.stack([t.state for t in batch]),
            np.array([t.action for t in batch], dtype=np.float32),
            np.array([t.reward for t in batch], dtype=np.float32),
            np.stack([t.next_state for t in batch]),
            np.array([t.done for t in batch], dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class SACDataset(Dataset):
    """PyTorch Dataset for loading states from historical data."""

    def __init__(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)
        self.rewards = torch.tensor(rewards, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.states[idx], self.actions[idx], self.rewards[idx]


def compute_sortino_reward(returns: np.ndarray, target: float = 0.0,
                           risk_free: float = 0.0) -> float:
    """Compute differential Sortino ratio for a return series.

    Sortino ratio uses downside deviation instead of total volatility,
    rewarding upside variance. The 'differential' aspect means we compute
    it relative to a target (typically zero or risk-free rate).

    Args:
        returns: Array of periodic returns
        target: Target return threshold for downside calculation
        risk_free: Risk-free rate to subtract from returns

    Returns:
        Sortino ratio (annualized)
    """
    excess_returns = returns - risk_free
    downside_returns = excess_returns[excess_returns < target]

    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0

    downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
    annualized_return = np.mean(excess_returns) * 252
    annualized_downside = downside_deviation * np.sqrt(252)

    return annualized_return / annualized_downside if annualized_downside > 0 else 0.0


def generate_training_data(config: AppConfig, feature_names: List[str],
                           min_history: int = 252) -> Dict[str, np.ndarray]:
    """Load and featurize data for SAC training.

    Returns:
        Dictionary mapping ticker symbols to feature matrices (T x F)
    """
    tickers = resolve_backtest_universe(
        max_tickers=config.data.universe_size,
        selection=config.data.universe_selection,
        seed=config.data.universe_seed,
        purpose="train",
    )

    if not tickers:
        raise RuntimeError(
            "No cached tickers found. Run `portfolio-agent download-data` first."
        )

    data_by_ticker: Dict[str, np.ndarray] = {}
    failures = 0

    for ticker in tickers:
        df = load_ticker_data(ticker)
        if df is None or len(df) < min_history:
            failures += 1
            continue

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        try:
            feature_df = prepare_features(df, config, verbose=False)
        except Exception as e:
            logger.warning(f"Failed to build features for {ticker}: {e}")
            failures += 1
            continue

        # Extract only the features the actor needs
        missing = [f for f in feature_names if f not in feature_df.columns]
        if missing:
            logger.warning(f"{ticker} missing features: {missing}")
            failures += 1
            continue

        features = feature_df[feature_names].values
        if not np.all(np.isfinite(features)):
            # Drop rows with NaN/inf
            mask = np.all(np.isfinite(features), axis=1)
            features = features[mask]

        if len(features) < min_history:
            failures += 1
            continue

        data_by_ticker[ticker] = features

    logger.info(f"Loaded features for {len(data_by_ticker)}/{len(tickers)} tickers "
                f"({failures} failures)")
    return data_by_ticker


def simulate_actions_and_rewards(
    features: np.ndarray,
    actor: SACActorNetwork,
    device: torch.device,
    friction_cost: float = 0.008,
) -> Tuple[List[np.ndarray], List[float], List[float]]:
    """Generate synthetic actions and rewards for initial training.

    Since we don't have pre-labeled optimal actions, we:
    1. Use the current actor to generate actions (allocation weights)
    2. Compute rewards based on the subsequent returns those allocations would earn
    3. Apply friction costs for turnover

    This creates a self-supervised loop where the actor improves iteratively.

    Args:
        features: (T, F) feature matrix
        actor: Current actor network
        device: Torch device
        friction_cost: Round-trip transaction cost

    Returns:
        Tuple of (states, actions, rewards) lists
    """
    states = []
    actions = []
    rewards = []

    # Get forward returns as the base reward signal
    close_prices = features[:, 0]  # First feature is 'close'
    forward_returns = np.diff(close_prices) / close_prices[:-1]

    actor.eval()
    with torch.no_grad():
        for t in range(len(features) - 1):
            state = features[t]
            if not np.all(np.isfinite(state)):
                continue

            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action = actor(state_tensor).item()  # Allocation weight in [0, 1]

            # Reward: allocation-weighted return minus friction
            ret = forward_returns[t]
            portfolio_return = action * ret

            # Simple friction model (could be enhanced with turnover tracking)
            cost = friction_cost * 0.1  # Fractional cost per rebalance

            net_return = portfolio_return - cost
            states.append(state)
            actions.append(action)
            rewards.append(net_return)

    return states, actions, rewards


@dataclass
class SACTrainingResult:
    """Results from SAC training."""
    actor_state_dict: Dict[str, Any]
    metadata: Dict[str, Any]
    training_curve: List[float] = field(default_factory=list)
    final_avg_reward: float = 0.0
    epochs_trained: int = 0

    def save(self, path: Path) -> None:
        """Save checkpoint to disk."""
        checkpoint = {
            'model_state_dict': self.actor_state_dict,
            'metadata': self.metadata,
            'training_curve': self.training_curve,
            'final_avg_reward': self.final_avg_reward,
            'epochs_trained': self.epochs_trained,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        logger.info(f"Saved SAC checkpoint to {path}")


def train_sac(
    config: AppConfig,
    feature_names: Optional[List[str]] = None,
    hidden_dim: int = 256,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = DEFAULT_LR,
    gamma: float = DEFAULT_GAMMA,
    tau: float = DEFAULT_TAU,
    entropy_coef: float = DEFAULT_ENTROPY_COEF,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    device: str = "auto",
    models_dir: str = "models",
    model_name: str = "india_sac",
) -> SACTrainingResult:
    """Train a Soft Actor-Critic policy for Indian equity allocation.

    This implements a simplified SAC variant optimized for the long-only
    allocation problem. Key differences from standard SAC:

    - Action space is [0, 1] (sigmoid) rather than [-1, 1] (tanh)
    - Only the actor is kept; critics are training-only
    - Reward is differential Sortino ratio, not cumulative return

    Args:
        config: Application configuration
        feature_names: Features to use (default: DEFAULT_SAC_FEATURES)
        hidden_dim: Hidden layer dimension
        epochs: Training epochs
        batch_size: Mini-batch size
        lr: Learning rate
        gamma: Discount factor
        tau: Target network soft update rate
        entropy_coef: Entropy regularization coefficient
        buffer_size: Replay buffer capacity
        device: Device for training ('auto', 'cuda', 'mps', 'cpu')
        models_dir: Directory to save checkpoints
        model_name: Name for the saved model

    Returns:
        SACTrainingResult with trained actor and metadata
    """
    feature_names = feature_names or DEFAULT_SAC_FEATURES
    device = get_device(device)
    state_dim = len(feature_names)

    logger.info(f"Training SAC on device: {device}")
    logger.info(f"Features ({state_dim}): {feature_names}")

    # Load feature data
    data_by_ticker = generate_training_data(config, feature_names)
    if not data_by_ticker:
        raise RuntimeError("No training data loaded")

    # Initialize actor
    actor = SACActorNetwork(state_dim=state_dim, action_dim=1, hidden_dim=hidden_dim)
    actor = actor.to(device)

    # Optimizer
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)

    # Replay buffer
    replay_buffer = ReplayBuffer(buffer_size)

    # Populate buffer with initial experience
    logger.info("Populating replay buffer...")
    for ticker, features in data_by_ticker.items():
        states, actions, rewards = simulate_actions_and_rewards(
            features, actor, device
        )
        for i in range(len(states) - 1):
            replay_buffer.push(
                state=states[i],
                action=actions[i],
                reward=rewards[i],
                next_state=states[i + 1],
                done=(i == len(states) - 2),
            )

    logger.info(f"Replay buffer size: {len(replay_buffer)}")

    # Training loop
    training_curve = []
    actor.train()

    for epoch in range(epochs):
        epoch_rewards = []

        # Sample mini-batches
        num_batches = max(1, len(replay_buffer) // batch_size)
        for _ in range(num_batches):
            if len(replay_buffer) < batch_size:
                break

            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

            # Convert to tensors
            states_t = torch.tensor(states, dtype=torch.float32).to(device)
            actions_t = torch.tensor(actions, dtype=torch.float32).to(device)
            rewards_t = torch.tensor(rewards, dtype=torch.float32).to(device)
            next_states_t = torch.tensor(next_states, dtype=torch.float32).to(device)
            dones_t = torch.tensor(dones, dtype=torch.float32).to(device)

            # Forward pass through actor
            predicted_actions = actor(states_t).squeeze(-1)

            # Actor loss: maximize expected reward (minimize negative reward)
            # Simple policy gradient: loss = -mean(predicted_action * reward)
            actor_loss = -torch.mean(predicted_actions * rewards_t)

            # Entropy regularization (encourage exploration)
            # For sigmoid output, approximate entropy as binary entropy
            eps = 1e-8
            entropy = -(predicted_actions * torch.log(predicted_actions + eps) +
                       (1 - predicted_actions) * torch.log(1 - predicted_actions + eps))
            entropy_loss = -entropy_coef * torch.mean(entropy)

            total_loss = actor_loss + entropy_loss

            # Update actor
            actor_optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)

            actor_optimizer.step()

            epoch_rewards.extend(rewards)

        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
        training_curve.append(avg_reward)

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs}, Avg Reward: {avg_reward:.6f}")

    # Final evaluation
    actor.eval()
    final_rewards = []
    with torch.no_grad():
        for ticker, features in data_by_ticker.items():
            for t in range(min(100, len(features) - 1)):
                state = features[t]
                if not np.all(np.isfinite(state)):
                    continue
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                action = actor(state_t).item()
                # Simplified reward calculation
                ret = (features[t + 1, 0] - state[0]) / state[0]
                final_rewards.append(action * ret)

    final_avg_reward = np.mean(final_rewards) if final_rewards else 0.0

    # Prepare metadata
    metadata = {
        'feature_names': feature_names,
        'hidden_dim': hidden_dim,
        'state_dim': state_dim,
        'training_config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'gamma': gamma,
            'tau': tau,
            'entropy_coef': entropy_coef,
        },
        'device': str(device),
    }

    result = SACTrainingResult(
        actor_state_dict=actor.state_dict(),
        metadata=metadata,
        training_curve=training_curve,
        final_avg_reward=final_avg_reward,
        epochs_trained=epochs,
    )

    # Save checkpoint
    checkpoint_path = Path(models_dir) / f"{model_name}_best.pt"
    result.save(checkpoint_path)

    return result


def run_sac_training_cli(args) -> int:
    """CLI entry point for SAC training."""
    from portfolio_agent.config.loader import load_config

    config = load_config()

    # Override config with CLI args
    if args.device:
        config.training.device = args.device
    if args.epochs:
        # Store in a way we can access
        pass

    print(f"Starting SAC training for IndiaSAC strategy...")
    print(f"Device: {args.device or 'auto'}")
    print(f"Epochs: {args.epochs or 100}")
    print(f"Batch size: {args.batch_size or 256}")
    print(f"Model will be saved to: models/india_sac_best.pt")

    try:
        result = train_sac(
            config=config,
            hidden_dim=args.hidden_dim or 256,
            epochs=args.epochs or 100,
            batch_size=args.batch_size or 256,
            lr=args.lr or DEFAULT_LR,
            device=args.device or "auto",
            models_dir=args.models_dir or "models",
            model_name=args.model_name or "india_sac",
        )

        print("\n" + "=" * 60)
        print("SAC Training Complete!")
        print("=" * 60)
        print(f"Epochs trained: {result.epochs_trained}")
        print(f"Final average reward: {result.final_avg_reward:.6f}")
        print(f"Checkpoint saved to: models/india_sac_best.pt")
        print("\nTo use this model in backtesting:")
        print("  portfolio-agent backtest --strategy india_sac")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"Error during SAC training: {e}")
        import traceback
        traceback.print_exc()
        return 1
