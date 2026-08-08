"""TimeSeries Dataset and DataLoader utilities for PyTorch."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from portfolio_agent.config.schema import TrainingConfig
from portfolio_agent.utils.device import get_device


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series forecasting.

    Creates sliding window sequences from feature matrices for sequence-to-one
    or sequence-to-sequence prediction tasks.

    Args:
        features: Feature matrix of shape (n_samples, n_features)
        targets: Target values of shape (n_samples,) or (n_samples, n_targets)
        sequence_length: Length of input sequences
    """

    def __init__(
        self,
        features: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor,
        sequence_length: int,
    ):
        self.sequence_length = sequence_length

        # Convert to numpy if torch tensor
        if isinstance(features, torch.Tensor):
            features = features.numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.numpy()

        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

        # Calculate valid indices for sequence creation
        # We need sequence_length points before each target
        self.valid_start_idx = sequence_length
        self.n_valid_samples = len(self.features) - self.valid_start_idx

        if self.n_valid_samples <= 0:
            raise ValueError(
                f"sequence_length ({sequence_length}) must be less than "
                f"number of samples ({len(self.features)})"
            )

    def __len__(self) -> int:
        return self.n_valid_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sequence-target pair.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (sequence, target) where:
                - sequence: Tensor of shape (sequence_length, n_features)
                - target: Tensor of shape (n_targets,) or scalar
        """
        # Actual index in the original data
        actual_idx = idx + self.valid_start_idx

        # Get the sequence ending at actual_idx - 1
        start_idx = actual_idx - self.sequence_length
        sequence = self.features[start_idx:actual_idx]

        # Get the target at actual_idx
        target = self.targets[actual_idx]

        return sequence, target


def create_dataloaders(
    df: pd.DataFrame, config: TrainingConfig
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders from a DataFrame.

    Performs chronological split (no shuffling across time) with:
    - Train: 70%
    - Validation: 15%
    - Test: 15%

    GPU Optimization:
    - pin_memory=True when CUDA is available for faster host-to-device transfer
    - num_workers set from config for parallel data loading

    Args:
        df: DataFrame containing features and target columns.
            Assumes last column is the target, rest are features.
        config: TrainingConfig with sequence_length, batch_size, device, num_workers

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Determine device for pin_memory setting
    device = get_device(config.device)
    use_cuda = device.type == "cuda" and torch.cuda.is_available()

    # Extract features and targets
    # Assume last column is target, rest are features
    feature_cols = df.columns[:-1].tolist()
    target_col = df.columns[-1]

    features = df[feature_cols].values
    targets = df[target_col].values

    # Chronological split: 70% train, 15% val, 15% test
    n_samples = len(df)
    train_end = int(n_samples * 0.70)
    val_end = int(n_samples * 0.85)

    # Split features and targets
    train_features = features[:train_end]
    train_targets = targets[:train_end]

    val_features = features[train_end:val_end]
    val_targets = targets[train_end:val_end]

    test_features = features[val_end:]
    test_targets = targets[val_end:]

    # Create datasets
    train_dataset = TimeSeriesDataset(
        train_features, train_targets, config.sequence_length
    )
    val_dataset = TimeSeriesDataset(
        val_features, val_targets, config.sequence_length
    )
    test_dataset = TimeSeriesDataset(
        test_features, test_targets, config.sequence_length
    )

    # Create dataloaders with GPU optimizations
    # pin_memory=True enables faster transfer to CUDA GPUs
    # num_workers controls parallel data loading
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,  # NO SHUFFLING - preserve temporal order
        num_workers=config.num_workers,
        pin_memory=use_cuda,
        drop_last=True,  # Drop incomplete batches for stable training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=use_cuda,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=use_cuda,
    )

    print(
        f"Created dataloaders: Train={len(train_dataset)}, "
        f"Val={len(val_dataset)}, Test={len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
