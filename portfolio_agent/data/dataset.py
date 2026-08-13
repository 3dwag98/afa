"""TimeSeries Dataset and DataLoader utilities for PyTorch."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from portfolio_agent.config.schema import TrainingConfig
from portfolio_agent.utils.device import resolve_device
from portfolio_agent.utils.workers import resolve_dataloader_workers


def sequence_target_positions(
    group_lengths: Sequence[int], sequence_length: int
) -> np.ndarray:
    """Row offsets that get a prediction, given how the panel is grouped.

    A sample is `sequence_length` consecutive rows of history plus the row
    after them, so a group of L rows spends its first `sequence_length` rows as
    history and yields L - sequence_length predictions; a group no longer than
    the window yields none and is skipped entirely.

    This is the single definition of *which rows are predicted*, and it exists
    as a function because three callers have to agree on it: the dataset that
    indexes the rows, `test_split_dates`, and the trainer's `_stacked_dates`.
    When those disagree, nothing raises — rank IC just correlates each
    prediction against a different day's cross-section and reports a confident
    number about the wrong thing.

    Args:
        group_lengths: Row count of each contiguous series in the panel, in
            panel order. Their sum must be the panel's row count.
        sequence_length: Length of the input window.

    Returns:
        Ascending row offsets of the predicted rows, in dataset order.
    """
    positions: List[np.ndarray] = []
    offset = 0
    for length in group_lengths:
        if length > sequence_length:
            positions.append(np.arange(offset + sequence_length, offset + length))
        offset += length
    if not positions:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(positions).astype(np.int64)


def group_lengths_in_slice(
    group_lengths: Sequence[int], start: int, stop: int
) -> List[int]:
    """How the rows in `panel[start:stop]` are grouped.

    A chronological split cuts the stacked panel by row offset, which can fall
    inside a ticker's block — the per-ticker 70/15/15 cuts floor independently,
    so their sum drifts from the panel-level cut by up to one row per ticker.
    Each side of such a cut is still a contiguous run of one ticker's rows, so
    both halves stay valid groups; what would not be valid is assuming the cut
    landed on a boundary and pairing the wrong lengths with the slice.
    """
    lengths: List[int] = []
    offset = 0
    for length in group_lengths:
        overlap = min(offset + length, stop) - max(offset, start)
        if overlap > 0:
            lengths.append(overlap)
        offset += length
    return lengths


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series forecasting.

    Creates sliding window sequences from feature matrices for sequence-to-one
    or sequence-to-sequence prediction tasks.

    **Windows never cross a group boundary.** `features` is normally several
    tickers' rows concatenated into one matrix, and a window that starts near
    the end of one ticker's block and ends inside the next hands the model one
    stock's price history and asks it to predict a different stock's move. That
    is not a small effect at the shapes this repo trains on: with the default
    60-row window and a panel at `data.min_history_days`, every sequence in the
    validation and test loaders straddled a join (see docs/tasks/T18). Passing
    `group_lengths` restricts every window to a single series.

    Boundary awareness is done by filtering the predicted row offsets rather
    than by concatenating one dataset per ticker. Both drop exactly the same
    windows; the filter is preferred because it keeps *which rows are
    predicted* in one array (`target_positions`) that the date-alignment code
    reads directly, instead of re-deriving it from a list of sub-dataset
    lengths. A ConcatDataset would also copy each ticker's rows into its own
    tensor, and the panel is one contiguous matrix by the time it gets here.

    Args:
        features: Feature matrix of shape (n_samples, n_features)
        targets: Target values of shape (n_samples,) or (n_samples, n_targets)
        sequence_length: Length of input sequences
        group_lengths: Row count of each contiguous series in `features`, in
            row order; their sum must equal `len(features)`. None means the
            whole matrix is one series, which is only correct for a single
            instrument — a stacked panel passed without it is trained on
            windows that mix instruments.
    """

    def __init__(
        self,
        features: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor,
        sequence_length: int,
        group_lengths: Optional[Sequence[int]] = None,
    ):
        self.sequence_length = sequence_length

        # Convert to numpy if torch tensor
        if isinstance(features, torch.Tensor):
            features = features.numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.numpy()

        # torch.tensor() copies, which also sidesteps the "given NumPy array is
        # not writable" warning raised when wrapping a read-only view.
        self.features = torch.tensor(np.asarray(features), dtype=torch.float32)
        self.targets = torch.tensor(np.asarray(targets), dtype=torch.float32)

        n_rows = len(self.features)
        if group_lengths is None:
            self.group_lengths = [n_rows]
        else:
            self.group_lengths = [int(length) for length in group_lengths]
            # Checked rather than trusted: lengths that do not add up would
            # silently shift every group boundary, which is the same
            # contamination this argument exists to remove, now harder to see.
            if sum(self.group_lengths) != n_rows:
                raise ValueError(
                    f"group_lengths sums to {sum(self.group_lengths)} but there "
                    f"are {n_rows} rows; boundaries that do not add up would put "
                    "the window splits in the wrong places"
                )

        # The rows this dataset predicts. Ordered, so sample i's row is
        # target_positions[i] — which is what the date alignment relies on.
        self.target_positions = sequence_target_positions(
            self.group_lengths, sequence_length
        )
        self.n_valid_samples = int(self.target_positions.size)

        if self.n_valid_samples <= 0:
            if len(self.group_lengths) == 1:
                raise ValueError(
                    f"sequence_length ({sequence_length}) must be less than "
                    f"number of samples ({n_rows})"
                )
            longest = max(self.group_lengths)
            # This is the shipped default's failure mode, and saying so is the
            # point. A 15% validation slice of a ticker with 250 sessions is 37
            # rows against a 60-row window, so no window fits inside one
            # ticker. The old code produced samples here anyway — every one of
            # them spanning a join, which is where "100% of validation samples
            # were contaminated" comes from. Refusing is the correction, not a
            # new restriction.
            raise ValueError(
                f"sequence_length ({sequence_length}) is not shorter than any of "
                f"the {len(self.group_lengths)} series in this panel (longest: "
                f"{longest} rows), so no window fits inside a single series. "
                f"Every sample here would have to span two instruments. Either "
                f"lower training.sequence_length below {longest}, or raise "
                f"data.min_history_days so each ticker's slice of the split is "
                f"longer than the window."
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
        actual_idx = int(self.target_positions[idx])

        # Get the sequence ending at actual_idx - 1
        start_idx = actual_idx - self.sequence_length
        sequence = self.features[start_idx:actual_idx]

        # Get the target at actual_idx
        target = self.targets[actual_idx]

        return sequence, target


def chronological_split_bounds(n_samples: int) -> Tuple[int, int]:
    """Row offsets where train ends and validation ends, for a 70/15/15 split.

    Stated once because two callers need to agree on it: `create_dataloaders`
    slices the arrays here, and anything that wants to know *which dates* the
    test predictions cover has to slice the index identically. Two copies of
    `int(n * 0.70)` drift the moment one of them is tuned.
    """
    return int(n_samples * 0.70), int(n_samples * 0.85)


def test_split_dates(
    index: pd.Index,
    sequence_length: int,
    group_lengths: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Dates that `create_dataloaders`' test loader actually predicts.

    The test slice starts at the validation boundary, and within it a
    `TimeSeriesDataset` spends the first `sequence_length` rows *of every
    ticker's block* as history rather than only the first `sequence_length`
    rows of the slice. Both cuts are applied here, in that order, through the
    same `sequence_target_positions` the dataset indexes with — an independent
    reimplementation of the offset is how the dates come to name different rows
    than the predictions came from.

    Args:
        index: The stacked panel's index, one entry per row.
        sequence_length: Length of the input window.
        group_lengths: Row count of each ticker's block in the panel. None
            means one contiguous series; on a multi-ticker panel that returns
            more dates than there are predictions, which the caller must not
            paper over by truncating.
    """
    _, val_end = chronological_split_bounds(len(index))
    test_index = np.asarray(index[val_end:])
    if group_lengths is None:
        groups: Sequence[int] = [len(test_index)]
    else:
        groups = group_lengths_in_slice(group_lengths, val_end, len(index))
    return test_index[sequence_target_positions(groups, sequence_length)]


def create_dataloaders(
    df: pd.DataFrame,
    config: TrainingConfig,
    group_lengths: Optional[Sequence[int]] = None,
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
        group_lengths: Row count of each ticker's block in `df`, in row order.
            Required for a stacked multi-ticker panel: without it every window
            that spans two blocks becomes a training sample built from two
            different stocks. Each split's share of the boundaries is derived
            here, because the 70/15/15 cut can land inside a block.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Raises:
        ValueError: if `df` is visibly a stacked panel (its dates step
            backwards, which one instrument's history cannot do) and no
            `group_lengths` were given. Refusing beats training on windows that
            mix instruments, which produces a model and no error at all.
    """
    # Determine device for pin_memory setting. resolve_device() never returns
    # an unusable accelerator, so this agrees with the device training runs on.
    use_cuda = resolve_device(config.device).type == "cuda"

    # One resolved worker count, used consistently for every DataLoader knob:
    # persistent_workers/prefetch_factor are only valid when workers > 0.
    # Zero on Windows: a DataLoader worker there is a spawned interpreter
    # that re-imports torch *and* receives a copy of the dataset tensors.
    num_workers = resolve_dataloader_workers(config.num_workers)
    worker_kwargs = (
        {"persistent_workers": True, "prefetch_factor": 2} if num_workers > 0 else {}
    )

    # A single instrument's dates only ever increase, so a backwards step is
    # proof the frame is several tickers stacked. Caught here because the
    # alternative failure is silent: the run trains, converges and reports
    # metrics, and nothing anywhere says that some fraction of its samples were
    # one stock's history labelled with another stock's forward return.
    if (
        group_lengths is None
        and isinstance(df.index, pd.DatetimeIndex)
        and not df.index.is_monotonic_increasing
    ):
        raise ValueError(
            "this frame's dates step backwards, so it is several tickers "
            "stacked rather than one series; pass group_lengths (one row count "
            "per ticker block) or the sliding window will build samples that "
            "span two instruments"
        )

    # Extract features and targets
    # Assume last column is target, rest are features
    feature_cols = df.columns[:-1].tolist()
    target_col = df.columns[-1]

    features = df[feature_cols].values
    targets = df[target_col].values

    # Chronological split: 70% train, 15% val, 15% test
    train_end, val_end = chronological_split_bounds(len(df))

    # Split features and targets
    train_features = features[:train_end]
    train_targets = targets[:train_end]

    val_features = features[train_end:val_end]
    val_targets = targets[train_end:val_end]

    test_features = features[val_end:]
    test_targets = targets[val_end:]

    # Each split gets the boundaries that fall inside it. A split is a row
    # range, not a whole number of tickers — the per-ticker cuts floor
    # independently of the panel-level one — so the lengths have to be derived
    # per slice rather than reused.
    panel_groups = [len(df)] if group_lengths is None else list(group_lengths)
    train_groups = group_lengths_in_slice(panel_groups, 0, train_end)
    val_groups = group_lengths_in_slice(panel_groups, train_end, val_end)
    test_groups = group_lengths_in_slice(panel_groups, val_end, len(df))

    # Create datasets
    train_dataset = TimeSeriesDataset(
        train_features, train_targets, config.sequence_length, train_groups
    )
    val_dataset = TimeSeriesDataset(
        val_features, val_targets, config.sequence_length, val_groups
    )
    test_dataset = TimeSeriesDataset(
        test_features, test_targets, config.sequence_length, test_groups
    )

    # Create dataloaders with GPU optimizations
    # pin_memory=True enables faster transfer to CUDA GPUs
    # num_workers controls parallel data loading
    # persistent_workers=True avoids worker restart overhead between epochs
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,  # NO SHUFFLING - preserve temporal order
        num_workers=num_workers,  # Limit workers on Windows to avoid overhead
        pin_memory=use_cuda,
        drop_last=True,  # Drop incomplete batches for stable training
        **worker_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        **worker_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        **worker_kwargs,
    )

    print(
        f"Created dataloaders: Train={len(train_dataset)}, "
        f"Val={len(val_dataset)}, Test={len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
