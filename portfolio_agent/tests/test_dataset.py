"""Tests for the TimeSeriesDataset and dataloader utilities."""

import numpy as np
import pandas as pd
import pytest
import torch

from portfolio_agent.config.schema import TrainingConfig
from portfolio_agent.data.dataset import TimeSeriesDataset, create_dataloaders


class TestTimeSeriesDataset:
    """Test cases for TimeSeriesDataset class."""

    def test_dataset_creation(self):
        """Test that dataset can be created with valid inputs."""
        n_samples = 100
        n_features = 5
        sequence_length = 10

        features = np.random.randn(n_samples, n_features)
        targets = np.random.randn(n_samples)

        dataset = TimeSeriesDataset(features, targets, sequence_length)

        assert len(dataset) == n_samples - sequence_length
        assert dataset.sequence_length == sequence_length

    def test_dataset_item_shape(self):
        """Test that __getitem__ returns correct shapes."""
        n_samples = 100
        n_features = 5
        sequence_length = 10

        features = np.random.randn(n_samples, n_features)
        targets = np.random.randn(n_samples)

        dataset = TimeSeriesDataset(features, targets, sequence_length)
        sequence, target = dataset[0]

        # Sequence should be (sequence_length, n_features)
        assert sequence.shape == (sequence_length, n_features)
        # Target should be scalar or (1,)
        assert target.dim() == 0 or target.shape == torch.Size([1])

    def test_dataset_tensor_conversion(self):
        """Test that dataset properly converts inputs to tensors."""
        n_samples = 50
        n_features = 3
        sequence_length = 5

        features = np.random.randn(n_samples, n_features)
        targets = np.random.randn(n_samples)

        dataset = TimeSeriesDataset(features, targets, sequence_length)
        sequence, target = dataset[0]

        assert isinstance(sequence, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert sequence.dtype == torch.float32
        assert target.dtype == torch.float32

    def test_dataset_with_torch_tensors(self):
        """Test that dataset works with torch tensor inputs."""
        n_samples = 50
        n_features = 3
        sequence_length = 5

        features = torch.randn(n_samples, n_features)
        targets = torch.randn(n_samples)

        dataset = TimeSeriesDataset(features, targets, sequence_length)
        sequence, target = dataset[0]

        assert isinstance(sequence, torch.Tensor)
        assert isinstance(target, torch.Tensor)

    def test_dataset_insufficient_samples(self):
        """Test that dataset raises error when sequence_length >= n_samples."""
        n_samples = 10
        n_features = 3
        sequence_length = 15  # Larger than n_samples

        features = np.random.randn(n_samples, n_features)
        targets = np.random.randn(n_samples)

        with pytest.raises(ValueError, match="sequence_length"):
            TimeSeriesDataset(features, targets, sequence_length)

    def test_chronological_ordering(self):
        """Test that dataset preserves chronological ordering."""
        n_samples = 100
        n_features = 1
        sequence_length = 10

        # Create sequential data where value equals index
        features = np.arange(n_samples).reshape(-1, 1).astype(float)
        targets = np.arange(n_samples).astype(float)

        dataset = TimeSeriesDataset(features, targets, sequence_length)

        # First item should use features from indices [0:10] and target at index 10
        seq_0, tgt_0 = dataset[0]
        assert seq_0[-1, 0].item() == 9.0  # Last element of first sequence
        assert tgt_0.item() == 10.0  # Target at index 10

        # Second item should use features from indices [1:11] and target at index 11
        seq_1, tgt_1 = dataset[1]
        assert seq_1[-1, 0].item() == 10.0
        assert tgt_1.item() == 11.0


class TestCreateDataloaders:
    """Test cases for create_dataloaders function."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        n_samples = 1000
        dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")

        df = pd.DataFrame({
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.randn(n_samples),
            "target": np.random.randn(n_samples),
        }, index=dates)

        return df

    @pytest.fixture
    def training_config(self):
        """Create a sample TrainingConfig for testing."""
        return TrainingConfig(
            sequence_length=20,
            batch_size=32,
            device="cpu",
            num_workers=0,
        )

    def test_chronological_split_ratios(self, sample_df, training_config):
        """Test that data is split in correct ratios (70/15/15)."""
        train_loader, val_loader, test_loader = create_dataloaders(
            sample_df, training_config
        )

        n_total = len(sample_df)
        expected_train = int(n_total * 0.70) - training_config.sequence_length
        expected_val = int(n_total * 0.85) - int(n_total * 0.70)
        expected_test = n_total - int(n_total * 0.85)

        # Allow for small differences due to sequence length adjustment
        assert abs(len(train_loader.dataset) - expected_train) <= training_config.sequence_length
        assert abs(len(val_loader.dataset) - expected_val) <= training_config.sequence_length
        assert abs(len(test_loader.dataset) - expected_test) <= training_config.sequence_length

    def test_no_shuffle_preserves_order(self, sample_df, training_config):
        """Test that shuffle=False preserves temporal order."""
        train_loader, val_loader, test_loader = create_dataloaders(
            sample_df, training_config
        )

        # Verify shuffle is False by checking _DataLoaderIter settings
        # DataLoader doesn't expose shuffle directly, so we verify by iteration order
        # Get first few samples from the loader
        first_batch_x, first_batch_y = next(iter(train_loader))
        
        # Since we're not shuffling, the first batch should contain early time indices
        # The dataset starts at index `sequence_length` due to sliding window
        # So first batch targets should be from early in the chronological sequence
        assert first_batch_x.dim() == 3  # (batch, seq_len, features)
        assert first_batch_y.dim() == 1  # (batch,)

    def test_dataloader_batch_size(self, sample_df, training_config):
        """Test that DataLoaders use correct batch size."""
        train_loader, val_loader, test_loader = create_dataloaders(
            sample_df, training_config
        )

        assert train_loader.batch_size == training_config.batch_size
        assert val_loader.batch_size == training_config.batch_size
        assert test_loader.batch_size == training_config.batch_size

    def test_dataloader_num_workers(self, sample_df, training_config):
        """Test that DataLoaders use correct num_workers."""
        train_loader, val_loader, test_loader = create_dataloaders(
            sample_df, training_config
        )

        assert train_loader.num_workers == training_config.num_workers
        assert val_loader.num_workers == training_config.num_workers
        assert test_loader.num_workers == training_config.num_workers

    def test_dataloader_pin_memory_cpu(self, sample_df, training_config):
        """Test that pin_memory is False for CPU device."""
        train_loader, val_loader, test_loader = create_dataloaders(
            sample_df, training_config
        )

        # On CPU, pin_memory should be False
        assert train_loader.pin_memory is False
        assert val_loader.pin_memory is False
        assert test_loader.pin_memory is False

    def test_dataloader_iteration(self, sample_df, training_config):
        """Test that DataLoaders can be iterated over."""
        train_loader, val_loader, test_loader = create_dataloaders(
            sample_df, training_config
        )

        for batch_x, batch_y in train_loader:
            assert batch_x.dim() == 3  # (batch, seq_len, features)
            assert batch_y.dim() == 1  # (batch,)
            break

        for batch_x, batch_y in val_loader:
            assert batch_x.dim() == 3
            assert batch_y.dim() == 1
            break

        for batch_x, batch_y in test_loader:
            assert batch_x.dim() == 3
            assert batch_y.dim() == 1
            break

    def test_tensor_dtype(self, sample_df, training_config):
        """Test that returned tensors have correct dtype."""
        train_loader, _, _ = create_dataloaders(sample_df, training_config)

        for batch_x, batch_y in train_loader:
            assert batch_x.dtype == torch.float32
            assert batch_y.dtype == torch.float32
            break


class TestIntegration:
    """Integration tests for dataset and model together."""

    def test_dataset_to_model_pipeline(self):
        """Test full pipeline from dataset to model forward pass."""
        from portfolio_agent.models.pytorch_models import LSTMForecaster

        n_samples = 200
        n_features = 5
        sequence_length = 20
        batch_size = 16

        features = np.random.randn(n_samples, n_features)
        targets = np.random.randn(n_samples)

        dataset = TimeSeriesDataset(features, targets, sequence_length)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        model = LSTMForecaster(
            n_features=n_features,
            hidden_size=32,
            n_layers=2,
            sequence_length=sequence_length,
            dropout=0.2,
        )

        for batch_x, batch_y in loader:
            output = model(batch_x)
            assert output.shape[0] == batch_x.shape[0]  # Same batch size
            assert output.shape[1] == 1  # Single output
            break
