"""PyTorch model implementations for portfolio forecasting."""

from __future__ import annotations

import torch
import torch.nn as nn

from .registry import register_model


@register_model('lstm')
class LSTMForecaster(nn.Module):
    """
    LSTM-based forecaster for time series prediction.

    Architecture:
        - Input: (batch_size, sequence_length, n_features)
        - LSTM layers with dropout
        - Fully connected output layer

    Args:
        n_features: Number of input features
        hidden_size: Size of LSTM hidden state
        n_layers: Number of LSTM layers
        sequence_length: Length of input sequences
        dropout: Dropout probability for regularization
        n_outputs: Number of output predictions (default 1)
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        sequence_length: int = 60,
        dropout: float = 0.2,
        n_outputs: int = 1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.sequence_length = sequence_length

        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_features)

        Returns:
            Output tensor of shape (batch_size, n_outputs)
        """
        # LSTM forward pass
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Take only the last time step output
        # last_output shape: (batch_size, hidden_size)
        last_output = lstm_out[:, -1, :]

        # Apply dropout
        out = self.dropout(last_output)

        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class PyTorchModelWrapper:
    """
    Wrapper class for PyTorch models handling device management.

    This wrapper handles:
        - Moving model to the appropriate device (CPU/CUDA/MPS)
        - Model training and evaluation modes
        - Saving and loading model weights

    Args:
        model: The PyTorch model to wrap
        device: torch.device or string specifying the device
        learning_rate: Learning rate for the optimizer
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str,
        learning_rate: float = 0.001,
    ):
        if isinstance(device, str):
            device = torch.device(device)

        self.device = device
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        print(f"Model wrapped and moved to device: {self.device}")

    def train_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> float:
        """
        Perform a single training step.

        Args:
            batch_x: Input batch tensor
            batch_y: Target batch tensor

        Returns:
            Loss value for this batch
        """
        self.model.train()

        # Move data to device
        batch_x = batch_x.to(self.device, non_blocking=True)
        batch_y = batch_y.to(self.device, non_blocking=True)

        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(batch_x)
        loss = self.criterion(outputs.squeeze(), batch_y)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> float:
        """
        Perform validation on a batch.

        Args:
            batch_x: Input batch tensor
            batch_y: Target batch tensor

        Returns:
            Loss value for this batch
        """
        self.model.eval()

        with torch.no_grad():
            # Move data to device
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            outputs = self.model(batch_x)
            loss = self.criterion(outputs.squeeze(), batch_y)

        return loss.item()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on input data.

        Args:
            x: Input tensor

        Returns:
            Predictions tensor
        """
        self.model.eval()

        with torch.no_grad():
            x = x.to(self.device, non_blocking=True)
            outputs = self.model(x)

        return outputs

    def save(self, path: str) -> None:
        """
        Save model weights to file.

        Args:
            path: Path to save the model weights
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "device": str(self.device),
            },
            path,
        )
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """
        Load model weights from file.

        Args:
            path: Path to load the model weights from
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Model loaded from {path}")
