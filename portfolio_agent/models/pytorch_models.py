"""PyTorch model implementations for portfolio forecasting.

Two things here are deliberate departures from the obvious defaults, both for
the same reason: a 5-day equity return is mostly noise, and the obvious
defaults are the ones that quietly reward a model for saying nothing.

**Loss.** Squared error is minimized by predicting the conditional mean, and
the conditional mean of a 5-day return is very close to a constant. A network
trained on MSELoss therefore converges to a near-constant output that scores
excellently and forecasts nothing — the mean-reversion trap. It also produces
a bare point estimate, which the trigger engine cannot turn into an expected
value without inventing a distribution around it. Pinball (quantile) loss fixes
both: it asks for the 10th, 50th and 90th percentiles of the forward return,
a constant answer cannot satisfy three different asymmetric penalties at once,
and the spread between the outer quantiles is a natively-calibrated confidence
interval.

**Architecture.** A vanilla LSTM compresses a 60-day, multi-feature window into
one hidden vector and predicts from the final timestep, so everything the
sequence contained has to survive a single bottleneck. PatchTST groups the
window into short patches and attends over them, which keeps local momentum
structure addressable and costs O(N^2) in the number of patches (12) rather
than timesteps (60).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .registry import register_model

# 10th / 50th / 90th percentile of the forward return. The outer pair brackets
# the plausible range without chasing the extreme tail, which needs far more
# data than a few years of daily bars can supply.
DEFAULT_QUANTILES = (0.1, 0.5, 0.9)


class QuantileLoss(nn.Module):
    """Pinball loss over several quantiles at once.

    For quantile q, prediction yhat and realized y, the penalty is asymmetric:

        L_q = max(q * (y - yhat), (q - 1) * (y - yhat))

    Under-predicting the 90th percentile costs 0.9 per unit while
    over-predicting costs 0.1, so the minimizer of L_0.9 is the true 90th
    percentile rather than the mean. Averaging over q fits the whole set
    jointly from one head.

    Nothing here forces the outputs to be ordered — a network can, especially
    early in training, emit a 90th percentile below its 10th. That is left to
    inference (see `sorted_quantiles`) rather than penalized during training:
    a monotonicity penalty trades off against the pinball objective and
    distorts the quantiles it is supposed to protect, while sorting at
    prediction time is exact and free.

    Args:
        quantiles: Quantile levels in (0, 1), in the order the model emits them.
    """

    def __init__(self, quantiles: Sequence[float] = DEFAULT_QUANTILES):
        super().__init__()
        quantiles = tuple(float(q) for q in quantiles)
        if not quantiles:
            raise ValueError("QuantileLoss needs at least one quantile")
        if any(not 0.0 < q < 1.0 for q in quantiles):
            raise ValueError(f"quantiles must lie strictly inside (0, 1), got {quantiles}")
        self.quantiles = quantiles
        self.register_buffer("_levels", torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Mean pinball loss.

        Args:
            predictions: (batch, n_quantiles), or (batch,) for a single quantile.
            targets: (batch,) realized values.

        Returns:
            Scalar loss.
        """
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(-1)
        targets = targets.reshape(-1, 1).to(predictions.dtype)

        errors = targets - predictions
        levels = self._levels.to(predictions.dtype).to(predictions.device)
        return torch.maximum(levels * errors, (levels - 1.0) * errors).mean()


class PointLoss(nn.Module):
    """Squared error against a single-output model, tolerant of trailing dims.

    Exists so the training loop can call `loss(outputs, targets)` uniformly
    regardless of head shape. The previous code reshaped with `.squeeze()` at
    the call site, which silently collapsed the batch dimension too whenever a
    final batch happened to contain one sample.
    """

    def __init__(self):
        super().__init__()
        self._mse = nn.MSELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._mse(predictions.reshape(-1), targets.reshape(-1).to(predictions.dtype))


def sorted_quantiles(predictions: torch.Tensor) -> torch.Tensor:
    """Repair quantile crossing by sorting each row ascending.

    A set of quantile estimates that is not monotone is not a distribution, and
    everything downstream — interval width as a confidence proxy, the median as
    a point forecast — assumes it is one. Sorting is the standard repair and is
    guaranteed not to increase the pinball loss.
    """
    if predictions.dim() == 1:
        return predictions
    return torch.sort(predictions, dim=-1).values


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


@register_model('patchtst')
class PatchTSTForecaster(nn.Module):
    """Patch-based transformer forecaster (Nie et al., PatchTST).

    Three ideas, each earning its place on noisy daily equity data:

    **Patching.** The 60-day window is cut into non-overlapping patches of
    `patch_length` days, and each patch — not each day — becomes one token. A
    single day's return is almost pure noise; a week of them carries a shape
    (a trend, a reversal, a volatility expansion) worth attending to. It also
    cuts attention from 60x60 to 12x12, which is where the quadratic cost of
    a plain transformer over raw timesteps stops mattering.

    **Channel independence.** Every feature is embedded and encoded by the
    *same* transformer weights, applied separately per feature rather than by
    concatenating features into one token. Mixing channels inside attention
    lets the model fit spurious cross-feature relationships in-sample, which
    is exactly the overfitting a few years of daily bars cannot afford. The
    encoder therefore never sees more than one channel at a time; the head
    below is where the channels are finally combined, because a single forecast
    has to come from somewhere.

    **Instance normalization.** Each window is centred and scaled by its own
    statistics before encoding and left that way — the target is a *return*, so
    there is nothing to de-normalize. This is what lets one set of weights serve
    a ₹30 small-cap and a ₹3,000 large-cap without the level dominating. It is
    computed strictly within the window, so it introduces no look-ahead.

    The window is truncated to a whole number of patches from the *most recent*
    end, so the oldest few days are dropped rather than the newest.

    Args:
        n_features: Number of input feature channels.
        hidden_size: Token embedding width D.
        n_layers: Transformer encoder layers.
        sequence_length: Input window length in days.
        dropout: Dropout probability.
        n_outputs: Head width — 3 for the default quantile triple, 1 for a
            point forecast.
        patch_length: Days per patch.
        n_heads: Attention heads.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        sequence_length: int = 60,
        dropout: float = 0.2,
        n_outputs: int = 3,
        patch_length: int = 5,
        n_heads: int = 4,
    ):
        super().__init__()

        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")
        if patch_length < 1 or patch_length > sequence_length:
            raise ValueError(
                f"patch_length must be in [1, sequence_length={sequence_length}], "
                f"got {patch_length}"
            )
        # Attention splits the embedding across heads, so the width has to
        # divide evenly. Failing here beats a shape error deep inside the
        # encoder on the first forward pass.
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})"
            )

        self.n_features = n_features
        self.sequence_length = sequence_length
        self.patch_length = patch_length
        self.n_patches = sequence_length // patch_length
        if self.n_patches < 1:
            raise ValueError("sequence_length is shorter than one patch")
        self.used_length = self.n_patches * patch_length
        self.hidden_size = hidden_size

        self.patch_embedding = nn.Linear(patch_length, hidden_size)
        # Learned rather than sinusoidal: with 12 positions there is nothing to
        # extrapolate to, and learned embeddings converge faster at this size.
        self.position_embedding = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm: markedly more stable at these depths
        )
        # Nested tensors are a fast path for padded batches; every sequence
        # here is the same length, and asking for it under norm_first only
        # produces a warning about the optimization not applying.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.dropout = nn.Dropout(dropout)

        # Per-channel flatten, then cross-channel mixing in the head.
        self.channel_head = nn.Linear(self.n_patches * hidden_size, hidden_size)
        self.output_head = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_features * hidden_size, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, sequence_length, n_features).

        Returns:
            (batch, n_outputs) — the quantile triple when n_outputs is 3.
        """
        batch_size = x.shape[0]

        # Keep the most recent whole patches.
        x = x[:, -self.used_length:, :]
        # (batch, channels, time)
        x = x.transpose(1, 2)

        # Per-window, per-channel instance normalization.
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / (std + 1e-6)

        # (batch * channels, n_patches, patch_length)
        x = x.reshape(batch_size * self.n_features, self.n_patches, self.patch_length)

        tokens = self.patch_embedding(x) + self.position_embedding
        tokens = self.dropout(tokens)
        encoded = self.encoder(tokens)

        # (batch * channels, n_patches * hidden) -> (batch, channels * hidden)
        flattened = encoded.reshape(batch_size * self.n_features, -1)
        per_channel = self.channel_head(flattened)
        combined = per_channel.reshape(batch_size, self.n_features * self.hidden_size)

        return self.output_head(combined)


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
        quantiles: Quantile levels the model's head emits, or None for a
            single-output point forecast trained on squared error.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str,
        learning_rate: float = 0.001,
        quantiles: Sequence[float] | None = DEFAULT_QUANTILES,
    ):
        if isinstance(device, str):
            device = torch.device(device)

        self.device = device
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.quantiles = tuple(quantiles) if quantiles else None
        self.criterion = (
            QuantileLoss(self.quantiles).to(device) if self.quantiles else PointLoss()
        )

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

        # Forward pass. The loss (not the call site) owns reshaping: a
        # `.squeeze()` here collapses the batch dimension too whenever a final
        # batch holds a single sample.
        self.optimizer.zero_grad()
        outputs = self.model(batch_x)
        loss = self.criterion(outputs, batch_y)

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
            loss = self.criterion(outputs, batch_y)

        return loss.item()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on input data.

        Quantile heads are sorted ascending before being returned, so a row
        of outputs is always a usable distribution (see sorted_quantiles).

        Args:
            x: Input tensor

        Returns:
            Predictions tensor
        """
        self.model.eval()

        with torch.no_grad():
            x = x.to(self.device, non_blocking=True)
            outputs = self.model(x)
            if self.quantiles:
                outputs = sorted_quantiles(outputs)

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
