"""ML-backed strategy wrapping a trained PyTorch forecasting model.

Plugs into the same BaseStrategy interface as RuleBasedStrategy so the
platform can train and backtest either kind of strategy identically. Unlike
the rule-based strategy (cheap per-ticker Python, best parallelized across
CPU workers), this strategy batches all eligible tickers' feature sequences
into a single tensor and does one GPU forward pass per scoring round — this
is the concrete GPU speedup for backtesting.

Requires the optional `gpu` extra (torch) to be installed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn

from .base import BaseStrategy
from .types import StrategyContext, StrategySignal
from portfolio_agent.config.schema import StrategyConfig
from portfolio_agent.models.registry import get_model
from portfolio_agent.utils.device import resolve_device

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and manage a trained PyTorch model for inference.

    Handles loading weights + metadata from checkpoint files, moving the model
    to the inference device, and running batched predictions.
    """

    def __init__(self, models_dir: str = "models", device: str = "cpu"):
        self.models_dir = Path(models_dir)
        # resolve_device() downgrades an unavailable accelerator to CPU instead
        # of handing back a torch.device that blows up on the first .to() call.
        self.device = resolve_device(device)
        self.model: Optional[nn.Module] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self._model_loaded = False

    def load_model(self, model_name: str = "lstm") -> bool:
        """Load a trained model from checkpoint. Returns True on success."""
        checkpoint_path = self.models_dir / f"{model_name}_best.pt"
        metadata_path = self.models_dir / "metadata.json"

        if not checkpoint_path.exists():
            logger.error(f"Model checkpoint not found: {checkpoint_path}")
            return False
        if not metadata_path.exists():
            logger.error(f"Metadata file not found: {metadata_path}")
            return False

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        n_features = len(self.metadata.get('feature_names', []))
        sequence_length = self.metadata.get('sequence_length', 60)
        if n_features == 0:
            logger.error("No feature names found in metadata")
            return False

        try:
            model_class = get_model(model_name)
        except KeyError as e:
            logger.error(f"Model class not found: {e}")
            return False

        self.model = model_class(
            n_features=n_features,
            hidden_size=64,
            n_layers=2,
            sequence_length=sequence_length,
            dropout=0.2,
            n_outputs=1,
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        self._model_loaded = True
        logger.info(f"Loaded model '{model_name}' from {checkpoint_path} ({n_features} features, seq_len={sequence_length})")
        return True

    def predict_batch(self, features: torch.Tensor) -> torch.Tensor:
        """Predict on a batch. features shape: (batch_size, sequence_length, n_features)."""
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        with torch.no_grad():
            features = features.to(self.device, non_blocking=True)
            return self.model(features)

    @property
    def feature_names(self) -> List[str]:
        return [] if self.metadata is None else self.metadata.get('feature_names', [])

    @property
    def sequence_length(self) -> int:
        return 60 if self.metadata is None else self.metadata.get('sequence_length', 60)

    @property
    def target_name(self) -> str:
        return "" if self.metadata is None else self.metadata.get('target', '')


def _predicted_value_to_probability(value: float) -> float:
    """Map a raw regression prediction to a 0-1 probability-like confidence."""
    if value < 0:
        prob = 0.5 + (value / 2)
    elif value > 1:
        prob = 1.0 - (1.0 / (1.0 + value))
    else:
        prob = value
    return min(max(prob, 0.0), 1.0)


class MLStrategy(BaseStrategy):
    """Strategy backed by a trained sequence-forecasting model (e.g. LSTM)."""

    def __init__(
        self,
        config: StrategyConfig,
        model_name: str = "lstm",
        models_dir: str = "models",
        device: str = "cpu",
    ):
        self._config = config
        self._model_name = config.params.get("model_name", model_name)
        resolved_device = config.params.get("device", device)
        self._loader = ModelLoader(models_dir=config.params.get("models_dir", models_dir), device=resolved_device)
        self._loaded = False

    def load(self) -> bool:
        """Load the trained model checkpoint. Must be called before scoring."""
        self._loaded = self._loader.load_model(self._model_name)
        return self._loaded

    @property
    def name(self) -> str:
        return f"ml_{self._model_name}"

    @property
    def supports_gpu_batch(self) -> bool:
        return True

    def required_features(self) -> List[str]:
        if self._loader.feature_names:
            return self._loader.feature_names
        raise RuntimeError("MLStrategy.load() must be called before required_features() is meaningful")

    def _sequence_tensor(self, features: pd.DataFrame) -> Optional[torch.Tensor]:
        seq_len = self._loader.sequence_length
        feature_names = self._loader.feature_names
        usable = features[feature_names].dropna()
        if len(usable) < seq_len:
            return None
        recent = usable.iloc[-seq_len:].values
        return torch.tensor(recent, dtype=torch.float32)

    def score(self, symbol: str, features: pd.DataFrame, context: StrategyContext) -> StrategySignal:
        return self.score_batch({symbol: features}, context)[symbol]

    def score_batch(
        self, features_by_symbol: Dict[str, pd.DataFrame], context: StrategyContext
    ) -> Dict[str, StrategySignal]:
        if not self._loaded:
            raise RuntimeError("MLStrategy.load() must be called before scoring")

        eligible: Dict[str, torch.Tensor] = {}
        last_close: Dict[str, float] = {}
        for symbol, features in features_by_symbol.items():
            if features.empty:
                continue
            seq = self._sequence_tensor(features)
            if seq is None:
                continue
            eligible[symbol] = seq
            last_close[symbol] = float(features["close"].iloc[-1]) if "close" in features.columns else 0.0

        results: Dict[str, StrategySignal] = {}

        if eligible:
            # One stacked tensor -> one GPU forward pass for every eligible ticker.
            symbols = list(eligible.keys())
            batch = torch.stack([eligible[s] for s in symbols])  # (n_tickers, seq_len, n_features)
            predictions = self._loader.predict_batch(batch)

            for i, symbol in enumerate(symbols):
                raw_value = float(predictions[i].squeeze().item())
                prob = _predicted_value_to_probability(raw_value)
                close = last_close[symbol]

                if prob > 0.6:
                    signal = "BUY"
                elif prob < 0.4:
                    signal = "SELL"
                else:
                    signal = "HOLD"

                stop_price = close * 0.98
                target_price = close * 1.03
                risk = close - stop_price
                reward_risk = (target_price - close) / risk if risk > 0 else 0.0

                results[symbol] = StrategySignal(
                    symbol=symbol,
                    signal=signal,
                    score=round(prob * 100, 2),
                    trigger="Model",
                    entry_price=close,
                    stop_price=stop_price,
                    target_price=target_price,
                    reward_risk=round(reward_risk, 4),
                    probability_profit=round(prob, 6),
                    component_scores={"Model": prob},
                    rationale=f"model={self._model_name} predicted={raw_value:.4f} prob={prob:.2f}",
                    extra={"model_probability": prob, "raw_prediction": raw_value},
                )

        # Tickers with insufficient history are skipped (HOLD, not scored by the model).
        for symbol, features in features_by_symbol.items():
            if symbol in results:
                continue
            close = float(features["close"].iloc[-1]) if not features.empty and "close" in features.columns else 0.0
            results[symbol] = StrategySignal(
                symbol=symbol, signal="HOLD", score=0.0, trigger="None",
                entry_price=close, stop_price=0.0, target_price=0.0,
                reward_risk=0.0, probability_profit=0.0,
                component_scores={}, rationale="Insufficient history for model input",
            )

        return results
