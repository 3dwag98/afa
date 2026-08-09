"""Tests for the ML strategy's batched GPU/CPU scoring path."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from portfolio_agent.config.schema import StrategyConfig
from portfolio_agent.models.pytorch_models import LSTMForecaster
from portfolio_agent.strategies.ml_strategy import MLStrategy
from portfolio_agent.strategies.types import StrategyContext, RiskParams


FEATURE_NAMES = ["sma_20", "sma_50", "rsi_14", "macd", "bollinger_pct_b", "atr_14", "return_1d"]
SEQUENCE_LENGTH = 10


@pytest.fixture
def trained_checkpoint(tmp_path):
    """Write a tiny, untrained-but-valid LSTM checkpoint + metadata to tmp_path.

    Architecture must match what ModelLoader.load_model() hardcodes
    (hidden_size=64, n_layers=2, dropout=0.2) since it doesn't persist/read
    those from the checkpoint itself.
    """
    model = LSTMForecaster(
        n_features=len(FEATURE_NAMES), hidden_size=64, n_layers=2,
        sequence_length=SEQUENCE_LENGTH, dropout=0.2, n_outputs=1,
    )
    checkpoint_path = tmp_path / "lstm_best.pt"
    torch.save({
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "val_loss": 0.01,
        "feature_names": FEATURE_NAMES,
        "target": "return_5d",
        "sequence_length": SEQUENCE_LENGTH,
        "device": "cpu",
    }, checkpoint_path)

    metadata = {
        "feature_names": FEATURE_NAMES,
        "target": "return_5d",
        "sequence_length": SEQUENCE_LENGTH,
        "device": "cpu",
    }
    with open(tmp_path / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return tmp_path


def _features_df(seed: int) -> pd.DataFrame:
    np.random.seed(seed)
    n = SEQUENCE_LENGTH + 5
    return pd.DataFrame({name: np.random.randn(n) for name in FEATURE_NAMES} | {"close": 100 + np.random.randn(n)})


class TestMLStrategy:
    def test_load_reads_checkpoint_and_metadata(self, trained_checkpoint):
        strategy = MLStrategy(StrategyConfig(), models_dir=str(trained_checkpoint))
        assert strategy.load() is True
        assert strategy.required_features() == FEATURE_NAMES
        assert strategy.supports_gpu_batch is True

    def test_load_fails_gracefully_without_checkpoint(self, tmp_path):
        strategy = MLStrategy(StrategyConfig(), models_dir=str(tmp_path))
        assert strategy.load() is False

    def test_score_batch_stacks_all_eligible_tickers_into_one_forward_pass(self, trained_checkpoint, monkeypatch):
        strategy = MLStrategy(StrategyConfig(), models_dir=str(trained_checkpoint))
        assert strategy.load()

        call_batch_sizes = []
        original_predict_batch = strategy._loader.predict_batch

        def spy_predict_batch(tensor):
            call_batch_sizes.append(tensor.shape[0])
            return original_predict_batch(tensor)

        monkeypatch.setattr(strategy._loader, "predict_batch", spy_predict_batch)

        features_by_symbol = {
            "AAA.NS": _features_df(1),
            "BBB.NS": _features_df(2),
            "CCC.NS": _features_df(3),
        }
        context = StrategyContext(
            risk=RiskParams(0.55, 1.5, 20.0, 1000000.0, 0.01, 0.03), weights={}
        )

        signals = strategy.score_batch(features_by_symbol, context)

        assert set(signals.keys()) == set(features_by_symbol.keys())
        assert call_batch_sizes == [3], "expected exactly one forward pass covering all 3 tickers"
        for sig in signals.values():
            assert sig.trigger == "Model"
            assert sig.signal in ("BUY", "SELL", "HOLD")

    def test_score_batch_skips_tickers_with_insufficient_history(self, trained_checkpoint):
        strategy = MLStrategy(StrategyConfig(), models_dir=str(trained_checkpoint))
        assert strategy.load()

        short_df = pd.DataFrame({name: [0.1, 0.2] for name in FEATURE_NAMES} | {"close": [100.0, 101.0]})
        context = StrategyContext(risk=RiskParams(0.55, 1.5, 20.0, 1000000.0, 0.01, 0.03), weights={})

        signals = strategy.score_batch({"SHORT.NS": short_df}, context)

        assert signals["SHORT.NS"].signal == "HOLD"
        assert signals["SHORT.NS"].rationale == "Insufficient history for model input"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
