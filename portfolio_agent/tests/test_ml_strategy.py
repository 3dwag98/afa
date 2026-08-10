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


class TestInputStandardization:
    """The network is fitted on standardized inputs, so inference has to apply
    the identical transform. Scoring raw price levels against weights trained
    on z-scores is not a small error — it is feeding the model values tens of
    thousands of standard deviations from anything it ever saw."""

    def test_scaler_from_metadata_is_applied_before_the_forward_pass(
        self, trained_checkpoint, monkeypatch
    ):
        scaler_payload = {
            "mean": [100.0] * len(FEATURE_NAMES),
            "std": [10.0] * len(FEATURE_NAMES),
            "clip": 10.0,
        }
        metadata_path = trained_checkpoint / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["feature_scaler"] = scaler_payload
        metadata_path.write_text(json.dumps(metadata))

        strategy = MLStrategy(StrategyConfig(), models_dir=str(trained_checkpoint))
        assert strategy.load()

        seen = []
        original = strategy._loader.predict_batch
        monkeypatch.setattr(
            strategy._loader, "predict_batch",
            lambda tensor: (seen.append(tensor.clone()), original(tensor))[1],
        )

        n = SEQUENCE_LENGTH + 2
        raw = pd.DataFrame(
            {name: np.full(n, 150.0) for name in FEATURE_NAMES} | {"close": np.full(n, 150.0)}
        )
        context = StrategyContext(risk=RiskParams(0.55, 1.5, 20.0, 1000000.0, 0.01, 0.03), weights={})

        strategy.score_batch({"AAA.NS": raw}, context)

        # (150 - 100) / 10 == 5.0, not the raw 150.
        assert seen and torch.allclose(seen[0], torch.full_like(seen[0], 5.0))

    def test_checkpoints_without_a_scaler_are_scored_on_raw_features(
        self, trained_checkpoint, monkeypatch
    ):
        """Checkpoints trained before standardization existed carry no scaler
        and must keep behaving exactly as they did."""
        strategy = MLStrategy(StrategyConfig(), models_dir=str(trained_checkpoint))
        assert strategy.load()
        assert strategy._loader.scaler is None

        seen = []
        original = strategy._loader.predict_batch
        monkeypatch.setattr(
            strategy._loader, "predict_batch",
            lambda tensor: (seen.append(tensor.clone()), original(tensor))[1],
        )

        n = SEQUENCE_LENGTH + 2
        raw = pd.DataFrame(
            {name: np.full(n, 150.0) for name in FEATURE_NAMES} | {"close": np.full(n, 150.0)}
        )
        context = StrategyContext(risk=RiskParams(0.55, 1.5, 20.0, 1000000.0, 0.01, 0.03), weights={})

        strategy.score_batch({"AAA.NS": raw}, context)

        assert seen and torch.allclose(seen[0], torch.full_like(seen[0], 150.0))

    def test_infinite_feature_rows_are_dropped_like_nans(self, trained_checkpoint):
        """A zero-priced bar makes the ratio features infinite; dropna() leaves
        those rows in place and the whole forward pass comes back NaN."""
        strategy = MLStrategy(StrategyConfig(), models_dir=str(trained_checkpoint))
        assert strategy.load()

        n = SEQUENCE_LENGTH + 4
        features = _features_df(9).iloc[:n].copy()
        features.iloc[2, 0] = np.inf

        sequence = strategy._sequence_tensor(features)

        assert sequence is not None
        assert torch.isfinite(sequence).all()
        assert len(sequence) == SEQUENCE_LENGTH


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
