"""Tests for the training pipeline (agents/trainer.py).

Focus: load_data() must load real cached tickers by default (the historical
bug being fixed here was that it always returned synthetic data), and must
fall back to synthetic data only when explicitly requested.
"""

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.agents.trainer import load_data, prepare_features, _generate_synthetic_ohlcv
from portfolio_agent.config.schema import AppConfig


def _make_ohlcv(n_days: int = 300, seed: int = 1) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    close = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
    return pd.DataFrame({
        'open': close + np.random.randn(n_days) * 0.1,
        'high': close + np.abs(np.random.randn(n_days)) * 0.3,
        'low': close - np.abs(np.random.randn(n_days)) * 0.3,
        'close': close,
        'volume': np.random.randint(100000, 1000000, n_days).astype(float),
    }, index=dates)


class TestPrepareFeatures:
    def test_builds_features_and_target(self):
        config = AppConfig()
        df = _make_ohlcv()
        feature_df = prepare_features(df, config, verbose=False)

        assert config.training.target in feature_df.columns
        assert len(feature_df) > 0
        assert not feature_df.isna().any().any()


class TestLoadDataUsesRealDataByDefault:
    """Regression test for the historical bug: load_data() always returned
    synthetic random-walk data instead of real cached tickers."""

    def test_synthetic_flag_returns_synthetic_data(self):
        config = AppConfig.model_validate({"training": {"use_synthetic_data": True}})
        df = load_data(config)
        assert len(df) > 0
        assert config.training.target in df.columns

    def test_default_loads_real_cached_tickers(self, monkeypatch, tmp_path):
        """With use_synthetic_data unset (default False), load_data() must
        call into the real data store rather than fabricating data."""
        calls = []

        def fake_resolve_backtest_universe(max_tickers=None):
            calls.append(max_tickers)
            return ["FAKE1.NS", "FAKE2.NS"]

        def fake_load_ticker_data(ticker, start_date=None, end_date=None):
            return _make_ohlcv(seed=hash(ticker) % 1000)

        monkeypatch.setattr("portfolio_agent.agents.trainer.resolve_backtest_universe", fake_resolve_backtest_universe)
        monkeypatch.setattr("portfolio_agent.agents.trainer.load_ticker_data", fake_load_ticker_data)

        config = AppConfig.model_validate({
            "training": {"use_synthetic_data": False, "sequence_length": 10},
            "data": {"universe_size": 5, "min_history_days": 50},
        })

        df = load_data(config)

        assert calls, "load_data() must call resolve_backtest_universe() when not using synthetic data"
        assert len(df) > 0
        assert config.training.target in df.columns

    def test_raises_when_no_cached_tickers_available(self, monkeypatch):
        monkeypatch.setattr("portfolio_agent.agents.trainer.resolve_backtest_universe", lambda max_tickers=None: [])

        config = AppConfig.model_validate({"training": {"use_synthetic_data": False}})

        with pytest.raises(RuntimeError, match="No cached tickers"):
            load_data(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
