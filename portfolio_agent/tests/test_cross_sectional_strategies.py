"""Tests for cross-sectional momentum and low-volatility strategies."""

import pandas as pd
import pytest
import yaml

from portfolio_agent.config.schema import StrategyConfig
from portfolio_agent.strategies.cross_sectional import MomentumStrategy, LowVolatilityStrategy
from portfolio_agent.strategies.ensemble import EnsembleStrategy
from portfolio_agent.strategies.types import RiskParams, StrategyContext


def _risk_params() -> RiskParams:
    return RiskParams(
        target_prob_profit=0.55,
        min_reward_risk=1.5,
        min_price_inr=20.0,
        portfolio_value_inr=1_000_000.0,
        risk_per_trade_pct=0.01,
        max_single_position_pct=0.10,
        atr_stop_multiplier=1.5,
        atr_target_multiplier=2.0,
    )


def _features(close: float, atr: float = 2.0, **extra) -> pd.DataFrame:
    row = {"close": close, "atr_14": atr, **extra}
    return pd.DataFrame([row])


def _momentum_config(**params) -> StrategyConfig:
    return StrategyConfig(type="momentum", params=params)


def _low_vol_config(**params) -> StrategyConfig:
    return StrategyConfig(type="low_volatility", params=params)


class TestMomentumStrategy:
    def test_defaults(self):
        strategy = MomentumStrategy(_momentum_config())
        assert strategy.name == "momentum"
        assert strategy.requires_full_batch is True
        # realized_vol_60 is required for per-position volatility targeting,
        # not for the ranking metric itself.
        assert strategy.required_features() == [
            "close", "mom_9m_skip1m", "atr_14", "realized_vol_60"
        ]

    def test_min_universe_default_is_statistically_meaningful(self):
        """A 'top decile' of 5 names is one stock, which is not a ranking."""
        strategy = MomentumStrategy(_momentum_config())
        assert strategy.entry_rules()["min_universe"] >= 30

    def test_default_universe_of_ten_is_rejected(self):
        strategy = MomentumStrategy(_momentum_config())
        features_by_symbol = {
            f"SYM{i}": _features(close=100.0, mom_9m_skip1m=i / 100.0) for i in range(1, 11)
        }
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(features_by_symbol, context)

        assert all(s.signal == "AVOID" for s in signals.values())
        assert "too small" in signals["SYM10"].rationale

    def test_custom_name_and_percentile(self):
        strategy = MomentumStrategy(_momentum_config(name="my_momentum", top_percentile=0.2))
        assert strategy.name == "my_momentum"
        assert strategy.entry_rules()["top_percentile"] == 0.2

    def test_top_decile_gets_buy_rest_avoid(self):
        # 10 symbols, momentum values 0.01..0.10; top_percentile=0.1 -> only the highest qualifies.
        strategy = MomentumStrategy(_momentum_config(top_percentile=0.1, min_universe=5))
        features_by_symbol = {
            f"SYM{i}": _features(close=100.0, mom_9m_skip1m=i / 100.0) for i in range(1, 11)
        }
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(features_by_symbol, context)

        assert signals["SYM10"].signal == "BUY"
        assert signals["SYM10"].trigger == "Momentum"
        for i in range(1, 10):
            assert signals[f"SYM{i}"].signal == "AVOID"
            assert signals[f"SYM{i}"].trigger == "None"

    def test_highest_momentum_scores_highest(self):
        strategy = MomentumStrategy(_momentum_config(top_percentile=0.5, min_universe=5))
        features_by_symbol = {
            f"SYM{i}": _features(close=100.0, mom_9m_skip1m=i / 100.0) for i in range(1, 6)
        }
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(features_by_symbol, context)

        assert signals["SYM5"].score > signals["SYM1"].score

    def test_small_universe_falls_back_to_avoid(self):
        strategy = MomentumStrategy(_momentum_config(min_universe=5))
        features_by_symbol = {
            "SYM1": _features(close=100.0, mom_9m_skip1m=0.05),
            "SYM2": _features(close=100.0, mom_9m_skip1m=0.10),
        }
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(features_by_symbol, context)

        assert all(s.signal == "AVOID" for s in signals.values())
        assert "too small" in signals["SYM1"].rationale

    def test_below_min_price_is_watch_not_buy(self):
        strategy = MomentumStrategy(_momentum_config(top_percentile=0.2, min_universe=5))
        features_by_symbol = {f"SYM{i}": _features(close=100.0, mom_9m_skip1m=i / 100.0) for i in range(1, 6)}
        features_by_symbol["SYM5"] = _features(close=5.0, mom_9m_skip1m=0.05)  # below min_price_inr=20
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(features_by_symbol, context)

        assert signals["SYM5"].signal == "WATCH"

    def test_score_single_ticker_delegates_to_score_batch(self):
        strategy = MomentumStrategy(_momentum_config(min_universe=1))
        features = _features(close=100.0, mom_9m_skip1m=0.05)
        context = StrategyContext(risk=_risk_params())

        sig = strategy.score("SOLO", features, context)

        assert sig.symbol == "SOLO"


class TestLowVolatilityStrategy:
    def test_defaults(self):
        strategy = LowVolatilityStrategy(_low_vol_config())
        assert strategy.name == "low_volatility"
        assert strategy.requires_full_batch is True
        assert strategy.required_features() == ["close", "realized_vol_60", "atr_14"]
        assert strategy.entry_rules()["min_universe"] >= 30

    def test_regime_filter_off_by_default(self):
        """Low-volatility is the defensive sleeve — it is meant to keep working
        through the drawdowns that stand momentum down."""
        strategy = LowVolatilityStrategy(_low_vol_config())
        assert strategy.entry_rules()["crash_protection"]["regime_filter"] is False

    def test_lowest_volatility_gets_buy(self):
        strategy = LowVolatilityStrategy(_low_vol_config(top_percentile=0.1, min_universe=5))
        # Lower realized_vol_60 is better; SYM1 has the lowest.
        features_by_symbol = {
            f"SYM{i}": _features(close=100.0, realized_vol_60=i * 0.05) for i in range(1, 11)
        }
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(features_by_symbol, context)

        assert signals["SYM1"].signal == "BUY"
        assert signals["SYM1"].trigger == "LowVolatility"
        assert signals["SYM10"].signal == "AVOID"

    def test_small_universe_falls_back_to_avoid(self):
        strategy = LowVolatilityStrategy(_low_vol_config(min_universe=5))
        features_by_symbol = {
            "SYM1": _features(close=100.0, realized_vol_60=0.10),
            "SYM2": _features(close=100.0, realized_vol_60=0.20),
        }
        context = StrategyContext(risk=_risk_params())

        signals = strategy.score_batch(features_by_symbol, context)

        assert all(s.signal == "AVOID" for s in signals.values())


class TestEnsembleRejectsFullBatchMembers:
    def test_ensemble_raises_for_cross_sectional_member(self, tmp_path):
        uma_path = tmp_path / "bad_uma.yaml"
        uma_path.write_text(yaml.safe_dump({
            "name": "bad_uma",
            "method": "weighted_blend",
            "members": [
                {"type": "momentum", "weight": 1.0, "params": {}},
            ],
        }))
        config = StrategyConfig(type="ensemble", config_path=str(uma_path))

        with pytest.raises(ValueError, match="requires the full eligible universe"):
            EnsembleStrategy(config)
