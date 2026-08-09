"""Tests for EnsembleStrategy ("UMA" — combining multiple strategies)."""

import pandas as pd
import pytest

from portfolio_agent.config.schema import StrategyConfig
from portfolio_agent.strategies.base import BaseStrategy
from portfolio_agent.strategies.ensemble import EnsembleStrategy
from portfolio_agent.strategies.registry import load_strategy, register_strategy
from portfolio_agent.strategies.types import RiskParams, StrategyContext, StrategySignal


class _FixedStrategy(BaseStrategy):
    """A test double that always returns the same signal, for deterministic ensemble tests."""

    def __init__(self, config: StrategyConfig):
        self._name = config.params.get("name", "fixed")
        self._signal = config.params.get("signal", "BUY")
        self._score = float(config.params.get("score", 80.0))

    @property
    def name(self) -> str:
        return self._name

    def required_features(self):
        return []

    def score(self, symbol, features, context) -> StrategySignal:
        return StrategySignal(
            symbol=symbol, signal=self._signal, score=self._score, trigger="Fixed",
            entry_price=100.0, stop_price=95.0, target_price=110.0,
            reward_risk=2.0, probability_profit=0.6,
        )


register_strategy("fixed_test", _FixedStrategy)


def _risk_params() -> RiskParams:
    return RiskParams(
        target_prob_profit=0.55, min_reward_risk=1.5, min_price_inr=20.0,
        portfolio_value_inr=1000000.0, risk_per_trade_pct=0.01, max_single_position_pct=0.03,
    )


def _write_uma_yaml(tmp_path, method="weighted_blend", vote_mode="majority", members=None):
    if members is None:
        members = [
            {"type": "fixed_test", "weight": 0.5, "params": {"name": "A", "signal": "BUY", "score": 80.0}},
            {"type": "fixed_test", "weight": 0.5, "params": {"name": "B", "signal": "AVOID", "score": 20.0}},
        ]
    import yaml
    spec = {"name": "Test UMA", "method": method, "vote": {"mode": vote_mode}, "members": members}
    path = tmp_path / "uma.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(spec, f)
    return path


class TestEnsembleStrategyBasics:
    def test_is_a_base_strategy(self, tmp_path):
        yaml_path = _write_uma_yaml(tmp_path)
        strategy = load_strategy(StrategyConfig(type="ensemble", params={"yaml_path": str(yaml_path)}))
        assert isinstance(strategy, BaseStrategy)
        assert strategy.name == "Test UMA"

    def test_missing_members_raises(self, tmp_path):
        import yaml
        path = tmp_path / "empty.yaml"
        with open(path, "w") as f:
            yaml.safe_dump({"name": "Empty"}, f)

        with pytest.raises(ValueError, match="members"):
            EnsembleStrategy(StrategyConfig(params={"yaml_path": str(path)}))

    def test_required_features_is_union_of_members(self, tmp_path):
        yaml_path = _write_uma_yaml(tmp_path, members=[
            {"type": "rule_based", "weight": 1.0, "config_path": "config/strategies/trend_breakout.yaml"},
            {"type": "fixed_test", "weight": 1.0, "params": {"name": "F"}},
        ])
        strategy = EnsembleStrategy(StrategyConfig(params={"yaml_path": str(yaml_path)}))
        features = strategy.required_features()
        assert "sma_50" in features
        assert "atr_14" in features


class TestWeightedBlend:
    def test_equal_weight_opposing_signals_cancel_to_hold(self, tmp_path):
        yaml_path = _write_uma_yaml(tmp_path, method="weighted_blend")
        strategy = EnsembleStrategy(StrategyConfig(params={"yaml_path": str(yaml_path)}))
        context = StrategyContext(risk=_risk_params(), weights={})

        sig = strategy.score("TEST", pd.DataFrame({"close": [100.0]}), context)

        # BUY (strength 1.0) at weight 0.5 + AVOID (strength -0.3) at weight 0.5
        # -> blended strength = 0.35 -> WATCH band, not BUY
        assert sig.trigger == "Ensemble:weighted_blend"
        assert sig.signal in ("WATCH", "HOLD")
        assert "member_signals" in sig.extra

    def test_unanimous_buy_gives_buy(self, tmp_path):
        yaml_path = _write_uma_yaml(tmp_path, members=[
            {"type": "fixed_test", "weight": 0.7, "params": {"name": "A", "signal": "BUY", "score": 90.0}},
            {"type": "fixed_test", "weight": 0.3, "params": {"name": "B", "signal": "BUY", "score": 70.0}},
        ])
        strategy = EnsembleStrategy(StrategyConfig(params={"yaml_path": str(yaml_path)}))
        context = StrategyContext(risk=_risk_params(), weights={})

        sig = strategy.score("TEST", pd.DataFrame({"close": [100.0]}), context)

        assert sig.signal == "BUY"
        assert sig.score == pytest.approx(90.0 * 0.7 + 70.0 * 0.3)

    def test_weights_bias_the_blend(self, tmp_path):
        """A heavily-weighted BUY member should dominate a lightly-weighted AVOID member."""
        yaml_path = _write_uma_yaml(tmp_path, members=[
            {"type": "fixed_test", "weight": 0.9, "params": {"name": "A", "signal": "BUY", "score": 90.0}},
            {"type": "fixed_test", "weight": 0.1, "params": {"name": "B", "signal": "AVOID", "score": 10.0}},
        ])
        strategy = EnsembleStrategy(StrategyConfig(params={"yaml_path": str(yaml_path)}))
        context = StrategyContext(risk=_risk_params(), weights={})

        sig = strategy.score("TEST", pd.DataFrame({"close": [100.0]}), context)

        assert sig.signal == "BUY"


class TestVote:
    def test_majority_vote_buy(self, tmp_path):
        yaml_path = _write_uma_yaml(tmp_path, method="vote", vote_mode="majority", members=[
            {"type": "fixed_test", "params": {"name": "A", "signal": "BUY"}},
            {"type": "fixed_test", "params": {"name": "B", "signal": "BUY"}},
            {"type": "fixed_test", "params": {"name": "C", "signal": "SELL"}},
        ])
        strategy = EnsembleStrategy(StrategyConfig(params={"yaml_path": str(yaml_path)}))
        context = StrategyContext(risk=_risk_params(), weights={})

        sig = strategy.score("TEST", pd.DataFrame({"close": [100.0]}), context)

        assert sig.signal == "BUY"
        assert sig.trigger == "Ensemble:vote:majority"

    def test_majority_vote_no_consensus_gives_hold(self, tmp_path):
        yaml_path = _write_uma_yaml(tmp_path, method="vote", vote_mode="majority", members=[
            {"type": "fixed_test", "params": {"name": "A", "signal": "BUY"}},
            {"type": "fixed_test", "params": {"name": "B", "signal": "SELL"}},
        ])
        strategy = EnsembleStrategy(StrategyConfig(params={"yaml_path": str(yaml_path)}))
        context = StrategyContext(risk=_risk_params(), weights={})

        sig = strategy.score("TEST", pd.DataFrame({"close": [100.0]}), context)

        assert sig.signal == "HOLD"

    def test_unanimous_vote_requires_all_agreement(self, tmp_path):
        yaml_path = _write_uma_yaml(tmp_path, method="vote", vote_mode="unanimous", members=[
            {"type": "fixed_test", "params": {"name": "A", "signal": "BUY"}},
            {"type": "fixed_test", "params": {"name": "B", "signal": "BUY"}},
            {"type": "fixed_test", "params": {"name": "C", "signal": "AVOID"}},
        ])
        strategy = EnsembleStrategy(StrategyConfig(params={"yaml_path": str(yaml_path)}))
        context = StrategyContext(risk=_risk_params(), weights={})

        sig = strategy.score("TEST", pd.DataFrame({"close": [100.0]}), context)

        assert sig.signal == "HOLD"

    def test_unanimous_vote_all_buy(self, tmp_path):
        yaml_path = _write_uma_yaml(tmp_path, method="vote", vote_mode="unanimous", members=[
            {"type": "fixed_test", "params": {"name": "A", "signal": "BUY"}},
            {"type": "fixed_test", "params": {"name": "B", "signal": "BUY"}},
        ])
        strategy = EnsembleStrategy(StrategyConfig(params={"yaml_path": str(yaml_path)}))
        context = StrategyContext(risk=_risk_params(), weights={})

        sig = strategy.score("TEST", pd.DataFrame({"close": [100.0]}), context)

        assert sig.signal == "BUY"


class TestExampleUmaYaml:
    def test_example_uma_yaml_loads(self):
        """The shipped example_uma.yaml (rule_based + lstm) must load without error."""
        strategy = load_strategy(StrategyConfig(
            type="ensemble", params={"yaml_path": "config/strategies/example_uma.yaml"}
        ))
        assert strategy.name == "Trend+ML Blend"
        assert strategy.method == "weighted_blend"
        assert len(strategy._members) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
