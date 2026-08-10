"""Tests for the unified strategy plugin system."""

import pytest
import pandas as pd
from pathlib import Path

from portfolio_agent.config.schema import StrategyConfig
from portfolio_agent.strategies.base import BaseStrategy
from portfolio_agent.strategies.registry import load_strategy, register_strategy, get_available_strategies
from portfolio_agent.strategies.rule_based import RuleBasedStrategy
from portfolio_agent.strategies.types import RiskParams, StrategyContext
from portfolio_agent.strategies.weighting import normalize_weights, combine_weighted, evaluate_and_learn
from src.monte_carlo import MonteCarloResult


def _strategy_config() -> StrategyConfig:
    return StrategyConfig(type="rule_based", params={"yaml_path": "config/strategies/trend_breakout.yaml"})


def _risk_params(target_prob_profit=0.55, min_reward_risk=1.5, min_price_inr=100.0) -> RiskParams:
    return RiskParams(
        target_prob_profit=target_prob_profit,
        min_reward_risk=min_reward_risk,
        min_price_inr=min_price_inr,
        portfolio_value_inr=1000000.0,
        risk_per_trade_pct=0.01,
        max_single_position_pct=0.10,
    )


def _mc_result(probability_profit: float = 0.70) -> MonteCarloResult:
    return MonteCarloResult(
        probability_profit=probability_profit,
        expected_return_pct=0.05,
        var_95=-0.10,
        cvar_95=-0.15,
        simulations_count=1000,
        horizon_days=20,
    )


def _features(
    close=150.0, sma_50=140.0, sma_200=120.0, donchian_upper_20=145.0,
    volume_ratio_20=2.0, atr_14=3.0,
) -> pd.DataFrame:
    """Build a single-row features DataFrame matching RuleBasedStrategy.required_features()."""
    return pd.DataFrame({
        "close": [close],
        "sma_50": [sma_50],
        "sma_200": [sma_200],
        "donchian_upper_20": [donchian_upper_20],
        "volume_ratio_20": [volume_ratio_20],
        "atr_14": [atr_14],
    })


class TestBaseStrategy:
    """Tests for the BaseStrategy abstract class."""

    def test_base_strategy_is_abstract(self):
        with pytest.raises(TypeError):
            BaseStrategy()

    def test_rule_based_is_a_base_strategy(self):
        strategy = RuleBasedStrategy(_strategy_config())
        assert isinstance(strategy, BaseStrategy)


class TestRuleBasedStrategyMetadata:
    """Tests for RuleBasedStrategy YAML-driven metadata."""

    def test_load_from_yaml(self):
        strategy = RuleBasedStrategy(_strategy_config())
        assert strategy.name == "Trend Breakout Volume MC"

    def test_required_features(self):
        strategy = RuleBasedStrategy(_strategy_config())
        features = strategy.required_features()
        for expected in ("close", "sma_50", "sma_200", "donchian_upper_20", "volume_ratio_20", "atr_14"):
            assert expected in features

    def test_entry_rules(self):
        strategy = RuleBasedStrategy(_strategy_config())
        entry_rules = strategy.entry_rules()
        assert "conditions" in entry_rules
        assert len(entry_rules["conditions"]) == 4

    def test_exit_rules(self):
        strategy = RuleBasedStrategy(_strategy_config())
        exit_rules = strategy.exit_rules()
        assert exit_rules["stop_loss"]["multiplier"] == 1.5
        # 3.0x, not 2.0x: a 1.5/2.0 structure is 1.33 gross but under 1.0 net
        # of round-trip friction on a typical ATR, i.e. unclearable by any
        # sensible gate.
        assert exit_rules["take_profit"]["multiplier"] == 3.0


class TestRuleBasedStrategyScoring:
    """Regression-pin tests: these mirror the exact scenarios the old
    src.scoring.score_candidate() covered, to guarantee the consolidated
    RuleBasedStrategy makes identical decisions."""

    def test_perfect_bullish_gives_buy_with_breakout_trigger(self):
        strategy = RuleBasedStrategy(_strategy_config())
        # The YAML's ATR multipliers (1.5 stop / 3.0 target) give a fixed
        # *gross* reward:risk of 2.0 regardless of ATR magnitude. The strategy
        # reports reward:risk NET of round-trip friction, which on a
        # 2%-of-price ATR takes it to ~1.37 — hence the shipped
        # compliance.min_reward_risk of 1.2.
        context = StrategyContext(
            risk=_risk_params(min_reward_risk=1.2),
            weights={"Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0},
            mc_result=_mc_result(0.70),
        )
        # close=150 > sma_50=140 > sma_200=120 (trend=1.0); close > donchian(145) (breakout=1.0)
        features = _features(close=150.0, sma_50=140.0, sma_200=120.0, donchian_upper_20=145.0, volume_ratio_20=2.0)

        sig = strategy.score("TEST", features, context)

        assert sig.signal == "BUY", sig.rationale
        assert sig.score >= 60
        assert sig.trigger == "Breakout"

    def test_reward_risk_is_net_of_round_trip_costs(self):
        """The reported ratio must be the after-friction one, and strictly worse
        than gross — otherwise the min_reward_risk gate flatters every trade."""
        strategy = RuleBasedStrategy(_strategy_config())
        context = StrategyContext(
            risk=_risk_params(min_reward_risk=1.2),
            weights={"Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0},
            mc_result=_mc_result(0.70),
        )
        features = _features(close=150.0, atr_14=3.0)

        sig = strategy.score("TEST", features, context)

        # 3.0x ATR target over 1.5x ATR stop = 2.0 gross, for any ATR.
        assert sig.extra["gross_reward_risk"] == pytest.approx(2.0, abs=1e-3)
        assert sig.reward_risk < sig.extra["gross_reward_risk"]
        assert sig.extra["round_trip_cost_pct"] > 0

    def test_high_cost_assumption_can_flip_buy_to_watch(self):
        """A setup that passes on gross reward:risk but not on net must not BUY."""
        strategy = RuleBasedStrategy(_strategy_config())
        features = _features(close=150.0, sma_50=140.0, sma_200=120.0,
                             donchian_upper_20=145.0, volume_ratio_20=2.0, atr_14=3.0)

        cheap = _risk_params(min_reward_risk=1.2)
        cheap.buy_cost_pct = 0.0
        cheap.sell_cost_pct = 0.0
        expensive = _risk_params(min_reward_risk=1.2)
        expensive.buy_cost_pct = 0.02
        expensive.sell_cost_pct = 0.02

        weights = {"Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0}
        cheap_sig = strategy.score(
            "TEST", features,
            StrategyContext(risk=cheap, weights=weights, mc_result=_mc_result(0.70)),
        )
        expensive_sig = strategy.score(
            "TEST", features,
            StrategyContext(risk=expensive, weights=weights, mc_result=_mc_result(0.70)),
        )

        assert cheap_sig.signal == "BUY"
        assert expensive_sig.signal == "WATCH", expensive_sig.rationale

    def test_missing_sma200_lowers_trend_score(self):
        strategy = RuleBasedStrategy(_strategy_config())
        context = StrategyContext(
            risk=_risk_params(),
            weights={"Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0},
            mc_result=_mc_result(0.70),
        )
        features = _features(sma_200=float("nan"))

        sig = strategy.score("TEST", features, context)

        assert sig.component_scores["Trend"] == 0.0
        assert sig.score < 85

    def test_stop_is_always_below_entry_via_atr_fallback(self):
        # ATR <= 0 falls back to a fixed 2%/3% stop/target, which must stay valid.
        strategy = RuleBasedStrategy(_strategy_config())
        context = StrategyContext(
            risk=_risk_params(),
            weights={"Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0},
            mc_result=_mc_result(0.70),
        )
        features = _features(atr_14=0.0)

        sig = strategy.score("TEST", features, context)

        assert sig.stop_price < sig.entry_price
        assert sig.signal != "AVOID" or sig.rationale  # AVOID only via score/prob/rr gates, not an invalid stop

    def test_trend_trigger_when_no_breakout(self):
        strategy = RuleBasedStrategy(_strategy_config())
        context = StrategyContext(
            risk=_risk_params(),
            weights={"Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0},
            mc_result=_mc_result(0.70),
        )
        # close=150 < donchian=160 -> no breakout; trend still satisfied
        features = _features(close=150.0, sma_50=140.0, sma_200=120.0, donchian_upper_20=160.0, volume_ratio_20=1.0)

        sig = strategy.score("TEST", features, context)

        assert sig.trigger == "Trend"

    def test_volume_trigger(self):
        strategy = RuleBasedStrategy(_strategy_config())
        context = StrategyContext(
            risk=_risk_params(),
            weights={"Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0},
            mc_result=_mc_result(0.70),
        )
        # sma_50 > sma_200 fails the trend>=1.0 condition (close between them), no breakout
        features = _features(close=150.0, sma_50=145.0, sma_200=160.0, donchian_upper_20=160.0, volume_ratio_20=1.6)

        sig = strategy.score("TEST", features, context)

        assert sig.trigger == "Volume"

    def test_empty_features_returns_avoid(self):
        strategy = RuleBasedStrategy(_strategy_config())
        context = StrategyContext(risk=_risk_params(), weights={}, mc_result=None)

        sig = strategy.score("TEST", pd.DataFrame(), context)

        assert sig.signal == "AVOID"
        assert sig.score == 0.0


class TestWeighting:
    """Tests for the pure weighting helper functions."""

    def test_normalize_weights_standard(self):
        weights = {"Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0}
        normalized = normalize_weights(weights)
        assert abs(sum(normalized.values()) - 100.0) < 0.001

    def test_normalize_weights_zero_total(self):
        weights = {"Trend": 0.0, "Breakout": 0.0, "Volume": 0.0, "MC_Prob": 0.0}
        normalized = normalize_weights(weights)
        assert all(v == 25.0 for v in normalized.values())

    def test_normalize_weights_empty(self):
        assert normalize_weights({}) == {}

    def test_combine_weighted_breakout_priority(self):
        scores = {"Trend": 1.0, "Breakout": 1.0, "Volume": 1.0, "MC_Prob": 1.0}
        weights = {"Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0}
        final_score, trigger = combine_weighted(scores, weights)
        assert final_score == pytest.approx(100.0)
        assert trigger == "Breakout"

    def test_evaluate_and_learn_insufficient_trades(self):
        weights = {"Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0}
        new_weights, message = evaluate_and_learn(weights, [], learning_rate=0.15, min_trades_for_learning=5)
        assert new_weights == weights
        assert "Not enough realized trades" in message

    def test_evaluate_and_learn_weights_sum_to_100(self):
        weights = {"Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0}
        trade_history = [
            {"signal_trigger": "Trend", "outcome": "WIN"},
            {"signal_trigger": "Trend", "outcome": "WIN"},
            {"signal_trigger": "Breakout", "outcome": "LOSS"},
            {"signal_trigger": "Breakout", "outcome": "WIN"},
            {"signal_trigger": "Volume", "outcome": "WIN"},
        ]
        new_weights, message = evaluate_and_learn(weights, trade_history, learning_rate=0.15, min_trades_for_learning=5)
        assert abs(sum(new_weights.values()) - 100.0) < 0.01
        assert message is not None


class TestStrategyRegistry:
    """Tests for strategy registry and loading."""

    def test_registry_contains_rule_based(self):
        assert "rule_based" in get_available_strategies()

    def test_load_strategy_rule_based(self):
        strategy = load_strategy(_strategy_config())
        assert isinstance(strategy, RuleBasedStrategy)
        assert strategy.name == "Trend Breakout Volume MC"

    def test_load_strategy_unknown_type(self):
        config = StrategyConfig(type="unknown_strategy", params={"yaml_path": "config/strategies/trend_breakout.yaml"})
        with pytest.raises(ValueError, match="Unknown strategy type"):
            load_strategy(config)

    def test_register_custom_strategy(self):
        class CustomStrategy(BaseStrategy):
            @property
            def name(self):
                return "Custom"

            def required_features(self):
                return []

            def score(self, symbol, features, context):
                from portfolio_agent.strategies.types import StrategySignal
                return StrategySignal(
                    symbol=symbol, signal="HOLD", score=50.0, trigger="None",
                    entry_price=0.0, stop_price=0.0, target_price=0.0,
                    reward_risk=0.0, probability_profit=0.0,
                )

        register_strategy("custom", CustomStrategy)
        assert "custom" in get_available_strategies()


class TestIntegrationWithConfig:
    """Integration tests with configuration loading."""

    def test_strategy_config_from_yaml_path(self):
        yaml_path = Path(__file__).parent.parent / "config" / "strategies" / "trend_breakout.yaml"
        assert yaml_path.exists(), f"Strategy YAML not found at {yaml_path}"

        config = StrategyConfig(type="rule_based", params={"yaml_path": str(yaml_path)})
        strategy = RuleBasedStrategy(config)
        assert strategy.name == "Trend Breakout Volume MC"

    def test_full_workflow_with_sample_data(self):
        strategy = load_strategy(_strategy_config())
        context = StrategyContext(
            risk=_risk_params(),
            weights={"Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0},
            mc_result=_mc_result(0.65),
        )
        features = _features()

        sig = strategy.score("TEST", features, context)

        assert sig.signal in ("BUY", "WATCH", "AVOID")
        assert 0 <= sig.score <= 100
        assert len(strategy.entry_rules().get("conditions", [])) > 0
        assert "stop_loss" in strategy.exit_rules()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
