"""Tests for strategy plugin system."""

import pytest
import pandas as pd
from pathlib import Path

from portfolio_agent.config.schema import StrategyConfig
from portfolio_agent.strategies.base import BaseStrategy
from portfolio_agent.strategies.registry import load_strategy, register_strategy, get_available_strategies
from portfolio_agent.strategies.rule_based import RuleBasedStrategy


class TestBaseStrategy:
    """Tests for the BaseStrategy abstract class."""
    
    def test_base_strategy_is_abstract(self):
        """Verify BaseStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseStrategy()
    
    def test_base_strategy_requires_implementation(self):
        """Verify all abstract methods must be implemented."""
        # Any concrete implementation should work
        config = StrategyConfig(
            params={
                "type": "rule_based",
                "yaml_path": "config/strategies/trend_breakout.yaml"
            }
        )
        strategy = RuleBasedStrategy(config)
        assert isinstance(strategy, BaseStrategy)


class TestRuleBasedStrategy:
    """Tests for RuleBasedStrategy implementation."""
    
    @pytest.fixture
    def strategy_config(self):
        """Create a valid strategy configuration."""
        return StrategyConfig(
            params={
                "type": "rule_based",
                "yaml_path": "config/strategies/trend_breakout.yaml"
            }
        )
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature data for testing."""
        return pd.DataFrame({
            'close': [100.0, 102.0, 105.0],
            'sma_50': [98.0, 99.0, 100.0],
            'sma_200': [90.0, 91.0, 92.0],
            'prev_donchian_upper_20': [103.0, 103.0, 103.0],
            'volume_ratio': [1.2, 1.6, 2.0],
            'atr_14': [2.0, 2.1, 2.2],
            'mc_probability': [0.55, 0.60, 0.65]
        })
    
    def test_load_from_yaml(self, strategy_config):
        """Test loading strategy from YAML configuration."""
        strategy = RuleBasedStrategy(strategy_config)
        assert strategy.name == "Trend Breakout Volume MC"
    
    def test_entry_rules(self, strategy_config):
        """Test entry rules are loaded correctly."""
        strategy = RuleBasedStrategy(strategy_config)
        entry_rules = strategy.entry_rules()
        
        assert "conditions" in entry_rules
        assert len(entry_rules["conditions"]) == 4
        
        # Check first condition (close > sma_200)
        cond = entry_rules["conditions"][0]
        assert cond["field"] == "close"
        assert cond["operator"] == ">"
        assert cond["reference_field"] == "sma_200"
    
    def test_exit_rules(self, strategy_config):
        """Test exit rules are loaded correctly."""
        strategy = RuleBasedStrategy(strategy_config)
        exit_rules = strategy.exit_rules()
        
        assert "stop_loss" in exit_rules
        assert "take_profit" in exit_rules
        assert exit_rules["stop_loss"]["multiplier"] == 1.5
        assert exit_rules["take_profit"]["multiplier"] == 2.0
    
    def test_score_calculation(self, strategy_config, sample_features):
        """Test score calculation with sample features."""
        strategy = RuleBasedStrategy(strategy_config)
        score = strategy.score(sample_features)
        
        # Score should be between 0 and 100
        assert 0 <= score <= 100
        
        # With good conditions (price above SMAs, breakout, high volume, good MC prob)
        # we expect a reasonably high score
        assert score > 50
    
    def test_generate_signals_buy(self, strategy_config, sample_features):
        """Test signal generation when conditions are met."""
        strategy = RuleBasedStrategy(strategy_config)
        signals = strategy.generate_signals(sample_features)
        
        assert "signal" in signals
        assert "entry_price" in signals
        assert "stop_price" in signals
        assert "target_price" in signals
        
        # Last row has all conditions met
        assert signals["signal"] == "BUY"
    
    def test_generate_signals_no_data(self, strategy_config):
        """Test signal generation with empty dataframe."""
        strategy = RuleBasedStrategy(strategy_config)
        signals = strategy.generate_signals(pd.DataFrame())
        
        assert signals["signal"] == "HOLD"
        assert signals["reason"] == "No data available"
    
    def test_scoring_weights(self, strategy_config, sample_features):
        """Test that scoring weights are applied correctly."""
        strategy = RuleBasedStrategy(strategy_config)
        
        # Get the rules to verify weights
        rules = strategy._rules
        weights = rules.get("scoring", {}).get("weights", {})
        
        assert weights.get("trend") == 25.0
        assert weights.get("breakout") == 25.0
        assert weights.get("volume") == 20.0
        assert weights.get("model_probability") == 30.0


class TestStrategyRegistry:
    """Tests for strategy registry and dynamic loading."""
    
    @pytest.fixture
    def strategy_config(self):
        """Create a valid strategy configuration."""
        return StrategyConfig(
            params={
                "type": "rule_based",
                "yaml_path": "config/strategies/trend_breakout.yaml"
            }
        )
    
    def test_registry_contains_rule_based(self):
        """Verify rule_based strategy is registered."""
        strategies = get_available_strategies()
        assert "rule_based" in strategies
    
    def test_load_strategy_rule_based(self, strategy_config):
        """Test loading rule_based strategy via registry."""
        strategy = load_strategy(strategy_config)
        assert isinstance(strategy, RuleBasedStrategy)
        assert strategy.name == "Trend Breakout Volume MC"
    
    def test_load_strategy_unknown_type(self):
        """Test error handling for unknown strategy type."""
        config = StrategyConfig(
            params={
                "type": "unknown_strategy",
                "yaml_path": "config/strategies/trend_breakout.yaml"
            }
        )
        
        with pytest.raises(ValueError, match="Unknown strategy type"):
            load_strategy(config)
    
    def test_register_custom_strategy(self):
        """Test registering a custom strategy."""
        
        class CustomStrategy(BaseStrategy):
            name = "Custom"
            
            def generate_signals(self, features):
                return {"signal": "HOLD"}
            
            def score(self, features):
                return 50.0
            
            def entry_rules(self):
                return {}
            
            def exit_rules(self):
                return {}
        
        register_strategy("custom", CustomStrategy)
        strategies = get_available_strategies()
        assert "custom" in strategies


class TestIntegrationWithConfig:
    """Integration tests with configuration loading."""
    
    def test_strategy_config_from_yaml_path(self):
        """Test that strategy can be loaded from config path."""
        yaml_path = Path(__file__).parent.parent / "config" / "strategies" / "trend_breakout.yaml"
        
        # Verify YAML file exists
        assert yaml_path.exists(), f"Strategy YAML not found at {yaml_path}"
        
        config = StrategyConfig(
            params={
                "type": "rule_based",
                "yaml_path": str(yaml_path)
            }
        )
        
        strategy = RuleBasedStrategy(config)
        assert strategy.name == "Trend Breakout Volume MC"
    
    def test_full_workflow_with_sample_data(self):
        """Test complete workflow: load strategy, generate signals, calculate score."""
        config = StrategyConfig(
            params={
                "type": "rule_based",
                "yaml_path": "config/strategies/trend_breakout.yaml"
            }
        )
        
        strategy = load_strategy(config)
        
        # Create realistic feature data
        features = pd.DataFrame({
            'close': [100.0, 102.0, 105.0],
            'sma_50': [98.0, 99.0, 100.0],
            'sma_200': [90.0, 91.0, 92.0],
            'prev_donchian_upper_20': [103.0, 103.0, 103.0],
            'volume_ratio': [1.2, 1.6, 2.0],
            'atr_14': [2.0, 2.1, 2.2],
            'mc_probability': [0.55, 0.60, 0.65]
        })
        
        # Generate signals
        signals = strategy.generate_signals(features)
        assert signals["signal"] == "BUY"
        
        # Calculate score
        score = strategy.score(features)
        assert 0 <= score <= 100
        
        # Verify entry/exit rules
        entry_rules = strategy.entry_rules()
        exit_rules = strategy.exit_rules()
        
        assert len(entry_rules.get("conditions", [])) > 0
        assert "stop_loss" in exit_rules
        assert "take_profit" in exit_rules
