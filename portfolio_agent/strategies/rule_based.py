"""Rule-based strategy implementation for portfolio agent."""

from typing import Any, Dict, Optional
import pandas as pd
import yaml
from pathlib import Path

from .base import BaseStrategy
from portfolio_agent.config.schema import StrategyConfig


class RuleBasedStrategy(BaseStrategy):
    """Rule-based trading strategy that reads configuration from YAML.
    
    This strategy implements the "Trend + Breakout + Volume + Monte Carlo" 
    scoring logic based on configurable rules from a YAML file.
    """
    
    def __init__(self, config: StrategyConfig):
        """Initialize the rule-based strategy.
        
        Args:
            config: StrategyConfig containing path to YAML configuration.
        """
        self._config = config
        self._yaml_path = config.params.get("yaml_path", config.config_path)
        self._rules = self._load_rules()
        
    def _load_rules(self) -> Dict[str, Any]:
        """Load rules from YAML configuration file.
        
        Returns:
            Dictionary containing entry/exit rules and scoring weights.
        """
        yaml_path = Path(self._yaml_path)
        if not yaml_path.is_absolute():
            # First try relative to portfolio_agent package root
            package_root = Path(__file__).parent.parent
            yaml_path = package_root / self._yaml_path
            
            # If not found, try relative to workspace root
            if not yaml_path.exists():
                workspace_root = package_root.parent
                yaml_path = workspace_root / self._yaml_path
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Strategy YAML file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    
    @property
    def name(self) -> str:
        """Return the name of the strategy."""
        return self._rules.get("name", "RuleBasedStrategy")
    
    def entry_rules(self) -> Dict[str, Any]:
        """Return the entry rules for this strategy.
        
        Returns:
            Dictionary describing entry conditions.
        """
        return self._rules.get("entry", {})
    
    def exit_rules(self) -> Dict[str, Any]:
        """Return the exit rules for this strategy.
        
        Returns:
            Dictionary describing exit conditions (stop loss, target).
        """
        return self._rules.get("exit", {})
    
    def generate_signals(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals based on feature data.
        
        Args:
            features: DataFrame containing feature data.
            
        Returns:
            Dictionary containing signal information.
        """
        if features.empty:
            return {
                "signal": "HOLD",
                "entry_price": 0.0,
                "stop_price": 0.0,
                "target_price": 0.0,
                "reason": "No data available"
            }
        
        latest = features.iloc[-1]
        entry_rules = self.entry_rules()
        exit_rules = self.exit_rules()
        
        # Check entry conditions
        conditions_met = self._check_entry_conditions(latest, entry_rules)
        
        if not conditions_met:
            return {
                "signal": "HOLD",
                "entry_price": float(latest.get("close", 0)),
                "stop_price": 0.0,
                "target_price": 0.0,
                "reason": "Entry conditions not met"
            }
        
        # Calculate prices
        close = float(latest.get("close", latest.get("adj_close", 0)))
        atr = float(latest.get("atr_14", close * 0.02))  # Default ATR ~2% of price
        
        stop_multiplier = exit_rules.get("stop_loss", {}).get("multiplier", 1.5)
        target_multiplier = exit_rules.get("take_profit", {}).get("multiplier", 2.0)
        
        stop_price = close - (atr * stop_multiplier)
        target_price = close + (atr * target_multiplier)
        
        return {
            "signal": "BUY",
            "entry_price": close,
            "stop_price": stop_price,
            "target_price": target_price,
            "reason": "All entry conditions met"
        }
    
    def _check_entry_conditions(self, row: pd.Series, entry_rules: Dict[str, Any]) -> bool:
        """Check if all entry conditions are met.
        
        Args:
            row: Single row of feature data.
            entry_rules: Entry rules from configuration.
            
        Returns:
            True if all conditions are met, False otherwise.
        """
        conditions = entry_rules.get("conditions", [])
        
        for condition in conditions:
            field_name = condition.get("field")
            operator = condition.get("operator")
            value = condition.get("value")
            reference_field = condition.get("reference_field")
            
            # Get actual value from row
            actual_value = row.get(field_name)
            if actual_value is None:
                return False
            
            # Compare against reference field or fixed value
            if reference_field:
                compare_value = row.get(reference_field)
                if compare_value is None:
                    return False
            else:
                compare_value = value
            
            # Evaluate condition
            if not self._evaluate_condition(actual_value, operator, compare_value):
                return False
        
        return True
    
    def _evaluate_condition(self, actual: float, operator: str, expected: float) -> bool:
        """Evaluate a single condition.
        
        Args:
            actual: Actual value from data.
            operator: Comparison operator.
            expected: Expected value or threshold.
            
        Returns:
            True if condition is satisfied.
        """
        if operator == ">":
            return actual > expected
        elif operator == ">=":
            return actual >= expected
        elif operator == "<":
            return actual < expected
        elif operator == "<=":
            return actual <= expected
        elif operator == "==":
            return actual == expected
        elif operator == "!=":
            return actual != expected
        return False
    
    def score(self, features: pd.DataFrame) -> float:
        """Calculate a score for the given feature set.
        
        Uses weighted scoring based on trend, breakout, volume, and 
        model probability components.
        
        Args:
            features: DataFrame containing feature data.
            
        Returns:
            Score between 0.0 and 100.0.
        """
        if features.empty:
            return 0.0
        
        latest = features.iloc[-1]
        weights_config = self._rules.get("scoring", {}).get("weights", {})
        
        # Get weights (should sum to 100)
        weight_trend = weights_config.get("trend", 25.0)
        weight_breakout = weights_config.get("breakout", 25.0)
        weight_volume = weights_config.get("volume", 20.0)
        weight_mc_prob = weights_config.get("model_probability", 30.0)
        
        # Calculate component scores
        trend_score = self._calculate_trend_score(latest)
        breakout_score = self._calculate_breakout_score(latest)
        volume_score = self._calculate_volume_score(latest)
        mc_prob_score = float(latest.get("mc_probability", 0.5))  # Default 0.5 if not present
        
        # Normalize MC probability to 0-100 scale
        mc_prob_score_normalized = mc_prob_score * 100.0
        
        # Calculate weighted final score
        final_score = (
            weight_trend * trend_score +
            weight_breakout * breakout_score +
            weight_volume * volume_score +
            weight_mc_prob * mc_prob_score_normalized
        )
        
        return min(max(final_score, 0.0), 100.0)
    
    def _calculate_trend_score(self, row: pd.Series) -> float:
        """Calculate trend score (0-1 scale).
        
        Args:
            row: Single row of feature data.
            
        Returns:
            Trend score between 0 and 1.
        """
        close = row.get("close", 0)
        sma_50 = row.get("sma_50")
        sma_200 = row.get("sma_200")
        
        if sma_200 is None or sma_50 is None or close == 0:
            return 0.0
        
        # Full points if close > sma_50 > sma_200 (strong uptrend)
        if close > sma_50 and sma_50 > sma_200:
            return 1.0
        # Partial points if just above sma_200
        elif close > sma_200:
            return 0.5
        else:
            return 0.0
    
    def _calculate_breakout_score(self, row: pd.Series) -> float:
        """Calculate breakout score (0-1 scale).
        
        Args:
            row: Single row of feature data.
            
        Returns:
            Breakout score between 0 and 1.
        """
        close = row.get("close", 0)
        prev_donchian_upper = row.get("prev_donchian_upper_20")
        
        if prev_donchian_upper is None or close == 0:
            return 0.0
        
        # Full points if price broke out above previous Donchian upper
        if close > prev_donchian_upper:
            return 1.0
        else:
            return 0.0
    
    def _calculate_volume_score(self, row: pd.Series) -> float:
        """Calculate volume score (0-1 scale).
        
        Args:
            row: Single row of feature data.
            
        Returns:
            Volume score between 0 and 1.
        """
        volume_ratio = row.get("volume_ratio")
        
        if volume_ratio is None:
            return 0.0
        
        # Normalize: ratio of 2.0 or higher gives full score
        return min(volume_ratio / 2.0, 1.0)
