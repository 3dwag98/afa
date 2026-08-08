"""Strategy module initialization."""

from .base import BaseStrategy
from .registry import load_strategy, register_strategy, get_available_strategies
from .rule_based import RuleBasedStrategy

# Register built-in strategies
register_strategy("rule_based", RuleBasedStrategy)

__all__ = [
    "BaseStrategy",
    "RuleBasedStrategy", 
    "load_strategy",
    "register_strategy",
    "get_available_strategies",
]
