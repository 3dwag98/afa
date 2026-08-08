"""Strategy registry for dynamic strategy loading."""

from typing import Dict, Type
import importlib
import os

from .base import BaseStrategy
from portfolio_agent.config.schema import StrategyConfig


# Registry of available strategies
STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {}


def register_strategy(name: str, strategy_class: Type[BaseStrategy]) -> None:
    """Register a strategy class in the registry.
    
    Args:
        name: Unique name for the strategy.
        strategy_class: The strategy class to register.
    """
    STRATEGY_REGISTRY[name] = strategy_class


def load_strategy(config: StrategyConfig) -> BaseStrategy:
    """Load a strategy based on configuration.
    
    Args:
        config: StrategyConfig containing strategy type and parameters.
        
    Returns:
        An instance of the configured strategy.
        
    Raises:
        ValueError: If the strategy type is not found in the registry.
        ImportError: If the strategy module cannot be imported.
    """
    strategy_type = config.params.get("type", "rule_based")
    
    # Check if strategy is in registry
    if strategy_type in STRATEGY_REGISTRY:
        strategy_class = STRATEGY_REGISTRY[strategy_type]
        return strategy_class(config)
    
    # Try to dynamically load from module
    module_path = config.module
    try:
        module = importlib.import_module(module_path)
        strategy_class = getattr(module, f"{strategy_type.replace('_', ' ').title().replace(' ', '')}Strategy", None)
        if strategy_class is None:
            # Try with exact name
            strategy_class = getattr(module, f"{strategy_type.title().replace('_', '')}Strategy", None)
        if strategy_class:
            return strategy_class(config)
    except ImportError:
        pass
    
    raise ValueError(f"Unknown strategy type: {strategy_type}")


def get_available_strategies() -> Dict[str, Type[BaseStrategy]]:
    """Return the registry of available strategies.
    
    Returns:
        Dictionary mapping strategy names to their classes.
    """
    return STRATEGY_REGISTRY.copy()
