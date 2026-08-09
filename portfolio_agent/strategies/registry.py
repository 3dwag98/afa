"""Strategy registry for loading configured strategies.

Registration is explicit (no import-guessing) so the set of available
strategies is always known statically. The ML strategy is optional: it's only
registered when torch is installed (the `gpu` extra).
"""

from typing import Dict, Type

from .base import BaseStrategy
from .rule_based import RuleBasedStrategy
from portfolio_agent.config.schema import StrategyConfig

STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {}


def register_strategy(name: str, strategy_class: Type[BaseStrategy]) -> None:
    """Register a strategy class in the registry.

    Args:
        name: Unique name for the strategy.
        strategy_class: The strategy class to register.
    """
    STRATEGY_REGISTRY[name] = strategy_class


register_strategy("rule_based", RuleBasedStrategy)

try:
    from .ml_strategy import MLStrategy
    register_strategy("lstm", MLStrategy)
except ImportError:
    # torch (the `gpu` extra) is not installed; the "lstm" strategy is unavailable.
    pass

from .ensemble import EnsembleStrategy
register_strategy("ensemble", EnsembleStrategy)


def load_strategy(config: StrategyConfig) -> BaseStrategy:
    """Load a strategy based on configuration.

    Args:
        config: StrategyConfig containing the strategy type and parameters.

    Returns:
        An instance of the configured strategy.

    Raises:
        ValueError: If the strategy type is not registered.
    """
    strategy_type = config.type or config.params.get("type", "rule_based")

    if strategy_type not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy type: {strategy_type!r}. Available: {sorted(STRATEGY_REGISTRY)}"
        )

    return STRATEGY_REGISTRY[strategy_type](config)


def get_available_strategies() -> Dict[str, Type[BaseStrategy]]:
    """Return the registry of available strategies.

    Returns:
        Dictionary mapping strategy names to their classes.
    """
    return STRATEGY_REGISTRY.copy()
