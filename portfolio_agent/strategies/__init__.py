"""Strategy module initialization.

Registration used to happen here *and* in `registry.py` — `rule_based` was
registered twice, from two files, with the second call silently overwriting the
first. Since T25 each class registers itself with `@register_strategy` at its
own definition, and the registry imports the modules lazily on first lookup, so
this file re-exports and nothing more.
"""

from .base import BaseStrategy, TrainableStrategy
from .registry import (
    get_available_strategies,
    get_strategy,
    is_strategy_registered,
    list_strategies,
    load_strategy,
    register_strategy,
    unavailable_strategies,
)
from .types import StrategyContext, StrategySignal

__all__ = [
    "BaseStrategy",
    "TrainableStrategy",
    "StrategyContext",
    "StrategySignal",
    "load_strategy",
    "register_strategy",
    "get_strategy",
    "get_available_strategies",
    "list_strategies",
    "is_strategy_registered",
    "unavailable_strategies",
]
