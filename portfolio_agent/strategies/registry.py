"""Registry of strategies.

Mirrors `training/registry.py` and `models/registry.py` — decorator to
register, `get_*` to look up, `list_*` to enumerate — so there is one registry
idiom in the codebase rather than four dialects of one. This was the last of
the four still using a bare `register(name, cls)` function call, which meant
registration lived in this file rather than beside the class it named, and a
new strategy had to be added in two places.

Registration is *lazy*, for the same reason it is in the trainer registry:
importing this module must not drag in PyTorch. Rule-based backtests are
supported on installs without the `gpu` extra, and an eager `import torch` at
the top of a registry would break `list-strategies` on those machines. It also
resolves the circularity a decorator introduces — a strategy module imports
`register_strategy` from here, so this module cannot import the strategy
modules at its top.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Type, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from portfolio_agent.config.schema import StrategyConfig

    from .base import BaseStrategy

STRATEGY_REGISTRY: Dict[str, Type["BaseStrategy"]] = {}

#: Built-ins that could not be imported, mapped to why. Absence and
#: unavailability are different facts: a name simply missing from
#: `list_strategies()` looks like a typo, where "needs the gpu extra" says what
#: to install.
UNAVAILABLE: Dict[str, str] = {}

# Guards the one-shot import of the built-in strategies. A bool rather than
# `if not STRATEGY_REGISTRY`, so an install without torch does not retry the
# failing import on every lookup.
_BUILTINS_LOADED = False


def register_strategy(name: str) -> Callable[[Type["BaseStrategy"]], Type["BaseStrategy"]]:
    """Decorator registering a strategy class under `name`.

    Args:
        name: Registry key, referenced from config as `strategy.type` and from
            the CLI as `--strategy`.

    Returns:
        The decorator, which returns the class unchanged.

    Raises:
        ValueError: On a duplicate name. Two classes under one key means
            whichever module imported last wins, which is not a thing to
            discover from a backtest result.

    Example:
        @register_strategy("momentum")
        class MomentumStrategy(BaseStrategy):
            ...
    """

    def decorator(strategy_class: Type["BaseStrategy"]) -> Type["BaseStrategy"]:
        existing = STRATEGY_REGISTRY.get(name)
        if existing is not None and existing is not strategy_class:
            raise ValueError(
                f"Strategy {name!r} is already registered to "
                f"{existing.__module__}.{existing.__qualname__}."
            )
        STRATEGY_REGISTRY[name] = strategy_class
        return strategy_class

    return decorator


def _ensure_builtins_loaded() -> None:
    """Import the built-in strategies once, tolerating a torch-less install."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    # Set first: the imports below re-enter this module via the decorator, and
    # a strategy whose own import fails must not leave the flag unset and cause
    # every later lookup to retry a multi-second failing import.
    _BUILTINS_LOADED = True

    from . import cross_sectional  # noqa: F401  (imports register by side effect)
    from . import ensemble  # noqa: F401
    from . import rule_based  # noqa: F401

    try:
        from . import india_sac  # noqa: F401
        from . import ml_strategy  # noqa: F401
    except ImportError as exc:
        reason = f"needs the `gpu` extra (uv sync --extra gpu): {exc}"
        UNAVAILABLE["lstm"] = reason
        UNAVAILABLE["india_sac"] = reason


def load_strategy(config: "StrategyConfig") -> "BaseStrategy":
    """Construct the strategy the configuration names.

    Args:
        config: StrategyConfig containing the strategy type and parameters.

    Returns:
        An instance of the configured strategy.

    Raises:
        ValueError: If the strategy type is not registered.
    """
    _ensure_builtins_loaded()
    strategy_type = config.type or config.params.get("type", "rule_based")

    if strategy_type not in STRATEGY_REGISTRY:
        hint = ""
        if strategy_type in UNAVAILABLE:
            hint = f" It is a built-in that {UNAVAILABLE[strategy_type]}"
        raise ValueError(
            f"Unknown strategy type: {strategy_type!r}. "
            f"Available: {sorted(STRATEGY_REGISTRY)}.{hint}"
        )

    return STRATEGY_REGISTRY[strategy_type](config)


def get_strategy(name: str) -> Type["BaseStrategy"]:
    """Look up a strategy class by registry name.

    Raises:
        KeyError: If no strategy is registered under that name, listing what is.
    """
    _ensure_builtins_loaded()
    if name not in STRATEGY_REGISTRY:
        raise KeyError(
            f"Strategy {name!r} is not registered. "
            f"Available: {sorted(STRATEGY_REGISTRY)}"
        )
    return STRATEGY_REGISTRY[name]


def get_available_strategies() -> Dict[str, Type["BaseStrategy"]]:
    """Return the registry of available strategies.

    Returns:
        Dictionary mapping strategy names to their classes.
    """
    _ensure_builtins_loaded()
    return STRATEGY_REGISTRY.copy()


def list_strategies() -> List[str]:
    """Return the sorted names of every registered strategy."""
    _ensure_builtins_loaded()
    return sorted(STRATEGY_REGISTRY)


def is_strategy_registered(name: str) -> bool:
    """Whether `name` resolves to a registered strategy."""
    _ensure_builtins_loaded()
    return name in STRATEGY_REGISTRY


def unavailable_strategies() -> Dict[str, str]:
    """Built-ins that could not be imported, mapped to why."""
    _ensure_builtins_loaded()
    return dict(UNAVAILABLE)
