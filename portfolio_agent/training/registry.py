"""Registry of training procedures.

The platform already made three things pluggable — features
(`features/registry.py`), model architectures (`models/registry.py`) and
strategies (`strategies/registry.py`). The one thing that was not pluggable is
*how a strategy gets trained*: `agents/trainer.py::run_training` is a single
supervised pipeline (panel -> sequence windows -> forward-return label ->
walk-forward -> calibration), and a strategy that learns some other way had
nowhere to attach. This registry is that fourth seam.

It deliberately mirrors `models/registry.py` — decorator to register, `get_*`
to look up, `list_*` to enumerate — so there is one registry idiom in the
codebase rather than four dialects of one.

Registration of the built-in trainers is *lazy*. Importing this module must not
drag in PyTorch: rule-based backtests are supported on installs without the
`gpu` extra, and an eager `import torch` at the top of a registry would break
`list-strategies` on those machines. The built-ins are imported on first
lookup instead, and a missing torch degrades to "no trainers registered"
rather than an ImportError.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Type, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from .base import BaseTrainer

_TRAINER_REGISTRY: Dict[str, Type["BaseTrainer"]] = {}

# Guards the one-shot import of the built-in trainers. A bool rather than
# `if not _TRAINER_REGISTRY`, so that an install without torch (which registers
# nothing) does not retry the failing import on every single lookup.
_BUILTINS_LOADED = False


def register_trainer(name: str) -> Callable[[Type["BaseTrainer"]], Type["BaseTrainer"]]:
    """Decorator registering a trainer class under `name`.

    Args:
        name: Registry key, referenced from config as `training.trainer`.

    Returns:
        The decorator, which returns the class unchanged.

    Example:
        @register_trainer("sac")
        class SACTrainer(BaseTrainer):
            ...
    """

    def decorator(trainer_class: Type["BaseTrainer"]) -> Type["BaseTrainer"]:
        _TRAINER_REGISTRY[name] = trainer_class
        return trainer_class

    return decorator


def _ensure_builtins_loaded() -> None:
    """Import the built-in trainers once, tolerating a torch-less install."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    # Set first: the import below re-enters this module via the decorator, and
    # a trainer whose own import fails must not leave the flag unset and cause
    # every later lookup to retry a multi-second failing import.
    _BUILTINS_LOADED = True
    try:
        from . import trainers  # noqa: F401  (import registers by side effect)
    except ImportError:
        # PyTorch (the `gpu` extra) is absent. Nothing to train, but callers
        # that only want to *list* what is available still work.
        pass


def get_trainer(name: str) -> Type["BaseTrainer"]:
    """Look up a trainer class by registry name.

    Args:
        name: Registry key.

    Returns:
        The trainer class.

    Raises:
        KeyError: If no trainer is registered under that name. The message
            lists what *is* available, because the overwhelmingly common cause
            is a typo in a YAML file rather than a genuinely missing trainer.
    """
    _ensure_builtins_loaded()
    if name not in _TRAINER_REGISTRY:
        available = sorted(_TRAINER_REGISTRY)
        if available:
            hint = (
                f" Available: {available}. A built-in trainer missing from that "
                "list needs an optional extra that is not installed — 'sac' needs "
                "`uv sync --extra gpu`."
            )
        else:
            hint = (
                " No trainers are registered at all, which means the package "
                "itself failed to import."
            )
        raise KeyError(f"Trainer {name!r} is not registered.{hint}")
    return _TRAINER_REGISTRY[name]


def list_trainers() -> List[str]:
    """Return the sorted names of every registered trainer."""
    _ensure_builtins_loaded()
    return sorted(_TRAINER_REGISTRY)


def unavailable_trainers() -> Dict[str, str]:
    """Built-ins that could not be imported, mapped to why.

    Absence and unavailability are different facts and deserve different
    messages: a name that is simply not in `list_trainers()` looks like a typo,
    where "sac needs the gpu extra" tells you what to install.
    """
    _ensure_builtins_loaded()
    try:
        from .trainers import UNAVAILABLE

        return dict(UNAVAILABLE)
    except ImportError:  # pragma: no cover - the package itself failed to load
        return {}


def is_trainer_registered(name: str) -> bool:
    """Whether `name` resolves to a registered trainer."""
    _ensure_builtins_loaded()
    return name in _TRAINER_REGISTRY
