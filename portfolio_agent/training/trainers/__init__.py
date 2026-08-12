"""Built-in trainers.

Importing this package registers them. It is imported lazily by
`registry._ensure_builtins_loaded`, so an install without PyTorch gets an empty
registry and a clear message rather than an ImportError from a module it never
asked for.
"""

from .supervised import SupervisedTrainer  # noqa: F401
from .sac import SACTrainer  # noqa: F401

__all__ = ["SupervisedTrainer", "SACTrainer"]
