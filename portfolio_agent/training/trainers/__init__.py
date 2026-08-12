"""Built-in trainers.

Importing this package registers them. It is imported lazily by
`registry._ensure_builtins_loaded`, so an install without PyTorch gets a clear
message rather than an ImportError from a module it never asked for.

Each import is guarded *separately*. A single `try` around all of them would
mean one trainer's missing dependency silently costs you every other trainer:
`sac` imports PyTorch at module scope, so on a torch-less install a shared
guard would swallow `supervised` and `gbm` with it — and `gbm` needs no torch
at all. What is missing should cost exactly what depends on it.
"""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

#: Built-ins whose module would not import, mapped to why. `list-trainers`
#: prints this, because "sac is not in the list" and "sac needs the gpu extra"
#: are very different messages to receive.
UNAVAILABLE: Dict[str, str] = {}

__all__: List[str] = ["UNAVAILABLE"]

try:
    from .supervised import SupervisedTrainer  # noqa: F401
except ImportError as exc:  # pragma: no cover - depends on optional extras
    UNAVAILABLE["supervised"] = str(exc)
    logger.debug("supervised trainer unavailable: %s", exc)
else:
    __all__.append("SupervisedTrainer")

try:
    from .sac import SACTrainer  # noqa: F401
except ImportError as exc:  # pragma: no cover - depends on optional extras
    UNAVAILABLE["sac"] = f"needs PyTorch (uv sync --extra gpu): {exc}"
    logger.debug("sac trainer unavailable: %s", exc)
else:
    __all__.append("SACTrainer")

# There is no ImportError to catch here in practice: gbm imports scikit-learn
# lazily so `list-trainers` still shows it, with the install hint, on a machine
# that has not got it yet. That hint comes from `GBMTrainer.availability()`.
try:
    from .gbm import GBMTrainer  # noqa: F401
except ImportError as exc:  # pragma: no cover - depends on optional extras
    UNAVAILABLE["gbm"] = f"needs scikit-learn (uv sync --extra gbm): {exc}"
    logger.debug("gbm trainer unavailable: %s", exc)
else:
    __all__.append("GBMTrainer")
