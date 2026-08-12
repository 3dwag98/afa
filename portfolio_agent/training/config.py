"""Resolving *which* trainer runs and *with what* hyperparameters.

Training settings used to live in one global `training:` block, which works
exactly as long as there is one training procedure. With several, three
questions need answering per run: which trainer, with which knobs, and who
wins when two sources disagree.

Precedence, strongest first:

    1. explicit overrides  (`--set epochs=200`, or a notebook keyword argument)
    2. the strategy's own YAML   (`config/strategies/<strategy>.yaml`)
    3. the global `training:` block in config.yaml
    4. the trainer's schema defaults

Layer 3 is what keeps this backward compatible: an existing install with
nothing but a global `training:` block resolves to exactly the settings it
resolved to before, and `portfolio-agent train` behaves unchanged.

Only keys the target trainer actually declares are taken from the global block.
That matters because the global block carries supervised-only settings
(`sequence_length`, `target_transform`, `quantiles`) that would trip the
`extra="forbid"` guard on an RL trainer's schema — a strict merge would make
config.yaml un-loadable the moment a second trainer existed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type

import yaml
from pydantic import ValidationError

from .base import BaseTrainer, TrainerConfig
from .registry import get_trainer

logger = logging.getLogger(__name__)

#: Where per-strategy YAML lives, relative to the package root.
STRATEGY_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config" / "strategies"

DEFAULT_TRAINER = "supervised"


def strategy_config_path(strategy: str) -> Optional[Path]:
    """Return `config/strategies/<strategy>.yaml` if it exists."""
    candidate = STRATEGY_CONFIG_DIR / f"{strategy}.yaml"
    return candidate if candidate.exists() else None


def load_strategy_training_block(
    strategy: str, config_path: Optional[Path | str] = None
) -> Dict[str, Any]:
    """Read the `training:` block out of a strategy's YAML file.

    Args:
        strategy: Strategy registry name, used to find the conventional file.
        config_path: Explicit YAML path, overriding the convention.

    Returns:
        The `training:` mapping, or an empty dict when the file or the block is
        absent. A strategy with no YAML is normal, not an error.
    """
    path = Path(config_path) if config_path else strategy_config_path(strategy)
    if path is None or not Path(path).exists():
        return {}

    with open(path, "r") as handle:
        document = yaml.safe_load(handle) or {}

    block = document.get("training") or {}
    if not isinstance(block, Mapping):
        raise ValueError(
            f"{path}: 'training:' must be a mapping of hyperparameters, got "
            f"{type(block).__name__}"
        )
    return dict(block)


def parse_overrides(pairs: Optional[List[str]]) -> Dict[str, Any]:
    """Turn `["epochs=200", "gamma=0.99"]` into a dict.

    Values are left as strings: pydantic coerces them against the target
    field's type, so `epochs=200` becomes an int and `epochs=banana` becomes a
    validation error naming the field — which is more useful than a guess made
    here without knowing the schema.
    """
    overrides: Dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(
                f"--set expects KEY=VALUE, got {pair!r}"
            )
        key, _, value = pair.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"--set expects a non-empty key, got {pair!r}")
        overrides[key] = value.strip()
    return overrides


def resolve_trainer_name(
    strategy: Optional[str],
    *,
    explicit: Optional[str] = None,
    strategy_block: Optional[Mapping[str, Any]] = None,
) -> str:
    """Decide which trainer trains `strategy`.

    Order: an explicit request, then the strategy YAML's `training.trainer`,
    then the strategy class's own `trainer_name` declaration, then the
    supervised default.
    """
    if explicit:
        return explicit

    if strategy_block and strategy_block.get("trainer"):
        return str(strategy_block["trainer"])

    if strategy:
        declared = _declared_trainer(strategy)
        if declared:
            return declared

    return DEFAULT_TRAINER


def _declared_trainer(strategy: str) -> Optional[str]:
    """Read `trainer_name` off a registered strategy class, if it declares one.

    Import failures are swallowed on purpose: resolving a trainer name must not
    require a working PyTorch install, and the caller has a sensible default.
    """
    try:
        from portfolio_agent.strategies.registry import get_available_strategies

        strategy_class = get_available_strategies().get(strategy)
    except Exception:  # pragma: no cover - depends on optional extras
        return None
    name = getattr(strategy_class, "trainer_name", None)
    return str(name) if name else None


def _strategy_class_defaults(strategy: str) -> Dict[str, Any]:
    """Trainer settings a `TrainableStrategy` subclass declares for itself."""
    try:
        from portfolio_agent.strategies.registry import get_available_strategies

        strategy_class = get_available_strategies().get(strategy)
    except Exception:  # pragma: no cover - depends on optional extras
        return {}
    getter = getattr(strategy_class, "training_defaults", None)
    if not callable(getter):
        return {}
    defaults = getter()
    return dict(defaults) if isinstance(defaults, Mapping) else {}


def _global_training_block(app_config: Any) -> Dict[str, Any]:
    """The global `training:` block as a plain dict, or empty if absent."""
    training = getattr(app_config, "training", None)
    if training is None:
        return {}
    if hasattr(training, "model_dump"):
        return dict(training.model_dump())
    if isinstance(training, Mapping):
        return dict(training)
    return {}


def resolve_training_config(
    app_config: Any,
    strategy: Optional[str] = None,
    *,
    trainer: Optional[str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    strategy_config_file: Optional[Path | str] = None,
) -> Tuple[str, Type[BaseTrainer], TrainerConfig]:
    """Resolve the trainer and its validated hyperparameters for one run.

    Args:
        app_config: The loaded AppConfig, supplying the global `training:`
            block and the data settings.
        strategy: Strategy being trained. None resolves to the supervised
            default, which is what plain `portfolio-agent train` does.
        trainer: Explicit trainer name, beating every other source.
        overrides: Highest-precedence key/value pairs (from `--set`, or a
            notebook call).
        strategy_config_file: Explicit strategy YAML, overriding the
            `config/strategies/<strategy>.yaml` convention.

    Returns:
        `(trainer_name, trainer_class, validated_config)`.

    Raises:
        KeyError: The named trainer is not registered.
        ValueError: A hyperparameter is unknown to the trainer or fails
            validation. The message names the key, because the whole point of
            per-trainer schemas is that a knob aimed at the wrong trainer
            stops the run instead of being silently dropped.
    """
    strategy_block = (
        load_strategy_training_block(strategy, strategy_config_file) if strategy else {}
    )

    trainer_name = resolve_trainer_name(
        strategy, explicit=trainer, strategy_block=strategy_block
    )
    trainer_class = get_trainer(trainer_name)
    schema = trainer_class.config_model()
    known_fields = set(schema.model_fields)

    # A strategy's YAML is written for the trainer it declares. When the caller
    # overrides that trainer, the block describes a different procedure's knobs,
    # and merging it wholesale would fail validation on every one of them —
    # turning "train india_sac the supervised way, just to compare" into a wall
    # of errors. In that case the block is filtered to what the target trainer
    # understands and the rest is reported as dropped.
    #
    # When the trainer is the one the strategy asked for, an unrecognized key is
    # a genuine mistake and must stop the run. That asymmetry is the point.
    declared = strategy_block.get("trainer")
    trainer_was_overridden = bool(declared) and str(declared) != trainer_name

    # Weakest layer first. `trainer` is metadata about which schema to use, not
    # a field within it, so it never reaches validation.
    merged: Dict[str, Any] = {}

    global_block = _global_training_block(app_config)
    merged.update({k: v for k, v in global_block.items() if k in known_fields})

    if strategy:
        class_defaults = _strategy_class_defaults(strategy)
        # Always filtered: a strategy's preferred defaults are advice, and
        # advice aimed at another trainer is simply not applicable.
        merged.update({k: v for k, v in class_defaults.items() if k in known_fields})

    yaml_settings = {k: v for k, v in strategy_block.items() if k != "trainer"}
    if trainer_was_overridden:
        dropped = sorted(set(yaml_settings) - known_fields)
        if dropped:
            logger.info(
                "Trainer overridden to %r for strategy %r, which declares %r. "
                "Ignoring %d setting(s) from its YAML that %r does not accept: %s",
                trainer_name, strategy, declared, len(dropped), trainer_name, dropped,
            )
        yaml_settings = {k: v for k, v in yaml_settings.items() if k in known_fields}
    merged.update(yaml_settings)

    merged.update(dict(overrides or {}))

    try:
        cfg = schema(**merged)
    except ValidationError as exc:
        raise ValueError(
            _explain_validation_error(exc, trainer_name, sorted(known_fields))
        ) from exc

    logger.debug("Resolved trainer=%s config=%s", trainer_name, cfg.model_dump())
    return trainer_name, trainer_class, cfg


def _explain_validation_error(
    exc: ValidationError, trainer_name: str, known_fields: List[str]
) -> str:
    """Turn a pydantic error into a message that says what to do about it."""
    lines = [f"Invalid training config for trainer {trainer_name!r}:"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(root)"
        if error["type"] == "extra_forbidden":
            lines.append(
                f"  - {location}: not a setting this trainer accepts. "
                f"Known settings: {known_fields}"
            )
        else:
            lines.append(f"  - {location}: {error['msg']}")
    return "\n".join(lines)
