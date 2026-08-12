"""Configuration loader for the Autonomous Financial Advisor (AFA) portfolio agent.

Supports loading configuration from YAML files with environment variable overrides.
Environment variables use the prefix AFA_ and double underscore for nested keys.
Example: AFA_TRAINING__DEVICE="cuda" overrides training.device
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .schema import AppConfig

logger = logging.getLogger(__name__)

#: Shipped inside the wheel so an installed copy has a real configuration
#: rather than silently falling back to schema defaults. Kept as a copy of the
#: project's `config.yaml` rather than a stripped-down file, so what an
#: installed package does matches what the repository does.
PACKAGED_DEFAULT = Path(__file__).resolve().parent / "default_config.yaml"


def resolve_config_path(path: str = "config.yaml") -> Optional[Path]:
    """Find the configuration file to load, or None if there is none.

    Search order, most specific first:

    1. The path as given, relative to the working directory — so `--config`
       and a project-local `config.yaml` both work.
    2. The same path relative to the project root, which is what makes running
       from a subdirectory of a checkout behave.
    3. The packaged default, which is the only one an installed copy has.

    Returned rather than logged-and-loaded so callers can report *which* file a
    run used. "Which config am I on" was previously unanswerable, and that is
    what let an install run on settings nobody chose.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None

    if candidate.exists():
        return candidate

    project_root = Path(__file__).resolve().parent.parent.parent
    from_root = project_root / path
    if from_root.exists():
        return from_root

    if PACKAGED_DEFAULT.exists():
        return PACKAGED_DEFAULT

    return None


def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "__") -> Dict[str, Any]:
    """Flatten a nested dictionary with a separator for keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _unflatten_dict(flat: Dict[str, Any], sep: str = "__") -> Dict[str, Any]:
    """Unflatten a dictionary with separated keys back to nested structure."""
    result: Dict[str, Any] = {}
    for key, value in flat.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def _convert_value(value: str) -> Any:
    """Convert string value to appropriate Python type."""
    # Handle boolean
    if value.lower() in ("true", "yes", "on"):
        return True
    if value.lower() in ("false", "no", "off"):
        return False
    
    # Handle None/null
    if value.lower() in ("none", "null", ""):
        return None
    
    # Handle integers
    try:
        return int(value)
    except ValueError:
        pass
    
    # Handle floats
    try:
        return float(value)
    except ValueError:
        pass
    
    # Return as string
    return value


def load_config(path: str = "config.yaml") -> AppConfig:
    """Load configuration from a YAML file with environment variable overrides.
    
    Args:
        path: Path to the YAML configuration file. Defaults to "config.yaml".
        
    Returns:
        AppConfig: Validated configuration object.
        
    Environment Variable Overrides:
        Environment variables use the prefix AFA_ and double underscore (__) 
        to denote nested keys. For example:
            - AFA_TRAINING__DEVICE="cuda" overrides training.device
            - AFA_RISK__PORTFOLIO_VALUE_INR=500000 overrides risk.portfolio_value_inr
            - AFA_DATA__DATA_DIR="/custom/path" overrides data.data_dir
    """
    config_path = resolve_config_path(path)

    # Load base configuration from YAML
    base_config: Dict[str, Any] = {}
    if config_path is not None:
        with open(config_path, "r") as f:
            base_config = yaml.safe_load(f) or {}
        logger.info("Loaded configuration from %s", config_path)
    else:
        # Every value falls back to its schema default. Previously this was
        # silent, and the result was a run that looked entirely normal while
        # using settings nobody chose — an installed copy resolved
        # universe_size to the schema's 10 rather than the project's 4000, and
        # still produced results, charts and a report.
        logger.warning(
            "No configuration file found (looked for %r relative to the working "
            "directory, the project root, and the packaged default). Every setting "
            "is falling back to its schema default, which is almost certainly not "
            "what you want — pass --config PATH to select one explicitly.",
            path,
        )

    # Get environment variable overrides
    env_prefix = "AFA_"
    env_overrides: Dict[str, Any] = {}
    
    for env_key, env_value in os.environ.items():
        if env_key.startswith(env_prefix):
            # Remove prefix and convert to lowercase with double underscores
            nested_key = env_key[len(env_prefix):].lower()
            env_overrides[nested_key] = _convert_value(env_value)
    
    # Merge configurations: env overrides take precedence
    # First flatten the base config
    flat_base = _flatten_dict(base_config)
    flat_env = _flatten_dict(env_overrides)
    
    # Merge with env taking precedence
    merged_flat = {**flat_base, **flat_env}
    
    # Unflatten back to nested structure
    merged_config = _unflatten_dict(merged_flat)
    
    # Validate and return AppConfig
    return AppConfig.model_validate(merged_config)
