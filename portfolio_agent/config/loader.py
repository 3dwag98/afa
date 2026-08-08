"""Configuration loader for the Autonomous Financial Advisor (AFA) portfolio agent.

Supports loading configuration from YAML files with environment variable overrides.
Environment variables use the prefix AFA_ and double underscore for nested keys.
Example: AFA_TRAINING__DEVICE="cuda" overrides training.device
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

from .schema import AppConfig


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
    # Resolve path relative to project root if not absolute
    config_path = Path(path)
    if not config_path.is_absolute():
        # Try relative to current directory first
        if not config_path.exists():
            # Try relative to workspace root
            workspace_root = Path(__file__).parent.parent.parent
            config_path = workspace_root / path
    
    # Load base configuration from YAML
    base_config: Dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            base_config = yaml.safe_load(f) or {}
    
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
