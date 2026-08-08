"""Model registry for portfolio forecasting models."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type

import torch.nn as nn


# Global registry of model classes
_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str) -> Callable:
    """Decorator to register a model class.
    
    Args:
        name: Name for the model in the registry.
        
    Returns:
        Decorator function.
        
    Example:
        @register_model('lstm')
        class LSTMForecaster(nn.Module):
            ...
    """
    def decorator(model_class: Type[nn.Module]) -> Type[nn.Module]:
        _MODEL_REGISTRY[name] = model_class
        return model_class
    return decorator


def get_model(name: str) -> Type[nn.Module]:
    """Get a model class by name from the registry.
    
    Args:
        name: Name of the model to retrieve.
        
    Returns:
        The model class if registered.
        
    Raises:
        KeyError: If model name is not found in registry.
    """
    if name not in _MODEL_REGISTRY:
        available_models = list(_MODEL_REGISTRY.keys())
        raise KeyError(
            f"Model '{name}' not found in registry. "
            f"Available models: {available_models}"
        )
    return _MODEL_REGISTRY[name]


def list_models() -> list[str]:
    """List all registered model names.
    
    Returns:
        List of registered model names.
    """
    return list(_MODEL_REGISTRY.keys())


def is_model_registered(name: str) -> bool:
    """Check if a model is registered.
    
    Args:
        name: Name of the model to check.
        
    Returns:
        True if model is registered, False otherwise.
    """
    return name in _MODEL_REGISTRY
