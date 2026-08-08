"""Feature registry for modular technical indicators."""

from typing import Callable, Dict, Optional, Any
import functools

# Global registry of features
_FEATURE_REGISTRY: Dict[str, Callable] = {}


def register_feature(name: Optional[str] = None) -> Callable:
    """Decorator to register a feature function.
    
    Args:
        name: Optional name for the feature. If not provided, uses function name.
    
    Returns:
        Decorator function.
    
    Example:
        @register_feature('sma_20')
        def calculate_sma_20(df):
            return df['close'].rolling(20).mean()
    """
    def decorator(func: Callable) -> Callable:
        feature_name = name if name is not None else func.__name__
        _FEATURE_REGISTRY[feature_name] = func
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def get_feature(name: str) -> Optional[Callable]:
    """Get a feature function by name from the registry.
    
    Args:
        name: Name of the feature to retrieve.
    
    Returns:
        The feature function if registered, None otherwise.
    
    Raises:
        KeyError: If feature name is not found in registry.
    """
    if name not in _FEATURE_REGISTRY:
        available_features = list(_FEATURE_REGISTRY.keys())
        raise KeyError(
            f"Feature '{name}' not found in registry. "
            f"Available features: {available_features}"
        )
    return _FEATURE_REGISTRY[name]


def list_features() -> list[str]:
    """List all registered feature names.
    
    Returns:
        List of registered feature names.
    """
    return list(_FEATURE_REGISTRY.keys())


def is_feature_registered(name: str) -> bool:
    """Check if a feature is registered.
    
    Args:
        name: Name of the feature to check.
    
    Returns:
        True if feature is registered, False otherwise.
    """
    return name in _FEATURE_REGISTRY
