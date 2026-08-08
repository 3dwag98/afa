"""Base strategy module for portfolio agent."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class for trading strategies.
    
    All strategy implementations must inherit from this class and implement
    the required methods.
    """
    
    @abstractmethod
    def generate_signals(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals based on feature data.
        
        Args:
            features: DataFrame containing feature data (technical indicators,
                     price data, volume, etc.)
        
        Returns:
            Dictionary containing signal information including:
            - 'signal': 'BUY', 'SELL', or 'HOLD'
            - 'entry_price': Suggested entry price
            - 'stop_price': Stop loss price
            - 'target_price': Target price
        """
        pass
    
    @abstractmethod
    def score(self, features: pd.DataFrame) -> float:
        """Calculate a score for the given feature set.
        
        Args:
            features: DataFrame containing feature data.
        
        Returns:
            Score between 0.0 and 100.0 (or 0 and 1 depending on implementation).
        """
        pass
    
    @abstractmethod
    def entry_rules(self) -> Dict[str, Any]:
        """Return the entry rules for this strategy.
        
        Returns:
            Dictionary describing entry conditions.
        """
        pass
    
    @abstractmethod
    def exit_rules(self) -> Dict[str, Any]:
        """Return the exit rules for this strategy.
        
        Returns:
            Dictionary describing exit conditions (stop loss, target).
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the strategy."""
        pass
