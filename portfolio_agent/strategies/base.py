"""Base strategy module for portfolio agent.

All trading strategies (rule-based, ML, or future additions) implement this
single interface so they can be called identically from the live orchestrator
and the backtest engine — eliminating the historical drift between them.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd

from .types import StrategyContext, StrategySignal


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the strategy."""

    @abstractmethod
    def required_features(self) -> List[str]:
        """Feature names (from features/registry.py) this strategy needs.

        The caller builds these once via features/pipeline.py::build_features()
        and passes the resulting DataFrame into score()/score_batch().
        """

    @abstractmethod
    def score(self, symbol: str, features: pd.DataFrame, context: StrategyContext) -> StrategySignal:
        """Score the latest available (lag-safe) row of features for one ticker."""

    def score_batch(
        self, features_by_symbol: Dict[str, pd.DataFrame], context: StrategyContext
    ) -> Dict[str, StrategySignal]:
        """Score many tickers at once.

        Default implementation is a plain CPU loop over score(). Strategies that
        can batch inference (e.g. an ML strategy doing one stacked GPU forward
        pass) should override this.
        """
        return {symbol: self.score(symbol, df, context) for symbol, df in features_by_symbol.items()}

    @property
    def supports_gpu_batch(self) -> bool:
        """Whether score_batch() does a real batched GPU forward pass.

        Used by callers (e.g. the backtest engine) to decide whether to build one
        stacked batch call instead of dispatching per-ticker work across a
        CPU process pool.
        """
        return False

    @property
    def requires_full_batch(self) -> bool:
        """Whether this strategy's signals depend on the full eligible universe
        at once (e.g. cross-sectional ranking), not just each ticker's own history.

        Callers must invoke score_batch() with every eligible ticker in one
        call for such strategies — scoring one ticker at a time (even in a
        loop) is semantically wrong, since ranking degenerates to a universe
        of one. Distinct from supports_gpu_batch, which is about batching for
        performance rather than correctness.
        """
        return False

    def entry_rules(self) -> Dict[str, Any]:
        """Optional: return the entry rules for this strategy (for reporting/introspection)."""
        return {}

    def exit_rules(self) -> Dict[str, Any]:
        """Optional: return the exit rules for this strategy (for reporting/introspection)."""
        return {}


class TrainableStrategy(BaseStrategy):
    """Optional mixin for strategies that require a training phase.
    
    Custom strategies should inherit from BaseStrategy AND implement this
    interface if they require a training phase (e.g., ML models, RL policies).
    
    This provides a standard interface for:
    - Declaring what model artifact(s) the strategy uses
    - Running the training loop
    - Loading trained weights from disk
    
    Example usage:
    
        class MyMLStrategy(TrainableStrategy):
            @property
            def model_artifact_name(self) -> str:
                return "my_strategy.pt"
            
            def train(self, data: dict, config: dict) -> dict:
                # Your training logic here
                return {"final_loss": 0.5}
            
            def load_model(self, artifact_path: str) -> None:
                # Load weights from artifact_path into your model
                pass
    """
    
    @property
    @abstractmethod
    def model_artifact_name(self) -> str:
        """Return the filename for the saved model artifact (e.g., 'my_strategy.pt').
        
        This is used by the generic trainer to save/load checkpoints.
        The file will be saved in the models/ directory.
        """
        pass

    @abstractmethod
    def train(self, data: dict, config: dict) -> dict:
        """Run the training loop.
        
        Args:
            data: Preprocessed market data containing:
                - 'features': Dict[str, pd.DataFrame] features by symbol
                - 'prices': Dict[str, pd.DataFrame] price data by symbol  
                - 'tickers': List[str] list of ticker symbols
                - Additional keys as needed by specific strategies
            config: Training hyperparameters including:
                - 'epochs': Number of training epochs
                - 'batch_size': Mini-batch size
                - 'lr': Learning rate
                - 'device': Device for training ('cpu', 'cuda', etc.)
                - Additional keys as needed
                
        Returns:
            dict: Training metrics including:
                - 'final_loss' or 'final_reward': Primary metric
                - 'epochs_trained': Number of epochs completed
                - Additional metrics for logging/checkpointing
        """
        pass

    @abstractmethod
    def load_model(self, artifact_path: str) -> None:
        """Load trained weights from disk into the strategy instance.
        
        Called automatically during backtesting/live trading when a checkpoint
        exists. Should raise FileNotFoundError if the checkpoint is missing.
        
        Args:
            artifact_path: Full path to the model file.
            
        Raises:
            FileNotFoundError: If the checkpoint file doesn't exist
            ValueError: If the checkpoint format is invalid
        """
        pass
    
    @classmethod
    def get_default_training_config(cls) -> dict:
        """Return default training configuration.
        
        Override this to provide sensible defaults for your strategy.
        These are used when --config is not provided to train-custom.
        
        Returns:
            dict: Default training parameters
        """
        return {
            "epochs": 100,
            "batch_size": 256,
            "lr": 3e-4,
            "device": "auto",
        }
