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

    def entry_rules(self) -> Dict[str, Any]:
        """Optional: return the entry rules for this strategy (for reporting/introspection)."""
        return {}

    def exit_rules(self) -> Dict[str, Any]:
        """Optional: return the exit rules for this strategy (for reporting/introspection)."""
        return {}
