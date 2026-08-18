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

    def required_cross_sectional_features(self) -> List[str]:
        """Names from `features/cross_section.py` this strategy needs.

        Separate from `required_features` because the two registries have
        different shapes: one is `Series = f(one_ticker_ohlcv)`, the other
        `DataFrame(date x symbol) = f(panel)`. Keeping them apart means a
        caller never has to guess which registry a name belongs to, and
        `features/sets.py` and `pipeline.warmup_rows` keep operating on
        per-ticker names only.

        A strategy that declares any of these must implement `score_batch` and
        report `requires_full_batch`: a cross-sectional feature scored one
        ticker at a time degenerates to a universe of one, which is the failure
        `requires_full_batch` exists to prevent.

        Returns:
            Empty by default — most strategies rank on per-ticker features.
        """
        return []

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
    """A strategy whose weights come from a registered training procedure.

    This is a *declaration*, not a training interface: a subclass says which
    trainer produces its checkpoint and, optionally, what that trainer's
    defaults should be for it. The loop itself lives in
    `portfolio_agent/training/trainers/`.

    Splitting it this way is deliberate. Putting `train()` on the strategy
    couples what is being learned to how it is learned: two strategies could
    not share one procedure without inheriting from each other, and retraining
    a strategy a different way would mean editing the class that scores it.
    With the trainer named rather than embedded, `trainer_name` is a one-line
    change and the scoring path never moves.

    Loading stays on `load()` — the contract the backtest engine already calls
    (`agents/backtester.py`). A second loading method taking a path would be a
    third convention alongside `load()` and `MLStrategy.load_model(name)`, and
    nothing would call it.

    Example:

        class MyStrategy(TrainableStrategy):
            trainer_name = "sac"

            @classmethod
            def training_defaults(cls) -> Dict[str, Any]:
                return {"epochs": 300, "hidden_dim": 128}
    """

    #: Registry name of the trainer that produces this strategy's checkpoint
    #: (see portfolio_agent/training/registry.py). Subclasses must set it.
    trainer_name: str = ""

    @classmethod
    def training_defaults(cls) -> Dict[str, Any]:
        """Trainer settings this strategy prefers, overriding schema defaults.

        Sits below the strategy's YAML and any explicit override in precedence,
        so it is a default rather than a lock. Return an empty dict to accept
        the trainer's own defaults unchanged.
        """
        return {}
