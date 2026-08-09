"""Self-learning module for agent adaptation.

The actual weight-adaptation math lives in strategies/weighting.py as pure
functions (no AppConfig/AgentBrain coupling) so it can be reused by any
rule-based strategy. This module is a thin wrapper that plugs that math into
the AgentBrain/AppConfig objects the orchestrator and backtest engine use.
"""

import logging
from datetime import datetime
from typing import Optional

from portfolio_agent.config.schema import AppConfig
from portfolio_agent.strategies.weighting import evaluate_and_learn as _evaluate_and_learn

# Use absolute imports for CLI execution
try:
    from .models import AgentBrain
    from .logging_utils import get_logger, ContextualLogger
except ImportError:
    from models import AgentBrain
    from logging_utils import get_logger, ContextualLogger


def _get_logger(run_id: Optional[str] = None, log_file: str = "logs/afa_pipeline.log") -> ContextualLogger:
    """Get a contextual logger for learning module."""
    return get_logger(
        module_name='learning',
        log_file=log_file,
        run_id=run_id,
        worker_id='main',
        level=logging.INFO
    )


def evaluate_and_learn(brain: AgentBrain, config: AppConfig, run_id: Optional[str] = None) -> AgentBrain:
    """Update agent signal weights based on trade outcomes.

    Args:
        brain: AgentBrain with trade_history and current weights.
        config: AppConfig with learning.learning_rate and learning.min_trades_for_learning.
        run_id: Unique run identifier for logging context.

    Returns:
        Updated AgentBrain with new weights and a learning_log entry.
    """
    logger = _get_logger(run_id=run_id, log_file=config.paths.log_file)
    logger.info("Starting learning evaluation")

    try:
        new_weights, message = _evaluate_and_learn(
            weights=brain.weights,
            trade_history=brain.trade_history,
            learning_rate=config.learning.learning_rate,
            min_trades_for_learning=config.learning.min_trades_for_learning,
        )
        brain.weights = new_weights

        if message is not None:
            brain.learning_log.append({"timestamp": datetime.now().isoformat(), "entry": message})
            logger.info(f"Learning complete: {message}")
        else:
            logger.info("Skipping learning: insufficient realized trades")

        brain.updated_at = datetime.now().isoformat()
        return brain
    except Exception as e:
        logger.exception(f"Error during learning evaluation: {e}")
        raise
