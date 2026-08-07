"""Airflow-safe wrapper functions for portfolio_agent modules.

These functions provide thin wrappers that can be safely called from Airflow DAGs.
They enforce paper trading, use logging, and return JSON-serializable dictionaries.
"""

import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Module-level imports for easier mocking in tests
# These are imported here but only used inside functions to avoid Airflow import issues
try:
    from src.config import get_config
    from src.orchestrator import run_orchestrator
    from src.storage import init_db, save_brain, load_brain, get_trade_history, get_open_trades
    from src.learning import evaluate_and_learn
    from src.outcomes import update_outcomes_from_market
except ImportError:
    from config import get_config
    from orchestrator import run_orchestrator
    from storage import init_db, save_brain, load_brain, get_trade_history, get_open_trades
    from learning import evaluate_and_learn
    from outcomes import update_outcomes_from_market


def run_daily_job(**context) -> Dict[str, Any]:
    """Run the daily portfolio optimization job.

    Args:
        **context: Airflow context dictionary (unused but accepted for compatibility).

    Returns:
        Dictionary with status, job name, and excel_path.

    Raises:
        RuntimeError: If paper_trading_mode is false.
    """
    logger.info("Starting daily job")

    # Load config
    config = get_config()

    # Enforce paper trading
    if not config.paper_trading_mode:
        raise RuntimeError("Live trading is disabled. Set paper_trading_mode=true.")

    # Run orchestrator
    excel_path = run_orchestrator(force_refresh=False)

    logger.info(f"Daily job completed successfully. Excel report: {excel_path}")

    return {
        "status": "success",
        "job": "DAILY_RUN",
        "excel_path": excel_path
    }


def run_update_outcomes_job(**context) -> Dict[str, Any]:
    """Update trade outcomes from market data.

    Args:
        **context: Airflow context dictionary (unused but accepted for compatibility).

    Returns:
        Dictionary with status, job name, and updated_outcomes count or reason.

    Raises:
        RuntimeError: If paper_trading_mode is false.
    """
    logger.info("Starting update outcomes job")

    # Load config
    config = get_config()

    # Enforce paper trading
    if not config.paper_trading_mode:
        raise RuntimeError("Live trading is disabled. Set paper_trading_mode=true.")

    # Check if update_outcomes_from_market exists and is callable
    try:
        # Initialize DB
        init_db(config.sqlite_path)

        # Fetch market data for open trades
        open_trades = get_open_trades(config.sqlite_path)
        if not open_trades:
            logger.info("No open trades to update")
            return {
                "status": "success",
                "job": "UPDATE_OUTCOMES",
                "updated_outcomes": 0
            }

        # Get symbols for open trades
        symbols = list(set(trade.symbol for trade in open_trades))
        logger.info(f"Fetching market data for {len(symbols)} symbols")

        # Fetch data using yfinance directly for the specific symbols
        import yfinance as yf
        import pandas as pd
        data_dict = {}
        try:
            # Download data for open trade symbols
            yf_data = yf.download(symbols, period="1mo", group_by="ticker")
            
            if len(symbols) == 1:
                df = yf_data.copy()
                if not df.empty:
                    df.columns = [col.lower() for col in df.columns]
                    df = df.sort_index().dropna(subset=["close"])
                    if not df.empty:
                        data_dict[symbols[0]] = df
            else:
                for symbol in symbols:
                    try:
                        df = yf_data[symbol].copy()
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = [col[0].lower() for col in df.columns]
                        else:
                            df.columns = [col.lower() for col in df.columns]
                        df = df.sort_index().dropna(subset=["close"])
                        if not df.empty:
                            data_dict[symbol] = df
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"Error fetching market data: {e}")
            return {
                "status": "skipped",
                "job": "UPDATE_OUTCOMES",
                "reason": f"failed to fetch market data: {e}"
            }

        if not data_dict:
            logger.warning("No market data available for open trades")
            return {
                "status": "skipped",
                "job": "UPDATE_OUTCOMES",
                "reason": "no market data available"
            }

        # Update outcomes from market
        updated = update_outcomes_from_market(config.sqlite_path, data_dict)

        logger.info(f"Updated {len(updated)} trade outcomes from market data")

        return {
            "status": "success",
            "job": "UPDATE_OUTCOMES",
            "updated_outcomes": len(updated)
        }

    except AttributeError as e:
        logger.warning(f"Outcome update not implemented: {e}")
        return {
            "status": "skipped",
            "job": "UPDATE_OUTCOMES",
            "reason": "outcome update not implemented"
        }


def run_relearn_job(**context) -> Dict[str, Any]:
    """Run the relearning job to update agent brain weights.

    Args:
        **context: Airflow context dictionary (unused but accepted for compatibility).

    Returns:
        Dictionary with status, job name, and current brain weights.

    Raises:
        RuntimeError: If paper_trading_mode is false.
    """
    logger.info("Starting relearn job")

    # Load config
    config = get_config()

    # Enforce paper trading
    if not config.paper_trading_mode:
        raise RuntimeError("Live trading is disabled. Set paper_trading_mode=true.")

    # Initialize DB
    init_db(config.sqlite_path)

    # Load brain from config.brain_file
    brain = load_brain(config.brain_file)
    logger.info(f"Loaded brain from {config.brain_file}")

    # Load trade history from SQLite into brain.trade_history
    trade_outcomes = get_trade_history(config.sqlite_path)
    for outcome in trade_outcomes:
        brain.trade_history.append({
            "trade_id": outcome.trade_id,
            "symbol": outcome.symbol,
            "signal_trigger": outcome.signal_trigger,
            "entry_date": outcome.entry_date,
            "entry_price": outcome.entry_price,
            "exit_date": outcome.exit_date,
            "exit_price": outcome.exit_price,
            "outcome": outcome.outcome,
            "return_pct": outcome.return_pct,
            "outcome_source": outcome.outcome_source
        })
    logger.info(f"Loaded {len(trade_outcomes)} trade outcomes from SQLite")

    # Run evaluate_and_learn
    brain = evaluate_and_learn(brain, config)
    logger.info("Learning evaluation complete")

    # Save brain
    save_brain(config.brain_file, brain)
    logger.info(f"Saved brain to {config.brain_file}")

    # Log learning summary
    if brain.learning_log:
        last_entry = brain.learning_log[-1]
        if isinstance(last_entry, dict) and "entry" in last_entry:
            logger.info(f"Learning summary: {last_entry['entry']}")
        elif isinstance(last_entry, dict) and "message" in last_entry:
            logger.info(f"Learning summary: {last_entry['message']}")

    return {
        "status": "success",
        "job": "RELEARN",
        "weights": brain.weights
    }
