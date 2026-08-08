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
    from src.backtest_engine import BacktestEngine
    from src.data_store import DataStore, batch_download_and_cache
    from src.universe import resolve_backtest_universe
    from src.risk_analytics import RiskAnalyzer
    from src.backtest_reporting import export_backtest_excel
except ImportError:
    from config import get_config
    from orchestrator import run_orchestrator
    from storage import init_db, save_brain, load_brain, get_trade_history, get_open_trades
    from learning import evaluate_and_learn
    from outcomes import update_outcomes_from_market
    from backtest_engine import BacktestEngine
    from data_store import DataStore, batch_download_and_cache
    from universe import resolve_backtest_universe
    from risk_analytics import RiskAnalyzer
    from backtest_reporting import export_backtest_excel


def run_daily_agent_job(**context) -> Dict[str, Any]:
    """Run the daily agent job in paper trading mode.

    Args:
        **context: Airflow context dictionary (unused but accepted for compatibility).

    Returns:
        Dictionary with status, job name, and excel_path.

    Raises:
        RuntimeError: If paper_trading_mode is false.
    """
    logger.info("Starting daily agent job")

    # Load config
    config = get_config()

    # Enforce paper trading
    if not config.paper_trading_mode:
        raise RuntimeError("Live trading is disabled. Set paper_trading_mode=true.")

    # Run orchestrator
    excel_path = run_orchestrator(force_refresh=False, config=config)

    logger.info(f"Daily agent job completed successfully. Excel report: {excel_path}")

    return {
        "status": "success",
        "job": "DAILY_AGENT",
        "excel_path": excel_path
    }


def run_backtest_job(**context) -> Dict[str, Any]:
    """Run a small backtest job with safe default parameters.

    Args:
        **context: Airflow context dictionary (unused but accepted for compatibility).

    Returns:
        Dictionary with status, job name, and backtest Excel path.

    Defaults:
        years: 1
        universe_size: 20
        force_download: False
    """
    from datetime import timedelta
    from pathlib import Path
    
    logger.info("Starting backtest job with safe defaults")

    # Load config
    config = get_config()

    # Safe default parameters
    years = 1
    universe_size = 20
    force_download = False
    initial_capital = config.initial_capital_inr if hasattr(config, 'initial_capital_inr') else 1000000
    output_file = "output/backtest_small_report.xlsx"

    # Calculate date range
    from datetime import datetime
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    logger.info(f"Backtest period: {start_date_str} to {end_date_str}")
    logger.info(f"Universe size: {universe_size}, Initial capital: {initial_capital}")

    # Fetch universe
    tickers = resolve_backtest_universe(
        force_full_download=force_download,
        max_tickers=universe_size
    )

    if not tickers:
        raise RuntimeError("No tickers with available data. Run download first.")

    logger.info(f"Resolved {len(tickers)} tickers")

    # Download/cache data
    data_store = DataStore()
    batch_download_and_cache(
        tickers=tickers,
        start_date=start_date_str,
        end_date=end_date_str,
        chunk_size=50,
        skip_existing=True
    )

    # Initialize and run backtest engine
    engine = BacktestEngine(
        start_date=start_date_str,
        end_date=end_date_str,
        initial_capital=initial_capital,
        universe_tickers=tickers
    )

    # Run simulation
    engine.run_backtest()

    # Calculate analytics
    analyzer = RiskAnalyzer(
        daily_equity_curve=engine.daily_equity_curve,
        trade_log=engine.trade_log
    )
    analytics_report = analyzer.generate_analytics_report()

    # Prepare analytics for export
    analytics_for_export = {
        'cagr': analytics_report.get('cagr', 0),
        'sharpe': analytics_report.get('sharpe_ratio', 0),
        'sortino': analytics_report.get('sortino_ratio', 0),
        'max_drawdown': analytics_report.get('max_drawdown_pct', 0),
        'profit_factor': analytics_report.get('profit_factor', 0),
        'probability_of_ruin': analytics_report.get('mc_probability_of_ruin_pct', 0),
        'total_return': analytics_report.get('total_return_pct', 0),
        'volatility': analytics_report.get('annualized_volatility_pct', 0),
        'win_rate': analytics_report.get('win_rate_pct', 0),
        'total_trades': analytics_report.get('total_trades', 0),
        'final_portfolio_value': analytics_report.get('final_capital', 0),
        'initial_capital': initial_capital,
        'monte_carlo_results': {
            'percentile_5': analytics_report.get('mc_percentile_5', 0),
            'percentile_50': analytics_report.get('mc_median_terminal_wealth', 0),
            'percentile_95': analytics_report.get('mc_percentile_95', 0)
        }
    }

    # Export Excel report
    export_backtest_excel(
        analytics=analytics_for_export,
        equity_curve=engine.daily_equity_curve,
        trade_log=engine.trade_log,
        brain_evolution=engine.brain_evolution,
        daily_activity_log=engine.daily_activity_log,
        filepath=output_file
    )

    logger.info(f"Backtest job completed. Report: {output_file}")

    return {
        "status": "success",
        "job": "BACKTEST_SMALL",
        "excel_path": output_file,
        "cagr": analytics_report.get('cagr_pct', 0),
        "sharpe": analytics_report.get('sharpe_ratio', 0),
        "total_trades": analytics_report.get('total_trades', 0)
    }


def run_full_backtest_job(**context) -> Dict[str, Any]:
    """Run a full 5-year backtest with all available tickers.

    Args:
        **context: Airflow context dictionary (unused but accepted for compatibility).

    Returns:
        Dictionary with status, job name, and backtest Excel path.

    Parameters:
        years: 5
        universe_size: None (all available tickers)
        force_download: False
    """
    from datetime import timedelta
    from pathlib import Path
    
    logger.info("Starting full backtest job (5 years, all tickers)")

    # Load config
    config = get_config()

    # Full backtest parameters
    years = 5
    universe_size = None  # All available tickers
    force_download = False
    initial_capital = config.initial_capital_inr if hasattr(config, 'initial_capital_inr') else 1000000
    output_file = "output/backtest_full_report.xlsx"

    # Calculate date range
    from datetime import datetime
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    logger.info(f"Backtest period: {start_date_str} to {end_date_str}")
    logger.info(f"Universe size: ALL, Initial capital: {initial_capital}")

    # Fetch universe (all available)
    tickers = resolve_backtest_universe(
        force_full_download=force_download,
        max_tickers=universe_size
    )

    if not tickers:
        raise RuntimeError("No tickers with available data. Run download first.")

    logger.info(f"Resolved {len(tickers)} tickers")

    # Download/cache data
    data_store = DataStore()
    batch_download_and_cache(
        tickers=tickers,
        start_date=start_date_str,
        end_date=end_date_str,
        chunk_size=50,
        skip_existing=True
    )

    # Initialize and run backtest engine
    engine = BacktestEngine(
        start_date=start_date_str,
        end_date=end_date_str,
        initial_capital=initial_capital,
        universe_tickers=tickers
    )

    # Run simulation
    engine.run_backtest()

    # Calculate analytics
    analyzer = RiskAnalyzer(
        daily_equity_curve=engine.daily_equity_curve,
        trade_log=engine.trade_log
    )
    analytics_report = analyzer.generate_analytics_report()

    # Prepare analytics for export
    analytics_for_export = {
        'cagr': analytics_report.get('cagr', 0),
        'sharpe': analytics_report.get('sharpe_ratio', 0),
        'sortino': analytics_report.get('sortino_ratio', 0),
        'max_drawdown': analytics_report.get('max_drawdown_pct', 0),
        'profit_factor': analytics_report.get('profit_factor', 0),
        'probability_of_ruin': analytics_report.get('mc_probability_of_ruin_pct', 0),
        'total_return': analytics_report.get('total_return_pct', 0),
        'volatility': analytics_report.get('annualized_volatility_pct', 0),
        'win_rate': analytics_report.get('win_rate_pct', 0),
        'total_trades': analytics_report.get('total_trades', 0),
        'final_portfolio_value': analytics_report.get('final_capital', 0),
        'initial_capital': initial_capital,
        'monte_carlo_results': {
            'percentile_5': analytics_report.get('mc_percentile_5', 0),
            'percentile_50': analytics_report.get('mc_median_terminal_wealth', 0),
            'percentile_95': analytics_report.get('mc_percentile_95', 0)
        }
    }

    # Export Excel report
    export_backtest_excel(
        analytics=analytics_for_export,
        equity_curve=engine.daily_equity_curve,
        trade_log=engine.trade_log,
        brain_evolution=engine.brain_evolution,
        daily_activity_log=engine.daily_activity_log,
        filepath=output_file
    )

    logger.info(f"Full backtest job completed. Report: {output_file}")

    return {
        "status": "success",
        "job": "BACKTEST_FULL",
        "excel_path": output_file,
        "cagr": analytics_report.get('cagr_pct', 0),
        "sharpe": analytics_report.get('sharpe_ratio', 0),
        "total_trades": analytics_report.get('total_trades', 0)
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
