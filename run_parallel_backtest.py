#!/usr/bin/env python3
"""
Parallel PIT Backtest Runner CLI.

Runs a fast, parallelized Point-In-Time backtest with learning,
using the same methods as run_orchestrator and training the agent
with 5 years of historical data.

Usage:
    python run_parallel_backtest.py --years 5 --universe-size 100 --workers -1
    python run_parallel_backtest.py --use-processes --workers 4
"""

import argparse
import sys
import signal
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

# Add workspace to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.universe import resolve_backtest_universe
from src.data_store import DataStore, batch_download_and_cache
from src.backtest_parallel import ParallelBacktestEngine, run_parallel_pit_backtest
from src.risk_analytics import RiskAnalyzer
from src.backtest_reporting import export_backtest_excel


# Global flag for graceful shutdown
_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle keyboard interrupt gracefully."""
    global _shutdown_requested
    _shutdown_requested = True
    print("\nShutdown requested. Saving partial results...")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run parallel PIT backtest simulation for Indian equity market.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Number of years for backtest simulation"
    )
    
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=1000000,
        help="Initial capital in INR"
    )
    
    parser.add_argument(
        "--universe-size",
        type=int,
        default=None,
        help="Number of tickers to include (None means ALL cached tickers)"
    )
    
    parser.add_argument(
        "--use-all-available",
        action="store_true",
        default=True,
        help="Use all available cached tickers"
    )
    
    parser.add_argument(
        "--force-download",
        action="store_true",
        default=False,
        help="Force full universe download from master list"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default="output/Parallel_Backtest_5Year_Report.xlsx",
        help="Output Excel file path"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="Number of parallel workers (-1 = auto-detect CPU count)"
    )
    
    parser.add_argument(
        "--use-processes",
        action="store_true",
        default=False,
        help="Use ProcessPoolExecutor instead of ThreadPoolExecutor (better for CPU-bound tasks)"
    )
    
    parser.add_argument(
        "--learning-interval",
        type=int,
        default=20,
        help="Trading days between learning updates"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). If not provided, calculates from --years"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Default: today"
    )
    
    return parser.parse_args()


def run_parallel_backtest(
    years: int,
    initial_capital: float,
    universe_size: int,
    force_download: bool,
    output_file: str,
    workers: int,
    use_processes: bool,
    learning_interval: int,
    start_date: str = None,
    end_date: str = None
) -> bool:
    """
    Run the complete parallel backtest pipeline.
    
    Args:
        years: Number of years for simulation.
        initial_capital: Starting capital in INR.
        universe_size: Number of tickers to include (None = ALL).
        force_download: Force full universe download.
        output_file: Path for Excel report.
        workers: Number of parallel workers.
        use_processes: Use processes instead of threads.
        learning_interval: Days between learning updates.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        
    Returns:
        True if completed successfully, False otherwise.
    """
    global _shutdown_requested
    
    try:
        # Calculate date range if not provided
        if end_date is None:
            end_date_obj = datetime.now()
            end_date = end_date_obj.strftime("%Y-%m-%d")
        else:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        
        if start_date is None:
            start_date_obj = end_date_obj - timedelta(days=years * 365)
            start_date = start_date_obj.strftime("%Y-%m-%d")
        
        print(f"Parallel Backtest Period: {start_date} to {end_date}")
        print(f"Initial Capital: ₹{initial_capital:,.2f}")
        print(f"Universe Size: {universe_size if universe_size else 'ALL'} tickers")
        print(f"Workers: {workers if workers != -1 else 'auto'}")
        print(f"Process-based: {use_processes}")
        print(f"Learning Interval: {learning_interval} trading days")
        print("-" * 60)
        
        # Step 1: Fetch Universe
        print("Fetching Universe...")
        tickers = resolve_backtest_universe(
            force_full_download=force_download,
            max_tickers=universe_size
        )
        
        if not tickers:
            raise SystemExit("No tickers with available data. Run with --force-download first.")
        
        print(f"  Resolved {len(tickers)} tickers with available data")
        
        if _shutdown_requested:
            return False
        
        # Step 2: Download/Cache Data
        print("Downloading/Caching Data...")
        data_store = DataStore()
        
        success = batch_download_and_cache(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            chunk_size=50,
            skip_existing=True
        )
        
        if not success:
            print("  Warning: Some tickers failed to download")
        
        if _shutdown_requested:
            return False
        
        # Step 3: Initialize Parallel Backtest Engine
        print("Initializing Parallel Backtest Engine...")
        engine = ParallelBacktestEngine(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            universe_tickers=tickers,
            num_workers=workers,
            use_processes=use_processes,
            learning_interval=learning_interval
        )
        
        if _shutdown_requested:
            return False
        
        # Step 4: Run Simulation
        print(f"Running {years}-Year Parallel Simulation...")
        print(f"  Trading days: {len(engine.master_date_index)}")
        print(f"  Tickers loaded: {len(engine.ticker_data)}")
        print(f"  Workers: {engine.num_workers}")
        print()
        
        result = engine.run_backtest()
        
        if _shutdown_requested:
            return False
        
        # Step 5: Calculate Advanced Risk Analytics
        print("Calculating Advanced Risk Analytics...")
        analyzer = RiskAnalyzer(
            daily_equity_curve=engine.daily_equity_curve,
            trade_log=engine.trade_log
        )
        
        analytics_report = analyzer.generate_analytics_report()
        
        # Step 6: Generate Excel Report
        print("Generating Excel Report...")
        
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
            },
            'parallel_execution': True,
            'num_workers': engine.num_workers,
            'learning_interval': learning_interval,
            'brain_evolution_snapshots': len(engine.brain_evolution)
        }
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        export_backtest_excel(
            analytics=analytics_for_export,
            equity_curve=engine.daily_equity_curve,
            trade_log=engine.trade_log,
            brain_evolution=engine.brain_evolution,
            daily_activity_log=engine.daily_activity_log,
            filepath=output_file
        )
        
        # Print summary
        print("-" * 60)
        print("PARALLEL BACKTEST COMPLETE")
        print("-" * 60)
        print(f"CAGR: {analytics_report.get('cagr_pct', 0):.2f}%")
        print(f"Sharpe Ratio: {analytics_report.get('sharpe_ratio', 0):.3f}")
        print(f"Sortino Ratio: {analytics_report.get('sortino_ratio', 0):.3f}")
        print(f"Max Drawdown: {analytics_report.get('max_drawdown_pct', 0):.2f}%")
        print(f"Calmar Ratio: {analytics_report.get('calmar_ratio', 0):.3f}")
        print(f"Probability of Ruin: {analytics_report.get('mc_probability_of_ruin_pct', 0):.2f}%")
        print(f"Total Trades: {analytics_report.get('total_trades', 0)}")
        print(f"Final Portfolio Value: ₹{analytics_report.get('final_capital', 0):,.2f}")
        print(f"Report saved to: {output_file}")
        print()
        print(f"Brain Evolution: {len(engine.brain_evolution)} snapshots recorded")
        print(f"Learning Updates: Every {learning_interval} trading days")
        
        return True
        
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return False
    except Exception as e:
        print(f"Error during backtest: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    args = parse_args()
    
    success = run_parallel_backtest(
        years=args.years,
        initial_capital=args.initial_capital,
        universe_size=args.universe_size,
        force_download=args.force_download,
        output_file=args.output_file,
        workers=args.workers,
        use_processes=args.use_processes,
        learning_interval=args.learning_interval,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
