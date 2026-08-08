#!/usr/bin/env python3
"""
Backtest Runner CLI for Portfolio Agent.

Orchestrates universe fetching, 5-year simulation, and Excel reporting.
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
from src.backtest_engine import BacktestEngine
from src.risk_analytics import RiskAnalyzer
from src.backtest_reporting import export_backtest_excel


# Global flag for graceful shutdown
_shutdown_requested = False
_partial_results_saved = False


def signal_handler(signum, frame):
    """Handle keyboard interrupt gracefully."""
    global _shutdown_requested
    _shutdown_requested = True
    print("\nShutdown requested. Saving partial results...")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run backtest simulation for Indian equity market.",
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
        help="Use all available cached tickers (calls resolve_backtest_universe)"
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
        default="output/Backtest_5Year_Report.xlsx",
        help="Output Excel file path"
    )
    
    return parser.parse_args()


def run_backtest(
    years: int,
    initial_capital: float,
    universe_size: int,
    force_download: bool,
    output_file: str
) -> bool:
    """
    Run the complete backtest pipeline.
    
    Args:
        years: Number of years for simulation.
        initial_capital: Starting capital in INR.
        universe_size: Number of tickers to include (None = ALL).
        force_download: Force full universe download.
        output_file: Path for Excel report.
        
    Returns:
        True if completed successfully, False otherwise.
    """
    global _shutdown_requested, _partial_results_saved
    
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        print(f"Backtest Period: {start_date_str} to {end_date_str}")
        print(f"Initial Capital: ₹{initial_capital:,.2f}")
        print(f"Universe Size: {universe_size if universe_size else 'ALL'} tickers")
        print("-" * 50)
        
        # Step 1: Fetch Universe using resolve_backtest_universe
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
            start_date=start_date_str,
            end_date=end_date_str,
            chunk_size=50,
            skip_existing=True
        )
        
        if not success:
            print("  Warning: Some tickers failed to download")
        
        if _shutdown_requested:
            return False
        
        # Step 3: Initialize Backtest Engine
        print("Initializing Backtest Engine...")
        engine = BacktestEngine(
            start_date=start_date_str,
            end_date=end_date_str,
            initial_capital=initial_capital,
            universe_tickers=tickers
        )
        
        if _shutdown_requested:
            return False
        
        # Step 4: Run Simulation with progress bar
        print(f"Running {years}-Year Simulation (This may take a few minutes)...")
        
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            print("  Note: tqdm not installed, running without progress bar")
        
        # Access the master date index for progress tracking
        total_days = len(engine.master_date_index)
        
        if use_tqdm:
            with tqdm(total=total_days, desc="Simulating", unit="day") as pbar:
                # Run backtest manually with progress tracking
                equity_curve = {}
                
                for i, current_date in enumerate(engine.master_date_index):
                    if _shutdown_requested:
                        # Save partial results
                        engine.daily_equity_curve = pd.Series(equity_curve)
                        _save_partial_results(
                            engine, initial_capital, output_file, 
                            f"Partial Results (Interrupted at Day {i})"
                        )
                        return False
                    
                    engine.trading_day_count = i + 1
                    
                    # Step A: Mark-to-Market
                    engine._mark_to_market(current_date)
                    equity_curve[current_date] = engine.portfolio_value
                    
                    # Step B: Check stop-losses and take-profits
                    engine._check_stop_loss_take_profit(current_date)
                    
                    # Step C: Generate signals
                    signals = engine._generate_signals(current_date)
                    
                    # Step D: Create pending orders
                    engine._create_pending_orders(signals, current_date)
                    
                    # Execute pending orders
                    engine._execute_pending_orders(current_date)
                    
                    # Handle delisted tickers
                    engine._handle_delisted_tickers(current_date)
                    
                    # Step E: Learning every 20 days
                    if engine.trading_day_count % 20 == 0:
                        engine._evaluate_and_learn()
                    
                    pbar.update(1)
                
                # Final brain snapshot
                final_snapshot = {
                    'trading_day': engine.trading_day_count,
                    'weights': dict(engine.agent_brain.weights),
                    'trade_count': len(engine.agent_brain.trade_history)
                }
                engine.brain_evolution.append(final_snapshot)
                engine.daily_equity_curve = pd.Series(equity_curve)
        else:
            # Fallback without tqdm
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
        
        if _shutdown_requested:
            _save_partial_results(engine, initial_capital, output_file, "Partial Results")
            return False
        
        # Step 6: Generate Excel Report
        print("Generating Excel Report...")
        
        # Prepare analytics dict for export (map risk_analytics keys to reporting keys)
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
            'initial_capital': analytics_report.get('initial_capital', initial_capital),
            'monte_carlo_results': {
                'percentile_5': analytics_report.get('mc_percentile_5', 0),
                'percentile_50': analytics_report.get('mc_median_terminal_wealth', 0),
                'percentile_95': analytics_report.get('mc_percentile_95', 0)
            }
        }
        
        export_backtest_excel(
            analytics=analytics_for_export,
            equity_curve=engine.daily_equity_curve,
            trade_log=engine.trade_log,
            brain_evolution=engine.brain_evolution,
            filepath=output_file
        )
        
        # Print summary
        print("-" * 50)
        print("BACKTEST COMPLETE")
        print("-" * 50)
        print(f"CAGR: {analytics_report.get('cagr_pct', 0):.2f}%")
        print(f"Sharpe Ratio: {analytics_report.get('sharpe_ratio', 0):.3f}")
        print(f"Sortino Ratio: {analytics_report.get('sortino_ratio', 0):.3f}")
        print(f"Max Drawdown: {analytics_report.get('max_drawdown_pct', 0):.2f}%")
        print(f"Calmar Ratio: {analytics_report.get('calmar_ratio', 0):.3f}")
        print(f"Probability of Ruin: {analytics_report.get('mc_probability_of_ruin_pct', 0):.2f}%")
        print(f"Total Trades: {analytics_report.get('total_trades', 0)}")
        print(f"Final Portfolio Value: ₹{analytics_report.get('final_capital', 0):,.2f}")
        print(f"Report saved to: {output_file}")
        
        return True
        
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        if 'engine' in locals():
            _save_partial_results(engine, initial_capital, output_file, "Partial Results (Interrupted)")
        return False
    except Exception as e:
        print(f"Error during backtest: {e}")
        if 'engine' in locals():
            _save_partial_results(engine, initial_capital, output_file, f"Partial Results (Error: {e})")
        return False


def _save_partial_results(engine, initial_capital, output_file, title_suffix):
    """Save partial results when interrupted."""
    global _partial_results_saved
    
    if _partial_results_saved:
        return
    
    try:
        import pandas as pd
        
        # Generate basic analytics from available data
        if len(engine.daily_equity_curve) > 0:
            analyzer = RiskAnalyzer(
                daily_equity_curve=engine.daily_equity_curve,
                trade_log=engine.trade_log
            )
            analytics_report = analyzer.generate_analytics_report()
        else:
            analytics_report = {}
        
        # Prepare analytics dict for export
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
        
        # Modify output filename for partial results
        output_path = Path(output_file)
        partial_output = output_path.parent / f"{output_path.stem}_PARTIAL{output_path.suffix}"
        
        export_backtest_excel(
            analytics=analytics_for_export,
            equity_curve=engine.daily_equity_curve,
            trade_log=engine.trade_log,
            brain_evolution=engine.brain_evolution,
            filepath=str(partial_output)
        )
        
        print(f"Partial results saved to: {partial_output}")
        _partial_results_saved = True
        
    except Exception as e:
        print(f"Failed to save partial results: {e}")


def main():
    """Main entry point."""
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    args = parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    success = run_backtest(
        years=args.years,
        initial_capital=args.initial_capital,
        universe_size=args.universe_size,
        force_download=args.force_download,
        output_file=args.output_file
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
