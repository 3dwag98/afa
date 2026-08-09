#!/usr/bin/env python3
"""CLI for Portfolio Agent - Autonomous Financial Advisor.

Commands:
    download-data: Download market data for the configured universe
    train: Train the ML model on historical data
    backtest: Run backtesting simulation
    run-agent: Run the daily portfolio agent
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def get_config() -> "AppConfig":
    """Load application configuration."""
    from portfolio_agent.config.loader import load_config
    return load_config()


def cmd_download_data(args) -> int:
    """Download market data command."""
    import pandas as pd
    from portfolio_agent.src.universe import resolve_backtest_universe
    from portfolio_agent.src.data_store import batch_download_and_cache
    
    config = get_config()
    
    # Resolve universe
    tickers = resolve_backtest_universe(
        force_full_download=args.force,
        max_tickers=args.universe_size or config.data.universe_size
    )
    
    if not tickers:
        print("Error: No tickers found with available data")
        return 1
    
    print(f"Resolved {len(tickers)} tickers")
    
    # Calculate date range
    from datetime import timedelta
    end_date = pd.Timestamp.now()
    start_date = end_date - timedelta(days=config.data.default_history_years * 365)
    
    # Download and cache
    success = batch_download_and_cache(
        tickers=tickers,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        chunk_size=50,
        skip_existing=not args.force
    )
    
    if success:
        print(f"Successfully downloaded data for {len(tickers)} tickers")
        return 0
    else:
        print("Warning: Some tickers failed to download")
        return 0


def cmd_train(args) -> int:
    """Train model command."""
    import torch
    from portfolio_agent.agents.trainer import run_training
    from portfolio_agent.utils.device import get_device
    
    config = get_config()
    
    # Override device if specified
    if args.device:
        config.training.device = args.device
    
    # Validate device
    device = get_device(config.training.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        config.training.device = "cpu"
    
    print(f"Starting training with device: {device}")
    
    try:
        metadata = run_training(config)
        print("\nTraining complete!")
        print(f"  Epochs trained: {metadata.get('epochs_trained', 0)}")
        print(f"  Best validation loss: {metadata.get('best_val_loss', 'N/A')}")
        return 0
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_backtest(args) -> int:
    """Run backtest command."""
    from portfolio_agent.agents.backtester import run_backtest_cli

    config = get_config()

    # Resolve strategy: --use-trained-model is a shorthand for --strategy lstm
    strategy_type = args.strategy
    if args.use_trained_model and strategy_type is None:
        strategy_type = "lstm"

    # Determine inference device
    if args.device:
        inference_device = args.device
    elif strategy_type == "lstm":
        # Use CPU for inference by default to save VRAM; pass --device cuda to override
        inference_device = "cpu"
        print(f"Using {inference_device} for model inference (pass --device cuda to override)")
    else:
        inference_device = "cpu"

    # Resolve date range
    start_date = args.start_date
    end_date = args.end_date
    if args.years and start_date is None:
        import pandas as pd
        end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
        start_date = (end - pd.Timedelta(days=args.years * 365)).strftime('%Y-%m-%d')

    output_file = args.output or config.paths.backtest_excel_output

    print(f"Running backtest...")
    print(f"  Strategy: {strategy_type or config.strategy.type}")
    print(f"  Inference device: {inference_device}")
    print(f"  Parallel: {args.parallel} (workers={args.workers or 'auto'})")
    print(f"  Output file: {output_file}")

    try:
        result = run_backtest_cli(
            config=config,
            strategy_type=strategy_type,
            device=inference_device,
            output_file=output_file,
            parallel=args.parallel,
            max_workers=args.workers,
            start_date=start_date,
            end_date=end_date,
        )

        if result.get('status') == 'success':
            print(f"\nBacktest complete!")
            print(f"  Trades executed: {result.get('trade_count', 0)}")
            print(f"  Report saved to: {result.get('output_file', output_file)}")
            return 0
        else:
            print("Backtest failed")
            return 1

    except Exception as e:
        print(f"Error during backtest: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_run_agent(args) -> int:
    """Run the daily portfolio agent."""
    from portfolio_agent.src.orchestrator import run_orchestrator

    config = get_config()

    print("Running orchestrator...")
    try:
        excel_path = run_orchestrator(
            force_refresh=args.force_refresh,
            simulate_outcome=args.simulate_outcome,
            update_outcomes=args.update_outcomes,
            config=config,
        )
        print(f"Done. Report saved to: {excel_path}")
        return 0
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="portfolio-agent",
        description="Portfolio Agent CLI - Autonomous Financial Advisor"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # download-data command
    download_parser = subparsers.add_parser(
        "download-data",
        help="Download market data for the configured universe"
    )
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of all data"
    )
    download_parser.add_argument(
        "--universe-size",
        type=int,
        default=None,
        help="Number of tickers to download (default: from config)"
    )
    download_parser.set_defaults(func=cmd_download_data)
    
    # train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train the ML model on historical data"
    )
    train_parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "mps", "cpu"],
        default=None,
        help="Device for training (default: auto)"
    )
    train_parser.set_defaults(func=cmd_train)
    
    # backtest command
    backtest_parser = subparsers.add_parser(
        "backtest",
        help="Run backtesting simulation"
    )
    backtest_parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Registered strategy to backtest (e.g. 'rule_based', 'lstm'). Defaults to config.strategy.type"
    )
    backtest_parser.add_argument(
        "--use-trained-model",
        action="store_true",
        help="Shorthand for --strategy lstm"
    )
    backtest_parser.add_argument(
        "--parallel",
        action="store_true",
        help="Parallelize rule-based signal generation across CPU workers"
    )
    backtest_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Max worker processes when --parallel is set (default: CPU count)"
    )
    backtest_parser.add_argument(
        "--years",
        type=int,
        default=None,
        help="Number of years of history to backtest (default: config.backtest.start_years_ago)"
    )
    backtest_parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Explicit backtest start date (YYYY-MM-DD), overrides --years"
    )
    backtest_parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Explicit backtest end date (YYYY-MM-DD), defaults to today"
    )
    backtest_parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "mps", "cpu"],
        default=None,
        help="Device for model inference (default: cpu for backtest)"
    )
    backtest_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output Excel file path"
    )
    backtest_parser.set_defaults(func=cmd_backtest)
    
    # run-agent command
    agent_parser = subparsers.add_parser(
        "run-agent",
        help="Run the daily portfolio agent"
    )
    agent_parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh of market data"
    )
    agent_parser.add_argument(
        "--simulate-outcome",
        action="store_true",
        help="Simulate outcome for top recommendation"
    )
    agent_parser.add_argument(
        "--update-outcomes",
        action="store_true",
        help="Update outcomes from market data"
    )
    agent_parser.set_defaults(func=cmd_run_agent)
    
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    if args.command is None:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
