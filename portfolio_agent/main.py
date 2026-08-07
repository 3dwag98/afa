#!/usr/bin/env python3
"""Portfolio Agent - Self-Learning Portfolio Optimization for Indian Markets.

This is a decision-support system that:
- Fetches historical market data for Indian stocks
- Performs technical analysis and Monte Carlo simulations
- Generates trading recommendations with risk management
- Outputs results to Excel
- Learns from historical outcomes

IMPORTANT: This system does NOT execute real trades.
It operates in paper trading / decision support mode only.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from config import get_config
from orchestrator import run_orchestrator


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Portfolio Agent - Self-Learning Portfolio Optimization"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh of market data (ignore cache)"
    )
    parser.add_argument(
        "--no-simulate",
        action="store_true",
        help="Disable simulated outcome generation for demo learning"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Portfolio Agent - Self-Learning Portfolio Optimization")
    print("Indian Markets - Decision Support System")
    print("=" * 60)
    print()

    # Load configuration
    print("Loading configuration...")
    try:
        config = get_config()
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid configuration: {e}")
        sys.exit(1)

    print(f"  Portfolio Value: ₹{config.portfolio_value_inr:,.2f}")
    print(f"  Risk per Trade: {config.risk_per_trade_pct * 100:.1f}%")
    print(f"  Max Position: {config.max_single_position_pct * 100:.1f}%")
    print(f"  Tickers: {len(config.tickers)}")
    print(f"  Paper Trading Mode: {config.paper_trading_mode}")
    print()

    # Run orchestrator
    print("Running orchestrator...")
    print("-" * 60)

    try:
        excel_path = run_orchestrator(
            force_refresh=args.force_refresh,
            simulate_outcome=not args.no_simulate
        )

        print()
        print("OPTIMIZATION COMPLETE")
        print("-" * 60)
        print(f"  Output File: {excel_path}")
        print()

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print()
    print("=" * 60)
    print("DISCLAIMER: This is a decision-support system only.")
    print("No real trades are executed. Past performance does not")
    print("guarantee future results. Use at your own risk.")
    print("=" * 60)

    # Print final Excel path
    print(f"\nFinal Excel path: {excel_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
