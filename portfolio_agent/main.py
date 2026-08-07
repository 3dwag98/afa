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
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from config import get_config
from orchestrator import PortfolioOrchestrator


def main():
    """Main entry point."""
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

    # Initialize and run orchestrator
    print("Initializing Portfolio Orchestrator...")
    orchestrator = PortfolioOrchestrator(config)

    print("Running portfolio optimization cycle...")
    print("-" * 60)

    try:
        result = orchestrator.run()

        if result.get('error'):
            print(f"Error: {result['error']}")
            sys.exit(1)

        print()
        print("OPTIMIZATION COMPLETE")
        print("-" * 60)
        print(f"  Status: {result.get('status', 'unknown')}")
        print(f"  Tickers Analyzed: {result.get('tickers_analyzed', 0)}")
        print(f"  Recommendations: {result.get('recommendations_count', 0)}")
        print(f"  Output File: {result.get('output_path', 'N/A')}")
        print(f"  Timestamp: {result.get('timestamp', 'N/A')}")
        print()

        # Save agent state
        orchestrator.save_state()
        print("Agent state saved.")

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

    return 0


if __name__ == "__main__":
    sys.exit(main())
