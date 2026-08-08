#!/usr/bin/env bash
set -e

# Run the backtest engine via Docker Compose
# Arguments are passed through to the CLI entrypoint
# Example: ./scripts/run_backtest.sh --help

docker compose run --rm backtest python -m portfolio_agent.cli backtest "$@"
