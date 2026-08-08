#!/usr/bin/env bash
set -e

# Run the backtest engine via Docker Compose
# Arguments are passed through to docker compose
# Example: ./scripts/run_backtest.sh --years 2 --universe-size 20

docker compose run --rm backtest "$@"
