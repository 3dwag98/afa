#!/usr/bin/env bash
set -e

# Run the live agent via Docker Compose
# Arguments are passed through to the CLI entrypoint
# Example: ./scripts/run_agent.sh --help

docker compose run --rm agent python -m portfolio_agent.cli run-agent "$@"
