#!/usr/bin/env bash
set -e

# Run the live agent via Docker Compose
# Arguments are passed through to docker compose
# Example: ./scripts/run_agent.sh

docker compose run --rm agent "$@"
