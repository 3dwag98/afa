#!/usr/bin/env bash
set -e

# Run tests via Docker Compose
# Arguments are passed through to docker compose
# Example: ./scripts/run_tests.sh -v --tb=short

docker compose run --rm test "$@"
