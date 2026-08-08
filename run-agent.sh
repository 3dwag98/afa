#!/bin/bash
# AFA Agent Runner Script for macOS/Linux
# Usage: ./run-agent.sh [OPTIONS]
# Options:
#   --force-refresh      Force refresh all data
#   --simulate-outcome   Simulate trade outcomes
#   --update-outcomes    Update actual trade outcomes

set -e

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Parse command line arguments
ARGS="$@"

echo "Starting AFA Agent..."
docker compose --profile base run --rm agent $ARGS

if [ $? -eq 0 ]; then
    echo ""
    echo "Agent completed successfully."
    echo "Check the 'output' folder for reports."
else
    echo ""
    echo "Agent failed. Check logs for details."
    exit 1
fi
