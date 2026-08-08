#!/usr/bin/env bash
set -e

echo "=========================================="
echo "AFA Docker Smoke Test"
echo "=========================================="

echo ""
echo "Building app image..."
docker compose build

echo ""
echo "Running tests..."
docker compose run --rm test

echo ""
echo "Running agent help or dry check..."
docker compose run --rm agent python main.py --help || true

echo ""
echo "Running backtest help or dry check..."
docker compose run --rm backtest python run_backtest.py --help || true

echo ""
echo "Running small backtest (1 year, 10 tickers)..."
docker compose run --rm backtest python run_backtest.py --years 1 --universe-size 10

echo ""
echo "=========================================="
echo "Smoke test complete."
echo "Check ./output for generated Excel files."
echo "=========================================="
