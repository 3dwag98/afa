.PHONY: install test run clean build backtest backtest-full shell help

help:
	@echo "Portfolio Agent - Available Commands"
	@echo "====================================="
	@echo ""
	@echo "Local Development:"
	@echo "  make install     - Create venv and install dependencies"
	@echo "  make test        - Run tests"
	@echo "  make run         - Run the agent locally"
	@echo "  make clean       - Clean up temporary files"
	@echo ""
	@echo "Backtesting:"
	@echo "  make backtest    - Run backtest with 50 tickers (quick test)"
	@echo "  make backtest-full - Run backtest with all 500 tickers"
	@echo ""
	@echo "Docker:"
	@echo "  make build       - Build Docker images"
	@echo "  make shell       - Open shell in agent container"
	@echo ""

install:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

test:
	.venv/bin/pytest tests/ -v

run:
	.venv/bin/python main.py

clean:
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf .pytest_cache/
	rm -rf data/*.db
	rm -rf output/*.xlsx
	rm -rf logs/*.log

# Backtest commands
backtest:
	.venv/bin/python run_backtest.py --universe-size 50

backtest-full:
	.venv/bin/python run_backtest.py --universe-size 500

# Docker targets
build:
	docker compose build

shell:
	docker compose run --rm portfolio-agent bash
