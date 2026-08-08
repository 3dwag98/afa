.PHONY: setup build build-airflow train-gpu backtest agent test airflow-up airflow-down logs clean-output clean-logs help

help:
	@echo "AFA Portfolio Agent - Available Commands"
	@echo "========================================="
	@echo ""
	@echo "Setup & Build:"
	@echo "  make setup         - Run platform setup script"
	@echo "  make build         - Build Docker images (GPU-enabled)"
	@echo "  make build-airflow - Build Airflow services"
	@echo ""
	@echo "Training & Backtesting:"
	@echo "  make train-gpu     - Run GPU training (requires NVIDIA Container Toolkit)"
	@echo "  make backtest      - Run default backtest"
	@echo ""
	@echo "Running:"
	@echo "  make agent         - Run the live agent"
	@echo "  make test          - Run pytest"
	@echo "  make logs          - Follow Docker logs"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean-output  - Remove output Excel files"
	@echo "  make clean-logs    - Remove log files"
	@echo ""
	@echo "Airflow:"
	@echo "  make airflow-up    - Start Airflow services (UI at localhost:8080)"
	@echo "  make airflow-down  - Stop Airflow services"
	@echo ""

setup:
	python scripts/setup_platform.py

build:
	docker compose build

build-airflow:
	docker compose --profile airflow build

train-gpu:
	docker compose run --rm train python -m portfolio_agent.cli train --model lstm --device auto

backtest:
	docker compose run --rm backtest python -m portfolio_agent.cli backtest --use-trained-model

agent:
	docker compose run --rm agent python -m portfolio_agent.cli run-agent

test:
	docker compose run --rm test

logs:
	docker compose logs -f

clean-output:
	rm -f output/*.xlsx || true

clean-logs:
	rm -f logs/*.log || true

airflow-up:
	docker compose --profile airflow up -d

airflow-down:
	docker compose --profile airflow down
