.PHONY: setup build agent backtest backtest-small backtest-full download-data test logs clean-output clean-logs airflow-build airflow-init airflow-up airflow-down airflow-logs help

help:
	@echo "AFA Portfolio Agent - Available Commands"
	@echo "========================================="
	@echo ""
	@echo "Setup & Build:"
	@echo "  make setup         - Run platform setup script"
	@echo "  make build         - Build Docker images"
	@echo ""
	@echo "Running:"
	@echo "  make agent         - Run the live agent"
	@echo "  make backtest      - Run default backtest"
	@echo "  make backtest-small- Run quick backtest (1 year, 20 tickers)"
	@echo "  make backtest-full - Run full backtest (5 years)"
	@echo "  make download-data - Download market data (50 tickers, 2 years)"
	@echo "  make test          - Run pytest"
	@echo "  make logs          - Follow Docker logs"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean-output  - Remove output Excel files"
	@echo "  make clean-logs    - Remove log files"
	@echo ""
	@echo "Airflow (requires --profile airflow):"
	@echo "  make airflow-build - Build Airflow services"
	@echo "  make airflow-init  - Initialize Airflow database"
	@echo "  make airflow-up    - Start Airflow services"
	@echo "  make airflow-down  - Stop Airflow services"
	@echo "  make airflow-logs  - Follow Airflow scheduler logs"
	@echo ""

setup:
	python scripts/setup_platform.py

build:
	docker compose build

agent:
	docker compose run --rm agent

backtest:
	docker compose run --rm backtest

backtest-small:
	docker compose run --rm backtest python run_backtest.py --years 1 --universe-size 20

backtest-full:
	docker compose run --rm backtest python run_backtest.py --years 5

download-data:
	docker compose run --rm backtest python run_backtest.py --force-download --universe-size 50 --years 2

test:
	docker compose run --rm test

logs:
	docker compose logs -f

clean-output:
	rm -f output/*.xlsx || true

clean-logs:
	rm -f logs/*.log || true

airflow-build:
	docker compose --profile airflow build

airflow-init:
	docker compose --profile airflow run --rm airflow-init

airflow-up:
	docker compose --profile airflow up -d

airflow-down:
	docker compose --profile airflow down

airflow-logs:
	docker compose --profile airflow logs -f airflow-scheduler
