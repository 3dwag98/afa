.PHONY: install test run clean build shell airflow-build airflow-init airflow-up airflow-down airflow-logs airflow-webserver-logs airflow-trigger-daily airflow-trigger-outcomes airflow-trigger-relearn airflow-test

install:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

test:
	.venv/bin/pytest tests/

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

# Docker targets
build:
	docker compose build

shell:
	docker compose run --rm portfolio-agent bash

# Airflow targets
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

airflow-webserver-logs:
	docker compose --profile airflow logs -f airflow-webserver

airflow-trigger-daily:
	docker compose --profile airflow exec airflow-scheduler airflow dags trigger portfolio_daily_run

airflow-trigger-outcomes:
	docker compose --profile airflow exec airflow-scheduler airflow dags trigger portfolio_update_outcomes

airflow-trigger-relearn:
	docker compose --profile airflow exec airflow-scheduler airflow dags trigger portfolio_relearn

airflow-test:
	docker compose --profile airflow run --rm airflow-test
