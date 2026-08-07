#!/bin/bash
set -e
echo "Starting Airflow DB migration"
airflow db migrate
echo "Creating admin user"
airflow users create \
  --username admin \
  --password admin \
  --firstname Admin \
  --lastname Admin \
  --role Admin \
  --email admin@example.com || true
echo "Airflow initialization complete"
