#!/bin/bash
set -e
echo "Initializing Airflow"
airflow db migrate
airflow users create \
  --username admin \
  --password admin \
  --firstname Admin \
  --lastname Admin \
  --role Admin \
  --email admin@example.com || true
echo "Airflow initialization complete"
