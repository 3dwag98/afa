#!/bin/bash
set -e

# =============================================================================
# IMPORTANT: LINE ENDINGS
# =============================================================================
# This file MUST be saved with LF (Unix) line endings, NOT CRLF (Windows).
# If you edit this file on Windows, ensure your editor (VS Code, Notepad++, etc.)
# is set to use LF line endings.
#
# If Docker throws an error like: "/bin/bash^M: bad interpreter",
# it means this file has CRLF endings. Fix it by running:
#   sed -i 's/\r$//' scripts/airflow-init.sh
# Or configure your editor to save with LF endings.
# =============================================================================

echo "Starting Airflow initialization..."

echo "Running database migrations..."
airflow db migrate

echo "Creating admin user..."
airflow users create \
  --username admin \
  --password admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com || true

echo "Airflow initialization completed successfully!"
