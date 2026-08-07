#!/bin/bash
echo "Fixing local folders for Airflow user"
mkdir -p data output logs
sudo chown -R 50000:0 data output logs || true
echo "Done. If your Docker Airflow image uses a different UID, update this script."
