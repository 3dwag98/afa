#!/bin/bash
# AFA Airflow Management Script for macOS/Linux
# Usage: ./airflow-manage.sh [COMMAND]
# Commands:
#   start     - Start all Airflow services (webserver, scheduler, postgres)
#   stop      - Stop all Airflow services
#   restart   - Restart all Airflow services
#   logs      - Show logs from all services
#   init      - Initialize/reset the Airflow database
#   status    - Check status of services

set -e

case "$1" in
    start)
        echo "Starting Airflow services..."
        docker compose --profile airflow up -d --build
        echo ""
        echo "Airflow is starting. Wait ~30 seconds for all services to be ready."
        echo "Web UI will be available at: http://localhost:8080"
        echo "Login: admin / admin"
        ;;
    stop)
        echo "Stopping Airflow services..."
        docker compose --profile airflow down
        echo ""
        echo "Airflow services stopped."
        ;;
    restart)
        echo "Restarting Airflow services..."
        docker compose --profile airflow down
        docker compose --profile airflow up -d --build
        echo ""
        echo "Airflow is restarting. Wait ~30 seconds for all services to be ready."
        ;;
    logs)
        docker compose --profile airflow logs -f
        ;;
    init)
        echo "WARNING: This will reset the Airflow database!"
        read -p "Are you sure? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Operation cancelled."
            exit 0
        fi
        echo "Initializing Airflow database..."
        docker compose --profile airflow down
        docker volume rm afa_postgres_data || true
        docker compose --profile airflow up -d --build airflow-init
        echo ""
        echo "Airflow database initialized."
        ;;
    status)
        echo "Checking Airflow service status..."
        docker compose --profile airflow ps
        ;;
    *)
        echo "Usage: ./airflow-manage.sh [COMMAND]"
        echo ""
        echo "Commands:"
        echo "  start     - Start all Airflow services"
        echo "  stop      - Stop all Airflow services"
        echo "  restart   - Restart all Airflow services"
        echo "  logs      - Show logs from all services"
        echo "  init      - Initialize/reset the Airflow database"
        echo "  status    - Check status of services"
        exit 1
        ;;
esac
