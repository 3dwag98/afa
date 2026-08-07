# portfolio_agent

Self-learning portfolio optimization agent.

## Docker Usage

Run agent once:
```bash
docker compose run --rm agent
```

Run tests:
```bash
docker compose run --rm test
```

## Local Development

### Prerequisites
- Python 3.11+
- Docker and Docker Compose

### Running Locally
```bash
python main.py
```

### Running Tests
```bash
pytest tests -v
```

## Scheduler

The scheduler is now Apache Airflow. Airflow runs only when explicitly started with Docker profile "airflow". The system remains paper-trading only.

## Airflow Local Setup

Fix permissions on Linux:
```bash
./scripts/fix_airflow_permissions.sh
```

Build Airflow image:
```bash
docker compose --profile airflow build
```

Initialize Airflow:
```bash
docker compose --profile airflow run --rm airflow-init
```

Start Airflow:
```bash
docker compose --profile airflow up -d
```

Open Airflow UI:
```
http://localhost:8080
```

Default login:
```
admin / admin
```

Stop Airflow:
```bash
docker compose --profile airflow down
```

**Warning:** Airflow admin credentials are for local development only. Do not use these credentials in production. The agent remains paper-trading only.
