# Portfolio Agent - Self-Learning Portfolio Optimization for Indian Markets

## Overview

Python self-learning portfolio agent for Indian markets.

- **Decision support only**: This system does NOT place real trades. It operates in paper trading / decision support mode only.
- **No real broker trading**: There is no broker API integration for trade execution.
- **Educational Purpose**: This tool is for educational and research purposes only. Past performance does not guarantee future results.

## Quick Start

### Option 1: Run with Docker (from project root)

```bash
# Run the agent once
docker compose --profile base run --rm agent

# Run with options
docker compose --profile base run --rm agent python main.py --force-refresh

# Run tests
docker compose --profile base run --rm test
```

### Option 2: Use Helper Scripts (Easiest!)

**Windows:**
```cmd
# Run the agent
run-agent.bat

# Run with options
run-agent.bat --force-refresh
run-agent.bat --simulate-outcome
run-agent.bat --update-outcomes

# Manage Airflow
airflow-manage.bat start
airflow-manage.bat stop
airflow-manage.bat status
```

**macOS/Linux:**
```bash
# Make scripts executable (first time only)
chmod +x ../run-agent.sh ../airflow-manage.sh

# Run the agent
../run-agent.sh

# Run with options
../run-agent.sh --force-refresh

# Manage Airflow
../airflow-manage.sh start
../airflow-manage.sh stop
```

### Option 3: Use Make Commands

```bash
# Show all available commands
make help

# Run the agent
make run

# Run tests
make test

# Manage Airflow
make airflow-up
make airflow-down
make airflow-trigger-daily
```

### Option 4: Run Locally with Python

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the agent
python main.py
```

## Table of Contents

- [Installation](#installation)
- [Running Manually](#running-manually)
- [Running with Airflow Scheduler](#running-with-airflow-scheduler)
- [Manual Trigger of Airflow DAGs](#manual-trigger-of-airflow-dags)
- [Configuration](#configuration)
- [Output](#output)
- [Learning Mechanism](#learning-mechanism)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Safety & Guardrails](#safety--guardrails)

## Installation

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for Docker-based runs)
- pip (Python package manager)
- make (optional, for using Makefile commands)

### Local Python Setup

1. **Navigate to the project directory:**

```bash
cd portfolio_agent
```

2. **Create a virtual environment:**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Verify installation (optional):**

```bash
pytest tests/test_scoring.py -v
```

### Docker Setup

No additional setup required beyond having Docker and Docker Compose installed.

## Running Manually

### With Docker

```bash
# Run the agent once
docker compose run --rm portfolio-agent

# Run with synthetic/refreshed data
docker compose run --rm portfolio-agent python main.py --force-refresh

# Simulate outcome for learning demo
docker compose run --rm portfolio-agent python main.py --simulate-outcome

# Update outcomes from market data
docker compose run --rm portfolio-agent python main.py --update-outcomes

# Run tests
docker compose run --rm test
```

### With Python (Local)

```bash
# Activate virtual environment first
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run the agent
python main.py

# Run with synthetic/refreshed data
python main.py --force-refresh

# Simulate outcome for learning demo
python main.py --simulate-outcome

# Update outcomes from market data
python main.py --update-outcomes

# Run tests
pytest tests/ -v
```

### Using Make Commands

```bash
# Build normal agent
make build

# Run agent once
make run

# Run tests
make test

# Clean build artifacts
make clean

# Open shell in container
make shell
```

## Running with Airflow Scheduler

Apache Airflow provides scheduled execution of the portfolio agent on weekdays at 15:45 IST.

### Airflow Architecture

The system includes three DAGs:

1. **portfolio_daily_run** - Daily portfolio execution (weekdays at 15:45 IST)
   - Runs the full orchestrator
   - Produces Excel output
   
2. **portfolio_update_outcomes** - Update trade outcomes (weekdays at 16:00 IST)
   - Updates trade outcomes if market data is available
   
3. **portfolio_relearn** - Manual relearning task (triggered on-demand)
   - Runs evaluate_and_learn only

### Airflow Quickstart

```bash
# 1. Fix permissions on Linux (if needed)
./scripts/fix_airflow_permissions.sh

# 2. Build Airflow images
make airflow-build
# Or: docker compose --profile airflow build

# 3. Initialize Airflow database and create admin user
make airflow-init
# Or: docker compose --profile airflow run --rm airflow-init

# 4. Start Airflow services (webserver + scheduler)
make airflow-up
# Or: docker compose --profile airflow up -d

# 5. Open Airflow UI
# Navigate to: http://localhost:8080
# Login: admin / admin

# 6. Enable the DAG (it's paused by default)
# In the Airflow UI, toggle the DAG to "On"

# 7. View logs
make airflow-logs
# Or: docker compose --profile airflow logs -f airflow-scheduler
```

### Stop Airflow

```bash
make airflow-down
# Or: docker compose --profile airflow down
```

### Disable Scheduler

To prevent automatic scheduling, you can either:

1. Set in config.yaml:
   ```yaml
   scheduler_enabled: false
   ```
   This marks DAGs as paused by default.

2. Stop Airflow:
   ```bash
   docker compose --profile airflow down
   ```

3. Keep DAGs paused in the Airflow UI

## Manual Trigger of Airflow DAGs

You can manually trigger DAGs via command line or the Airflow UI.

### Via Command Line

```bash
# Trigger daily orchestrator run
make airflow-trigger-daily
# Or: docker compose --profile airflow exec airflow-scheduler airflow dags trigger portfolio_daily_run

# Trigger outcome update
make airflow-trigger-outcomes
# Or: docker compose --profile airflow exec airflow-scheduler airflow dags trigger portfolio_update_outcomes

# Trigger relearning
make airflow-trigger-relearn
# Or: docker compose --profile airflow exec airflow-scheduler airflow dags trigger portfolio_relearn
```

### Via Airflow UI

1. Open http://localhost:8080
2. Login with `admin / admin`
3. Find the DAG you want to trigger
4. Click the "Play" button in the Actions column

### All Airflow Make Commands

```bash
# Build Airflow
make airflow-build

# Initialize Airflow
make airflow-init

# Start Airflow
make airflow-up

# Trigger daily run
make airflow-trigger-daily

# Trigger outcome update
make airflow-trigger-outcomes

# Trigger relearning
make airflow-trigger-relearn

# View scheduler logs
make airflow-logs

# View webserver logs
make airflow-webserver-logs

# Run tests in Airflow environment
make airflow-test

# Stop Airflow
make airflow-down
```

## Configuration

Edit `config.yaml` to customize:

- Portfolio value (INR)
- Risk parameters
- Target tickers (NSE/BSE stocks)
- Simulation parameters
- File paths
- Scheduler settings

### Key Configuration Options

```yaml
# Enable/disable scheduler
scheduler_enabled: false

# Paper trading mode (must remain true)
paper_trading_mode: true

# Portfolio settings
portfolio_value: 1000000  # INR

# Risk settings
max_risk_per_trade: 0.02  # 2% of portfolio
max_position_size: 0.1    # 10% of portfolio
```

## Output

The agent generates an Excel report with detailed analysis and recommendations:

```
output/Agent_Orchestrator_Output.xlsx
```

### Learning Data

- Agent weights stored in: `data/agent_brain.json`
- Trade outcomes stored in: `data/portfolio_agent.db`
- Logs stored in: `logs/`

## Learning Mechanism

The agent stores:

- Historical decisions and their outcomes
- Performance metrics per ticker
- Pattern recognition data

Over time, the agent adjusts its scoring based on past success/failure rates.

- Weights update based on WIN/LOSS outcomes
- The agent learns from simulated and actual outcomes
- Learning can be triggered manually via the `portfolio_relearn` DAG

## Testing

```bash
# With Docker
docker compose run --rm test

# With Python (local)
pytest tests/ -v

# Specific test file
pytest tests/test_scoring.py -v

# With Make
make test

# In Airflow environment
make airflow-test
```

## Project Structure

```
portfolio_agent/
├── config.yaml          # Configuration file
├── main.py              # Entry point
├── requirements.txt     # Python dependencies
├── Makefile             # Make commands
├── docker-compose.yml   # Docker Compose configuration
├── Dockerfile.airflow   # Docker image for Airflow
├── README.md            # This file
├── src/                 # Source modules
│   ├── __init__.py
│   ├── config.py        # Configuration loader
│   ├── models.py        # Data models
│   ├── storage.py       # SQLite and JSON storage
│   ├── data_ingestion.py # Market data fetching
│   ├── indicators.py    # Technical indicators
│   ├── monte_carlo.py   # Monte Carlo simulations
│   ├── scoring.py       # Stock scoring logic
│   ├── risk.py          # Risk management
│   ├── compliance.py    # Compliance checks
│   ├── learning.py      # Self-learning logic
│   ├── reporting.py     # Excel report generation
│   └── orchestrator.py  # Main orchestration logic
├── tests/               # Test files
├── data/                # SQLite DB and agent brain JSON
├── output/              # Excel output files
├── logs/                # Log files
└── airflow/             # Airflow configuration
    ├── dags/            # Airflow DAGs
    └── scripts/         # Airflow scripts
```

## Safety & Guardrails

- **Paper trading only**: No live trading is enabled
- **No leverage**: Positions are fully funded
- **No short selling**: Only long positions allowed
- **Penny stock filter**: Low-quality stocks excluded
- **Position size cap**: Maximum 10% per position
- **Risk per trade cap**: Maximum 2% risk per trade
- **Airflow UI credentials**: Local-only (admin/admin)

## Alternative Scheduling Options

### Windows Task Scheduler

If Airflow is too heavy, use Windows Task Scheduler:

1. Create a new task
2. Set trigger: Daily at 15:45, Monday-Friday
3. Set action:
   ```
   docker compose run --rm portfolio-agent
   ```
4. Optional second task at 16:00 for outcomes:
   ```
   docker compose run --rm portfolio-agent python main.py --update-outcomes
   ```

### Linux/macOS cron

On macOS/Linux, use `crontab -e`:

```
# Daily run at 15:45 IST on weekdays
45 15 * * 1-5 cd /path/to/portfolio_agent && docker compose run --rm portfolio-agent

# Outcome update at 16:00 IST on weekdays
0 16 * * 1-5 cd /path/to/portfolio_agent && docker compose run --rm portfolio-agent python main.py --update-outcomes
```

## Troubleshooting

### Module Import Errors

If you encounter `ModuleNotFoundError: No module named 'src'`:

1. Ensure you're running from the correct directory
2. Verify PYTHONPATH is set correctly in docker-compose.yml
3. For local runs, ensure you've installed dependencies

### Airflow Permission Issues (Linux)

```bash
./scripts/fix_airflow_permissions.sh
```

### Docker Build Issues

Clear cache and rebuild:
```bash
docker compose --profile airflow build --no-cache
```

### Line Ending Issues (Windows)

If scripts fail with "bad interpreter" errors:
```bash
sed -i 's/\r$//' scripts/airflow-init.sh
```

## Features

- Fetches historical market data from Yahoo Finance for Indian stocks (NSE/BSE)
- Technical indicator calculation
- Monte Carlo simulation for risk analysis
- Self-learning agent that stores outcomes and improves over time
- SQLite database for structured history
- JSON-based agent memory/brain
- Excel output with detailed analysis and recommendations

## License

MIT License - For educational purposes only.

**Warning**: This system is for educational and research purposes only. Do not use for actual financial decisions without proper professional advice and regulatory compliance.
