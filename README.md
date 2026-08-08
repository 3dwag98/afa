# AFA - Autonomous Financial Advisor (Portfolio Agent)

Self-learning portfolio optimization agent for Indian markets.

## Overview

- **Decision support only**: This system does NOT place real trades. It operates in paper trading / decision support mode only.
- **No real broker trading**: There is no broker API integration for trade execution.
- **Educational Purpose**: This tool is for educational and research purposes only. Past performance does not guarantee future results.

## Quick Start

### Option 1: Run with Docker (Recommended)

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

# Manage Airflow
airflow-manage.bat start
airflow-manage.bat stop
airflow-manage.bat status
```

**macOS/Linux:**
```bash
# Make scripts executable (first time only)
chmod +x run-agent.sh airflow-manage.sh

# Run the agent
./run-agent.sh

# Run with options
./run-agent.sh --force-refresh

# Manage Airflow
./airflow-manage.sh start
./airflow-manage.sh stop
./airflow-manage.sh status
```

### Option 3: Run Locally with Python

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
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Safety & Guardrails](#safety--guardrails)

## Installation

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for Docker-based runs)
- pip (Python package manager)

### Local Python Setup

1. **Navigate to the project directory:**

```bash
cd /workspace
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
pytest tests/ -v
```

### Docker Setup

No additional setup required beyond having Docker and Docker Compose installed.

## Running Manually

### With Docker

```bash
# Run the agent once
docker compose --profile base run --rm agent

# Run with synthetic/refreshed data
docker compose --profile base run --rm agent python main.py --force-refresh

# Run tests
docker compose --profile base run --rm test
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

## Running with Airflow Scheduler

Apache Airflow provides scheduled execution of the portfolio agent on weekdays at 15:45 IST.

### Airflow Architecture

The system includes three DAGs:

1. **afa_daily_orchestrator** - Daily portfolio execution (weekdays at 15:45 IST)
2. **portfolio_update_outcomes** - Update trade outcomes (weekdays at 16:00 IST)
3. **portfolio_relearn** - Manual relearning task (triggered on-demand)

### Airflow Quickstart

#### Using Helper Scripts (Easiest!)

**Windows:**
```cmd
# Start Airflow services
airflow-manage.bat start

# Check status
airflow-manage.bat status

# View logs
airflow-manage.bat logs

# Stop services
airflow-manage.bat stop

# Reset database (if needed)
airflow-manage.bat init
```

**macOS/Linux:**
```bash
# Start Airflow services
./airflow-manage.sh start

# Check status
./airflow-manage.sh status

# View logs
./airflow-manage.sh logs

# Stop services
./airflow-manage.sh stop

# Reset database (if needed)
./airflow-manage.sh init
```

#### Manual Docker Commands

```bash
# 1. Build Airflow images
docker compose --profile airflow build

# 2. Initialize Airflow database and create admin user
docker compose --profile airflow run --rm airflow-init

# 3. Start Airflow services (webserver + scheduler)
docker compose --profile airflow up -d

# 4. Open Airflow UI
# Navigate to: http://localhost:8080
# Login: admin / admin

# 5. Enable the DAG (it's paused by default)
# In the Airflow UI, toggle the DAG to "On"

# 6. View logs
docker compose --profile airflow logs -f airflow-scheduler
```

### Stop Airflow

```bash
# Using helper script (recommended)
./airflow-manage.sh stop    # macOS/Linux
airflow-manage.bat stop     # Windows

# Or manually
docker compose --profile airflow down
```

### Disable Scheduler

To prevent automatic scheduling, you can either:

1. Stop Airflow:
   ```bash
   docker compose --profile airflow down
   ```

2. Or keep DAGs paused in the Airflow UI

## Manual Trigger of Airflow DAGs

You can manually trigger DAGs via command line or the Airflow UI.

### Via Command Line

```bash
# Trigger daily orchestrator run
docker compose --profile airflow exec airflow-scheduler airflow dags trigger afa_daily_orchestrator

# Trigger outcome update
docker compose --profile airflow exec airflow-scheduler airflow dags trigger portfolio_update_outcomes

# Trigger relearning
docker compose --profile airflow exec airflow-scheduler airflow dags trigger portfolio_relearn
```

### Via Airflow UI

1. Open http://localhost:8080
2. Login with `admin / admin`
3. Find the DAG you want to trigger
4. Click the "Play" button in the Actions column

### Using Make Commands (portfolio_agent subdirectory)

If working from the `portfolio_agent` subdirectory:

```bash
cd portfolio_agent

# Show all available commands
make help

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

# View logs
make airflow-logs

# Stop Airflow
make airflow-down
```

### Using Helper Scripts (Root Directory - Recommended!)

From the project root directory, use the helper scripts for easier management:

**Windows:**
```cmd
# Run agent
run-agent.bat

# Run with options
run-agent.bat --force-refresh
run-agent.bat --simulate-outcome
run-agent.bat --update-outcomes

# Manage Airflow
airflow-manage.bat start
airflow-manage.bat stop
airflow-manage.bat status
airflow-manage.bat logs
airflow-manage.bat init
```

**macOS/Linux:**
```bash
# Make scripts executable (first time only)
chmod +x run-agent.sh airflow-manage.sh

# Run agent
./run-agent.sh

# Run with options
./run-agent.sh --force-refresh
./run-agent.sh --simulate-outcome
./run-agent.sh --update-outcomes

# Manage Airflow
./airflow-manage.sh start
./airflow-manage.sh stop
./airflow-manage.sh status
./airflow-manage.sh logs
./airflow-manage.sh init
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

## Testing

```bash
# With Docker
docker compose --profile base run --rm test

# With Python (local)
pytest tests/ -v

# Specific test file
pytest tests/test_scoring.py -v
```

## Project Structure

```
/workspace/
├── docker-compose.yml      # Main Docker Compose configuration
├── Dockerfile              # Docker image for agent
├── README.md               # This file
├── config.yaml             # Configuration file
├── main.py                 # Entry point
├── requirements.txt        # Python dependencies
├── scripts/                # Utility scripts
│   ├── airflow-init.sh     # Airflow initialization script
│   └── fix_airflow_permissions.sh
├── src/                    # Source modules (symlink to portfolio_agent/src)
│   ├── __init__.py
│   ├── config.py           # Configuration loader
│   ├── models.py           # Data models
│   ├── storage.py          # SQLite and JSON storage
│   ├── data_ingestion.py   # Market data fetching
│   ├── indicators.py       # Technical indicators
│   ├── monte_carlo.py      # Monte Carlo simulations
│   ├── scoring.py          # Stock scoring logic
│   ├── risk.py             # Risk management
│   ├── compliance.py       # Compliance checks
│   ├── learning.py         # Self-learning logic
│   ├── reporting.py        # Excel report generation
│   └── orchestrator.py     # Main orchestration logic
├── airflow/                # Airflow DAGs and scripts
│   ├── dags/
│   │   └── afa_daily_dag.py
│   └── scripts/
│       └── airflow-init.sh
├── data/                   # SQLite DB and agent brain JSON
├── output/                 # Excel output files
└── logs/                   # Log files
```

## Safety & Guardrails

- **Paper trading only**: No live trading is enabled
- **No leverage**: Positions are fully funded
- **No short selling**: Only long positions allowed
- **Penny stock filter**: Low-quality stocks excluded
- **Position size cap**: Maximum 3% per position
- **Risk per trade cap**: Maximum 2% risk per trade
- **Airflow UI credentials**: Local-only (admin/admin)

## Backtesting

Run historical simulations with the `run_backtest.py` script:

```bash
# Quick test with 50 tickers, 1 year
python run_backtest.py --universe-size 50 --years 1

# Full 5-year backtest with 500 tickers
python run_backtest.py --universe-size 500 --years 5

# Custom initial capital
python run_backtest.py --initial-capital 5000000 --years 5
```

### Advanced Risk Metrics

The backtest report includes institutional-grade risk analytics:

| Metric | Description |
|--------|-------------|
| **CAGR** | Compound Annual Growth Rate - annualized return over the period |
| **Sharpe Ratio** | Risk-adjusted return using total volatility (excess return / volatility) |
| **Sortino Ratio** | Risk-adjusted return using downside deviation (penalizes only negative volatility) |
| **Calmar Ratio** | Return relative to maximum drawdown (CAGR / Max Drawdown) |
| **Max Drawdown** | Largest peak-to-trough decline in portfolio value |
| **Profit Factor** | Gross profits divided by gross losses |
| **Win Rate** | Percentage of profitable trades |
| **Probability of Ruin** | Monte Carlo simulation result showing chance of portfolio dropping below threshold (default 50%) |

#### Understanding Sortino Ratio

Unlike Sharpe ratio which penalizes all volatility equally, Sortino ratio only considers **downside deviation** - the volatility of negative returns. This is more appropriate for investors who care about losses, not gains.

```
Sortino = (CAGR - Risk Free Rate) / Downside Deviation
```

A Sortino > 2.0 is considered excellent.

#### Understanding Calmar Ratio

Calmar ratio measures return per unit of worst-case loss (maximum drawdown). It answers: "How much return did I get for surviving the worst period?"

```
Calmar = CAGR / Maximum Drawdown
```

A Calmar > 3.0 indicates strong risk-adjusted performance.

#### Understanding Probability of Ruin

Using Monte Carlo bootstrap resampling (10,000 simulations), this metric estimates the probability that your portfolio will drop below a critical threshold (typically 50% of initial capital) based on historical trade distributions.

- **< 5%**: Very low risk of catastrophic loss
- **5-15%**: Moderate risk, acceptable for most strategies
- **> 20%**: High risk, consider reducing position sizes

## Alternative Scheduling Options

### Windows Task Scheduler

1. Create a new task
2. Set trigger: Daily at 15:45, Monday-Friday
3. Set action:
   ```
   docker compose --profile base run --rm agent
   ```
4. Optional second task at 16:00 for outcomes:
   ```
   docker compose --profile base run --rm agent python main.py --update-outcomes
   ```

### Linux/macOS cron

Edit crontab:
```bash
crontab -e
```

Add entries:
```
# Daily run at 15:45 IST on weekdays
45 15 * * 1-5 cd /workspace && docker compose --profile base run --rm agent

# Outcome update at 16:00 IST on weekdays
0 16 * * 1-5 cd /workspace && docker compose --profile base run --rm agent python main.py --update-outcomes
```

## Troubleshooting

### Module Import Errors

If you encounter `ModuleNotFoundError: No module named 'src'`:

1. Ensure the symlink exists: `ls -la /workspace/src`
2. For Airflow, verify PYTHONPATH is set correctly in docker-compose.yml
3. The DAG file adds `/workspace` to sys.path as a fallback

### Config File Not Found

If you get `FileNotFoundError: Configuration file not found`:

1. Ensure `config.yaml` exists at the project root: `ls -la config.yaml`
2. If missing, copy it from portfolio_agent: `cp portfolio_agent/config.yaml .`
3. Verify Docker volumes are mounted correctly in docker-compose.yml

### Docker Volume Mount Issues (Windows)

If you see errors like "Cannot create a file when that file already exists":

**Option 1:** Remove existing src symlink and recreate as junction:
```powershell
Remove-Item .\src -Force -Recurse
cmd /c mklink /J src portfolio_agent\src
```

**Option 2:** Use the helper scripts which handle this automatically

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

### Helper Scripts Not Working

**Windows:** Run from Command Prompt or PowerShell, not Git Bash
**macOS/Linux:** Make sure scripts are executable: `chmod +x *.sh`

### Checking Service Status

```bash
# Check if Docker is running
docker info

# Check container status
docker compose --profile base ps
docker compose --profile airflow ps

# View logs
docker compose --profile base logs -f agent
docker compose --profile airflow logs -f airflow-scheduler
```

## License

MIT License - For educational purposes only.

**Warning**: This system is for educational and research purposes only. Do not use for actual financial decisions without proper professional advice and regulatory compliance.
