# AFA - Autonomous Financial Advisor (Portfolio Agent)

Self-learning portfolio optimization agent for Indian markets.

## Overview

- **Decision support only**: This system does NOT place real trades. It operates in paper trading / decision support mode only.
- **No real broker trading**: There is no broker API integration for trade execution.
- **Educational Purpose**: This tool is for educational and research purposes only. Past performance does not guarantee future results.

## Modern Docker & GPU Training

This project uses a modern architecture with `uv` for package management and Docker Compose for containerized execution. The Docker images are GPU-enabled for accelerated ML training.

### Building the GPU-Enabled Image

```bash
# Build the main Docker image (includes GPU support)
make build
```

The Dockerfile includes CUDA and PyTorch GPU dependencies. When running on a Linux host with an NVIDIA GPU, the container will automatically use GPU acceleration if available.

### GPU Training

To trigger a GPU-accelerated training run:

```bash
make train-gpu
```

This runs the LSTM model training with automatic device selection (`--device auto`). If a GPU is available and the NVIDIA Container Toolkit is installed, it will use CUDA; otherwise, it falls back to CPU (or MPS on macOS).

**Note on GPU Passthrough:**
- **Linux**: Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to be installed on the host for GPU passthrough to work.
- **Windows/macOS**: GPU passthrough via Docker is limited. The training will automatically fallback to CPU or MPS (Metal Performance Shaders on macOS).

### Airflow UI

To start the Airflow services and access the web UI:

```bash
# Start Airflow (webserver + scheduler)
make airflow-up

# Access the UI at http://localhost:8080
# Login credentials: admin / admin

# Stop Airflow services
make airflow-down
```

Airflow provides scheduled execution of the portfolio agent on weekdays at 15:45 IST using TaskFlow API for DAG definition.

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
# Quick test with 5 tickers, 1 year
python run_backtest.py --universe-size 5 --years 1

# Full 5-year backtest with ALL cached tickers (auto-discovered)
python run_backtest.py --years 5

# Force download of full universe before backtest
python run_backtest.py --force-download --years 5

# Custom initial capital
python run_backtest.py --initial-capital 5000000 --years 5
```

### Universe Auto-Discovery

The backtest engine now **automatically discovers all cached tickers** from the `data/market_data/` directory. This means:

- By default, `--universe-size` is `None`, which uses **ALL** available tickers
- Use `--universe-size N` to limit to N tickers for quick tests
- Use `--force-download` to fetch fresh data for the entire master ticker list before running

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

## Airflow Commands

### Build Airflow

```bash
docker compose --profile airflow build
```

### Initialize Airflow

```bash
docker compose --profile airflow run --rm airflow-init
```

### Start Airflow

```bash
docker compose --profile airflow up -d
```

Access the Airflow UI at http://localhost:8080 (login: admin / admin)

### Trigger Small Backtest Manually

```bash
docker compose --profile airflow exec airflow-scheduler airflow dags trigger afa_backtest_small
```

### Trigger Full Backtest Manually

```bash
docker compose --profile airflow exec airflow-scheduler airflow dags trigger afa_backtest_full
```

### Stop Airflow

```bash
docker compose --profile airflow down
```

### View Airflow Logs

```bash
docker compose --profile airflow logs -f airflow-scheduler
```

## Windows PowerShell Commands

### Setup

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
```

### Run Agent

```powershell
.\scripts\run_agent.ps1
```

### Run Backtest

```powershell
# Default backtest
.\scripts\run_backtest.ps1

# Small backtest (1 year, 20 tickers)
.\scripts\run_backtest.ps1 --years 1 --universe-size 20

# Full 5-year backtest
.\scripts\run_backtest.ps1 --years 5

# Force download data first
.\scripts\run_backtest.ps1 --force-download --universe-size 50 --years 2
```

### Run Tests

```powershell
.\scripts\run_tests.ps1
```

### Build Images

```powershell
docker compose build
```

### Start Airflow

```powershell
docker compose --profile airflow up -d
```

### Stop Airflow

```powershell
docker compose --profile airflow down
```

### View Airflow Logs

```powershell
docker compose --profile airflow logs -f airflow-scheduler
```

**Note:** Windows users do not need bash or make. The PowerShell scripts provide full functionality. If you prefer using `make`, you can install it via Chocolatey (`choco install make`) or use the included batch files.

---

## Docker Smoke Test

Validate your Docker setup with a comprehensive smoke test that builds the image, runs tests, and executes a small backtest.

### Linux/macOS

```bash
./scripts/smoke_test.sh
```

### Windows PowerShell

```powershell
powershell -ExecutionPolicy Bypass -File scripts/smoke_test.ps1
```

Or directly from PowerShell:

```powershell
.\scripts\smoke_test.ps1
```

### What the Smoke Test Does

1. **Builds** the Docker image (`docker compose build`)
2. **Runs tests** (`docker compose run --rm test`)
3. **Checks agent** help/dry run (`docker compose run --rm agent python main.py --help`)
4. **Checks backtest** help/dry run (`docker compose run --rm backtest python run_backtest.py --help`)
5. **Runs a small backtest** (1 year, 10 tickers) to verify end-to-end functionality

### Expected Output

After successful completion, check the `./output` directory for generated Excel files from the small backtest.

### Manual Smoke Test Steps

If you prefer to run steps individually:

```bash
# Build
docker compose build

# Run tests
docker compose run --rm test

# Quick agent check
docker compose run --rm agent python main.py --help || true

# Quick backtest check  
docker compose run --rm backtest python run_backtest.py --help || true

# Small backtest
docker compose run --rm backtest python run_backtest.py --years 1 --universe-size 10
```

---

## Running With Docker

This section provides comprehensive instructions for running the AFA agent and backtests using Docker on any platform.

### 1. Initial Setup

**Linux/macOS:**
```bash
python scripts/setup_platform.py
chmod +x scripts/*.sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
```

**Optional build (if you want to pre-build the image):**
```bash
docker compose build
```

### 2. Run the Live Paper-Trading Agent

**Linux/macOS:**
```bash
./scripts/run_agent.sh
```

**Windows:**
```powershell
.\scripts\run_agent.ps1
```

**Or directly with docker compose:**
```bash
docker compose run --rm agent
```

**Output:** The agent generates an Excel report at:
```
output/Agent_Orchestrator_Output.xlsx
```

### 3. Run Backtests

**Small quick backtest (1 year, 20 tickers):**
```bash
docker compose run --rm backtest python run_backtest.py --years 1 --universe-size 20
```

**5-year backtest using all available cached tickers:**
```bash
docker compose run --rm backtest python run_backtest.py --years 5
```

**Force download more historical data first:**
```bash
docker compose run --rm backtest python run_backtest.py --force-download --universe-size 50 --years 2
```

**Using helper scripts:**

Linux/macOS:
```bash
./scripts/run_backtest.sh --years 1 --universe-size 20
./scripts/run_backtest.sh --years 5
```

Windows:
```powershell
.\scripts\run_backtest.ps1 --years 1 --universe-size 20
.\scripts\run_backtest.ps1 --years 5
```

**Output:** Backtest reports are generated at:
```
output/Backtest_Report.xlsx
```

### 4. Run Tests

```bash
docker compose run --rm test
```

Or using helper scripts:

Linux/macOS:
```bash
./scripts/run_tests.sh
```

Windows:
```powershell
.\scripts\run_tests.ps1
```

### 5. Optional Airflow Scheduler

For automated scheduling of daily agent runs and backtests.

**Build:**
```bash
docker compose --profile airflow build
```

**Initialize:**
```bash
docker compose --profile airflow run --rm airflow-init
```

**Start:**
```bash
docker compose --profile airflow up -d
```

**Open UI:**
Navigate to http://localhost:8080

**Default local login:**
- Username: `admin`
- Password: `admin`

**Trigger daily agent:**
```bash
docker compose --profile airflow exec airflow-scheduler airflow dags trigger afa_daily_agent
```

**Trigger small backtest:**
```bash
docker compose --profile airflow exec airflow-scheduler airflow dags trigger afa_backtest_small
```

**Trigger full backtest:**
```bash
docker compose --profile airflow exec airflow-scheduler airflow dags trigger afa_backtest_full
```

**Stop Airflow:**
```bash
docker compose --profile airflow down
```

**View Airflow logs:**
```bash
docker compose --profile airflow logs -f airflow-scheduler
```

### 6. Persistent Data

The following host folders persist results and state between Docker runs:

| Directory | Purpose |
|-----------|---------|
| `data/` | Agent brain (`agent_brain.json`), SQLite database (`portfolio_agent.db`), settings state |
| `data/market_data/` | Cached historical ticker parquet files (downloaded once, reused) |
| `output/` | Excel reports (`Agent_Orchestrator_Output.xlsx`, `Backtest_Report.xlsx`) |
| `logs/` | Runtime logs (`agent.log`, backtest logs, etc.) |

These directories are mounted as Docker volumes, so your data persists even if you remove containers.

### 7. Troubleshooting

**Docker daemon not running:**
- Start Docker Desktop (Windows/macOS) or the Docker service (Linux)
- Verify with: `docker info`

**Permission denied on Linux:**
```bash
sudo chown -R 1000:1000 data output logs
```
If using Airflow:
```bash
sudo chown -R 50000:0 data output logs airflow
```

**Backtest is slow:**
- Use `--universe-size 20` first for quick tests
- Full universe backtest downloads and processes many tickers
- Consider using `--force-download` only when needed

**Excel output missing:**
- Check logs for errors:
  ```bash
  docker compose logs agent
  docker compose logs backtest
  ```
- Verify the `output/` directory exists and is writable

**Windows line-ending errors:**
- Ensure `.gitattributes` exists in the repository
- Shell scripts should use LF line endings, not CRLF
- Use PowerShell scripts (`.ps1`) instead of bash scripts on Windows

**Module import errors:**
- Ensure `config.yaml` exists at project root
- Verify Docker volumes are mounted correctly in `docker-compose.yml`

**Airflow not starting:**
- Run initialization first: `docker compose --profile airflow run --rm airflow-init`
- Check logs: `docker compose --profile airflow logs airflow-init`

---

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
