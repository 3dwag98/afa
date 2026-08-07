# Portfolio Agent - Self-Learning Portfolio Optimization for Indian Markets

## Overview

Python self-learning portfolio agent for Indian markets.

- **Decision support only**: This system does NOT place real trades. It operates in paper trading / decision support mode only.
- **No real broker trading**: There is no broker API integration for trade execution.
- **Educational Purpose**: This tool is for educational and research purposes only. Past performance does not guarantee future results.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Run with synthetic data

```bash
python main.py --force-refresh
```

## Simulate outcome for learning demo

```bash
python main.py --simulate-outcome
```

## Update outcomes from market data

```bash
python main.py --update-outcomes
```

## Output

```
output/Agent_Orchestrator_Output.xlsx
```

## Learning

- Agent stores weights in `data/agent_brain.json`.
- Trade outcomes stored in `data/portfolio_agent.db`.
- Weights update based on WIN/LOSS outcomes.

## Guardrails

- Paper trading only.
- No leverage.
- No short selling.
- Penny stock filter.
- Position size cap.
- Risk per trade cap.

## Features

- Fetches historical market data from Yahoo Finance for Indian stocks (NSE/BSE)
- Technical indicator calculation
- Monte Carlo simulation for risk analysis
- Self-learning agent that stores outcomes and improves over time
- SQLite database for structured history
- JSON-based agent memory/brain
- Excel output with detailed analysis and recommendations

## Project Structure

```
portfolio_agent/
├── config.yaml          # Configuration file
├── main.py              # Entry point
├── requirements.txt     # Python dependencies
├── src/                 # Source modules
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
├── data/                # SQLite DB and agent brain JSON
├── output/              # Excel output files
└── logs/                # Log files
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

1. **Navigate to the project directory:**

```bash
cd portfolio_agent
```

2. **(Recommended) Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Verify installation (optional):**

```bash
pytest tests/test_scoring.py
```

## Usage

```bash
python main.py
```

The system will:
1. Load configuration from `config.yaml`
2. Fetch historical data for configured tickers
3. Run analysis and Monte Carlo simulations
4. Generate recommendations based on learned patterns
5. Output results to `output/Agent_Orchestrator_Output.xlsx`

## Configuration

Edit `config.yaml` to customize:
- Portfolio value (INR)
- Risk parameters
- Target tickers
- Simulation parameters
- File paths

## Learning Mechanism

The agent stores:
- Historical decisions and their outcomes
- Performance metrics per ticker
- Pattern recognition data

Over time, the agent adjusts its scoring based on past success/failure rates.

## Testing

```bash
pytest tests/
```

## License

MIT License - For educational purposes only.
