"""Test orchestrator module using synthetic data only."""

import json
import os
import sqlite3
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from portfolio_agent.config.schema import AppConfig
from src.orchestrator import run_orchestrator
from src.storage import init_db, get_trade_history

TEST_TICKERS = ["TEST1.NS", "TEST2.NS", "TEST3.NS"]


def generate_synthetic_data(tickers: list[str], days: int = 300) -> dict[str, pd.DataFrame]:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    result = {}

    end_date = pd.Timestamp("2024-01-01")
    start_date = end_date - pd.Timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq="B")

    for ticker in tickers:
        returns = np.random.normal(0.0005, 0.02, len(dates))
        close_prices = 100 * np.cumprod(1 + returns)

        daily_vol = np.abs(np.random.normal(0.01, 0.005, len(dates)))
        high_prices = close_prices * (1 + daily_vol)
        low_prices = close_prices * (1 - daily_vol)

        open_prices = low_prices + np.random.uniform(0, 1, len(dates)) * (high_prices - low_prices)
        high_prices = np.maximum(high_prices, low_prices)

        volume = np.random.randint(100000, 10000000, len(dates)).astype(float)

        df = pd.DataFrame({
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        }, index=dates)
        df.index.name = "Date"
        result[ticker] = df

    return result


@pytest.fixture
def temp_config():
    """Create a temporary nested AppConfig for testing."""
    temp_dir = tempfile.mkdtemp()

    config = AppConfig.model_validate({
        "data": {"tickers": TEST_TICKERS, "min_history_days": 250, "allow_synthetic_fallback": False},
        "risk": {"portfolio_value_inr": 308733, "risk_per_trade_pct": 0.01, "max_single_position_pct": 0.03},
        "compliance": {
            "min_price_inr": 20,
            "target_prob_profit": 0.55,
            "min_reward_risk": 1.5,
            "paper_trading_mode": True,
        },
        "learning": {"learning_rate": 0.15, "min_trades_for_learning": 5},
        "simulation": {"mc_horizon_days": 20, "mc_simulations": 100, "random_seed": 42},
        "paths": {
            "brain_file": os.path.join(temp_dir, "test_brain.json"),
            "sqlite_path": os.path.join(temp_dir, "test_portfolio.db"),
            "excel_output": os.path.join(temp_dir, "test_output.xlsx"),
            "log_file": os.path.join(temp_dir, "test.log"),
        },
    })

    yield config

    for f in (config.paths.brain_file, config.paths.sqlite_path, config.paths.excel_output, config.paths.log_file):
        if os.path.exists(f):
            os.remove(f)
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)


def test_orchestrator_with_synthetic_data(temp_config):
    """Test orchestrator runs successfully with synthetic data."""
    synthetic_data = generate_synthetic_data(temp_config.data.tickers)

    with patch('src.orchestrator.load_or_fetch_data', return_value=synthetic_data):
        excel_path = run_orchestrator(
            force_refresh=False,
            simulate_outcome=True,
            config=temp_config
        )

    assert os.path.exists(excel_path), f"Excel file not created at {excel_path}"
    assert excel_path == temp_config.paths.excel_output


def test_orchestrator_creates_sqlite_recommendations(temp_config):
    """Test that orchestrator creates recommendations in SQLite."""
    init_db(temp_config.paths.sqlite_path)

    synthetic_data = generate_synthetic_data(temp_config.data.tickers)

    with patch('src.orchestrator.load_or_fetch_data', return_value=synthetic_data):
        run_orchestrator(
            force_refresh=False,
            simulate_outcome=True,
            config=temp_config
        )

    conn = sqlite3.connect(temp_config.paths.sqlite_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM recommendations")
    count = cursor.fetchone()[0]
    conn.close()

    assert count > 0, "No recommendations found in SQLite"
    assert count == len(temp_config.data.tickers)


def test_orchestrator_creates_brain_file(temp_config):
    """Test that orchestrator creates/updates brain file."""
    synthetic_data = generate_synthetic_data(temp_config.data.tickers)

    with patch('src.orchestrator.load_or_fetch_data', return_value=synthetic_data):
        run_orchestrator(
            force_refresh=False,
            simulate_outcome=True,
            config=temp_config
        )

    assert os.path.exists(temp_config.paths.brain_file)

    with open(temp_config.paths.brain_file, 'r') as f:
        brain_data = json.load(f)

    assert "weights" in brain_data
    assert "trade_history" in brain_data
    assert "updated_at" in brain_data


def test_orchestrator_without_simulation(temp_config):
    """Test orchestrator runs without simulated outcomes."""
    synthetic_data = generate_synthetic_data(temp_config.data.tickers)

    with patch('src.orchestrator.load_or_fetch_data', return_value=synthetic_data):
        excel_path = run_orchestrator(
            force_refresh=False,
            simulate_outcome=False,
            config=temp_config
        )

    assert os.path.exists(excel_path)

    trade_outcomes = get_trade_history(temp_config.paths.sqlite_path)
    simulated_count = sum(1 for o in trade_outcomes if o.outcome_source == "SIMULATED")
    assert simulated_count == 0, "Simulated outcome was added when it shouldn't be"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
