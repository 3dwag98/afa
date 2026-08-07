"""Test orchestrator module using synthetic data only."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import pandas as pd
import numpy as np

# Add src to path
src_path = Path(__file__).parent.parent / "portfolio_agent" / "src"
sys.path.insert(0, str(src_path))

from src.config import AppConfig, clear_config_cache
from src.orchestrator import run_orchestrator
from src.storage import init_db, get_trade_history


def generate_synthetic_data(tickers: list[str], days: int = 300) -> dict[str, pd.DataFrame]:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    result = {}
    
    end_date = pd.Timestamp("2024-01-01")
    start_date = end_date - pd.Timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    
    for ticker in tickers:
        # Random walk for close prices starting at 100
        returns = np.random.normal(0.0005, 0.02, len(dates))
        close_prices = 100 * np.cumprod(1 + returns)
        
        # Generate high, low based on close
        daily_vol = np.abs(np.random.normal(0.01, 0.005, len(dates)))
        high_prices = close_prices * (1 + daily_vol)
        low_prices = close_prices * (1 - daily_vol)
        
        # Open is between low and high
        open_prices = low_prices + np.random.uniform(0, 1, len(dates)) * (high_prices - low_prices)
        
        # Ensure high >= low
        high_prices = np.maximum(high_prices, low_prices)
        
        # Volume: random positive values
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
    """Create temporary configuration for testing."""
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    # Create config with temp paths
    config_dict = {
        "portfolio_value_inr": 308733,
        "risk_per_trade_pct": 0.01,
        "max_single_position_pct": 0.03,
        "min_price_inr": 20,
        "target_prob_profit": 0.55,
        "min_reward_risk": 1.5,
        "learning_rate": 0.15,
        "min_trades_for_learning": 5,
        "mc_horizon_days": 20,
        "mc_simulations": 100,  # Reduced for faster tests
        "random_seed": 42,
        "tickers": ["TEST1.NS", "TEST2.NS", "TEST3.NS"],
        "brain_file": os.path.join(temp_dir, "test_brain.json"),
        "sqlite_path": os.path.join(temp_dir, "test_portfolio.db"),
        "excel_output": os.path.join(temp_dir, "test_output.xlsx"),
        "log_file": os.path.join(temp_dir, "test.log"),
        "paper_trading_mode": True,
        "min_history_days": 250,
        "allow_synthetic_fallback": False,
    }
    
    config = AppConfig.from_dict(config_dict)
    
    yield config
    
    # Cleanup
    try:
        for f in [config.brain_file, config.sqlite_path, config.excel_output, config.log_file]:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
    except Exception:
        pass
    
    clear_config_cache()


def test_orchestrator_with_synthetic_data(temp_config):
    """Test orchestrator runs successfully with synthetic data."""
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(temp_config.tickers)
    
    # Mock load_or_fetch_data to return synthetic data
    with patch('src.orchestrator.load_or_fetch_data', return_value=synthetic_data):
        # Run orchestrator with the temp config
        excel_path = run_orchestrator(
            force_refresh=False, 
            simulate_outcome=True,
            config=temp_config
        )
    
    # Assert Excel file exists
    assert os.path.exists(excel_path), f"Excel file not created at {excel_path}"
    assert excel_path == temp_config.excel_output


def test_orchestrator_creates_sqlite_recommendations(temp_config):
    """Test that orchestrator creates recommendations in SQLite."""
    # Initialize database first
    init_db(temp_config.sqlite_path)
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(temp_config.tickers)
    
    # Mock load_or_fetch_data to return synthetic data
    with patch('src.orchestrator.load_or_fetch_data', return_value=synthetic_data):
        # Run orchestrator with the temp config
        excel_path = run_orchestrator(
            force_refresh=False, 
            simulate_outcome=True,
            config=temp_config
        )
    
    # Check SQLite has recommendations
    import sqlite3
    
    conn = sqlite3.connect(temp_config.sqlite_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM recommendations")
    count = cursor.fetchone()[0]
    conn.close()
    
    assert count > 0, "No recommendations found in SQLite"
    assert count == len(temp_config.tickers), f"Expected {len(temp_config.tickers)} recommendations, got {count}"


def test_orchestrator_creates_brain_file(temp_config):
    """Test that orchestrator creates/updates brain file."""
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(temp_config.tickers)
    
    # Mock load_or_fetch_data to return synthetic data
    with patch('src.orchestrator.load_or_fetch_data', return_value=synthetic_data):
        # Run orchestrator with the temp config
        excel_path = run_orchestrator(
            force_refresh=False, 
            simulate_outcome=True,
            config=temp_config
        )
    
    # Assert brain file exists
    assert os.path.exists(temp_config.brain_file), f"Brain file not created at {temp_config.brain_file}"
    
    # Verify brain file content
    import json
    with open(temp_config.brain_file, 'r') as f:
        brain_data = json.load(f)
    
    assert "weights" in brain_data
    assert "trade_history" in brain_data
    assert "updated_at" in brain_data


def test_orchestrator_without_simulation(temp_config):
    """Test orchestrator runs without simulated outcomes."""
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(temp_config.tickers)
    
    # Mock load_or_fetch_data to return synthetic data
    with patch('src.orchestrator.load_or_fetch_data', return_value=synthetic_data):
        # Run orchestrator without simulation
        excel_path = run_orchestrator(
            force_refresh=False, 
            simulate_outcome=False,
            config=temp_config
        )
    
    # Assert Excel file exists
    assert os.path.exists(excel_path)
    
    # Check no simulated outcomes were added
    trade_outcomes = get_trade_history(temp_config.sqlite_path)
    simulated_count = sum(1 for o in trade_outcomes if o.outcome_source == "SIMULATED")
    assert simulated_count == 0, "Simulated outcome was added when it shouldn't be"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
