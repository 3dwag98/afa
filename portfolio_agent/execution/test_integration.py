"""Integration tests for portfolio agent orchestrator."""

import os
import json
import sqlite3
import tempfile
import shutil
from pathlib import Path

import pytest

from .orchestrator import run_orchestrator
from portfolio_agent.config.schema import AppConfig


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    temp_dir = tempfile.mkdtemp()
    
    data_dir = os.path.join(temp_dir, "data")
    output_dir = os.path.join(temp_dir, "output")
    logs_dir = os.path.join(temp_dir, "logs")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    yield {
        "temp_dir": temp_dir,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "logs_dir": logs_dir,
    }
    
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config(temp_dirs):
    """Create a test AppConfig with synthetic data fallback."""
    brain_file = os.path.join(temp_dirs["data_dir"], "agent_brain.json")
    sqlite_path = os.path.join(temp_dirs["data_dir"], "portfolio_agent.db")
    excel_output = os.path.join(temp_dirs["output_dir"], "Agent_Orchestrator_Output.xlsx")
    log_file = os.path.join(temp_dirs["logs_dir"], "test.log")
    
    return AppConfig.model_validate({
        "risk": {"portfolio_value_inr": 1000000.0, "risk_per_trade_pct": 0.01, "max_single_position_pct": 0.10},
        "compliance": {"min_price_inr": 100.0, "target_prob_profit": 0.55, "min_reward_risk": 1.5, "paper_trading_mode": True},
        "learning": {"learning_rate": 0.01, "min_trades_for_learning": 5},
        "simulation": {"mc_horizon_days": 20, "mc_simulations": 100, "random_seed": 42},
        "data": {
            "tickers": ["RELIANCE", "TCS", "INFY"],
            "min_history_days": 200,
            "allow_synthetic_fallback": True,
        },
        "paths": {
            "brain_file": brain_file,
            "sqlite_path": sqlite_path,
            "excel_output": excel_output,
            "log_file": log_file,
        },
    })


class TestIntegrationOrchestrator:
    """Integration tests for full orchestrator run with synthetic data."""

    def test_full_orchestrator_run(self, test_config, temp_dirs):
        """Test running full orchestrator with synthetic data."""
        # Run orchestrator with force_refresh to use synthetic data
        result = run_orchestrator(
            config=test_config,
            force_refresh=True,
            simulate_outcome=False,
        )
        
        assert result is not None
        
        # Assert output Excel exists
        assert os.path.exists(test_config.paths.excel_output), \
            f"Excel output file should exist at {test_config.paths.excel_output}"
        
        # Assert SQLite database exists and contains recommendations
        assert os.path.exists(test_config.paths.sqlite_path), \
            f"SQLite database should exist at {test_config.paths.sqlite_path}"
        
        conn = sqlite3.connect(test_config.paths.sqlite_path)
        cursor = conn.cursor()
        
        # Check recommendations table has entries
        cursor.execute("SELECT COUNT(*) FROM recommendations")
        rec_count = cursor.fetchone()[0]
        assert rec_count > 0, "Recommendations table should have entries"
        
        conn.close()
        
        # Assert brain JSON exists
        assert os.path.exists(test_config.paths.brain_file), \
            f"Brain JSON file should exist at {test_config.paths.brain_file}"
        
        # Assert weights sum to 100
        with open(test_config.paths.brain_file, "r") as f:
            brain = json.load(f)
        
        assert "weights" in brain, "Brain should contain weights"
        weights = brain["weights"]
        
        total_weight = sum(weights.values())
        assert abs(total_weight - 100.0) < 0.01, \
            f"Weights should sum to 100, got {total_weight}"

    def test_orchestrator_with_simulation(self, test_config, temp_dirs):
        """Test orchestrator run with outcome simulation."""
        # Run orchestrator with simulation enabled
        result = run_orchestrator(
            config=test_config,
            force_refresh=True,
            simulate_outcome=True,
        )
        
        assert result is not None
        
        # Assert output Excel exists
        assert os.path.exists(test_config.paths.excel_output)
        
        # Assert SQLite contains trade outcomes from simulation
        conn = sqlite3.connect(test_config.paths.sqlite_path)
        cursor = conn.cursor()
        
        # Check trade_outcomes table has entries
        cursor.execute("SELECT COUNT(*) FROM trade_outcomes")
        outcome_count = cursor.fetchone()[0]
        assert outcome_count > 0, "Trade outcomes table should have entries from simulation"
        
        # Verify outcomes have WIN/LOSS status
        cursor.execute("SELECT DISTINCT outcome FROM trade_outcomes WHERE outcome IS NOT NULL")
        outcomes = [row[0] for row in cursor.fetchall()]
        assert len(outcomes) > 0, "Should have recorded outcomes"
        
        conn.close()
        
        # Assert brain JSON exists and weights sum to 100
        assert os.path.exists(test_config.paths.brain_file)
        
        with open(test_config.paths.brain_file, "r") as f:
            brain = json.load(f)
        
        weights = brain["weights"]
        total_weight = sum(weights.values())
        assert abs(total_weight - 100.0) < 0.01, \
            f"Weights should sum to 100, got {total_weight}"

    def test_weights_persist_across_runs(self, test_config, temp_dirs):
        """Test that weights persist and update across multiple runs."""
        # First run
        run_orchestrator(
            config=test_config,
            force_refresh=True,
            simulate_outcome=True,
        )
        
        # Load initial weights
        with open(test_config.paths.brain_file, "r") as f:
            initial_brain = json.load(f)
        initial_weights = dict(initial_brain["weights"])
        
        # Second run with simulation (should potentially update weights)
        run_orchestrator(
            config=test_config,
            force_refresh=True,
            simulate_outcome=True,
        )
        
        # Load updated weights
        with open(test_config.paths.brain_file, "r") as f:
            updated_brain = json.load(f)
        updated_weights = dict(updated_brain["weights"])
        
        # Weights should still sum to 100
        total = sum(updated_weights.values())
        assert abs(total - 100.0) < 0.01, \
            f"Weights should sum to 100 after multiple runs, got {total}"
