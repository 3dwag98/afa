"""Tests for storage module."""

import json
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from portfolio_agent.src.models import AgentBrain, Recommendation, TradeOutcome
from portfolio_agent.src.storage import (
    load_brain,
    save_brain,
    init_db,
    save_recommendations,
    save_trade_outcome,
    get_open_trades,
    get_trade_history,
    log_run,
)


class TestJSONBrainStorage:
    """Tests for JSON brain storage functions."""

    def test_default_brain_creation(self, tmp_path):
        """Test that default brain is created when file does not exist."""
        brain_file = tmp_path / "brain.json"
        
        # Load brain from non-existent path - should create default
        brain = load_brain(str(brain_file))
        
        # Verify default weights
        assert brain.weights == {
            "Trend": 25.0,
            "Breakout": 25.0,
            "Volume": 20.0,
            "MC_Prob": 30.0
        }
        assert brain.trade_history == []
        assert brain.learning_log == []
        assert brain.updated_at is not None
        
        # Verify file was created
        assert brain_file.exists()
        
        # Verify file contents
        with open(brain_file, 'r') as f:
            data = json.load(f)
        assert data["weights"] == brain.weights
        assert data["trade_history"] == []
        assert data["learning_log"] == []

    def test_save_load_brain_roundtrip(self, tmp_path):
        """Test saving and loading brain roundtrip."""
        brain_file = tmp_path / "brain.json"
        
        # Create a custom brain
        original_brain = AgentBrain(
            weights={"Trend": 30.0, "Breakout": 20.0, "Volume": 25.0, "MC_Prob": 25.0},
            trade_history=[{"trade_id": "t1", "profit": 100}],
            learning_log=[{"event": "test_event"}],
            updated_at="2024-01-01T00:00:00Z"
        )
        
        # Save the brain
        save_brain(str(brain_file), original_brain)
        
        # Load the brain back
        loaded_brain = load_brain(str(brain_file))
        
        # Verify data integrity
        assert loaded_brain.weights == original_brain.weights
        assert loaded_brain.trade_history == original_brain.trade_history
        assert loaded_brain.learning_log == original_brain.learning_log
        # Note: updated_at will be different since save_brain updates it


class TestSQLiteStorage:
    """Tests for SQLite storage functions."""

    def test_sqlite_table_creation(self, tmp_path):
        """Test that SQLite tables are created correctly."""
        db_file = tmp_path / "test.db"
        
        # Initialize database
        init_db(str(db_file))
        
        # Verify tables exist
        with sqlite3.connect(str(db_file)) as conn:
            cursor = conn.cursor()
            
            # Check recommendations table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='recommendations'")
            assert cursor.fetchone() is not None
            
            # Check trade_outcomes table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trade_outcomes'")
            assert cursor.fetchone() is not None
            
            # Check run_logs table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='run_logs'")
            assert cursor.fetchone() is not None
            
            # Verify recommendations table columns
            cursor.execute("PRAGMA table_info(recommendations)")
            columns = {row[1] for row in cursor.fetchall()}
            expected_columns = {
                "recommendation_id", "created_at", "symbol", "signal", "score",
                "trigger", "entry_price", "stop_price", "target_price", "reward_risk",
                "quantity", "investment_inr", "max_loss_inr", "mc_probability_profit",
                "mc_var_95_pct", "mc_cvar_95_pct", "compliance_status", "rationale"
            }
            assert expected_columns.issubset(columns)
            
            # Verify trade_outcomes table columns
            cursor.execute("PRAGMA table_info(trade_outcomes)")
            columns = {row[1] for row in cursor.fetchall()}
            expected_columns = {
                "trade_id", "recommendation_id", "symbol", "signal_trigger",
                "entry_date", "entry_price", "exit_date", "exit_price",
                "outcome", "return_pct", "outcome_source"
            }
            assert expected_columns.issubset(columns)
            
            # Verify run_logs table columns
            cursor.execute("PRAGMA table_info(run_logs)")
            columns = {row[1] for row in cursor.fetchall()}
            expected_columns = {"run_id", "run_at", "status", "message", "recommendations_count"}
            assert expected_columns.issubset(columns)

    def test_saving_and_reading_recommendations(self, tmp_path):
        """Test saving and reading recommendations."""
        db_file = tmp_path / "test.db"
        
        # Create test recommendations
        rec1 = Recommendation(
            symbol="AAPL",
            signal="BUY",
            score=0.85,
            trigger="breakout",
            entry_price=150.0,
            stop_price=145.0,
            target_price=165.0,
            reward_risk=3.0,
            quantity=100,
            investment_inr=125000.0,
            max_loss_inr=50000.0,
            mc_probability_profit=0.75,
            mc_var_95_pct=-0.05,
            mc_cvar_95_pct=-0.08,
            compliance_status="PASS",
            rationale="Strong breakout signal with volume confirmation",
            recommendation_id="rec-001",
            created_at="2024-01-15T10:00:00Z"
        )
        
        rec2 = Recommendation(
            symbol="GOOGL",
            signal="HOLD",
            score=0.60,
            trigger="trend_following",
            entry_price=140.0,
            stop_price=135.0,
            target_price=150.0,
            reward_risk=2.0,
            quantity=50,
            investment_inr=58000.0,
            max_loss_inr=25000.0,
            mc_probability_profit=0.60,
            mc_var_95_pct=-0.04,
            mc_cvar_95_pct=-0.06,
            compliance_status="PASS",
            rationale="Trend following signal",
            recommendation_id="rec-002",
            created_at="2024-01-15T11:00:00Z"
        )
        
        # Save recommendations
        save_recommendations(str(db_file), [rec1, rec2])
        
        # Verify data was saved by querying directly
        with sqlite3.connect(str(db_file)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM recommendations ORDER BY created_at")
            rows = cursor.fetchall()
            
            assert len(rows) == 2
            
            # Verify first recommendation
            row = rows[0]
            assert row["recommendation_id"] == "rec-001"
            assert row["symbol"] == "AAPL"
            assert row["signal"] == "BUY"
            assert row["score"] == 0.85
            assert row["trigger"] == "breakout"
            assert row["entry_price"] == 150.0
            assert row["compliance_status"] == "PASS"

    def test_saving_and_reading_trade_outcomes(self, tmp_path):
        """Test saving and reading trade outcomes."""
        db_file = tmp_path / "test.db"
        
        # Create test trade outcomes
        outcome1 = TradeOutcome(
            trade_id="trade-001",
            recommendation_id="rec-001",
            symbol="AAPL",
            signal_trigger="breakout",
            entry_date="2024-01-15",
            entry_price=150.0,
            exit_date="2024-01-20",
            exit_price=160.0,
            outcome="WIN",
            return_pct=6.67,
            outcome_source="backtest"
        )
        
        outcome2 = TradeOutcome(
            trade_id="trade-002",
            recommendation_id="rec-002",
            symbol="GOOGL",
            signal_trigger="trend_following",
            entry_date="2024-01-16",
            entry_price=140.0,
            exit_date="",
            exit_price=0.0,
            outcome="PENDING",
            return_pct=0.0,
            outcome_source="live"
        )
        
        # Save trade outcomes
        save_trade_outcome(str(db_file), outcome1)
        save_trade_outcome(str(db_file), outcome2)
        
        # Test get_trade_history
        history = get_trade_history(str(db_file))
        assert len(history) == 2
        
        # Test get_open_trades - should only return PENDING trades
        open_trades = get_open_trades(str(db_file))
        assert len(open_trades) == 1
        assert open_trades[0].trade_id == "trade-002"
        assert open_trades[0].outcome == "PENDING"
        
        # Verify trade data
        closed_trades = [t for t in history if t.outcome != "PENDING"]
        assert len(closed_trades) == 1
        assert closed_trades[0].symbol == "AAPL"
        assert closed_trades[0].return_pct == 6.67
        assert closed_trades[0].outcome == "WIN"

    def test_log_run(self, tmp_path):
        """Test logging runs to database."""
        db_file = tmp_path / "test.db"
        
        # Log a successful run
        log_run(str(db_file), "run-001", "SUCCESS", "Completed successfully", 5)
        
        # Log a failed run
        log_run(str(db_file), "run-002", "FAILED", "Error: Connection timeout", 0)
        
        # Verify logs
        with sqlite3.connect(str(db_file)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM run_logs ORDER BY run_at")
            rows = cursor.fetchall()
            
            assert len(rows) == 2
            
            # Verify first run
            row = rows[0]
            assert row["run_id"] == "run-001"
            assert row["status"] == "SUCCESS"
            assert row["message"] == "Completed successfully"
            assert row["recommendations_count"] == 5
            
            # Verify second run
            row = rows[1]
            assert row["run_id"] == "run-002"
            assert row["status"] == "FAILED"
            assert row["message"] == "Error: Connection timeout"
            assert row["recommendations_count"] == 0

    def test_directory_creation(self, tmp_path):
        """Test that directories are created automatically."""
        nested_dir = tmp_path / "nested" / "deep" / "path"
        db_file = nested_dir / "test.db"
        
        # This should create all parent directories
        init_db(str(db_file))
        
        assert db_file.exists()
        assert nested_dir.exists()
