"""Storage module for SQLite and JSON persistence."""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from portfolio_agent.src.models import AgentBrain, Recommendation, TradeOutcome


def _ensure_directory(path: str) -> None:
    """Ensure the directory for a file path exists."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)


# =============================================================================
# JSON Brain Storage Functions
# =============================================================================

def load_brain(path: str) -> AgentBrain:
    """Load agent brain from JSON file.
    
    If file does not exist, create default brain.
    
    Args:
        path: Path to JSON brain file.
        
    Returns:
        AgentBrain object with loaded or default data.
    """
    _ensure_directory(path)
    brain_path = Path(path)
    
    if brain_path.exists():
        with open(brain_path, 'r') as f:
            data = json.load(f)
        return AgentBrain(
            weights=data.get("weights", {
                "Trend": 25.0,
                "Breakout": 25.0,
                "Volume": 20.0,
                "MC_Prob": 30.0
            }),
            trade_history=data.get("trade_history", []),
            learning_log=data.get("learning_log", []),
            updated_at=data.get("updated_at")
        )
    else:
        # Create default brain
        default_brain = AgentBrain(
            weights={
                "Trend": 25.0,
                "Breakout": 25.0,
                "Volume": 20.0,
                "MC_Prob": 30.0
            },
            trade_history=[],
            learning_log=[],
            updated_at=datetime.now(timezone.utc).isoformat()
        )
        save_brain(path, default_brain)
        return default_brain


def save_brain(path: str, brain: AgentBrain) -> None:
    """Save agent brain to JSON file.
    
    Args:
        path: Path to JSON brain file.
        brain: AgentBrain object to save.
    """
    _ensure_directory(path)
    
    data = {
        "weights": brain.weights,
        "trade_history": brain.trade_history,
        "learning_log": brain.learning_log,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


# =============================================================================
# SQLite Storage Functions
# =============================================================================

def init_db(sqlite_path: str) -> None:
    """Initialize SQLite database with required tables.
    
    Creates tables if they do not exist:
    - recommendations
    - trade_outcomes
    - run_logs
    
    Args:
        sqlite_path: Path to SQLite database file.
    """
    _ensure_directory(sqlite_path)
    
    with sqlite3.connect(sqlite_path) as conn:
        cursor = conn.cursor()
        
        # Create recommendations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                recommendation_id TEXT PRIMARY KEY,
                created_at TEXT,
                symbol TEXT,
                signal TEXT,
                score REAL,
                trigger TEXT,
                entry_price REAL,
                stop_price REAL,
                target_price REAL,
                reward_risk REAL,
                quantity INTEGER,
                investment_inr REAL,
                max_loss_inr REAL,
                mc_probability_profit REAL,
                mc_var_95_pct REAL,
                mc_cvar_95_pct REAL,
                compliance_status TEXT,
                rationale TEXT
            )
        """)
        
        # Create trade_outcomes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_outcomes (
                trade_id TEXT PRIMARY KEY,
                recommendation_id TEXT,
                symbol TEXT,
                signal_trigger TEXT,
                entry_date TEXT,
                entry_price REAL,
                exit_date TEXT,
                exit_price REAL,
                outcome TEXT,
                return_pct REAL,
                outcome_source TEXT
            )
        """)
        
        # Create run_logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS run_logs (
                run_id TEXT PRIMARY KEY,
                run_at TEXT,
                status TEXT,
                message TEXT,
                recommendations_count INTEGER
            )
        """)
        
        conn.commit()


def save_recommendations(sqlite_path: str, recommendations: List[Recommendation]) -> None:
    """Save recommendations to SQLite database.
    
    Args:
        sqlite_path: Path to SQLite database file.
        recommendations: List of Recommendation objects to save.
    """
    _ensure_directory(sqlite_path)
    init_db(sqlite_path)
    
    with sqlite3.connect(sqlite_path) as conn:
        cursor = conn.cursor()
        
        for rec in recommendations:
            # Generate ID if not present
            rec_id = rec.recommendation_id or str(uuid.uuid4())
            created_at = rec.created_at or datetime.now(timezone.utc).isoformat()
            
            cursor.execute("""
                INSERT OR REPLACE INTO recommendations 
                (recommendation_id, created_at, symbol, signal, score, trigger,
                 entry_price, stop_price, target_price, reward_risk, quantity,
                 investment_inr, max_loss_inr, mc_probability_profit,
                 mc_var_95_pct, mc_cvar_95_pct, compliance_status, rationale)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rec_id,
                created_at,
                rec.symbol,
                rec.signal,
                rec.score,
                rec.trigger,
                rec.entry_price,
                rec.stop_price,
                rec.target_price,
                rec.reward_risk,
                rec.quantity,
                rec.investment_inr,
                rec.max_loss_inr,
                rec.mc_probability_profit,
                rec.mc_var_95_pct,
                rec.mc_cvar_95_pct,
                rec.compliance_status,
                rec.rationale
            ))
        
        conn.commit()


def save_trade_outcome(sqlite_path: str, outcome: TradeOutcome) -> None:
    """Save a trade outcome to SQLite database.
    
    Args:
        sqlite_path: Path to SQLite database file.
        outcome: TradeOutcome object to save.
    """
    _ensure_directory(sqlite_path)
    init_db(sqlite_path)
    
    with sqlite3.connect(sqlite_path) as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO trade_outcomes 
            (trade_id, recommendation_id, symbol, signal_trigger, entry_date,
             entry_price, exit_date, exit_price, outcome, return_pct, outcome_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            outcome.trade_id,
            outcome.recommendation_id,
            outcome.symbol,
            outcome.signal_trigger,
            outcome.entry_date,
            outcome.entry_price,
            outcome.exit_date,
            outcome.exit_price,
            outcome.outcome,
            outcome.return_pct,
            outcome.outcome_source
        ))
        
        conn.commit()


def get_open_trades(sqlite_path: str) -> List[TradeOutcome]:
    """Get all open trades from SQLite database.
    
    Open trades are those with outcome = 'PENDING'.
    
    Args:
        sqlite_path: Path to SQLite database file.
        
    Returns:
        List of TradeOutcome objects for open trades.
    """
    init_db(sqlite_path)
    
    with sqlite3.connect(sqlite_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM trade_outcomes WHERE outcome = 'PENDING'
        """)
        
        rows = cursor.fetchall()
        return [TradeOutcome(
            trade_id=row["trade_id"],
            recommendation_id=row["recommendation_id"],
            symbol=row["symbol"],
            signal_trigger=row["signal_trigger"],
            entry_date=row["entry_date"],
            entry_price=row["entry_price"],
            exit_date=row["exit_date"],
            exit_price=row["exit_price"],
            outcome=row["outcome"],
            return_pct=row["return_pct"],
            outcome_source=row["outcome_source"]
        ) for row in rows]


def get_trade_history(sqlite_path: str) -> List[TradeOutcome]:
    """Get all trade outcomes from SQLite database.
    
    Args:
        sqlite_path: Path to SQLite database file.
        
    Returns:
        List of all TradeOutcome objects.
    """
    init_db(sqlite_path)
    
    with sqlite3.connect(sqlite_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM trade_outcomes ORDER BY entry_date DESC
        """)
        
        rows = cursor.fetchall()
        return [TradeOutcome(
            trade_id=row["trade_id"],
            recommendation_id=row["recommendation_id"],
            symbol=row["symbol"],
            signal_trigger=row["signal_trigger"],
            entry_date=row["entry_date"],
            entry_price=row["entry_price"],
            exit_date=row["exit_date"],
            exit_price=row["exit_price"],
            outcome=row["outcome"],
            return_pct=row["return_pct"],
            outcome_source=row["outcome_source"]
        ) for row in rows]


def log_run(sqlite_path: str, run_id: str, status: str, 
            message: str, recommendations_count: int) -> None:
    """Log a run to the SQLite database.
    
    Args:
        sqlite_path: Path to SQLite database file.
        run_id: Unique identifier for this run.
        status: Status of the run (e.g., 'SUCCESS', 'FAILED').
        message: Message describing the run outcome.
        recommendations_count: Number of recommendations generated.
    """
    _ensure_directory(sqlite_path)
    init_db(sqlite_path)
    
    with sqlite3.connect(sqlite_path) as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO run_logs 
            (run_id, run_at, status, message, recommendations_count)
            VALUES (?, ?, ?, ?, ?)
        """, (
            run_id,
            datetime.now(timezone.utc).isoformat(),
            status,
            message,
            recommendations_count
        ))
        
        conn.commit()
