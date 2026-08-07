"""Storage module for SQLite and JSON persistence."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


class SQLiteStorage:
    """SQLite database storage for portfolio history."""

    def __init__(self, db_path: str):
        """Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        self._ensure_directory()
        self._init_db()

    def _ensure_directory(self) -> None:
        """Ensure the directory for the database exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    def _init_db(self) -> None:
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Decisions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id TEXT UNIQUE NOT NULL,
                    ticker TEXT NOT NULL,
                    action TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    outcome TEXT,
                    reward REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    features TEXT
                )
            """)

            # Positions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    avg_price REAL NOT NULL,
                    current_price REAL,
                    entry_date DATETIME NOT NULL,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'OPEN'
                )
            """)

            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    portfolio_value REAL,
                    daily_return REAL,
                    total_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()

    def save_decision(self, decision: Dict[str, Any]) -> None:
        """Save a trading decision to the database.

        Args:
            decision: Dictionary containing decision data.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO decisions 
                (decision_id, ticker, action, entry_price, exit_price, outcome, reward, features)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                decision.get('decision_id'),
                decision.get('ticker'),
                decision.get('action'),
                decision.get('entry_price'),
                decision.get('exit_price'),
                decision.get('outcome'),
                decision.get('reward'),
                json.dumps(decision.get('features', {}))
            ))
            conn.commit()

    def get_decisions(self, ticker: Optional[str] = None, 
                      limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve historical decisions.

        Args:
            ticker: Optional ticker filter.
            limit: Maximum number of records to return.

        Returns:
            List of decision dictionaries.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if ticker:
                cursor.execute("""
                    SELECT * FROM decisions 
                    WHERE ticker = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (ticker, limit))
            else:
                cursor.execute("""
                    SELECT * FROM decisions 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))

            rows = cursor.fetchall()
            decisions = []
            for row in rows:
                decision = dict(row)
                if decision.get('features'):
                    decision['features'] = json.loads(decision['features'])
                decisions.append(decision)

            return decisions

    def save_position(self, position: Dict[str, Any]) -> None:
        """Save or update a position.

        Args:
            position: Dictionary containing position data.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO positions 
                (ticker, quantity, avg_price, current_price, entry_date, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                position.get('ticker'),
                position.get('quantity'),
                position.get('avg_price'),
                position.get('current_price'),
                position.get('entry_date'),
                position.get('status', 'OPEN')
            ))
            conn.commit()

    def get_positions(self, status: str = 'OPEN') -> List[Dict[str, Any]]:
        """Retrieve positions by status.

        Args:
            status: Position status filter ('OPEN', 'CLOSED').

        Returns:
            List of position dictionaries.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM positions WHERE status = ?
            """, (status,))
            return [dict(row) for row in cursor.fetchall()]

    def save_performance(self, perf: Dict[str, Any]) -> None:
        """Save daily performance metrics.

        Args:
            perf: Dictionary containing performance data.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO performance 
                (date, portfolio_value, daily_return, total_return, sharpe_ratio, max_drawdown)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                perf.get('date'),
                perf.get('portfolio_value'),
                perf.get('daily_return'),
                perf.get('total_return'),
                perf.get('sharpe_ratio'),
                perf.get('max_drawdown')
            ))
            conn.commit()


class JSONBrain:
    """JSON-based agent memory/brain storage."""

    def __init__(self, brain_path: str):
        """Initialize JSON brain storage.

        Args:
            brain_path: Path to JSON brain file.
        """
        self.brain_path = brain_path
        self._ensure_directory()
        self.data = self._load_brain()

    def _ensure_directory(self) -> None:
        """Ensure the directory for the brain file exists."""
        brain_dir = Path(self.brain_path).parent
        brain_dir.mkdir(parents=True, exist_ok=True)

    def _load_brain(self) -> Dict[str, Any]:
        """Load brain data from JSON file.

        Returns:
            Dictionary containing brain data.
        """
        if Path(self.brain_path).exists():
            with open(self.brain_path, 'r') as f:
                return json.load(f)
        return {
            'decisions': [],
            'ticker_performance': {},
            'pattern_weights': {},
            'learning_stats': {
                'total_decisions': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0
            }
        }

    def _save_brain(self) -> None:
        """Save brain data to JSON file."""
        with open(self.brain_path, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)

    def add_decision(self, decision: Dict[str, Any]) -> None:
        """Add a decision to the brain.

        Args:
            decision: Decision dictionary to store.
        """
        self.data['decisions'].append({
            **decision,
            'stored_at': datetime.now().isoformat()
        })
        self._update_learning_stats(decision)
        self._save_brain()

    def _update_learning_stats(self, decision: Dict[str, Any]) -> None:
        """Update learning statistics based on decision outcome.

        Args:
            decision: Decision dictionary with outcome.
        """
        stats = self.data['learning_stats']
        stats['total_decisions'] += 1

        outcome = decision.get('outcome')
        if outcome == 'WIN':
            stats['wins'] += 1
        elif outcome == 'LOSS':
            stats['losses'] += 1

        if stats['total_decisions'] > 0:
            stats['win_rate'] = stats['wins'] / stats['total_decisions']

    def get_ticker_history(self, ticker: str) -> List[Dict[str, Any]]:
        """Get decision history for a specific ticker.

        Args:
            ticker: Ticker symbol.

        Returns:
            List of historical decisions for the ticker.
        """
        return [d for d in self.data['decisions'] if d.get('ticker') == ticker]

    def update_pattern_weight(self, pattern: str, weight: float) -> None:
        """Update weight for a recognized pattern.

        Args:
            pattern: Pattern identifier.
            weight: New weight value.
        """
        self.data['pattern_weights'][pattern] = weight
        self._save_brain()

    def get_pattern_weight(self, pattern: str, default: float = 1.0) -> float:
        """Get weight for a pattern.

        Args:
            pattern: Pattern identifier.
            default: Default weight if pattern not found.

        Returns:
            Pattern weight.
        """
        return self.data['pattern_weights'].get(pattern, default)

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get current learning statistics.

        Returns:
            Dictionary with learning statistics.
        """
        return self.data['learning_stats'].copy()
