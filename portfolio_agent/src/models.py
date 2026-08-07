"""Data models for portfolio agent."""

from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime


@dataclass
class StockData:
    """Historical stock data model."""
    ticker: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: Optional[float] = None


@dataclass
class Signal:
    """Trading signal model."""
    ticker: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # 0.0 to 1.0
    price: float
    timestamp: datetime
    reason: str = ""


@dataclass
class Position:
    """Portfolio position model."""
    ticker: str
    quantity: int
    avg_price: float
    current_price: float
    entry_date: datetime
    last_updated: datetime


@dataclass
class Recommendation:
    """Portfolio recommendation model."""
    ticker: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    quantity: int
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    confidence: float = 0.0
    expected_return: float = 0.0
    risk_score: float = 0.0
    rationale: str = ""


@dataclass
class SimulationResult:
    """Monte Carlo simulation result model."""
    ticker: str
    mean_return: float
    std_return: float
    percentile_5: float
    percentile_95: float
    probability_profit: float
    simulations_count: int
    horizon_days: int


@dataclass
class AgentMemory:
    """Agent learning memory model."""
    decision_id: str
    ticker: str
    action: str
    entry_price: float
    exit_price: Optional[float] = None
    outcome: Optional[str] = None  # 'WIN', 'LOSS', 'PENDING'
    reward: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    features: dict = field(default_factory=dict)
