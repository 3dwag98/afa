"""Data models for portfolio agent."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
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
    symbol: str
    signal: str  # 'BUY', 'SELL', 'HOLD'
    score: float
    trigger: str
    entry_price: float
    stop_price: float
    target_price: float
    reward_risk: float
    quantity: int
    investment_inr: float
    max_loss_inr: float
    mc_probability_profit: float
    mc_var_95_pct: float
    mc_cvar_95_pct: float
    compliance_status: str
    rationale: str
    recommendation_id: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class TradeOutcome:
    """Trade outcome model."""
    trade_id: str
    recommendation_id: str
    symbol: str
    signal_trigger: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    outcome: str  # 'WIN', 'LOSS', 'PENDING'
    return_pct: float
    outcome_source: str


@dataclass
class AgentBrain:
    """Agent brain/learning state model."""
    weights: Dict[str, float] = field(default_factory=lambda: {
        "Trend": 25.0,
        "Breakout": 25.0,
        "Volume": 20.0,
        "MC_Prob": 30.0
    })
    trade_history: List[Dict[str, Any]] = field(default_factory=list)
    learning_log: List[Dict[str, Any]] = field(default_factory=list)
    updated_at: Optional[str] = None


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
