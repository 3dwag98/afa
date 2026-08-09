"""Shared data types for the strategy layer.

These decouple strategy/risk math from whichever config representation is in use
and give every strategy implementation (rule-based, ML, or future additions) a
single, canonical input/output shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from portfolio_agent.config.schema import AppConfig
    from src.monte_carlo import MonteCarloResult


@dataclass
class RiskParams:
    """Risk/eligibility parameters a strategy needs to size and gate a trade.

    Plain-value container so strategy code never has to know whether it was
    constructed from the live AppConfig, a backtest config, or a test fixture.
    """

    target_prob_profit: float
    min_reward_risk: float
    min_price_inr: float
    portfolio_value_inr: float
    risk_per_trade_pct: float
    max_single_position_pct: float
    atr_stop_multiplier: float = 1.5
    atr_target_multiplier: float = 2.0

    @classmethod
    def from_app_config(cls, config: "AppConfig") -> "RiskParams":
        """Build RiskParams from the nested AppConfig."""
        return cls(
            target_prob_profit=config.compliance.target_prob_profit,
            min_reward_risk=config.compliance.min_reward_risk,
            min_price_inr=config.compliance.min_price_inr,
            portfolio_value_inr=config.risk.portfolio_value_inr,
            risk_per_trade_pct=config.risk.risk_per_trade_pct,
            max_single_position_pct=config.risk.max_single_position_pct,
        )


@dataclass
class StrategyContext:
    """Everything besides the feature DataFrame that score()/score_batch() need."""

    risk: RiskParams
    weights: Dict[str, float] = field(default_factory=dict)
    mc_result: Optional["MonteCarloResult"] = None
    run_id: Optional[str] = None


@dataclass
class StrategySignal:
    """Canonical strategy output — the single shape both rule-based and ML
    strategies produce, consumed identically by the live orchestrator and the
    backtest engine."""

    symbol: str
    signal: str  # "BUY" | "SELL" | "HOLD" | "WATCH" | "AVOID"
    score: float  # 0-100 canonical scale
    trigger: str  # "Trend" | "Breakout" | "Volume" | "MC_Prob" | "Model" | "None"
    entry_price: float
    stop_price: float
    target_price: float
    reward_risk: float
    probability_profit: float
    component_scores: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
