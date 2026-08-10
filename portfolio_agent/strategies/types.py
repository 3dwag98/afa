"""Shared data types for the strategy layer.

These decouple strategy/risk math from whichever config representation is in use
and give every strategy implementation (rule-based, ML, or future additions) a
single, canonical input/output shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

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
    # Estimated per-leg friction as a fraction of turnover (brokerage, STT,
    # exchange and SEBI charges, GST, stamp duty and assumed slippage). Charged
    # against a signal's reward:risk before any quantity exists, so the
    # min_reward_risk gate compares net-of-cost trades — see
    # src/execution_sim.py::cost_fraction_per_side and
    # src/risk.py::net_reward_risk. Defaults match the platform's default
    # slippage assumption so hand-built RiskParams (tests, fixtures) are
    # cost-aware without extra wiring.
    buy_cost_pct: float = field(default_factory=lambda: _default_cost_pct("BUY"))
    sell_cost_pct: float = field(default_factory=lambda: _default_cost_pct("SELL"))

    @classmethod
    def from_app_config(cls, config: "AppConfig") -> "RiskParams":
        """Build RiskParams from the nested AppConfig."""
        from portfolio_agent.src.execution_sim import cost_fraction_per_side

        slippage = config.risk.slippage_pct_per_side
        return cls(
            target_prob_profit=config.compliance.target_prob_profit,
            min_reward_risk=config.compliance.min_reward_risk,
            min_price_inr=config.compliance.min_price_inr,
            portfolio_value_inr=config.risk.portfolio_value_inr,
            risk_per_trade_pct=config.risk.risk_per_trade_pct,
            max_single_position_pct=config.risk.max_single_position_pct,
            buy_cost_pct=cost_fraction_per_side("BUY", slippage),
            sell_cost_pct=cost_fraction_per_side("SELL", slippage),
        )


def _default_cost_pct(side: str) -> float:
    """Per-leg cost fraction at the platform's default slippage assumption.

    Imported lazily: strategies/types.py is imported by the strategy layer,
    which src/execution_sim.py must not depend on in reverse.
    """
    from portfolio_agent.src.execution_sim import cost_fraction_per_side

    return cost_fraction_per_side(side)


@dataclass
class StrategyContext:
    """Everything besides the feature DataFrame that score()/score_batch() need."""

    risk: RiskParams
    weights: Dict[str, float] = field(default_factory=dict)
    mc_result: Optional["MonteCarloResult"] = None
    run_id: Optional[str] = None
    # Benchmark index close series (e.g. the Nifty 50), truncated to the
    # decision date by the caller. The momentum crash filter prefers this over
    # its equal-weighted composite of the traded universe: "the market is below
    # its 200-day average" is a statement about the index the research actually
    # studied, and a composite of whatever happens to be in today's universe is
    # only a proxy for it. None when no benchmark is cached.
    benchmark_close: Optional["pd.Series"] = None


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
