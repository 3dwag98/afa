"""Tests for the learning module."""

import pytest
from datetime import datetime
from typing import Dict, Any

from src.models import AgentBrain
from portfolio_agent.config.schema import AppConfig
from src.learning import evaluate_and_learn


def make_config(learning_rate: float = 0.15, min_trades: int = 5) -> AppConfig:
    """Create a test AppConfig."""
    return AppConfig.model_validate({
        "risk": {"portfolio_value_inr": 300000, "risk_per_trade_pct": 0.01, "max_single_position_pct": 0.03},
        "compliance": {"min_price_inr": 20, "target_prob_profit": 0.55, "min_reward_risk": 1.5, "paper_trading_mode": True},
        "learning": {"learning_rate": learning_rate, "min_trades_for_learning": min_trades},
        "simulation": {"mc_horizon_days": 20, "mc_simulations": 1000, "random_seed": 42},
        "data": {"tickers": ["NIFTYBEES.NS"], "min_history_days": 250},
        "paths": {
            "brain_file": "data/agent_brain.json",
            "sqlite_path": "data/portfolio_agent.db",
            "excel_output": "output/output.xlsx",
            "log_file": "logs/agent.log",
        },
    })


def make_trade(
    signal_trigger: str,
    outcome: str,
    entry_price: float = 100.0,
    exit_price: float = 110.0,
) -> Dict[str, Any]:
    """Create a test trade dictionary."""
    return {
        "trade_id": f"trade_{signal_trigger}_{outcome}",
        "recommendation_id": f"rec_{signal_trigger}",
        "symbol": "NIFTYBEES.NS",
        "signal_trigger": signal_trigger,
        "entry_date": "2024-01-01",
        "entry_price": entry_price,
        "exit_date": "2024-01-10",
        "exit_price": exit_price,
        "outcome": outcome,
        "return_pct": (exit_price - entry_price) / entry_price * 100,
        "outcome_source": "backtest",
    }


class TestEvaluateAndLearn:
    """Test cases for evaluate_and_learn function."""

    def test_empty_history_weights_unchanged(self):
        """Test case 1: Empty history - weights should remain unchanged."""
        brain = AgentBrain(
            weights={"Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0},
            trade_history=[],
            learning_log=[],
        )
        config = make_config(min_trades=5)

        original_weights = brain.weights.copy()
        result = evaluate_and_learn(brain, config)

        # Weights should be unchanged
        assert result.weights == original_weights
        # Should have logged that not enough trades exist
        assert any(
            "Not enough realized trades to learn" in str(entry)
            for entry in result.learning_log
        )

    def test_all_breakout_losses_weight_decreases(self):
        """Test case 2: All breakout losses - breakout weight decreases."""
        brain = AgentBrain(
            weights={"Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0},
            trade_history=[
                make_trade("Breakout", "LOSS", 100, 90 - i) for i in range(30)
            ],
            learning_log=[],
        )
        config = make_config(learning_rate=0.15, min_trades=5)

        original_breakout_weight = brain.weights["Breakout"]
        result = evaluate_and_learn(brain, config)

        # Breakout weight should decrease (all losses means win_rate = 0)
        assert result.weights["Breakout"] < original_breakout_weight

    def test_all_trend_wins_weight_increases(self):
        """Test case 3: All trend wins - trend weight increases."""
        brain = AgentBrain(
            weights={"Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0},
            # 30 trades, not 5: a component's weight only moves once its own
            # sample clears MIN_TRADES_PER_COMPONENT and a binomial test
            # rejects "coin flip". At n=5 a perfect record is not evidence.
            trade_history=[
                make_trade("Trend", "WIN", 100, 110 + i) for i in range(30)
            ],
            learning_log=[],
        )
        config = make_config(learning_rate=0.15, min_trades=5)

        original_trend_weight = brain.weights["Trend"]
        result = evaluate_and_learn(brain, config)

        # Trend weight should increase (all wins means win_rate = 1.0)
        assert result.weights["Trend"] > original_trend_weight

    def test_small_sample_does_not_move_weights(self):
        """A perfect 5-trade record is not evidence: at n=5 the win-rate
        standard error is +/-22pp, and this is a closed feedback loop."""
        brain = AgentBrain(
            weights={"Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0},
            trade_history=[make_trade("Trend", "WIN", 100, 110) for _ in range(5)],
            learning_log=[],
        )
        config = make_config(learning_rate=0.15, min_trades=5)

        result = evaluate_and_learn(brain, config)

        assert result.weights == brain.weights

    def test_insignificant_win_rate_does_not_move_weights(self):
        """35 trades at a 60% win rate does not clear a two-sided binomial
        test against 0.5, so the weight is held rather than nudged on noise."""
        brain = AgentBrain(
            weights={"Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0},
            trade_history=(
                [make_trade("Trend", "WIN", 100, 110) for _ in range(21)]
                + [make_trade("Trend", "LOSS", 100, 95) for _ in range(14)]
            ),
            learning_log=[],
        )
        config = make_config(learning_rate=0.15, min_trades=5)

        result = evaluate_and_learn(brain, config)

        assert result.weights["Trend"] == brain.weights["Trend"]

    def test_weights_sum_to_100(self):
        """Test case 4: Weights always sum to exactly 100."""
        brain = AgentBrain(
            weights={"Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0},
            trade_history=[
                make_trade("Trend", "WIN"),
                make_trade("Trend", "WIN"),
                make_trade("Breakout", "LOSS"),
                make_trade("Breakout", "WIN"),
                make_trade("Volume", "WIN"),
            ],
            learning_log=[],
        )
        config = make_config(min_trades=5)

        result = evaluate_and_learn(brain, config)

        # Weights must sum to exactly 100
        total = sum(result.weights.values())
        assert abs(total - 100.0) < 0.01, f"Weights sum to {total}, expected 100"

    def test_weights_never_below_5(self):
        """Test case 5: Weights never go below minimum of 5."""
        # Create scenario where one trigger has very poor performance
        brain = AgentBrain(
            weights={"Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0},
            trade_history=[
                # Many losses for Breakout to try to push weight down
                *[make_trade("Breakout", "LOSS") for _ in range(30)],
            ],
            learning_log=[],
        )
        config = make_config(learning_rate=0.5, min_trades=5)  # High learning rate

        result = evaluate_and_learn(brain, config)

        # All weights should be >= 5.0
        for trigger, weight in result.weights.items():
            assert weight >= 5.0, f"{trigger} weight is {weight}, expected >= 5.0"

    def test_open_trades_ignored(self):
        """Test that OPEN trades are ignored."""
        brain = AgentBrain(
            weights={"Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0},
            trade_history=[
                make_trade("Trend", "OPEN"),  # Should be ignored
                make_trade("Trend", "OPEN"),
                make_trade("Trend", "OPEN"),
                make_trade("Trend", "OPEN"),
                make_trade("Trend", "OPEN"),
            ],
            learning_log=[],
        )
        config = make_config(min_trades=5)

        original_weights = brain.weights.copy()
        result = evaluate_and_learn(brain, config)

        # OPEN trades should be ignored, so not enough realized trades
        assert result.weights == original_weights
        assert any(
            "Not enough realized trades to learn" in str(entry)
            for entry in result.learning_log
        )

    def test_updated_at_is_set(self):
        """Test that updated_at timestamp is set after learning."""
        brain = AgentBrain(
            weights={"Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0},
            trade_history=[
                make_trade("Trend", "WIN"),
                make_trade("Trend", "WIN"),
                make_trade("Trend", "WIN"),
                make_trade("Trend", "WIN"),
                make_trade("Trend", "WIN"),
            ],
            learning_log=[],
            updated_at=None,
        )
        config = make_config(min_trades=5)

        result = evaluate_and_learn(brain, config)

        assert result.updated_at is not None

    def test_learning_log_entry_created(self):
        """Test that learning log entry is created with proper format."""
        brain = AgentBrain(
            weights={"Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0},
            trade_history=[
                make_trade("Trend", "WIN"),
                make_trade("Trend", "WIN"),
                make_trade("Breakout", "LOSS"),
                make_trade("Breakout", "WIN"),
                make_trade("Volume", "WIN"),
            ],
            learning_log=[],
        )
        config = make_config(min_trades=5)

        result = evaluate_and_learn(brain, config)

        # Check that learning log has an entry
        assert len(result.learning_log) > 0
        # Find the entry with Learning Update
        update_entries = [
            e for e in result.learning_log if "Learning Update" in str(e)
        ]
        assert len(update_entries) > 0
        # Check format contains WR and Wt
        entry_str = str(update_entries[0])
        assert "WR:" in entry_str
        assert "Wt:" in entry_str
