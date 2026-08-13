"""Tests for the learning module."""

import pytest
from datetime import datetime
from typing import Dict, Any

from portfolio_agent.src.models import AgentBrain
from portfolio_agent.config.schema import AppConfig
from portfolio_agent.src.learning import evaluate_and_learn


def make_config(
    learning_rate: float = 0.15,
    min_trades: int = 5,
    min_trades_per_component: int = 30,
    significance_level: float = 0.05,
) -> AppConfig:
    """Create a test AppConfig."""
    return AppConfig.model_validate({
        "risk": {"portfolio_value_inr": 300000, "risk_per_trade_pct": 0.01, "max_single_position_pct": 0.03},
        "compliance": {"min_price_inr": 20, "target_prob_profit": 0.55, "min_reward_risk": 1.5, "paper_trading_mode": True},
        "learning": {
            "learning_rate": learning_rate,
            "min_trades_for_learning": min_trades,
            "min_trades_per_component": min_trades_per_component,
            "significance_level": significance_level,
        },
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

    def test_a_five_trade_streak_does_not_move_any_weight(self):
        """The guard that makes this loop safe to close.

        Five straight wins is a 3% event under a coin flip, so it clears a
        significance test on its own — and at n=5 the win-rate standard error
        is ~22 percentage points, which is exactly the lucky streak the sample
        floor exists to refuse. Weights adapting on this would feed a noise
        estimate back into the signals that generate the next trades.
        """
        brain = AgentBrain(
            weights={"Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0},
            trade_history=[make_trade("Trend", "WIN", 100, 110) for _ in range(5)],
            learning_log=[],
        )

        result = evaluate_and_learn(brain, make_config(min_trades=5))

        assert result.weights == {
            "Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0
        }

    def test_a_large_but_insignificant_sample_does_not_move_any_weight(self):
        """52 wins in 100 trades is not evidence of an edge, and moving weights
        on it is how a feedback loop chases noise."""
        brain = AgentBrain(
            weights={"Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0},
            trade_history=(
                [make_trade("Trend", "WIN", 100, 110) for _ in range(52)]
                + [make_trade("Trend", "LOSS", 100, 95) for _ in range(48)]
            ),
            learning_log=[],
        )

        result = evaluate_and_learn(brain, make_config(min_trades=5))

        assert result.weights["Trend"] == 25.0

    def test_shrinkage_damps_the_size_of_the_move(self):
        """A 70% win rate over 40 trades is read as ~63%, not 70% — the same
        Beta prior the Kelly path applies, so the two finally agree about what
        a win rate is worth."""
        history = (
            [make_trade("Trend", "WIN", 100, 110) for _ in range(28)]
            + [make_trade("Trend", "LOSS", 100, 95) for _ in range(12)]
        )

        shrunk = evaluate_and_learn(
            AgentBrain(
                weights={"Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0},
                trade_history=history, learning_log=[],
            ),
            make_config(min_trades=5),
        )
        config_raw = make_config(min_trades=5)
        config_raw.learning.shrinkage_strength = 0.0
        raw = evaluate_and_learn(
            AgentBrain(
                weights={"Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0},
                trade_history=history, learning_log=[],
            ),
            config_raw,
        )

        assert 25.0 < shrunk.weights["Trend"] < raw.weights["Trend"]

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
                make_trade("Breakout", "LOSS"),
                make_trade("Breakout", "LOSS"),
                make_trade("Breakout", "LOSS"),
                make_trade("Breakout", "LOSS"),
                make_trade("Breakout", "LOSS"),
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


class TestBinomialTailProbability:
    """Exact, not approximate: the samples this gate is asked about are small,
    which is where a normal approximation is worst and where a spurious
    'significant' verdict does the most damage."""

    def test_matches_the_closed_form_for_a_clean_streak(self):
        from portfolio_agent.strategies.weighting import binomial_tail_probability

        # P(X >= 5 | n=5, p=0.5) = 0.5^5
        assert binomial_tail_probability(5, 5) == pytest.approx(0.03125)
        # P(X >= 8 | n=10, p=0.5) = (45 + 10 + 1)/1024
        assert binomial_tail_probability(8, 10) == pytest.approx(56 / 1024)

    def test_a_coin_flip_result_is_never_significant(self):
        from portfolio_agent.strategies.weighting import binomial_tail_probability

        assert binomial_tail_probability(50, 100) > 0.5
        assert binomial_tail_probability(52, 100) > 0.05

    def test_a_real_edge_clears_the_gate_once_the_sample_is_large_enough(self):
        from portfolio_agent.strategies.weighting import binomial_tail_probability

        # The same 60% win rate: noise at 20 trades, evidence at 200.
        assert binomial_tail_probability(12, 20) > 0.05
        assert binomial_tail_probability(120, 200) < 0.05

    def test_degenerate_input_refuses_to_answer(self):
        from portfolio_agent.strategies.weighting import binomial_tail_probability

        assert binomial_tail_probability(0, 0) == 1.0
        assert binomial_tail_probability(0, 10) == 1.0
        assert binomial_tail_probability(11, 10) == 0.0

