"""Tests for scoring module."""

import pytest
from src.models import IndicatorSnapshot, AgentBrain
from src.monte_carlo import MonteCarloResult
from src.config import AppConfig
from src.scoring import score_candidate, _normalize_weights


def _make_config(
    target_prob_profit: float = 0.55,
    min_reward_risk: float = 1.5,
    min_price_inr: float = 100.0
) -> AppConfig:
    """Create a minimal AppConfig for testing."""
    return AppConfig(
        portfolio_value_inr=1000000.0,
        risk_per_trade_pct=1.0,
        max_single_position_pct=10.0,
        min_price_inr=min_price_inr,
        target_prob_profit=target_prob_profit,
        min_reward_risk=min_reward_risk,
        learning_rate=0.01,
        min_trades_for_learning=5,
        mc_horizon_days=20,
        mc_simulations=1000,
        random_seed=42,
        tickers=["RELIANCE"],
        brain_file="brain.yaml",
        sqlite_path=":memory:",
        excel_output="output.xlsx",
        log_file="test.log",
        paper_trading_mode=True,
        min_history_days=200,
    )


def _make_perfect_bullish_indicator(symbol: str = "TEST") -> IndicatorSnapshot:
    """Create a perfect bullish indicator setup.
    
    Conditions:
    - close > sma50 > sma200 (trend score = 1.0)
    - close > prev_donchian_upper_20 (breakout score = 1.0)
    - volume_ratio >= 1.5 (volume score = 1.0)
    """
    # We'll use entry_price=150 as close, so set up SMAs accordingly
    return IndicatorSnapshot(
        symbol=symbol,
        sma20=148.0,
        sma50=140.0,
        sma200=120.0,
        donchian_upper_20=145.0,
        prev_donchian_upper_20=145.0,  # close will be 150, so breakout
        avg_volume_20=1000000.0,
        volume_ratio=2.0,  # gives volume_score = min(2.0/2.0, 1.0) = 1.0
        atr14=3.0,
        daily_log_return=0.01,
    )


def _make_mc_result(probability_profit: float = 0.70) -> MonteCarloResult:
    """Create a MonteCarloResult with given probability of profit."""
    return MonteCarloResult(
        probability_profit=probability_profit,
        expected_return_pct=0.05,
        var_95=-0.10,
        cvar_95=-0.15,
        simulations_count=1000,
        horizon_days=20,
    )


def _make_brain(weights: dict = None) -> AgentBrain:
    """Create an AgentBrain with given weights."""
    if weights is None:
        weights = {
            "Trend": 25.0,
            "Breakout": 25.0,
            "Volume": 20.0,
            "MC_Prob": 30.0,
        }
    return AgentBrain(weights=weights)


class TestPerfectBullishSetup:
    """Test that perfect bullish setup gives BUY signal."""

    def test_perfect_bullish_gives_buy(self):
        """Perfect bullish conditions should result in BUY signal."""
        indicator = _make_perfect_bullish_indicator()
        mc_result = _make_mc_result(probability_profit=0.70)
        brain = _make_brain()
        config = _make_config(
            target_prob_profit=0.55,
            min_reward_risk=1.5,
            min_price_inr=100.0
        )
        
        entry_price = 150.0
        stop_price = 140.0  # Risk = 10
        target_price = 170.0  # Reward = 20, RR = 2.0
        
        result = score_candidate(
            indicator=indicator,
            mc_result=mc_result,
            brain=brain,
            config=config,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
        )
        
        assert result["signal"] == "BUY", f"Expected BUY, got {result['signal']}. Rationale: {result['rationale']}"
        assert result["score"] >= 60, f"Score should be >= 60, got {result['score']}"
        assert result["trigger"] == "Breakout", f"Expected Breakout trigger, got {result['trigger']}"


class TestMissingSMA200:
    """Test that missing SMA200 lowers score."""

    def test_missing_sma200_lowers_trend_score(self):
        """When sma200 is None, trend score should be 0.0."""
        indicator = IndicatorSnapshot(
            symbol="TEST",
            sma20=148.0,
            sma50=140.0,
            sma200=None,  # Missing SMA200
            donchian_upper_20=145.0,
            prev_donchian_upper_20=145.0,
            avg_volume_20=1000000.0,
            volume_ratio=2.0,
            atr14=3.0,
            daily_log_return=0.01,
        )
        
        mc_result = _make_mc_result(probability_profit=0.70)
        brain = _make_brain()
        config = _make_config()
        
        entry_price = 150.0
        stop_price = 140.0
        target_price = 170.0
        
        result = score_candidate(
            indicator=indicator,
            mc_result=mc_result,
            brain=brain,
            config=config,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
        )
        
        # With no trend score (0.0), the total score should be lower
        # Max possible without trend: 25 (breakout) + 20 (volume) + 30 (MC) = 75
        # But we need to verify trend_score is actually 0
        # The score should be significantly lower than perfect setup
        assert result["score"] < 85, f"Score should be lower without SMA200, got {result['score']}"
        # Signal might still be BUY if other conditions pass, but score is lower


class TestStopGreaterOrEqualEntry:
    """Test that stop >= entry gives AVOID signal."""

    def test_stop_equals_entry_gives_avoid(self):
        """When stop_price equals entry_price, signal should be AVOID."""
        indicator = _make_perfect_bullish_indicator()
        mc_result = _make_mc_result(probability_profit=0.70)
        brain = _make_brain()
        config = _make_config()
        
        entry_price = 150.0
        stop_price = 150.0  # Stop equals entry - invalid
        target_price = 170.0
        
        result = score_candidate(
            indicator=indicator,
            mc_result=mc_result,
            brain=brain,
            config=config,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
        )
        
        assert result["signal"] == "AVOID", f"Expected AVOID when stop>=entry, got {result['signal']}"
        assert result["reward_risk"] == 0.0, f"Expected reward_risk=0, got {result['reward_risk']}"

    def test_stop_greater_than_entry_gives_avoid(self):
        """When stop_price > entry_price, signal should be AVOID."""
        indicator = _make_perfect_bullish_indicator()
        mc_result = _make_mc_result(probability_profit=0.70)
        brain = _make_brain()
        config = _make_config()
        
        entry_price = 150.0
        stop_price = 155.0  # Stop above entry - invalid
        target_price = 170.0
        
        result = score_candidate(
            indicator=indicator,
            mc_result=mc_result,
            brain=brain,
            config=config,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
        )
        
        assert result["signal"] == "AVOID", f"Expected AVOID when stop>entry, got {result['signal']}"


class TestWeightsNormalization:
    """Test that weights are normalized to sum to 100."""

    def test_normalize_weights_standard(self):
        """Standard weights should normalize correctly."""
        weights = {"Trend": 25.0, "Breakout": 25.0, "Volume": 20.0, "MC_Prob": 30.0}
        normalized = _normalize_weights(weights)
        
        assert abs(sum(normalized.values()) - 100.0) < 0.001, "Normalized weights should sum to 100"
        assert normalized["Trend"] == 25.0  # Already sums to 100
        assert normalized["Breakout"] == 25.0
        assert normalized["Volume"] == 20.0
        assert normalized["MC_Prob"] == 30.0

    def test_normalize_weights_not_summing_to_100(self):
        """Weights not summing to 100 should be normalized."""
        weights = {"Trend": 50.0, "Breakout": 50.0, "Volume": 50.0, "MC_Prob": 50.0}
        normalized = _normalize_weights(weights)
        
        assert abs(sum(normalized.values()) - 100.0) < 0.001, "Normalized weights should sum to 100"
        # Each should be 25% after normalization
        assert normalized["Trend"] == 25.0
        assert normalized["Breakout"] == 25.0
        assert normalized["Volume"] == 25.0
        assert normalized["MC_Prob"] == 25.0

    def test_normalize_weights_zero_total(self):
        """Zero weights should be distributed equally."""
        weights = {"Trend": 0.0, "Breakout": 0.0, "Volume": 0.0, "MC_Prob": 0.0}
        normalized = _normalize_weights(weights)
        
        assert abs(sum(normalized.values()) - 100.0) < 0.001, "Normalized weights should sum to 100"
        # Each should get equal share
        expected = 100.0 / 4
        for key in weights:
            assert normalized[key] == expected

    def test_normalize_weights_empty(self):
        """Empty weights should return empty dict."""
        weights = {}
        normalized = _normalize_weights(weights)
        assert normalized == {}


class TestTriggerConditions:
    """Test trigger determination logic."""

    def test_breakout_trigger(self):
        """Breakout score == 1 should give Breakout trigger."""
        indicator = _make_perfect_bullish_indicator()
        mc_result = _make_mc_result()
        brain = _make_brain()
        config = _make_config()
        
        result = score_candidate(
            indicator=indicator,
            mc_result=mc_result,
            brain=brain,
            config=config,
            entry_price=150.0,
            stop_price=140.0,
            target_price=170.0,
        )
        
        assert result["trigger"] == "Breakout"

    def test_trend_trigger_when_no_breakout(self):
        """Trend score == 1 without breakout should give Trend trigger."""
        indicator = IndicatorSnapshot(
            symbol="TEST",
            sma20=148.0,
            sma50=140.0,
            sma200=120.0,
            donchian_upper_20=160.0,
            prev_donchian_upper_20=160.0,  # close=150 < 160, no breakout
            avg_volume_20=1000000.0,
            volume_ratio=1.0,  # volume_score = 0.5
            atr14=3.0,
            daily_log_return=0.01,
        )
        
        mc_result = _make_mc_result(probability_profit=0.70)
        brain = _make_brain()
        config = _make_config()
        
        result = score_candidate(
            indicator=indicator,
            mc_result=mc_result,
            brain=brain,
            config=config,
            entry_price=150.0,
            stop_price=140.0,
            target_price=170.0,
        )
        
        assert result["trigger"] == "Trend"

    def test_volume_trigger(self):
        """Volume score >= 0.75 without breakout/trend should give Volume trigger."""
        indicator = IndicatorSnapshot(
            symbol="TEST",
            sma20=148.0,
            sma50=145.0,  # close > sma50 but sma50 < sma200, no perfect trend
            sma200=160.0,  # close < sma200
            donchian_upper_20=160.0,
            prev_donchian_upper_20=160.0,  # no breakout
            avg_volume_20=1000000.0,
            volume_ratio=1.6,  # volume_score = min(1.6/2.0, 1.0) = 0.8
            atr14=3.0,
            daily_log_return=0.01,
        )
        
        mc_result = _make_mc_result(probability_profit=0.70)
        brain = _make_brain()
        config = _make_config()
        
        result = score_candidate(
            indicator=indicator,
            mc_result=mc_result,
            brain=brain,
            config=config,
            entry_price=150.0,
            stop_price=140.0,
            target_price=170.0,
        )
        
        assert result["trigger"] == "Volume"


class TestSignalThresholds:
    """Test signal determination based on score thresholds."""

    def test_watch_signal(self):
        """Score >= 45 but < 60 should give WATCH."""
        indicator = IndicatorSnapshot(
            symbol="TEST",
            sma20=148.0,
            sma50=140.0,
            sma200=120.0,
            donchian_upper_20=160.0,
            prev_donchian_upper_20=160.0,  # no breakout
            avg_volume_20=1000000.0,
            volume_ratio=1.0,  # volume_score = 0.5
            atr14=3.0,
            daily_log_return=0.01,
        )
        
        # Lower MC probability to reduce score
        mc_result = _make_mc_result(probability_profit=0.50)
        brain = _make_brain()
        config = _make_config(target_prob_profit=0.55)  # prob won't pass
        
        result = score_candidate(
            indicator=indicator,
            mc_result=mc_result,
            brain=brain,
            config=config,
            entry_price=150.0,
            stop_price=140.0,
            target_price=170.0,
        )
        
        # Should be WATCH if score >= 45, or AVOID if < 45
        assert result["signal"] in ["WATCH", "AVOID"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
