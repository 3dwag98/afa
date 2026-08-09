"""Stock scoring module."""

import logging
import pandas as pd
from typing import Dict, Any, Optional

# Use absolute imports for CLI execution
try:
    from .models import IndicatorSnapshot, AgentBrain, Recommendation
    from .config import AppConfig
    from .monte_carlo import MonteCarloResult
    from .logging_utils import get_logger, ContextualLogger
except ImportError:
    from models import IndicatorSnapshot, AgentBrain, Recommendation
    from config import AppConfig
    from monte_carlo import MonteCarloResult
    from logging_utils import get_logger, ContextualLogger


def _get_logger(run_id: Optional[str] = None, log_file: str = "logs/afa_pipeline.log") -> ContextualLogger:
    """Get a contextual logger for scoring module."""
    return get_logger(
        module_name='scoring',
        log_file=log_file,
        run_id=run_id,
        worker_id='main',
        level=logging.INFO
    )


def calculate_technical_score(df: pd.DataFrame) -> float:
    """Calculate technical analysis score.

    Args:
        df: DataFrame with indicator columns.

    Returns:
        Score between 0 and 1.
    """
    if df.empty or len(df) < 50:
        return 0.5

    latest = df.iloc[-1]
    score = 0.0
    max_score = 5.0

    # RSI scoring (30-70 range is neutral)
    rsi = latest.get('RSI', 50)
    if 40 <= rsi <= 60:
        score += 1.0
    elif rsi < 30:  # Oversold - potential buy
        score += 2.0
    elif rsi > 70:  # Overbought - potential sell
        score += 0.0
    else:
        score += 0.5

    # MACD scoring
    macd_hist = latest.get('MACD_Hist', 0)
    if macd_hist > 0:
        score += 1.5
    elif macd_hist < 0:
        score += 0.5
    else:
        score += 1.0

    # Price vs SMA scoring
    close = latest.get('Close', 0)
    sma_20 = latest.get('SMA_20', close)
    sma_50 = latest.get('SMA_50', close)

    if close > sma_20 > sma_50:  # Uptrend
        score += 1.5
    elif close < sma_20 < sma_50:  # Downtrend
        score += 0.0
    else:
        score += 0.75

    # Bollinger Bands position
    bb_lower = latest.get('BB_Lower', close * 0.95)
    bb_upper = latest.get('BB_Upper', close * 1.05)

    if close <= bb_lower:  # Near lower band - potential bounce
        score += 1.0
    elif close >= bb_upper:  # Near upper band - potential pullback
        score += 0.5
    else:
        score += 0.75

    return min(score / max_score, 1.0)


def calculate_momentum_score(df: pd.DataFrame, lookback: int = 20) -> float:
    """Calculate momentum score based on price change.

    Args:
        df: DataFrame with 'Close' column.
        lookback: Number of days for momentum calculation.

    Returns:
        Score between 0 and 1.
    """
    if len(df) < lookback:
        return 0.5

    returns = df['Close'].pct_change().iloc[-lookback:]
    cumulative_return = (1 + returns).prod() - 1

    # Normalize to 0-1 scale
    # Assume +/- 20% over lookback period as extremes
    normalized = (cumulative_return + 0.2) / 0.4
    return max(0.0, min(1.0, normalized))


def calculate_volume_score(df: pd.DataFrame, period: int = 20) -> float:
    """Calculate volume strength score.

    Args:
        df: DataFrame with 'Volume' column.
        period: Period for average volume calculation.

    Returns:
        Score between 0 and 1.
    """
    if len(df) < period:
        return 0.5

    avg_volume = df['Volume'].rolling(window=period).mean().iloc[-1]
    recent_volume = df['Volume'].iloc[-5:].mean()

    if avg_volume == 0:
        return 0.5

    volume_ratio = recent_volume / avg_volume

    # Higher volume indicates stronger conviction
    if volume_ratio > 1.5:
        return 1.0
    elif volume_ratio > 1.0:
        return 0.75
    elif volume_ratio > 0.8:
        return 0.5
    else:
        return 0.25


def calculate_combined_score(df: pd.DataFrame, 
                             weights: Dict[str, float] = None) -> float:
    """Calculate combined score from multiple factors.

    Args:
        df: DataFrame with price and indicator data.
        weights: Optional weights for each score component.

    Returns:
        Combined score between 0 and 1.
    """
    if weights is None:
        weights = {
            'technical': 0.4,
            'momentum': 0.35,
            'volume': 0.25
        }

    tech_score = calculate_technical_score(df)
    mom_score = calculate_momentum_score(df)
    vol_score = calculate_volume_score(df)

    combined = (
        weights['technical'] * tech_score +
        weights['momentum'] * mom_score +
        weights['volume'] * vol_score
    )

    return max(0.0, min(1.0, combined))


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize weights to sum to 100.
    
    Args:
        weights: Dictionary of weight names to values.
        
    Returns:
        Normalized weights that sum to 100.
    """
    total = sum(weights.values())
    if total == 0:
        # Return equal weights if all are zero
        n = len(weights)
        if n == 0:
            return {}
        return {k: 100.0 / n for k in weights}
    return {k: (v / total) * 100.0 for k, v in weights.items()}


def score_candidate(
    indicator: IndicatorSnapshot,
    mc_result: MonteCarloResult,
    brain: AgentBrain,
    config: AppConfig,
    entry_price: float,
    stop_price: float,
    target_price: float,
    run_id: Optional[str] = None
) -> Dict[str, Any]:
    """Score a stock candidate and generate recommendation.
    
    Args:
        indicator: Technical indicators snapshot.
        mc_result: Monte Carlo simulation result.
        brain: Agent brain with weights.
        config: Application configuration.
        entry_price: Proposed entry price.
        stop_price: Stop loss price.
        target_price: Target price.
        run_id: Unique run identifier for logging context.
        
    Returns:
        Recommendation-like dict with score, signal, trigger, and rationale.
    """
    logger = _get_logger(run_id=run_id, log_file=config.log_file)
    logger.debug(f"Scoring candidate {indicator.symbol}")
    
    try:
        close = indicator.sma20  # Use sma20 as proxy for close if not available
        # Actually, we need the actual close price. Let's use entry_price as close.
        close = entry_price
        
        # 1. Trend score
        sma50 = indicator.sma50
        sma200 = indicator.sma200
        
        if sma200 is None:
            trend_score = 0.0
        elif close > sma50 and close > sma200 and sma50 > sma200:
            trend_score = 1.0
        elif close > sma200:
            trend_score = 0.5
        else:
            trend_score = 0.0
        
        # 2. Breakout score
        prev_donchian_upper_20 = indicator.prev_donchian_upper_20
        if prev_donchian_upper_20 is None:
            breakout_score = 0.0
        elif close > prev_donchian_upper_20:
            breakout_score = 1.0
        else:
            breakout_score = 0.0
        
        # 3. Volume score
        volume_ratio = indicator.volume_ratio
        if volume_ratio is None:
            volume_score = 0.0
        else:
            volume_score = min(volume_ratio / 2.0, 1.0)
        
        # 4. MC score
        mc_score = max(0.0, min(1.0, mc_result.probability_profit))
        
        # Normalize weights to sum to 100
        raw_weights = brain.weights.copy()
        normalized_weights = _normalize_weights(raw_weights)
        
        weight_trend = normalized_weights.get("Trend", 0.0)
        weight_breakout = normalized_weights.get("Breakout", 0.0)
        weight_volume = normalized_weights.get("Volume", 0.0)
        weight_mc_prob = normalized_weights.get("MC_Prob", 0.0)
        
        # 5. Final score
        final_score = (
            weight_trend * trend_score +
            weight_breakout * breakout_score +
            weight_volume * volume_score +
            weight_mc_prob * mc_score
        )
        
        # 6. Trigger
        if breakout_score == 1.0:
            trigger = "Breakout"
        elif trend_score == 1.0:
            trigger = "Trend"
        elif volume_score >= 0.75:
            trigger = "Volume"
        else:
            trigger = "None"
        
        # Calculate reward_risk safely
        if entry_price > stop_price:
            reward = target_price - entry_price
            risk = entry_price - stop_price
            reward_risk = reward / risk if risk != 0 else 0.0
        else:
            # Stop >= entry is invalid
            reward_risk = 0.0
        
        # Build rationale
        rationale_parts = []
        
        # Check conditions for BUY signal
        prob_profit = mc_result.probability_profit
        target_prob_profit = config.target_prob_profit
        min_reward_risk = config.min_reward_risk
        min_price_inr = config.min_price_inr
        
        passed_score = final_score >= 60
        passed_prob = prob_profit >= target_prob_profit
        passed_rr = reward_risk >= min_reward_risk
        passed_price = close >= min_price_inr
        stop_valid = stop_price < entry_price
        
        rationale_parts.append(f"Score={final_score:.1f}")
        rationale_parts.append(f"trend={trend_score:.1f}")
        rationale_parts.append(f"breakout={breakout_score:.1f}")
        rationale_parts.append(f"volume={volume_score:.2f}")
        rationale_parts.append(f"mc_prob={mc_score:.2f}")
        
        if passed_score:
            rationale_parts.append("score>=60:PASS")
        else:
            rationale_parts.append("score>=60:FAIL")
        
        if passed_prob:
            rationale_parts.append(f"prob({prob_profit:.2f})>={target_prob_profit}:PASS")
        else:
            rationale_parts.append(f"prob({prob_profit:.2f})>={target_prob_profit}:FAIL")
        
        if passed_rr:
            rationale_parts.append(f"rr({reward_risk:.2f})>={min_reward_risk}:PASS")
        else:
            rationale_parts.append(f"rr({reward_risk:.2f})>={min_reward_risk}:FAIL")
        
        if passed_price:
            rationale_parts.append(f"price({close:.2f})>={min_price_inr}:PASS")
        else:
            rationale_parts.append(f"price({close:.2f})>={min_price_inr}:FAIL")
        
        if stop_valid:
            rationale_parts.append("stop<entry:VALID")
        else:
            rationale_parts.append("stop>=entry:INVALID")
        
        rationale = "; ".join(rationale_parts)
        
        # 7. Signal determination
        if not stop_valid:
            signal = "AVOID"
        elif passed_score and passed_prob and passed_rr and passed_price:
            signal = "BUY"
        elif final_score >= 45:
            signal = "WATCH"
        else:
            signal = "AVOID"
        
        logger.debug(f"Scored {indicator.symbol}: signal={signal}, score={final_score:.2f}, trigger={trigger}")
        
        return {
            "symbol": indicator.symbol,
            "signal": signal,
            "score": round(final_score, 2),
            "trigger": trigger,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "reward_risk": round(reward_risk, 4) if reward_risk != 0 else 0.0,
            "mc_probability_profit": round(prob_profit, 6),
            "rationale": rationale,
            # These fields are set by caller (quantity calculated elsewhere)
            "quantity": 0,
            "investment_inr": 0.0,
            "max_loss_inr": 0.0,
            "mc_var_95_pct": round(mc_result.var_95, 6),
            "mc_cvar_95_pct": round(mc_result.cvar_95, 6),
            "compliance_status": "PENDING",
        }
    except Exception as e:
        logger.exception(f"Error scoring candidate {indicator.symbol}: {e}")
        raise
