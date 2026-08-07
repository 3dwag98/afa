"""Self-learning module for agent adaptation."""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional

# Use absolute imports for CLI execution
try:
    from .models import AgentBrain, TradeOutcome
    from .config import AppConfig
except ImportError:
    from models import AgentBrain, TradeOutcome
    from config import AppConfig


def evaluate_and_learn(brain: AgentBrain, config: AppConfig) -> AgentBrain:
    """Update agent signal weights based on trade outcomes.
    
    Args:
        brain: AgentBrain with trade_history and current weights.
        config: AppConfig with learning_rate and min_trades_for_learning.
    
    Returns:
        Updated AgentBrain with new weights and learning_log entry.
    """
    # Filter trades where outcome is WIN or LOSS (ignore OPEN/PENDING)
    realized_trades = [
        t for t in brain.trade_history 
        if t.get("outcome") in ("WIN", "LOSS")
    ]
    
    # Check minimum trades threshold
    if len(realized_trades) < config.min_trades_for_learning:
        # Add log entry without changing weights
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": "Not enough realized trades to learn."
        }
        brain.learning_log.append(log_entry)
        brain.updated_at = datetime.now().isoformat()
        return brain
    
    # Group by signal_trigger
    trigger_stats: Dict[str, Dict[str, int]] = {}
    for trade in realized_trades:
        trigger = trade.get("signal_trigger", "Unknown")
        if trigger not in trigger_stats:
            trigger_stats[trigger] = {"total": 0, "wins": 0}
        trigger_stats[trigger]["total"] += 1
        if trade.get("outcome") == "WIN":
            trigger_stats[trigger]["wins"] += 1
    
    # Start with existing weights
    new_weights: Dict[str, float] = dict(brain.weights)
    
    # Calculate win rates and weight adjustments for triggers with trades
    for trigger, stats in trigger_stats.items():
        total = stats["total"]
        wins = stats["wins"]
        win_rate = wins / total if total > 0 else 0.5
        
        # Get old weight (use existing or default)
        old_weight = brain.weights.get(trigger, 25.0)
        
        # Weight update formula
        adjustment = (win_rate - 0.5) * config.learning_rate
        new_weight = old_weight * (1 + adjustment)
        
        # Apply floor: minimum weight = 5.0
        new_weight = max(5.0, new_weight)
        
        new_weights[trigger] = new_weight
    
    # Normalize weights to sum exactly 100
    total_weight = sum(new_weights.values())
    if total_weight > 0:
        scale_factor = 100.0 / total_weight
        for trigger in new_weights:
            new_weights[trigger] = new_weights[trigger] * scale_factor
    
    # Round to 1 decimal place for cleaner output while ensuring sum is 100
    # First round all values
    rounded_weights = {k: round(v, 1) for k, v in new_weights.items()}
    
    # Adjust for rounding errors to ensure exact sum of 100
    current_sum = sum(rounded_weights.values())
    if abs(current_sum - 100.0) > 0.001:
        # Find the weight with largest fractional part and adjust
        diff = 100.0 - current_sum
        # Simply add the difference to the largest weight
        max_key = max(rounded_weights.keys(), key=lambda k: rounded_weights[k])
        rounded_weights[max_key] = round(rounded_weights[max_key] + diff, 1)
    
    brain.weights = rounded_weights
    
    # Create learning log entry
    log_parts = []
    for trigger in sorted(trigger_stats.keys()):
        stats = trigger_stats.get(trigger, {"total": 0, "wins": 0})
        total = stats["total"]
        wins = stats["wins"]
        wr_pct = int(round(wins / total * 100)) if total > 0 else 50
        wt = rounded_weights.get(trigger, 0)
        log_parts.append(f"{trigger} WR:{wr_pct}% (Wt:{wt:.1f})")
    
    log_entry = f"{datetime.now().strftime('%Y-%m-%d')} Learning Update: {' | '.join(log_parts)}"
    brain.learning_log.append({
        "timestamp": datetime.now().isoformat(),
        "entry": log_entry
    })
    
    # Update updated_at
    brain.updated_at = datetime.now().isoformat()
    
    return brain


class LearningAgent:
    """Self-learning agent that adapts based on historical outcomes."""

    def __init__(self, brain_data: Dict[str, Any], learning_rate: float = 0.15):
        """Initialize learning agent.

        Args:
            brain_data: Initial brain/memory data.
            learning_rate: Rate of learning adaptation.
        """
        self.brain = brain_data
        self.learning_rate = learning_rate
        self._ensure_structure()

    def _ensure_structure(self) -> None:
        """Ensure brain has required structure."""
        if 'pattern_weights' not in self.brain:
            self.brain['pattern_weights'] = {}
        if 'ticker_performance' not in self.brain:
            self.brain['ticker_performance'] = {}
        if 'learning_stats' not in self.brain:
            self.brain['learning_stats'] = {
                'total_decisions': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0
            }

    def record_outcome(self, decision_id: str, ticker: str, action: str,
                       entry_price: float, exit_price: float,
                       features: Dict[str, Any]) -> Dict[str, Any]:
        """Record a completed trade outcome.

        Args:
            decision_id: Unique decision identifier.
            ticker: Stock ticker.
            action: 'BUY' or 'SELL'.
            entry_price: Entry price.
            exit_price: Exit price.
            features: Features used for the decision.

        Returns:
            Outcome dictionary.
        """
        # Calculate P&L
        if action == 'BUY':
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        # Determine outcome
        is_win = pnl_pct > 0
        reward = pnl_pct if is_win else pnl_pct * 2  # Penalize losses more

        outcome = {
            'decision_id': decision_id,
            'ticker': ticker,
            'action': action,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'outcome': 'WIN' if is_win else 'LOSS',
            'reward': reward,
            'timestamp': datetime.now().isoformat(),
            'features': features
        }

        # Update learning stats
        self._update_stats(is_win, ticker)

        # Update pattern weights based on features
        self._update_pattern_weights(features, reward)

        # Update ticker-specific performance
        self._update_ticker_performance(ticker, pnl_pct)

        return outcome

    def _update_stats(self, is_win: bool, ticker: str) -> None:
        """Update overall learning statistics."""
        stats = self.brain['learning_stats']
        stats['total_decisions'] += 1

        if is_win:
            stats['wins'] += 1
        else:
            stats['losses'] += 1

        if stats['total_decisions'] > 0:
            stats['win_rate'] = stats['wins'] / stats['total_decisions']

    def _update_pattern_weights(self, features: Dict[str, Any], 
                                 reward: float) -> None:
        """Update weights for patterns/features based on reward.

        Args:
            features: Feature dictionary from decision.
            reward: Reward value (positive or negative).
        """
        for feature_name, feature_value in features.items():
            pattern_key = f"{feature_name}:{feature_value}"

            current_weight = self.brain['pattern_weights'].get(pattern_key, 1.0)

            # Adjust weight based on reward and learning rate
            adjustment = self.learning_rate * reward
            new_weight = current_weight + adjustment

            # Clamp weights between 0.1 and 3.0
            new_weight = max(0.1, min(3.0, new_weight))

            self.brain['pattern_weights'][pattern_key] = new_weight

    def _update_ticker_performance(self, ticker: str, pnl_pct: float) -> None:
        """Update ticker-specific performance tracking.

        Args:
            ticker: Stock ticker.
            pnl_pct: P&L percentage.
        """
        if ticker not in self.brain['ticker_performance']:
            self.brain['ticker_performance'][ticker] = {
                'trades': 0,
                'wins': 0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'win_rate': 0.0
            }

        perf = self.brain['ticker_performance'][ticker]
        perf['trades'] += 1
        perf['total_pnl'] += pnl_pct

        if pnl_pct > 0:
            perf['wins'] += 1

        perf['avg_pnl'] = perf['total_pnl'] / perf['trades']
        perf['win_rate'] = perf['wins'] / perf['trades']

    def get_adjusted_score(self, base_score: float, ticker: str,
                           features: Dict[str, Any]) -> float:
        """Get score adjusted by learned patterns.

        Args:
            base_score: Original calculated score.
            ticker: Stock ticker.
            features: Current features.

        Returns:
            Adjusted score.
        """
        adjustment_factor = 1.0

        # Apply pattern weight adjustments
        for feature_name, feature_value in features.items():
            pattern_key = f"{feature_name}:{feature_value}"
            weight = self.brain['pattern_weights'].get(pattern_key, 1.0)
            adjustment_factor *= weight

        # Apply ticker-specific bias
        ticker_perf = self.brain['ticker_performance'].get(ticker, {})
        ticker_win_rate = ticker_perf.get('win_rate', 0.5)
        if ticker_win_rate > 0:
            # Adjust based on historical performance with this ticker
            ticker_bias = 0.5 + (ticker_win_rate * 0.5)
            adjustment_factor *= ticker_bias

        # Apply global win rate adjustment
        global_win_rate = self.brain['learning_stats'].get('win_rate', 0.5)
        if global_win_rate > 0:
            confidence_factor = global_win_rate
            adjustment_factor *= (0.8 + 0.4 * confidence_factor)

        adjusted = base_score * adjustment_factor
        return max(0.0, min(1.0, adjusted))

    def should_learn_from(self, outcome: str, confidence: float,
                          target_confidence: float) -> bool:
        """Determine if agent should update weights from this outcome.

        Args:
            outcome: 'WIN' or 'LOSS'.
            confidence: Confidence level of original decision.
            target_confidence: Minimum confidence threshold.

        Returns:
            True if should learn from this outcome.
        """
        # Always learn from high-confidence mistakes
        if outcome == 'LOSS' and confidence >= target_confidence:
            return True

        # Learn from wins proportionally
        if outcome == 'WIN':
            return True

        return False

    def get_brain(self) -> Dict[str, Any]:
        """Get current brain state.

        Returns:
            Brain dictionary.
        """
        return self.brain.copy()

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get current learning statistics.

        Returns:
            Dictionary with learning statistics.
        """
        return self.brain.get('learning_stats', {}).copy()

    def save_brain(self, path: str) -> None:
        """Save brain to JSON file.

        Args:
            path: File path to save to.
        """
        with open(path, 'w') as f:
            json.dump(self.brain, f, indent=2, default=str)

    @classmethod
    def load_brain(cls, path: str, learning_rate: float = 0.15) -> 'LearningAgent':
        """Load agent from JSON file.

        Args:
            path: File path to load from.
            learning_rate: Learning rate.

        Returns:
            LearningAgent instance.
        """
        try:
            with open(path, 'r') as f:
                brain_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            brain_data = {}

        return cls(brain_data, learning_rate)
