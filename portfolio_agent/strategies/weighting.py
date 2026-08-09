"""Pluggable weight combination and self-learning weight adaptation.

Extracted so the win-rate-based weight adaptation (previously hardcoded across
scoring.py/learning.py) works against any rule-based strategy's component
scores, not one hardcoded implementation. These are pure functions: no
AppConfig, no AgentBrain, no I/O.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize weights to sum to 100.

    Args:
        weights: Dictionary of weight names to values.

    Returns:
        Normalized weights that sum to 100. Equal weights if all inputs are zero.
    """
    total = sum(weights.values())
    if total == 0:
        n = len(weights)
        if n == 0:
            return {}
        return {k: 100.0 / n for k in weights}
    return {k: (v / total) * 100.0 for k, v in weights.items()}


def combine_weighted(component_scores: Dict[str, float], weights: Dict[str, float]) -> Tuple[float, str]:
    """Combine component scores (each 0.0-1.0) into a final 0-100 score.

    Args:
        component_scores: Mapping of component name (e.g. "Trend", "Breakout",
            "Volume", "MC_Prob") to a 0.0-1.0 sub-score.
        weights: Raw (not necessarily normalized) weights per component.

    Returns:
        Tuple of (final_score 0-100, trigger name). The trigger is the highest
        scoring fully-satisfied (score == 1.0) component, falling back to the
        single highest-weighted-contribution component, or "None".
    """
    normalized = normalize_weights(weights)
    final_score = sum(normalized.get(name, 0.0) * score for name, score in component_scores.items())

    # Trigger = first fully-satisfied component in priority order (matches the
    # historical Breakout > Trend > Volume precedence), else "None".
    trigger = "None"
    for name in ("Breakout", "Trend", "Volume"):
        if component_scores.get(name, 0.0) >= 1.0:
            trigger = name
            break
    else:
        if component_scores.get("Volume", 0.0) >= 0.75:
            trigger = "Volume"

    return final_score, trigger


def evaluate_and_learn(
    weights: Dict[str, float],
    trade_history: List[Dict[str, Any]],
    learning_rate: float,
    min_trades_for_learning: int,
) -> Tuple[Dict[str, float], Optional[str]]:
    """Adjust component weights based on realized trade win rate per trigger.

    Pure function: takes/returns plain dicts, no AppConfig/AgentBrain coupling.

    Args:
        weights: Current component weights.
        trade_history: List of trade dicts with "outcome" ("WIN"/"LOSS"/other)
            and "signal_trigger" keys.
        learning_rate: Rate at which weights move toward realized win rate.
        min_trades_for_learning: Minimum realized (WIN/LOSS) trades required.

    Returns:
        Tuple of (new_weights, log_message). log_message is None when weights
        were adjusted; otherwise a human-readable explanation of why not.
    """
    realized_trades = [t for t in trade_history if t.get("outcome") in ("WIN", "LOSS")]

    if len(realized_trades) < min_trades_for_learning:
        return dict(weights), (
            f"Not enough realized trades to learn "
            f"(have {len(realized_trades)}, need {min_trades_for_learning})."
        )

    trigger_stats: Dict[str, Dict[str, int]] = {}
    for trade in realized_trades:
        trigger = trade.get("signal_trigger", "Unknown")
        stats = trigger_stats.setdefault(trigger, {"total": 0, "wins": 0})
        stats["total"] += 1
        if trade.get("outcome") == "WIN":
            stats["wins"] += 1

    new_weights: Dict[str, float] = dict(weights)
    for trigger, stats in trigger_stats.items():
        total, wins = stats["total"], stats["wins"]
        win_rate = wins / total if total > 0 else 0.5
        old_weight = weights.get(trigger, 25.0)
        adjustment = (win_rate - 0.5) * learning_rate
        new_weight = max(5.0, old_weight * (1 + adjustment))
        new_weights[trigger] = new_weight

    total_weight = sum(new_weights.values())
    if total_weight > 0:
        scale_factor = 100.0 / total_weight
        for trigger in new_weights:
            new_weights[trigger] *= scale_factor

    rounded_weights = {k: round(v, 1) for k, v in new_weights.items()}
    current_sum = sum(rounded_weights.values())
    if abs(current_sum - 100.0) > 0.001 and rounded_weights:
        diff = 100.0 - current_sum
        max_key = max(rounded_weights.keys(), key=lambda k: rounded_weights[k])
        rounded_weights[max_key] = round(rounded_weights[max_key] + diff, 1)

    log_parts = []
    for trigger in sorted(trigger_stats.keys()):
        stats = trigger_stats[trigger]
        wr_pct = int(round(stats["wins"] / stats["total"] * 100)) if stats["total"] > 0 else 50
        wt = rounded_weights.get(trigger, 0)
        log_parts.append(f"{trigger} WR:{wr_pct}% (Wt:{wt:.1f})")

    log_message = f"{datetime.now().strftime('%Y-%m-%d')} Learning Update: {' | '.join(log_parts)}"
    return rounded_weights, log_message
