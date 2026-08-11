"""Pluggable weight combination and self-learning weight adaptation.

Extracted so the win-rate-based weight adaptation (previously hardcoded across
scoring.py/learning.py) works against any rule-based strategy's component
scores, not one hardcoded implementation. These are pure functions: no
AppConfig, no AgentBrain, no I/O.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from src.risk import shrink_win_probability
except ImportError:  # pragma: no cover - flat-path import
    from risk import shrink_win_probability

# Minimum realized trades attributable to ONE component before its weight may
# move. The platform-wide `learning.min_trades_for_learning` floor is a
# *total* across every trigger, so a 5-trade floor meant a component could be
# re-weighted off two wins and a loss. At n = 5 the win-rate standard error is
# +/-22 percentage points; at 30 it is +/-9.
MIN_TRADES_PER_COMPONENT = 30

# Beta-prior strength, in pseudo-trades, for shrinking a component's win rate
# toward a coin flip. Same prior the Kelly path already uses
# (src/risk.py::shrink_win_probability) — the two were inconsistent, with
# Kelly correctly shrinking and the weight learner taking the raw rate.
WIN_RATE_PRIOR_STRENGTH = 20.0

# One-sided significance level for "is this component's win rate actually
# different from a coin flip?". Weights used to move on every evaluation
# regardless of whether the difference was distinguishable from zero.
SIGNIFICANCE_ALPHA = 0.05


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


def is_win_rate_significant(wins: int, total: int, alpha: float = SIGNIFICANCE_ALPHA) -> bool:
    """Is this win rate distinguishable from a coin flip at level `alpha`?

    A two-sided binomial test against p = 0.5, so a component is re-weighted
    down on significant *under*-performance just as it is re-weighted up on
    significant over-performance. Implemented against scipy's exact test
    rather than a normal approximation: at n = 30 the approximation's tail is
    noticeably wrong, and n = 30 is exactly where this gate operates.
    """
    if total <= 0:
        return False
    from scipy import stats as scipy_stats

    return float(scipy_stats.binomtest(wins, total, 0.5).pvalue) < alpha


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

    Three guards, all of which the pre-existing version lacked and all of
    which matter because this is a closed feedback loop — weights change which
    trades are taken, which changes the outcomes the next adaptation sees:

    1. **Shrinkage.** The raw win rate is replaced by the same Beta-Binomial
       posterior mean the Kelly path uses. The two were inconsistent: Kelly
       argued (correctly) that a win rate has +/-7pp of standard error at 50
       trades and applied a 20-pseudo-trade prior, while this function took a
       raw rate at a floor of 5 trades, where the error is +/-22pp.
    2. **A real sample-size floor**, per component rather than in total.
    3. **A significance test.** A component's weight moves only if a one-sided
       binomial test rejects "this is a coin flip" at alpha = 0.05. Without it
       every evaluation moved every weight, which is a random walk driven by
       sampling noise dressed up as learning.

    Args:
        weights: Current component weights.
        trade_history: List of trade dicts with "outcome" ("WIN"/"LOSS"/other)
            and "signal_trigger" keys.
        learning_rate: Rate at which weights move toward realized win rate.
        min_trades_for_learning: Minimum realized (WIN/LOSS) trades in total.

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
    skipped: Dict[str, str] = {}
    for trigger, stats in trigger_stats.items():
        total, wins = stats["total"], stats["wins"]

        if total < MIN_TRADES_PER_COMPONENT:
            skipped[trigger] = f"n={total}<{MIN_TRADES_PER_COMPONENT}"
            continue

        if not is_win_rate_significant(wins, total, alpha=SIGNIFICANCE_ALPHA):
            skipped[trigger] = "not significant"
            continue

        # Shrunk, not raw: at these sample sizes the raw rate's deviation from
        # 0.5 is mostly sampling error, and feeding it back scales that error
        # into the scores that generate the next round of trades.
        win_rate = shrink_win_probability(
            wins=wins, total=total, prior_strength=WIN_RATE_PRIOR_STRENGTH
        )
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
        reason = skipped.get(trigger)
        suffix = f" [held: {reason}]" if reason else ""
        log_parts.append(f"{trigger} WR:{wr_pct}% (Wt:{wt:.1f}){suffix}")

    log_message = f"{datetime.now().strftime('%Y-%m-%d')} Learning Update: {' | '.join(log_parts)}"
    return rounded_weights, log_message
