"""Pluggable weight combination and self-learning weight adaptation.

Extracted so the win-rate-based weight adaptation (previously hardcoded across
scoring.py/learning.py) works against any rule-based strategy's component
scores, not one hardcoded implementation. These are pure functions: no
AppConfig, no AgentBrain, no I/O.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from portfolio_agent.src.risk import shrink_win_probability
except ImportError:  # flat `src` layout
    from portfolio_agent.src.risk import shrink_win_probability

# Minimum realized trades attributed to a single component before its win rate
# is allowed to move that component's weight. The platform's own Kelly path
# argues that a raw win rate has a +/-7 percentage point standard error at 50
# trades; at 5 it is +/-22 points, which is not an estimate of anything.
DEFAULT_MIN_TRADES_PER_COMPONENT = 30

# One-sided significance required before a weight moves at all.
DEFAULT_SIGNIFICANCE_LEVEL = 0.05


def binomial_tail_probability(wins: int, total: int, null_win_rate: float = 0.5) -> float:
    """P(X >= wins) for X ~ Binomial(total, null_win_rate) — an exact one-sided test.

    Computed exactly rather than by normal approximation: the samples this is
    asked about are small, which is precisely where the approximation is worst
    and where a spurious "significant" result does the most damage.

    Returns 1.0 for degenerate input, so a caller gating on p < alpha does
    nothing rather than something arbitrary.
    """
    if total <= 0 or wins <= 0:
        return 1.0
    if wins > total:
        return 0.0
    if not 0.0 < null_win_rate < 1.0:
        return 1.0

    tail = 0.0
    for k in range(int(wins), int(total) + 1):
        tail += (
            math.comb(total, k)
            * null_win_rate**k
            * (1.0 - null_win_rate) ** (total - k)
        )
    return min(1.0, tail)


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


def combine_weighted(
    component_scores: Dict[str, float],
    weights: Dict[str, float],
    unavailable: Optional[Iterable[str]] = None,
) -> Tuple[float, str]:
    """Combine component scores (each 0.0-1.0) into a final 0-100 score.

    **"Cannot compute" is not the same as "scores zero", and this is where the
    difference has to be handled.** Every gate downstream reads the result as
    an absolute number — the rule-based strategy buys at `score >= 60` and
    watches at `>= 45` — so a component that silently contributes 0 because its
    input was missing does not merely add no information, it moves the total
    across fixed thresholds.

The live instance: a rule-based member inside a batched UMA receives no
    per-ticker Monte Carlo result, so `MC_Prob` scored 0 and the identical
    stock on the identical day scored ~12 points lower inside an ensemble than
    standalone. That is a silent, systematic level shift between two code paths
    that are supposed to agree, and it has nothing to do with the stock.

    Naming the unavailable components instead drops them from the weight
    normalization, so the remaining components renormalize to 100 and the score
    stays on a scale the thresholds still mean something on.

    **This is only correct when the pipeline failed to compute a value, not
    when the stock lacks the data.** Renormalizing away a component the *stock*
    cannot support — a missing SMA-200, meaning a recent listing — would scale
    the remaining components up and score a young, illiquid name higher than a
    seasoned one, applying least caution exactly where an Indian micro-cap
    universe warrants most. Callers must keep that case at its conservative
    zero; see rule_based.py, which passes only `MC_Prob` here.

    Args:
        component_scores: Mapping of component name (e.g. "Trend", "Breakout",
            "Volume", "MC_Prob") to a 0.0-1.0 sub-score.
        weights: Raw (not necessarily normalized) weights per component.
        unavailable: Components whose inputs were missing. These are excluded
            from the weight normalization rather than scored as zero. A
            component absent here but scored 0.0 is a genuine zero — a stock
            that really is below its Donchian channel — and still counts.

    Returns:
        Tuple of (final_score 0-100, trigger name). The trigger is the highest
        scoring fully-satisfied (score == 1.0) component, falling back to the
        single highest-weighted-contribution component, or "None".
    """
    missing = set(unavailable or ())
    usable = {name: score for name, score in component_scores.items() if name not in missing}

    # Renormalize over what could actually be measured. Weights for components
    # that were never configured stay out of it either way.
    effective_weights = {
        name: value for name, value in weights.items() if name not in missing
    }
    normalized = normalize_weights(effective_weights)
    final_score = sum(normalized.get(name, 0.0) * score for name, score in usable.items())

    return final_score, select_trigger(component_scores)


def select_trigger(component_scores: Dict[str, float]) -> str:
    """Name the component that fired, in Breakout > Trend > Volume precedence.

    Separated from the score because the two answer different questions and,
    under rank-composite scoring, read different inputs. "Breakout fired" is a
    statement about the raw indicator — the close cleared its Donchian channel
    — and stays true whether or not that clearing ranks well against the rest
    of the universe today. Feeding percentile ranks in here would rename the
    trigger to whichever component happened to rank highest, which is a
    different and much less useful claim, and it would corrupt the weight
    learner: `evaluate_and_learn` attributes realized outcomes by trigger name.

    Returns the first fully-satisfied (1.0) component in precedence order, then
    a near-satisfied Volume, else "None".
    """
    for name in ("Breakout", "Trend", "Volume"):
        if component_scores.get(name, 0.0) >= 1.0:
            return name
    if component_scores.get("Volume", 0.0) >= 0.75:
        return "Volume"
    return "None"


def evaluate_and_learn(
    weights: Dict[str, float],
    trade_history: List[Dict[str, Any]],
    learning_rate: float,
    min_trades_for_learning: int,
    min_trades_per_component: int = DEFAULT_MIN_TRADES_PER_COMPONENT,
    shrinkage_strength: float = 20.0,
    significance_level: float = DEFAULT_SIGNIFICANCE_LEVEL,
) -> Tuple[Dict[str, float], Optional[str]]:
    """Adjust component weights based on realized trade win rate per trigger.

    Pure function: takes/returns plain dicts, no AppConfig/AgentBrain coupling.

    **Three guards, because this is a feedback loop.** Weights adapt on
    realized outcomes, which changes which trades are taken, which changes the
    outcomes the next adaptation sees. That makes noise self-reinforcing unless
    each update has to clear a real bar:

    1. *Shrinkage.* The raw win rate used to move weights directly, at a floor
       of 5 trades where its standard error is ~22 percentage points. The same
       Beta prior the Kelly path already applies (src/risk.py) is used here, so
       the two paths finally agree about what a win rate is worth.
    2. *A sample floor per component.* The floor was on the *total* trade
       count, so a component with three trades to its name could move on the
       strength of a fifty-trade sample it contributed almost nothing to.
    3. *A significance test.* An exact one-sided binomial test against a
       coin-flip null. Without it, weights moved on every evaluation regardless
       of whether the win-rate difference was distinguishable from zero.

    None of this makes the loop out-of-sample — for that the weights have to be
    fitted on walk-forward training folds and frozen for the test fold, the way
    model checkpoints already are. It does stop the loop from chasing noise.

    Args:
        weights: Current component weights.
        trade_history: List of trade dicts with "outcome" ("WIN"/"LOSS"/other)
            and "signal_trigger" keys.
        learning_rate: Rate at which weights move toward realized win rate.
        min_trades_for_learning: Minimum realized (WIN/LOSS) trades overall.
        min_trades_per_component: Minimum realized trades attributed to a
            single component before its own weight may move.
        shrinkage_strength: Beta-prior strength in pseudo-trades; 0 uses the
            raw win rate.
        significance_level: One-sided alpha a component's win rate must clear
            before its weight moves. Set to 1.0 to disable the test.

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
    moved: set[str] = set()
    for trigger, stats in trigger_stats.items():
        total, wins = stats["total"], stats["wins"]
        if total < min_trades_per_component:
            continue

        # Two-sided in effect: a component is promoted only on significant
        # evidence of an edge and demoted only on significant evidence against.
        win_tail = binomial_tail_probability(wins, total)
        loss_tail = binomial_tail_probability(total - wins, total)
        if min(win_tail, loss_tail) > significance_level:
            continue

        win_rate = shrink_win_probability(
            wins=wins, total=total, prior_strength=shrinkage_strength
        )
        old_weight = weights.get(trigger, 25.0)
        adjustment = (win_rate - 0.5) * learning_rate
        new_weight = max(5.0, old_weight * (1 + adjustment))
        new_weights[trigger] = new_weight
        moved.add(trigger)

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
        # A component whose sample was too small or whose edge was not
        # significant is marked "held", so the log distinguishes "no evidence"
        # from "evidence of no edge" instead of showing a win rate that did
        # nothing and letting the reader assume it did.
        held = "" if trigger in moved else " held"
        log_parts.append(f"{trigger} WR:{wr_pct}% n={stats['total']} (Wt:{wt:.1f}{held})")

    log_message = f"{datetime.now().strftime('%Y-%m-%d')} Learning Update: {' | '.join(log_parts)}"
    return rounded_weights, log_message
