"""Signal arbitration: deciding when several models' opinions become a trade.

The problem this exists to fix is a specific, expensive failure of linear
blending. A UMA that averages its members (strategies/ensemble.py's
`weighted_blend`) maps each member's signal to a strength and takes a weighted
mean. Give it a momentum model screaming BUY at 0.90 conviction and a
mean-reversion model screaming SELL at 0.85, and it reports a mildly positive
number — a small BUY. But the two models do not disagree mildly. They disagree
maximally, which is the single strongest available evidence that nobody knows
what this stock is about to do, and it is exactly the setup that produces
whipsaw: entered on a blended signal neither model would have taken alone, then
stopped out by whichever of them was right.

Averaging is the right operation for *estimates of the same quantity*. It is
the wrong operation for *votes on a decision*. This module treats them as
votes:

1. **Conflict penalty.** A buy-side conviction is discounted by the strongest
   opposing conviction, ``c_eff = c_buy * (1 - max(c_opposing))``. Two models
   at 0.9 and 0.85 in opposite directions leave 0.135 of usable conviction,
   not a positive average.
2. **Global vetoes.** Untradeable instruments, muted models and trades whose
   expected value does not clear a hurdle are blocked outright, before any
   arithmetic on confidences. A veto is not a low score to be outweighed.
3. **Firing modes.** A trade needs either one genuinely strong model
   (`strong_single`), or several independent models agreeing (`consensus`), or
   either (`strong_or_consensus`). "Several models each mildly positive" is
   evidence; "one model mildly positive" is noise.
4. **Size, not just direction.** What survives arbitration is a size
   multiplier, so a trade that barely clears its threshold is taken small and
   a fully-agreed one is taken in full. The decision is not binary because the
   evidence is not binary.

The engine is deliberately conservative in every fail case: no verdicts, no
eligible model, an unknown regime or an unestimable expected value all resolve
toward BLOCK or toward a smaller position, never toward a larger one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence

from portfolio_agent.strategies.types import ModelVerdict

TriggerMode = Literal["strong_single", "consensus", "strong_or_consensus"]

# How a model the current regime does not permit to buy is handled.
RegimePolicy = Literal["mute", "veto"]


@dataclass(frozen=True)
class TriggerConfig:
    """Thresholds governing when model verdicts become a trade.

    Attributes:
        mode: Which firing rule applies (see module docstring).
        strong_confidence: Conflict-adjusted conviction a *single* model must
            reach to fire on its own. High by design — this is the branch that
            lets one model act without corroboration.
        consensus_confidence: Conviction a model must reach to count as one of
            the agreeing voices, and the mean the agreeing group must clear.
        min_consensus_models: How many models must agree for the consensus
            branch to fire. Two is the minimum that means anything; one is not
            a consensus.
        min_net_ev_pct: Expected-value hurdle in percent of entry price, net of
            round-trip friction. A trade whose expected value does not clear
            this is not worth its own costs however confident anyone is.
        conflict_veto_confidence: An opposing conviction at or above this
            blocks outright. The multiplicative penalty alone would usually
            suppress such a trade anyway; the explicit veto exists so the
            decision's stated reason is "models disagree" rather than a
            confidence number that happened to land low.
        regime_policy: "mute" drops regime-incompatible models from the vote;
            "veto" lets any incompatible model block the trade outright.
        min_size_multiplier: Size applied to a trade that only just clears its
            firing threshold.
        max_size_multiplier: Size applied to a maximally-convinced,
            unopposed trade.
    """

    mode: TriggerMode = "strong_or_consensus"
    strong_confidence: float = 0.75
    consensus_confidence: float = 0.55
    min_consensus_models: int = 2
    min_net_ev_pct: float = 0.0
    conflict_veto_confidence: float = 0.5
    regime_policy: RegimePolicy = "mute"
    min_size_multiplier: float = 0.5
    max_size_multiplier: float = 1.0

    @classmethod
    def from_params(cls, params: Optional[Dict[str, Any]]) -> "TriggerConfig":
        """Build from a YAML/params block, ignoring unknown keys."""
        params = params or {}
        defaults = cls()
        return cls(
            mode=str(params.get("mode", defaults.mode)),
            strong_confidence=float(
                params.get("strong_confidence", defaults.strong_confidence)
            ),
            consensus_confidence=float(
                params.get("consensus_confidence", defaults.consensus_confidence)
            ),
            min_consensus_models=int(
                params.get("min_consensus_models", defaults.min_consensus_models)
            ),
            min_net_ev_pct=float(params.get("min_net_ev_pct", defaults.min_net_ev_pct)),
            conflict_veto_confidence=float(
                params.get("conflict_veto_confidence", defaults.conflict_veto_confidence)
            ),
            regime_policy=str(params.get("regime_policy", defaults.regime_policy)),
            min_size_multiplier=float(
                params.get("min_size_multiplier", defaults.min_size_multiplier)
            ),
            max_size_multiplier=float(
                params.get("max_size_multiplier", defaults.max_size_multiplier)
            ),
        )


@dataclass
class TriggerDecision:
    """The arbitrated outcome for one instrument.

    Attributes:
        action: "BUY" when the trade fires, "BLOCK" otherwise. There is no
            "weak buy" — that is the outcome this module exists to prevent.
        size_multiplier: Fraction of the otherwise-sized position to take, in
            [min_size_multiplier, max_size_multiplier]; 0.0 when blocked.
        effective_confidence: Conviction after the conflict penalty, on 0-1.
        expected_net_ev_pct: Confidence-weighted expected value of the
            contributing models, or None when none of them could estimate one.
        fired_rule: Which branch fired ("strong_single" / "consensus"), or ""
            when nothing did.
        reason: Human-readable explanation, suitable for a trade rationale.
        vetoes: Every hard block that applied, in the order they were checked.
        contributing_models: Models that voted to buy and were heard.
        opposing_models: Models that voted to sell.
        muted_models: Models dropped because the regime does not permit them.
    """

    action: str
    size_multiplier: float
    effective_confidence: float
    expected_net_ev_pct: Optional[float] = None
    fired_rule: str = ""
    reason: str = ""
    vetoes: List[str] = field(default_factory=list)
    contributing_models: List[str] = field(default_factory=list)
    opposing_models: List[str] = field(default_factory=list)
    muted_models: List[str] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        """Whether the trade may be placed."""
        return self.action == "BUY"


def _blocked(reason: str, **kwargs: Any) -> TriggerDecision:
    """A BLOCK decision, so every early return has identical shape."""
    return TriggerDecision(
        action="BLOCK",
        size_multiplier=0.0,
        effective_confidence=0.0,
        reason=reason,
        **kwargs,
    )


class TriggerEngine:
    """Arbitrates ModelVerdicts into a single allow/block decision and a size.

    Stateless and pure: the same verdicts always produce the same decision, so
    the live orchestrator and the backtest engine cannot drift apart, and a
    decision can be replayed from a logged verdict list.
    """

    def __init__(self, config: Optional[TriggerConfig] = None):
        self.config = config or TriggerConfig()

    def evaluate(self, verdicts: Sequence[ModelVerdict]) -> TriggerDecision:
        """Arbitrate one instrument's verdicts.

        Args:
            verdicts: Every model's opinion on this instrument. Order is
                irrelevant; duplicates are treated as independent voices, so
                callers must not pass the same model twice.

        Returns:
            A TriggerDecision. Anything short of an affirmative firing rule is
            a BLOCK — there is no partial credit.
        """
        cfg = self.config

        if not verdicts:
            return _blocked("no model produced a verdict")

        # --- Hard vetoes, checked before any arithmetic on confidences. ------
        # Tradability is a property of the instrument, not of the model that
        # noticed, so one screen failing blocks regardless of who reported it.
        illiquid = [v.model_name for v in verdicts if not v.liquidity_pass]
        if illiquid:
            return _blocked(
                f"liquidity/tradability screen failed (reported by {', '.join(sorted(illiquid))})",
                vetoes=["liquidity_pass"],
            )

        buys = [v for v in verdicts if v.is_buy]
        sells = [v for v in verdicts if v.is_sell]

        incompatible = [v for v in buys if not v.regime_compatible]
        if incompatible and cfg.regime_policy == "veto":
            return _blocked(
                f"regime does not permit {', '.join(sorted(v.model_name for v in incompatible))}",
                vetoes=["regime_compatible"],
                muted_models=sorted(v.model_name for v in incompatible),
            )

        # Default policy: an incompatible model is silenced, not amplified into
        # a veto. Muting is what makes the regime map of Phase 4 usable — one
        # sleeve being out of season should not stop the sleeve that is in it.
        muted = sorted(v.model_name for v in incompatible)
        contributors = [v for v in buys if v.regime_compatible]

        if not contributors:
            reason = (
                f"every buying model is muted in this regime ({', '.join(muted)})"
                if muted
                else "no model voted to buy"
            )
            return _blocked(
                reason,
                vetoes=["regime_compatible"] if muted else [],
                muted_models=muted,
                opposing_models=sorted(v.model_name for v in sells),
            )

        opposing_confidence = max((v.confidence for v in sells), default=0.0)
        opposing_names = sorted(v.model_name for v in sells)
        contributing_names = sorted(v.model_name for v in contributors)

        if sells and opposing_confidence >= cfg.conflict_veto_confidence:
            strongest = max(sells, key=lambda v: v.confidence)
            best_buy = max(contributors, key=lambda v: v.confidence)
            return _blocked(
                f"models conflict: {best_buy.model_name} BUY at {best_buy.confidence:.2f} against "
                f"{strongest.model_name} SELL at {strongest.confidence:.2f} "
                f"(>= {cfg.conflict_veto_confidence:.2f})",
                vetoes=["model_conflict"],
                contributing_models=contributing_names,
                opposing_models=opposing_names,
                muted_models=muted,
            )

        # --- Expected value hurdle. -----------------------------------------
        expected_ev = self._weighted_expected_ev(contributors)
        if expected_ev is not None and expected_ev < cfg.min_net_ev_pct:
            return _blocked(
                f"expected value {expected_ev:.2f}% below the {cfg.min_net_ev_pct:.2f}% hurdle "
                f"after costs",
                vetoes=["min_net_ev_pct"],
                contributing_models=contributing_names,
                opposing_models=opposing_names,
                muted_models=muted,
            )

        # --- Firing rules. ---------------------------------------------------
        penalty = max(0.0, 1.0 - opposing_confidence)
        strongest_buy = max(v.confidence for v in contributors)
        strong_effective = strongest_buy * penalty

        agreeing = [v for v in contributors if v.confidence >= cfg.consensus_confidence]
        consensus_effective = 0.0
        if agreeing:
            consensus_effective = (
                sum(v.confidence for v in agreeing) / len(agreeing)
            ) * penalty

        strong_fires = cfg.mode in ("strong_single", "strong_or_consensus") and (
            strong_effective >= cfg.strong_confidence
        )
        consensus_fires = cfg.mode in ("consensus", "strong_or_consensus") and (
            len(agreeing) >= cfg.min_consensus_models
            and consensus_effective >= cfg.consensus_confidence
        )

        if not strong_fires and not consensus_fires:
            return _blocked(
                self._miss_reason(
                    strong_effective, consensus_effective, len(agreeing), opposing_confidence
                ),
                contributing_models=contributing_names,
                opposing_models=opposing_names,
                muted_models=muted,
            )

        # Prefer whichever branch cleared its own bar by more, so the reported
        # rule and the size multiplier describe the same evidence.
        if strong_fires and (
            not consensus_fires
            or (strong_effective - cfg.strong_confidence)
            >= (consensus_effective - cfg.consensus_confidence)
        ):
            fired_rule = "strong_single"
            effective = strong_effective
            threshold = cfg.strong_confidence
        else:
            fired_rule = "consensus"
            effective = consensus_effective
            threshold = cfg.consensus_confidence

        size_multiplier = self._size_multiplier(effective, threshold)
        conflict_note = (
            f"; discounted by opposing conviction {opposing_confidence:.2f}"
            if opposing_confidence > 0
            else ""
        )
        ev_note = f"; net EV {expected_ev:.2f}%" if expected_ev is not None else ""

        return TriggerDecision(
            action="BUY",
            size_multiplier=size_multiplier,
            effective_confidence=round(effective, 6),
            expected_net_ev_pct=expected_ev,
            fired_rule=fired_rule,
            reason=(
                f"{fired_rule} fired at effective confidence {effective:.2f} "
                f"(threshold {threshold:.2f}){conflict_note}{ev_note}; "
                f"size x{size_multiplier:.2f}"
            ),
            contributing_models=contributing_names,
            opposing_models=opposing_names,
            muted_models=muted,
        )

    @staticmethod
    def _weighted_expected_ev(contributors: Sequence[ModelVerdict]) -> Optional[float]:
        """Confidence-weighted expected value across models that could estimate one.

        Models reporting None are skipped rather than counted as zero: a
        strategy without a probability estimate has no opinion on expected
        value, and treating that as "zero EV" would let the hurdle veto trades
        purely because a ranking model cannot speak the language of
        probabilities. When *no* contributor can estimate an EV the hurdle is
        skipped entirely — the other gates still apply.
        """
        known = [
            v for v in contributors if v.expected_net_ev_pct is not None and v.confidence > 0
        ]
        if not known:
            return None
        total_weight = sum(v.confidence for v in known)
        if total_weight <= 0:
            return None
        return sum(v.expected_net_ev_pct * v.confidence for v in known) / total_weight

    def _size_multiplier(self, effective: float, threshold: float) -> float:
        """Scale position size by how far past its threshold the evidence got.

        Linear from min_size_multiplier at the threshold to max_size_multiplier
        at full conviction. A trade that only just qualifies is a trade the
        evidence only just supports, and it should be sized accordingly.
        """
        cfg = self.config
        span = 1.0 - threshold
        fraction = 1.0 if span <= 0 else (effective - threshold) / span
        fraction = min(1.0, max(0.0, fraction))
        multiplier = cfg.min_size_multiplier + (
            cfg.max_size_multiplier - cfg.min_size_multiplier
        ) * fraction
        return round(min(cfg.max_size_multiplier, max(0.0, multiplier)), 4)

    def _miss_reason(
        self,
        strong_effective: float,
        consensus_effective: float,
        n_agreeing: int,
        opposing_confidence: float,
    ) -> str:
        """Why nothing fired, in the terms of whichever rules were eligible."""
        cfg = self.config
        parts: List[str] = []
        if cfg.mode in ("strong_single", "strong_or_consensus"):
            parts.append(
                f"strongest model at {strong_effective:.2f} < {cfg.strong_confidence:.2f}"
            )
        if cfg.mode in ("consensus", "strong_or_consensus"):
            if n_agreeing < cfg.min_consensus_models:
                parts.append(
                    f"only {n_agreeing} model(s) above {cfg.consensus_confidence:.2f}, "
                    f"need {cfg.min_consensus_models}"
                )
            else:
                parts.append(
                    f"consensus at {consensus_effective:.2f} < {cfg.consensus_confidence:.2f}"
                )
        if opposing_confidence > 0:
            parts.append(f"after an opposing conviction of {opposing_confidence:.2f}")
        return "no trigger fired: " + "; ".join(parts)
