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
    from portfolio_agent.src.monte_carlo import MonteCarloResult


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
    # ATR multiples defining the exit plan. These reach the fill: the backtest
    # engine sizes a filled position's stop and target from the signal's own
    # levels (BacktestEngine._exit_levels), so changing them changes both what
    # gets screened by min_reward_risk and what actually happens on exit.
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
            atr_stop_multiplier=config.risk.atr_stop_multiplier,
            atr_target_multiplier=config.risk.atr_target_multiplier,
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
    """Everything besides the feature DataFrame that score()/score_batch() need.

    **Every field except `risk` is optional, and which ones arrive depends on
    the caller.** That is the part worth knowing before writing a strategy: a
    field is not "usually there", it is there on some paths and absent on
    others, and a strategy that assumes one silently produces a different
    answer depending on who called it.

    | field | backtest, per-ticker | backtest, batched | evaluate | live |
    | --- | --- | --- | --- | --- |
    | `risk` | yes | yes | yes | yes |
    | `weights` | yes | yes | **no** | yes |
    | `mc_result` | yes | **no** | **no** | yes |
    | `benchmark_close` / `_ohlcv` | **no** | yes | yes | yes |
    | `regime_label` | yes | yes | **no** | yes |
    | `run_id` | **no** | **no** | **no** | yes |

    This table is not a wish list; it is what the callers do today, and the
    gaps are why it exists. `rule_based` read its component weights from
    `weights` alone, so under `evaluate` — which fills none — the weighted sum
    ran over an empty mapping and returned 0.0 for every name, and that floor
    was published as the strategy's measured score dispersion. It now falls
    back to its own configured weights (`rule_based.py::_load_weights`).

    The rule for a strategy: **treat every optional field as absent and degrade
    to something defensible**, the way `combine_weighted`'s `unavailable`
    argument does — do not treat a missing input as a zero measurement. Where
    degrading is not defensible, say so in the rationale rather than emitting a
    number that looks like one.
    """

    risk: RiskParams
    #: Component weights. The backtest evolves these across a run and passes
    #: them down; `evaluate` has no learning loop and leaves them empty, so a
    #: strategy that needs weights must carry its own default.
    weights: Dict[str, float] = field(default_factory=dict)
    #: Monte Carlo result for this ticker. Only the per-ticker backtest path
    #: and the live orchestrator run one, so a strategy whose gate depends on
    #: it cannot pass under `evaluate` or a batched backtest. That is the
    #: correct default for a compliance gate with no evidence either way, but
    #: it belongs in the rationale rather than looking like a bad score.
    mc_result: Optional["MonteCarloResult"] = None
    run_id: Optional[str] = None
    # Benchmark index close series (e.g. the Nifty 50), truncated to the
    # decision date by the caller. The momentum crash filter prefers this over
    # its equal-weighted composite of the traded universe: "the market is below
    # its 200-day average" is a statement about the index the research actually
    # studied, and a composite of whatever happens to be in today's universe is
    # only a proxy for it. None when no benchmark is cached.
    benchmark_close: Optional["pd.Series"] = None
    # Benchmark OHLC, when the cache has it. Only the trend and volatility
    # tests can be run from closes alone; ADX — which is what separates a
    # sideways chop from a trend at the same distance from the moving average —
    # needs the daily high/low range. None falls back to a close-only proxy.
    benchmark_ohlcv: Optional["pd.DataFrame"] = None
    # Market regime label for this scoring round (src/regime.py). Set by the
    # caller so every strategy in a round sees the same classification rather
    # than each re-deriving it, and read by the meta-orchestrator to decide
    # which models the regime permits to buy. None means "not assessed".
    regime_label: Optional[str] = None


@dataclass(frozen=True)
class ModelVerdict:
    """One model's opinion, in the single shape the trigger engine arbitrates.

    StrategySignal is a *trade plan* — entry, stop, target, a 0-100 score whose
    meaning is strategy-specific. That is the wrong input for deciding whether
    several models agree: "score 82" means a momentum percentile in one
    strategy and a forecast probability in another, and averaging them is how a
    strong BUY and a strong SELL become a weak BUY (src/trigger_engine.py).

    A verdict strips that down to the four things arbitration actually needs —
    a direction, a conviction on a comparable 0-1 scale, an expected value in
    money terms, and whether the trade is admissible at all.

    Attributes:
        model_name: Which model produced this. Appears in decision rationales.
        action: "BUY", "SELL" or "AVOID". AVOID is an *abstention*: it carries
            no weight for or against, because a model declining to call a name
            is not evidence that the name is bad. Admissibility is carried by
            the two flags below, not by an AVOID action.
        confidence: Conviction in `action`, on 0-1. Comparable across models by
            construction — see from_signal() for how each strategy's native
            score is mapped onto it.
        expected_net_ev_pct: Expected value of the trade as a percentage of the
            entry price, net of round-trip friction. **None means "not
            estimable"**, not "zero": a strategy with no probability estimate
            (cross-sectional ranking without a Monte Carlo result, for example)
            genuinely cannot produce one, and fabricating a number there would
            make the engine's EV hurdle bite hardest on the models that are
            most honest about their uncertainty.
        regime_compatible: Whether this model is one the current market regime
            allows to buy (Phase 4's meta-orchestrator mapping). An
            incompatible model is muted rather than heard.
        liquidity_pass: Whether the instrument is tradable at all — the
            circuit-lock and illiquidity screens of src/liquidity.py. False is
            a hard veto: untradeable is a property of the stock, so it holds no
            matter which model noticed.
        rationale: Human-readable explanation, concatenated into the decision.
    """

    model_name: str
    action: str
    confidence: float
    expected_net_ev_pct: Optional[float] = None
    regime_compatible: bool = True
    liquidity_pass: bool = True
    rationale: str = ""

    @property
    def is_buy(self) -> bool:
        return self.action == "BUY"

    @property
    def is_sell(self) -> bool:
        return self.action == "SELL"

    @classmethod
    def from_signal(
        cls,
        signal: "StrategySignal",
        model_name: Optional[str] = None,
        regime_compatible: bool = True,
        win_probability: Optional[float] = None,
    ) -> "ModelVerdict":
        """Map a StrategySignal onto the standardized verdict contract.

        **Action.** BUY and SELL pass through. HOLD, WATCH and AVOID all become
        AVOID — each is the model declining to take a side, whether because it
        sees nothing (HOLD), because it ranked the name but a gate stopped it
        (WATCH), or because it screened the name out (AVOID).

        **Confidence.** The 0-100 `score` is a *goodness* scale in every
        strategy here: high means attractive. Conviction in a BUY is therefore
        score/100, and conviction in a SELL is its complement — an ML strategy
        emitting SELL at score 15 is 85% convinced, not 15%. An abstention
        carries zero conviction by definition.

        **Expected value.** Computed in "R units" off the reward:risk the
        signal already carries, which this platform computes *net* of estimated
        round-trip friction (src/risk.py::net_reward_risk):

            EV_R   = p * b - (1 - p)
            EV_pct = EV_R * (entry - stop) / entry * 100

        Deriving it this way rather than re-costing target and stop is
        deliberate: `b` has already had brokerage, STT, exchange and SEBI
        charges, GST, stamp duty and slippage charged against it, and charging
        them a second time here would double-count the friction stack. When no
        win probability is available the EV is left None rather than guessed.

        **Liquidity.** Read from the tradability screen's own marker on the
        signal, so the engine and the strategy cannot disagree about whether a
        name was screened.

        Args:
            signal: The strategy output to convert.
            model_name: Name to record; defaults to the signal's trigger.
            regime_compatible: Whether the regime permits this model to buy.
            win_probability: Override for p. Defaults to the signal's own
                Monte Carlo probability-of-profit when it carries a usable one.

        Returns:
            The equivalent ModelVerdict.
        """
        action = signal.signal if signal.signal in ("BUY", "SELL") else "AVOID"

        score_fraction = min(1.0, max(0.0, signal.score / 100.0))
        if action == "BUY":
            confidence = score_fraction
        elif action == "SELL":
            confidence = 1.0 - score_fraction
        else:
            confidence = 0.0

        probability = win_probability
        if probability is None and signal.probability_profit > 0:
            probability = signal.probability_profit

        expected_net_ev_pct: Optional[float] = None
        if probability is not None and signal.entry_price > 0 and signal.reward_risk > 0:
            risk_fraction = (signal.entry_price - signal.stop_price) / signal.entry_price
            if risk_fraction > 0:
                ev_in_r = probability * signal.reward_risk - (1.0 - probability)
                expected_net_ev_pct = ev_in_r * risk_fraction * 100.0

        extra = signal.extra or {}
        liquidity_pass = "tradability_reject_reason" not in extra

        return cls(
            model_name=model_name or signal.trigger or "unnamed",
            action=action,
            confidence=round(confidence, 6),
            expected_net_ev_pct=expected_net_ev_pct,
            regime_compatible=regime_compatible,
            liquidity_pass=liquidity_pass,
            rationale=signal.rationale,
        )


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
