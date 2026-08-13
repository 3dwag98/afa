"""Rule-based strategy implementation for portfolio agent.

This is the single, canonical "Trend + Breakout + Volume + Monte Carlo
Probability" strategy. It replaces three previously-drifted copies of this
logic (src/scoring.py::score_candidate, the old unwired version of this file,
and the crude stub inside BacktestEngine._generate_signals) so the live
orchestrator and the backtest engine make identical decisions from identical
inputs.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from .base import BaseStrategy
from .types import StrategyContext, StrategySignal
from .weighting import combine_weighted, select_trigger
from portfolio_agent.config.schema import StrategyConfig

from portfolio_agent.src.risk import calculate_stop_target, net_reward_risk


# The four components, in the order they are reported. Named once so the
# rank transform and the score both walk the same set.
COMPONENT_NAMES = ("Trend", "Breakout", "Volume", "MC_Prob")

SCORING_MODES = ("weighted_sum", "rank_composite", "probit_composite")

# Modes whose score is a statement about the cross-section, so a per-ticker
# call cannot produce one.
_CROSS_SECTIONAL_MODES = ("rank_composite", "probit_composite")

# Phi and Phi^-1 from the stdlib, matching src/performance_stats.py, which is
# the other module in this package that needs them. NormalDist.inv_cdf is
# Wichura's AS241 and accurate to full double precision, so this is the same
# number scipy.stats.norm.ppf returns, without a second convention for the same
# quantity living in two files.
_NORMAL = statistics.NormalDist()

# A cross-section with dispersion below this has nothing to standardize by;
# np.std of a constant column lands near 1e-17 rather than exactly zero, and
# dividing by that turns a tie into an arbitrary ±1e17 ordering.
_MIN_COMPOSITE_DISPERSION = 1e-12


@dataclass
class _ComponentRead:
    """Raw component values for one ticker, before any combination.

    Split out so weighted-sum and rank-composite scoring share one reading of
    the indicators and one construction of the signal, differing only in what
    the weights are applied to. Two code paths computing components separately
    is how the standalone and ensemble scores drifted apart in the first place.
    """

    components: Dict[str, float]
    unavailable: List[str] = field(default_factory=list)
    close: float = 0.0
    atr: Optional[float] = None
    prob_profit: float = 0.0


def _clean(value: Any) -> Optional[float]:
    """Convert a possibly-NaN/None pandas scalar to a plain float or None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


class RuleBasedStrategy(BaseStrategy):
    """Rule-based trading strategy configured via YAML.

    Implements the trend/breakout/volume/Monte-Carlo scoring logic that was
    previously hardcoded in src/scoring.py, now driven by configurable
    component weights and ATR multipliers from a YAML rules file.
    """

    def __init__(self, config: StrategyConfig):
        """Initialize the rule-based strategy.

        Args:
            config: StrategyConfig containing the path to the YAML rules file.
        """
        self._config = config
        self._yaml_path = config.params.get("yaml_path", config.config_path)
        self._rules = self._load_rules()

        # "weighted_sum" (default) or "rank_composite". The strategy params
        # win so a UMA member can override one shared YAML; otherwise it comes
        # from `scoring.method` in the rules file, alongside the weights the
        # method is applied to.
        scoring_block = self._rules.get("scoring")
        yaml_mode = (
            scoring_block.get("method") if isinstance(scoring_block, dict) else None
        )
        scoring = config.params.get("scoring_mode") or yaml_mode or "weighted_sum"
        if scoring not in SCORING_MODES:
            raise ValueError(
                f"unknown scoring mode {scoring!r} for {self._yaml_path}; "
                f"expected one of {SCORING_MODES}"
            )
        self._scoring = scoring

    def _load_rules(self) -> Dict[str, Any]:
        """Load rules from the YAML configuration file."""
        yaml_path = Path(self._yaml_path)
        if not yaml_path.is_absolute():
            # First try relative to the current working directory
            if not yaml_path.exists():
                # Then relative to the portfolio_agent package root
                package_root = Path(__file__).parent.parent
                candidate = package_root / self._yaml_path
                if candidate.exists():
                    yaml_path = candidate
                else:
                    # Then relative to the workspace root
                    workspace_root = package_root.parent
                    yaml_path = workspace_root / self._yaml_path

        if not yaml_path.exists():
            raise FileNotFoundError(f"Strategy YAML file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)

    @property
    def name(self) -> str:
        return self._rules.get("name", "rule_based")

    def required_features(self) -> List[str]:
        return ["close", "sma_50", "sma_200", "donchian_upper_20", "volume_ratio_20", "atr_14"]

    def entry_rules(self) -> Dict[str, Any]:
        return self._rules.get("entry", {})

    def exit_rules(self) -> Dict[str, Any]:
        return self._rules.get("exit", {})

    def score(self, symbol: str, features: pd.DataFrame, context: StrategyContext) -> StrategySignal:
        """Score one ticker.

        Under ``scoring: weighted_sum`` (the default) this is self-contained.

        Under ``rank_composite`` or ``probit_composite`` a single ticker is a
        cross-section of one — every component ranks at the same quantile and
        the score says nothing — so callers must use score_batch() with the
        full eligible universe. ``requires_full_batch`` is what tells them so,
        and every production path honours it: the orchestrator and the backtest
        engine both branch on it before dispatching, and EnsembleStrategy
        refuses to build a UMA that would reach a cross-sectional member
        through this method. A direct call here still returns the weighted sum
        of the raw components, which is a different quantity on a different
        scale from what score_batch() produces.
        """
        read = self._read_components(symbol, features, context)
        if read is None:
            return self._empty_signal(symbol)
        return self._build_signal(symbol, read, read.components, context)

    def score_batch(
        self, features_by_symbol: Dict[str, pd.DataFrame], context: StrategyContext
    ) -> Dict[str, StrategySignal]:
        """Score many tickers, ranking their components against each other
        when ``scoring: rank_composite`` is configured."""
        if self._scoring not in _CROSS_SECTIONAL_MODES:
            return super().score_batch(features_by_symbol, context)

        reads = {
            symbol: self._read_components(symbol, features, context)
            for symbol, features in features_by_symbol.items()
        }
        usable = {symbol: read for symbol, read in reads.items() if read is not None}
        if not usable:
            return {symbol: self._empty_signal(symbol) for symbol in features_by_symbol}

        signals = {
            symbol: self._empty_signal(symbol)
            for symbol, read in reads.items() if read is None
        }

        if self._scoring == "rank_composite":
            ranked = self._rank_components(usable)
            for symbol, read in usable.items():
                signals[symbol] = self._build_signal(
                    symbol, read, ranked[symbol], context
                )
            return signals

        # probit_composite: normal-score each component, combine, then
        # standardize the combination across the date. The standardization is
        # cross-sectional, so it cannot happen inside the per-symbol
        # _build_signal — the composite is computed for the whole batch first
        # and handed down.
        scored = self._probit_components(usable)
        composites = {
            symbol: combine_weighted(
                scored[symbol], context.weights, unavailable=read.unavailable
            )[0]
            for symbol, read in usable.items()
        }
        standardized = self._standardize(composites)

        for symbol, read in usable.items():
            z = standardized[symbol]
            signals[symbol] = self._build_signal(
                symbol, read, scored[symbol], context,
                # Phi is monotone, so the 0-100 score orders names exactly as
                # the z does while staying on the scale the >=60 / >=45 gates
                # and every downstream report already read.
                final_score=100.0 * _NORMAL.cdf(z),
                composite_z=z,
            )
        return signals

    @property
    def requires_full_batch(self) -> bool:
        """Cross-sectional scoring is a statement about a whole universe, so a
        per-ticker loop is semantically wrong rather than merely slow."""
        return self._scoring in _CROSS_SECTIONAL_MODES

    def _empty_signal(self, symbol: str) -> StrategySignal:
        return StrategySignal(
            symbol=symbol, signal="AVOID", score=0.0, trigger="None",
            entry_price=0.0, stop_price=0.0, target_price=0.0,
            reward_risk=0.0, probability_profit=0.0,
            component_scores={}, rationale="No feature data available",
        )

    def _rank_components(self, reads: Dict[str, "_ComponentRead"]) -> Dict[str, Dict[str, float]]:
        """Convert each component to its percentile rank within the batch.

The weighted sum this replaces adds four incommensurable quantities: an
        ordinal on three levels, a binary, a right-skewed continuous, and —
        after the drift shrinkage of section 21 — a near-constant with a
        standard deviation around 0.05. Percentile ranks make them commensurable
        by construction and invariant to each component's marginal
        distribution, so a component's influence no longer depends on the
        accident of its units: whether Volume is expressed as a ratio, its
        logarithm, or anything else monotone, the composite is unchanged.

        **What this does not fix, despite being the review's proposed remedy
        for it.** A component that is near-constant across the universe still
        consumes its full share of the score budget. Ranking ties gives every
        name the same percentile (0.6 for a five-name universe, whatever the
        constant is), so MC_Prob contributes a flat number here exactly as it
        did under the weighted sum — a different flat number, but still flat,
        and still 30% of the budget spent on a component separating nobody.
        Making influence track discrimination needs the weights themselves to
        respond to realized dispersion or information coefficient, which is a
        change to the weight learner rather than to the combination rule.
        Lowering MC_Prob's configured weight is the blunt version available
        today.

        Percentile rather than the inverse-normal form: it keeps the composite
        on 0-100, so the existing `score >= 60` / `>= 45` thresholds stay
        syntactically valid and gain a clean reading — "60th percentile of the
        weighted composite" — instead of requiring every YAML to be rewritten
        against a z-scale. The inverse-normal form is available as the separate
        ``probit_composite`` mode (see _probit_components), which keeps the
        0-100 score by mapping back through Phi and exposes the z alongside it;
        prefer that one when the score is being consumed as a magnitude rather
        than as an ordering, since a weighted sum of percentiles has a spread
        that depends on how many components were measurable that day.

        Ties take the average rank, which matters here: Breakout is binary and
        Trend has three levels, so ties are the common case rather than an edge
        case, and a first-past-the-post rank would order tied names by
        whatever the dict iteration produced.

        An unavailable component is left at its raw value and excluded from the
        combination by the caller — ranking a column no one has would hand
        every ticker the same percentile, which is strictly worse than dropping
        the weight: it spends the budget to say nothing.
        """
        symbols = list(reads)
        ranked: Dict[str, Dict[str, float]] = {symbol: {} for symbol in symbols}

        for component in COMPONENT_NAMES:
            measurable = [s for s in symbols if component not in reads[s].unavailable]
            if not measurable:
                for symbol in symbols:
                    ranked[symbol][component] = reads[symbol].components[component]
                continue

            values = pd.Series(
                {s: reads[s].components[component] for s in measurable}, dtype=float
            )
            percentiles = values.rank(method="average", pct=True)
            for symbol in symbols:
                ranked[symbol][component] = (
                    float(percentiles[symbol]) if symbol in percentiles.index
                    else reads[symbol].components[component]
                )

        return ranked

    def _probit_components(
        self, reads: Dict[str, "_ComponentRead"]
    ) -> Dict[str, Dict[str, float]]:
        """Cross-sectional percentile ranks pushed through Phi^-1.

        The rank composite made the components commensurable; this makes them
        *additive*. A percentile is a uniform variate, and a weighted sum of
        uniforms has a spread that depends on how many components were
        measurable and how correlated they are — so the same 0.72 means
        different things on different dates and in different universes. Normal
        scores are the standard fix (Van der Waerden): rank, map to a normal
        quantile, and the sum inherits a scale that is stable across dates.

        **The plotting position is load-bearing, not a rounding detail.**
        Ranks are converted with r/(N+1), not the r/N that `pct=True` gives.
        Under r/N the best name in the universe ranks at exactly 1.0 and
        Phi^-1(1) is +inf — which does not fail loudly on that one ticker, it
        propagates into the weighted sum, makes the cross-sectional mean and
        standard deviation nan, and takes down every score on the date. r/(N+1)
        is the Blom/Van der Waerden convention and keeps the argument strictly
        inside (0, 1) for every N.

        Ties take the average rank, as under rank scoring: Breakout is binary
        and Trend has three levels, so ties are the common case, and a
        first-past-the-post rank would order tied names by dict iteration
        order.

        An unavailable component is left at its raw value and dropped from the
        combination by the caller, rather than being ranked — ranking a column
        nobody has hands every ticker the same quantile, which spends the
        weight to say nothing.
        """
        symbols = list(reads)
        scored: Dict[str, Dict[str, float]] = {symbol: {} for symbol in symbols}

        for component in COMPONENT_NAMES:
            measurable = [s for s in symbols if component not in reads[s].unavailable]
            if not measurable:
                for symbol in symbols:
                    scored[symbol][component] = reads[symbol].components[component]
                continue

            values = pd.Series(
                {s: reads[s].components[component] for s in measurable}, dtype=float
            )
            # Ranks are 1..N; dividing by N+1 keeps Phi^-1's argument off both
            # singularities however large or small the cross-section is.
            quantiles = values.rank(method="average") / (len(values) + 1)
            for symbol in symbols:
                scored[symbol][component] = (
                    _NORMAL.inv_cdf(float(quantiles[symbol]))
                    if symbol in quantiles.index
                    else reads[symbol].components[component]
                )

        return scored

    @staticmethod
    def _standardize(composites: Dict[str, float]) -> Dict[str, float]:
        """Centre and scale the composite so the date is mean-zero, variance-one.

        This is the step that makes a score comparable across dates. The probit
        transform fixes each component's marginal shape, but the *combination*
        still has a variance set by the weights and by how correlated the
        components happened to be that day — standardizing removes both, so a
        z of 1.5 means "1.5 standard deviations better than today's universe"
        on every date, which is the contract a portfolio optimizer needs from
        an alpha input.

        A degenerate cross-section (every name identical, or a universe of one)
        has no dispersion to divide by and collapses to zero rather than nan.
        """
        symbols = list(composites)
        values = np.array([composites[s] for s in symbols], dtype=float)

        centred = values - values.mean()
        dispersion = float(np.std(centred, ddof=0))
        if dispersion < _MIN_COMPOSITE_DISPERSION:
            return {symbol: 0.0 for symbol in symbols}

        standardized = centred / dispersion
        return {symbol: float(z) for symbol, z in zip(symbols, standardized)}

    def _read_components(
        self, symbol: str, features: pd.DataFrame, context: StrategyContext
    ) -> Optional["_ComponentRead"]:
        """Raw component values for one ticker, before any combination."""
        if features.empty:
            return None

        latest = features.iloc[-1]
        close = _clean(latest.get("close")) or 0.0
        sma50 = _clean(latest.get("sma_50"))
        sma200 = _clean(latest.get("sma_200"))
        donchian_upper = _clean(latest.get("donchian_upper_20"))
        volume_ratio = _clean(latest.get("volume_ratio_20"))
        atr = _clean(latest.get("atr_14"))

        # Two different reasons a component can be unmeasurable, and they must
        # NOT be handled the same way:
        #
        # - **The pipeline did not compute it.** Inside a batched UMA the
        #   caller builds one context for the whole round, so there is no
        #   per-ticker Monte Carlo result. Nothing about the stock is different;
        #   a different code path ran. Scoring that at 0 made the identical
        #   stock on the identical day score ~12 points lower in an ensemble
        #   than standalone — an artifact, and the weight is renormalized away.
        #
        # - **The stock does not have the history.** A missing SMA-200 means a
        #   recent listing, and that *is* information about the stock. Dropping
        #   the trend weight there would renormalize the remaining components up
        #   and make the system score a young, illiquid name *higher* than a
        #   seasoned one — least caution where an Indian micro-cap universe
        #   warrants most. Those keep their conservative zero.
        unavailable: List[str] = []

        # 1. Trend score
        if sma200 is None:
            trend_score = 0.0
        elif sma50 is not None and close > sma50 and close > sma200 and sma50 > sma200:
            trend_score = 1.0
        elif close > sma200:
            trend_score = 0.5
        else:
            trend_score = 0.0

        # 2. Breakout score
        if donchian_upper is None:
            breakout_score = 0.0
        elif close > donchian_upper:
            breakout_score = 1.0
        else:
            breakout_score = 0.0

        # 3. Volume score
        volume_score = 0.0 if volume_ratio is None else min(volume_ratio / 2.0, 1.0)

        # 4. Monte Carlo probability-of-profit score.
        if context.mc_result is None:
            prob_profit = 0.0
            mc_score = 0.0
            unavailable.append("MC_Prob")
        else:
            prob_profit = context.mc_result.probability_profit
            mc_score = max(0.0, min(1.0, prob_profit))

        return _ComponentRead(
            components={
                "Trend": trend_score,
                "Breakout": breakout_score,
                "Volume": volume_score,
                "MC_Prob": mc_score,
            },
            unavailable=unavailable,
            close=close,
            atr=atr,
            prob_profit=prob_profit,
        )

    def _build_signal(
        self,
        symbol: str,
        read: "_ComponentRead",
        score_inputs: Dict[str, float],
        context: StrategyContext,
        final_score: Optional[float] = None,
        composite_z: Optional[float] = None,
    ) -> StrategySignal:
        """Turn component values into a signal.

        `score_inputs` is what the weights are applied to — the raw components
        under weighted-sum scoring, their within-batch percentile ranks under
        rank-composite scoring, their normal scores under probit-composite.
        `read.components` stays the raw values throughout, because the trigger,
        the reported component scores and the rationale are all statements
        about the indicators themselves.

        `final_score` and `composite_z` are supplied only by probit-composite
        scoring, where the score depends on the whole cross-section and so
        cannot be derived from one symbol's inputs here.
        """
        close, atr = read.close, read.atr
        prob_profit = read.prob_profit
        unavailable = read.unavailable
        component_scores = read.components

        if final_score is None:
            final_score, _ = combine_weighted(
                score_inputs, context.weights, unavailable=unavailable
            )
        trigger = select_trigger(component_scores)

        # 5. Stop / target from ATR (YAML multipliers, falling back to context defaults)
        exit_rules = self.exit_rules()
        stop_mult = exit_rules.get("stop_loss", {}).get("multiplier", context.risk.atr_stop_multiplier)
        target_mult = exit_rules.get("take_profit", {}).get("multiplier", context.risk.atr_target_multiplier)
        stop_price, target_price = calculate_stop_target(close, atr, stop_mult, target_mult)

        # Reward:risk is measured net of estimated round-trip friction
        # (brokerage, STT, exchange and SEBI charges, GST, stamp duty and
        # slippage) rather than gross, so the min_reward_risk gate below is a
        # statement about money actually kept — see src/risk.py::net_reward_risk.
        stop_valid = stop_price < close
        if stop_valid:
            reward_risk = net_reward_risk(
                entry_price=close,
                stop_price=stop_price,
                target_price=target_price,
                buy_cost_pct=context.risk.buy_cost_pct,
                sell_cost_pct=context.risk.sell_cost_pct,
            )
            gross_reward_risk = (
                (target_price - close) / (close - stop_price) if close != stop_price else 0.0
            )
        else:
            reward_risk = 0.0
            gross_reward_risk = 0.0

        passed_score = final_score >= 60
        passed_prob = prob_profit >= context.risk.target_prob_profit
        passed_rr = reward_risk >= context.risk.min_reward_risk
        passed_price = close >= context.risk.min_price_inr

        # An unavailable component reports as such rather than as a 0.00 that
        # reads like a measurement. The probability gate is the one that
        # matters most: with no Monte Carlo result it fails closed, so a
        # rule-based member inside a batched UMA cannot issue BUY at all.
        # That is the right default — a compliance gate with no evidence
        # either way should refuse, not wave the trade through untested — but
        # it is a real limitation of mixing an MC-dependent member into a
        # batched ensemble, and it should be legible in the rationale rather
        # than looking like the stock scored badly.
        def _component(name: str, value: float, fmt: str = "{:.2f}") -> str:
            return f"{name}=n/a" if name in unavailable else f"{name}={fmt.format(value)}"

        probability_note = (
            "prob(no MC result):FAIL"
            if "MC_Prob" in unavailable
            else f"prob({prob_profit:.2f})>={context.risk.target_prob_profit}:"
                 f"{'PASS' if passed_prob else 'FAIL'}"
        )

        rationale = "; ".join([
            f"Score={final_score:.1f}",
            _component("Trend", component_scores["Trend"], "{:.1f}"),
            _component("Breakout", component_scores["Breakout"], "{:.1f}"),
            _component("Volume", component_scores["Volume"]),
            _component("MC_Prob", component_scores["MC_Prob"]),
            f"score>=60:{'PASS' if passed_score else 'FAIL'}",
            probability_note,
            f"rr({reward_risk:.2f})>={context.risk.min_reward_risk}:{'PASS' if passed_rr else 'FAIL'}",
            f"price({close:.2f})>={context.risk.min_price_inr}:{'PASS' if passed_price else 'FAIL'}",
            "stop<entry:VALID" if stop_valid else "stop>=entry:INVALID",
        ])

        if not stop_valid:
            signal = "AVOID"
        elif passed_score and passed_prob and passed_rr and passed_price:
            signal = "BUY"
        elif final_score >= 45:
            signal = "WATCH"
        else:
            signal = "AVOID"

        return StrategySignal(
            symbol=symbol,
            signal=signal,
            score=round(final_score, 2),
            trigger=trigger,
            entry_price=close,
            stop_price=stop_price,
            target_price=target_price,
            reward_risk=round(reward_risk, 4),
            probability_profit=round(prob_profit, 6),
            component_scores=component_scores,
            rationale=rationale,
            extra={
                "mc_var_95_pct": round(context.mc_result.var_95, 6) if context.mc_result else 0.0,
                "mc_cvar_95_pct": round(context.mc_result.cvar_95, 6) if context.mc_result else 0.0,
                # Reported alongside the (net) reward_risk so the gap between
                # them is visible: on ATR-tight stops, round-trip friction is a
                # large fraction of the risk being taken, and a strategy whose
                # gross ratio clears the gate but whose net ratio does not is
                # the exact failure this platform is meant to surface.
                "gross_reward_risk": round(gross_reward_risk, 4),
                "round_trip_cost_pct": round(
                    context.risk.buy_cost_pct + context.risk.sell_cost_pct, 6
                ),
                # The standardized cross-sectional composite, under
                # probit_composite scoring only. Kept unrounded and alongside
                # the 0-100 score rather than replacing it: `score` is what the
                # gates and the reports read, while this is the mean-zero,
                # variance-one quantity an optimizer wants as its alpha input.
                **({} if composite_z is None else {"composite_z": float(composite_z)}),
            },
        )
