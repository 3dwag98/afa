"""Per-trade risk: stop and target placement, and position sizing.

One family, not two. Until T17 this module also carried `calculate_stop_loss`,
`calculate_target_price`, `calculate_position_size`, `calculate_portfolio_risk`
and `check_risk_limits` — a second, older family answering the same questions
with different constants, and with no importer anywhere, not even a test.

They disagreed by amounts that matter. A stop off a 5-rupee ATR at 100 was
**92.5** through `calculate_stop_target` and **90.0** through
`calculate_stop_loss`. With no ATR it was **98.0** against **95.0** — a 2.5x
difference in risk-per-share, which is the denominator `calculate_quantity`
divides by, so the same trade sized 2.5x larger depending on which function the
caller reached for.

`calculate_portfolio_risk` was worse, because its error was structural rather
than a tuning difference: it assumed **zero correlation by default**. On twenty
names at 20% volatility with pairwise correlation 0.85 — an ordinary Indian
long-only book, where everything loads on the same market — it reported **4.5%**
portfolio volatility against a true **18.5%**. A 4.1x understatement, and in
the direction that makes a book look diversified when it is one bet.

Book-level risk lives in `src/portfolio.py`, whose `portfolio_volatility`
measures the covariance instead of assuming it away. Nothing here reasons about
more than one position.
"""

import math
import numpy as np
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from portfolio_agent.config.schema import AppConfig


def calculate_stop_target(
    entry_price: float,
    atr: Optional[float],
    stop_multiplier: float = 1.5,
    target_multiplier: float = 2.0,
) -> Tuple[float, float]:
    """Calculate stop loss and target prices based on ATR.

    Args:
        entry_price: Entry price of the position.
        atr: Average True Range value, or None.
        stop_multiplier: ATR multiple below entry for the stop (default 1.5).
        target_multiplier: ATR multiple above entry for the target (default 2.0).

    Returns:
        Tuple of (stop_price, target_price), both rounded.
    """
    # Use fallback if atr is None or <= 0
    if atr is None or atr <= 0:
        stop = entry_price * 0.98  # 2% fallback stop
        target = entry_price * 1.03  # 3% fallback target
    else:
        stop = entry_price - stop_multiplier * atr
        target = entry_price + target_multiplier * atr

    # Stop cannot be negative
    stop = max(0.0, stop)

    return (round(stop, 2), round(target, 2))


def net_reward_risk(
    entry_price: float,
    stop_price: float,
    target_price: float,
    buy_cost_pct: float,
    sell_cost_pct: float,
) -> float:
    """Reward:risk ratio measured *after* round-trip transaction costs.

    A gross reward:risk of 1.33 (the ATR 1.5x/2.0x default) looks like a
    perfectly good trade until brokerage, STT, exchange and SEBI charges, GST,
    stamp duty and slippage are charged against both legs — on a tight stop
    those costs are a meaningful fraction of the move. This nets them out on
    both sides so the platform's `min_reward_risk` gate compares like with
    like (docs/QUANT_RESEARCH.md section 12):

        effective entry cost  = entry  * (1 + buy_cost_pct)
        net target proceeds   = target * (1 - sell_cost_pct)
        net stop proceeds     = stop   * (1 - sell_cost_pct)

    Args:
        entry_price: Entry price per share.
        stop_price: Stop price per share.
        target_price: Target price per share.
        buy_cost_pct: Buy-leg friction as a fraction of turnover.
        sell_cost_pct: Sell-leg friction as a fraction of turnover.

    Returns:
        Net reward:risk ratio, or 0.0 when the trade has no net upside or the
        stop sits at/above the cost-adjusted entry (no measurable risk).
    """
    if entry_price <= 0:
        return 0.0

    effective_entry = entry_price * (1.0 + buy_cost_pct)
    net_target = target_price * (1.0 - sell_cost_pct)
    net_stop = stop_price * (1.0 - sell_cost_pct)

    net_reward = net_target - effective_entry
    net_risk = effective_entry - net_stop

    if net_risk <= 0 or net_reward <= 0:
        return 0.0

    return net_reward / net_risk


def net_realized_return_pct(
    entry_price: float,
    exit_price: float,
    buy_cost_pct: float,
    sell_cost_pct: float,
) -> float:
    """A completed round trip's return in percent, net of both legs' friction.

    Same convention as net_reward_risk(), applied after the fact instead of
    before: the buy leg's cost is capitalized into the basis and the sell leg's
    is deducted from the proceeds, so the result is the return the portfolio
    actually kept.

        net_pct = (exit * (1 - sell_cost) - entry * (1 + buy_cost))
                  / (entry * (1 + buy_cost)) * 100

    Args:
        entry_price: Fill price per share on the buy leg.
        exit_price: Fill price per share on the sell leg.
        buy_cost_pct: Buy-leg friction as a fraction of turnover.
        sell_cost_pct: Sell-leg friction as a fraction of turnover.

    Returns:
        Net return in percent; 0.0 when the entry price is unusable.
    """
    if entry_price <= 0:
        return 0.0
    cost_basis = entry_price * (1.0 + buy_cost_pct)
    proceeds = exit_price * (1.0 - sell_cost_pct)
    return (proceeds - cost_basis) / cost_basis * 100.0


def to_net_realized_trades(
    trade_history: List[Dict[str, Any]],
    buy_cost_pct: float,
    sell_cost_pct: float,
) -> List[Dict[str, Any]]:
    """Restate a gross trade history in net-of-friction terms.

    The stored trade log is deliberately gross: `return_pct` there is the price
    move, which is what a report should show. Kelly is not a report. Both of
    its inputs have to be measured after friction, and the failure mode is
    asymmetric in the dangerous direction (docs/QUANT_RESEARCH.md section 4):

    - **b, the payoff ratio.** With ~0.8% of round-trip friction on an
      Indian delivery trade, a gross 2.0 reward:risk realizes nearer 1.8. A
      gross b therefore overstates the edge, and f* = p - (1-p)/b is more
      sensitive to b than to anything except p.
    - **p, the win probability.** A trade that gained 0.3% gross *lost* money.
      Classifying it as a WIN inflates p and b simultaneously — it adds a
      phantom win and drags the average win magnitude down by less than the
      loss it actually was.

    Every record is therefore re-priced and re-classified from the entry and
    exit prices, so an outcome label computed gross upstream cannot leak into
    the sizing decision. Records without a usable entry/exit pair (still open,
    or missing prices) are dropped rather than guessed at.

    Args:
        trade_history: Trade dicts carrying "entry_price" and "exit_price"
            (the shape AgentBrain.trade_history and storage.get_trade_history
            produce).
        buy_cost_pct: Buy-leg friction as a fraction of turnover.
        sell_cost_pct: Sell-leg friction as a fraction of turnover.

    Returns:
        New trade dicts with net "return_pct" and a net-consistent "outcome",
        every other key preserved.
    """
    restated: List[Dict[str, Any]] = []
    for trade in trade_history:
        entry_price = trade.get("entry_price")
        exit_price = trade.get("exit_price")
        if not entry_price or not exit_price:
            continue
        try:
            net_pct = net_realized_return_pct(
                float(entry_price), float(exit_price), buy_cost_pct, sell_cost_pct
            )
        except (TypeError, ValueError):
            continue
        restated.append({
            **trade,
            "return_pct": net_pct,
            "outcome": "WIN" if net_pct > 0 else "LOSS",
        })
    return restated


def calculate_quantity(
    entry_price: float,
    stop_price: float,
    config: AppConfig
) -> int:
    """Calculate position quantity based on risk parameters.

    Args:
        entry_price: Entry price per share.
        stop_price: Stop loss price per share.
        config: Application configuration.

    Returns:
        Integer quantity >= 0.
    """
    # Risk amount = portfolio_value_inr * risk_per_trade_pct
    risk_amount = config.risk.portfolio_value_inr * config.risk.risk_per_trade_pct

    # Risk per share = entry_price - stop_price
    risk_per_share = entry_price - stop_price

    # If risk per share <= 0, return 0
    if risk_per_share <= 0:
        return 0

    # quantity = floor(risk_amount / risk_per_share)
    quantity = math.floor(risk_amount / risk_per_share)

    # Max position value = portfolio_value_inr * max_single_position_pct
    max_position_value = config.risk.portfolio_value_inr * config.risk.max_single_position_pct

    # Reduce quantity if quantity * entry_price > max position value
    if quantity * entry_price > max_position_value:
        quantity = math.floor(max_position_value / entry_price)

    return max(0, quantity)


def shrink_win_probability(
    wins: int,
    total: int,
    prior_strength: float = 20.0,
    prior_win_rate: float = 0.5,
) -> float:
    """Beta-Binomial posterior-mean win rate (Bayesian shrinkage).

    The raw win rate wins/total is an unbiased but high-variance estimate at
    the sample sizes a retail-scale strategy actually accumulates: at 50
    trades its standard error is already ~7 percentage points. Kelly is
    asymmetric in that error — over-betting off an optimistic p costs far more
    long-run growth than under-betting off a pessimistic one — so the estimate
    is shrunk toward a no-edge prior instead of taken at face value:

        p_hat = (wins + a) / (total + a + b),  a = m*q, b = m*(1-q)

    with prior strength m (in pseudo-trades) and prior win rate q. m = 20 and
    q = 0.5 means "start from a coin flip worth 20 trades of evidence": with
    50 real trades a raw 70% win rate is reported as 64%, and the pull fades
    as real evidence accumulates. See docs/QUANT_RESEARCH.md section 4.

    Args:
        wins: Number of realized winning trades.
        total: Number of realized (WIN or LOSS) trades.
        prior_strength: Prior weight m in pseudo-trades; 0 disables shrinkage.
        prior_win_rate: Prior win rate q in [0, 1].

    Returns:
        Shrunk win probability in [0, 1].
    """
    if total <= 0:
        return 0.0

    m = max(0.0, prior_strength)
    q = min(1.0, max(0.0, prior_win_rate))
    alpha = m * q
    beta = m * (1.0 - q)
    return (wins + alpha) / (total + alpha + beta)


class KellyInputs(NamedTuple):
    """Everything Kelly needs about the realized payoff distribution.

    The magnitudes travel alongside the ratio deliberately. `reward_risk_ratio`
    alone (b = g/l) is enough for the *binary-bet* Kelly fraction, but a
    stop-loss equity trade is not a binary bet, and sizing it needs the loss
    magnitude `l` in its own right — see kelly_allocation_fraction() for why
    dropping it understates the allocation by a factor of 1/l.
    """

    win_probability: float  # p, Beta-shrunk realized win rate
    reward_risk_ratio: float  # b = avg_win_pct / avg_loss_pct
    avg_win_pct: float  # g, mean magnitude of a winning trade, in percent
    avg_loss_pct: float  # l, mean magnitude of a losing trade, in percent


def estimate_kelly_inputs(
    trade_history: List[Dict[str, Any]],
    min_trades: int = 50,
    shrinkage_strength: float = 20.0,
) -> Optional[KellyInputs]:
    """Estimate the realized payoff distribution Kelly sizes from.

    Per docs/QUANT_RESEARCH.md section 4: p is the realized win rate and b is
    the average win magnitude divided by the average loss magnitude (in the
    same reward:risk units this platform already reports on StrategySignal).

    **The input contract is net, not gross.** Both "outcome" and "return_pct"
    must already have round-trip friction — brokerage, STT, exchange and SEBI
    charges, GST, stamp duty, slippage and (where modelled) capital gains tax —
    deducted. This function cannot verify that, so the two call sites do it
    explicitly and identically: the backtest engine divides `net_pnl` by the
    cost basis (backtest_engine.py::_net_return_pct), and the live orchestrator
    restates its stored gross history through to_net_realized_trades(). Feeding
    gross figures here inflates b by roughly the friction stack and mislabels
    marginally-positive trades as wins, and f* over-bets on both.

    Two guards against sizing off noise, both of which matter because Kelly
    punishes over-betting much harder than under-betting:

    1. A hard sample-size floor (`min_trades`, default 50). Below ~50 realized
       trades the win-rate standard error is wide enough (±5-7 percentage
       points) that f* is dominated by estimation error.
    2. Beta-prior shrinkage of the win rate toward 0.5 (see
       shrink_win_probability), which keeps a lucky early streak from being
       read as a large edge even once the floor is cleared.

    Args:
        trade_history: Trade dicts with "outcome" ("WIN"/"LOSS"/other) and
            "return_pct" keys (same shape as AgentBrain.trade_history).
        min_trades: Minimum realized (WIN/LOSS) trades required to trust the
            estimate at all.
        shrinkage_strength: Beta-prior strength in pseudo-trades; 0 returns
            the raw win rate.

    Returns:
        A KellyInputs, or None when there isn't enough realized history, or
        losses average to zero (b undefined) — callers should fall back to
        fixed-fractional sizing in that case.
    """
    realized = [t for t in trade_history if t.get("outcome") in ("WIN", "LOSS")]
    if len(realized) < min_trades:
        return None

    wins = [t for t in realized if t.get("outcome") == "WIN"]
    losses = [t for t in realized if t.get("outcome") == "LOSS"]
    if not wins or not losses:
        return None

    win_probability = shrink_win_probability(
        wins=len(wins), total=len(realized), prior_strength=shrinkage_strength
    )
    avg_win_pct = float(np.mean([abs(t.get("return_pct", 0.0)) for t in wins]))
    avg_loss_pct = float(np.mean([abs(t.get("return_pct", 0.0)) for t in losses]))
    if avg_loss_pct <= 0:
        return None

    return KellyInputs(
        win_probability=win_probability,
        reward_risk_ratio=avg_win_pct / avg_loss_pct,
        avg_win_pct=avg_win_pct,
        avg_loss_pct=avg_loss_pct,
    )


# Hard ceiling on the fractional-Kelly multiplier kappa. Kelly assumes p and b
# are known and the payoff distribution is roughly symmetric. Neither holds
# here: p and b are estimated from a few dozen trades, and a loss that locks at
# the lower circuit for several sessions realizes far worse than the modelled
# stop, so the measured b is biased upward and f* with it. Quarter-Kelly keeps
# roughly half of full-Kelly's growth rate at a small fraction of its drawdown.
MAX_KELLY_FRACTION = 0.25


# This platform runs an unlevered long-only cash book: it never shorts and
# never borrows, so no single position's *allocation* may exceed the whole
# portfolio. Once the Kelly units are correct (see kelly_allocation_fraction)
# f* routinely exceeds 1, and it needs a leverage constraint that says so
# explicitly rather than a silent min(1.0, ...) buried in the formula.
MAX_KELLY_ALLOCATION_FRACTION = 1.0


def calculate_kelly_fraction(win_probability: float, reward_risk_ratio: float) -> float:
    """Kelly *stake* fraction for a binary, all-or-nothing bet: f = p - (1-p)/b.

    **This is not a position size.** It is the classical Kelly result for a bet
    that returns b times the stake with probability p and loses the *entire*
    stake with probability 1-p. A stop-loss equity trade is not that bet: a
    loser gives back the distance from entry to stop (a few percent of the
    position), not the position. Sizing a portfolio with this number allocates
    roughly l times too little, where l is the loss-given-stop — see
    kelly_allocation_fraction(), which is what position sizing calls.

    Retained because it is the correct quantity in its own right and the
    numerator of the allocation formula, and because reporting it alongside
    the allocation makes the units visible rather than implicit.

    Clamped to [0, 1] — the natural range for a stake fraction. A negative f
    (an unprofitable edge) becomes 0.
    """
    if reward_risk_ratio <= 0:
        return 0.0
    f_star = win_probability - (1.0 - win_probability) / reward_risk_ratio
    return max(0.0, min(1.0, f_star))


def kelly_allocation_fraction(
    win_probability: float,
    avg_win_pct: float,
    avg_loss_pct: float,
) -> float:
    """Growth-optimal fraction of *wealth* to allocate to a stop-loss trade.

    For a position that gains a fraction g of its value with probability p and
    loses a fraction l with probability 1-p, the log-growth-maximizing
    allocation is

        f*_alloc = (p*g - (1-p)*l) / (g*l) = (p - (1-p)/b) / l,   b = g/l

    The numerator is the per-rupee-of-position edge; dividing by g*l converts
    it from "edge per rupee at risk" into "rupees of position per rupee of
    wealth". The binary-bet form f = p - (1-p)/b is that expression with l = 1
    — correct only when a loss wipes out the whole stake.

    The distinction is not cosmetic. At p = 0.55, b = 1.8 and a 6%
    loss-given-stop, the binary form returns 0.30 while the allocation is 5.0:
    a factor of 1/l = 16.7. Applied as an allocation, the binary form runs the
    book at ~6% of the quarter-Kelly it claims. Worse, the error is not a
    constant rescaling — l varies per signal, so a wide-stop trade (which
    *should* get a smaller allocation) is distorted differently from a tight
    one, and no amount of raising kappa recovers it.

    Deliberately unbounded above: an f* of 5.0 is a true statement about a
    growth-optimal bet, and hiding it behind a min(1.0, ...) would make the
    clamp — rather than the risk policy — the thing that sizes the book. The
    leverage and concentration limits are applied explicitly by the caller
    (see calculate_kelly_quantity).

    Args:
        win_probability: p, the (shrunk) probability the trade wins.
        avg_win_pct: g, mean magnitude of a winning trade in percent.
        avg_loss_pct: l, mean magnitude of a losing trade in percent. Measured
            from realized history rather than the modelled stop distance
            because a stop that gaps or locks at the lower circuit realizes
            worse than it was set — history carries that, the stop does not.

    Returns:
        Growth-optimal allocation as a fraction of portfolio value, >= 0.
        Zero when the edge is negative or either magnitude is unusable.
    """
    g = abs(float(avg_win_pct)) / 100.0
    l = abs(float(avg_loss_pct)) / 100.0
    if g <= 0.0 or l <= 0.0:
        return 0.0

    edge = win_probability * g - (1.0 - win_probability) * l
    if edge <= 0.0:
        return 0.0
    return edge / (g * l)


def calculate_kelly_quantity(
    entry_price: float,
    portfolio_value_inr: float,
    max_single_position_pct: float,
    win_probability: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    kelly_fraction: float = MAX_KELLY_FRACTION,
) -> int:
    """Fractional-Kelly position sizing, in allocation units.

    Constraints are applied in this order, each one explicit:

    1. ``f*_alloc`` from kelly_allocation_fraction() — growth-optimal, unbounded.
    2. ``kappa``, the fractional-Kelly multiplier, hard-capped at quarter-Kelly.
    3. ``MAX_KELLY_ALLOCATION_FRACTION`` — the unlevered-cash-book constraint.
    4. ``max_single_position_pct`` — the concentration cap.

    A note on what this means in practice, because it is the honest reading of
    the corrected arithmetic: with the units right, f*_alloc for any trade with
    a real edge is large (order 1-10), so steps 3 and 4 bind almost always and
    the concentration cap — not Kelly — is what sizes a position. That is not a
    defect of this function; it is what fractional Kelly on a stop-loss book
    actually implies, and it is why the missing piece is portfolio-level
    (covariance-aware) construction rather than a better per-trade formula.
    Kelly still binds where it matters most: as the edge approaches zero,
    f*_alloc collapses continuously to 0 and sizes the position down, which the
    binary-bet form also did but at 1/l of the correct scale.

    Args:
        entry_price: Entry price per share.
        portfolio_value_inr: Total portfolio value in INR.
        max_single_position_pct: Hard cap on position value as a fraction of
            portfolio value — Kelly sizing can never exceed this, matching
            the platform's existing fixed-fractional cap.
        win_probability: Realized win rate p (see estimate_kelly_inputs).
        avg_win_pct: Realized average win magnitude g, in percent.
        avg_loss_pct: Realized average loss magnitude l, in percent.
        kelly_fraction: Fractional-Kelly multiplier kappa, clamped to
            [0, MAX_KELLY_FRACTION]. The clamp is applied here rather than
            trusted to the caller, so no config, YAML or test fixture can
            route around it.

    Returns:
        Integer quantity >= 0.
    """
    if entry_price <= 0:
        return 0

    f_star = kelly_allocation_fraction(win_probability, avg_win_pct, avg_loss_pct)
    kappa = max(0.0, min(MAX_KELLY_FRACTION, kelly_fraction))
    position_fraction = min(f_star * kappa, MAX_KELLY_ALLOCATION_FRACTION)

    position_value = portfolio_value_inr * position_fraction
    max_position_value = portfolio_value_inr * max_single_position_pct
    position_value = min(position_value, max_position_value)

    return max(0, math.floor(position_value / entry_price))


def calculate_position_quantity(
    entry_price: float,
    stop_price: float,
    config: AppConfig,
    trade_history: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """Single position-sizing entry point: fixed-fractional by default,
    switching to fractional-Kelly once config.risk.use_kelly_sizing is set
    and enough realized trade history exists to estimate it reliably.

    **Kelly is applied as a ceiling, not a licence.** The size taken is the
    smaller of the Kelly allocation and the fixed-fractional risk budget. Two
    reasons, both a consequence of getting the Kelly units right:

    - The risk budget is priced off *this* trade's stop distance, which is
      known exactly, while Kelly's loss magnitude l is a historical average
      across trades with different stops. Where they disagree, the known
      quantity should win.
    - Correcting the units (kelly_allocation_fraction) scales f* up by 1/l,
      roughly 17x at a 6% stop. Without this ceiling that correction would
      land as a silent increase in position size across the whole book —
      turning a units bug fix into a leverage change nobody asked for.

    So enabling Kelly can only ever size *down* relative to fixed-fractional:
    it de-risks as the estimated edge approaches zero and is otherwise
    inactive. That is the intended behaviour for an unlevered long-only cash
    book; growth-optimal sizing across positions is a portfolio-level problem
    (covariance-aware allocation), not something a per-trade formula can do.

    Args:
        entry_price: Entry price per share.
        stop_price: Stop loss price per share.
        config: Application configuration.
        trade_history: Realized trade history (e.g. AgentBrain.trade_history
            or a backtest engine's trade_log) used to estimate Kelly inputs.
            Ignored when config.risk.use_kelly_sizing is False.

    Returns:
        Integer quantity >= 0.
    """
    fixed_fractional = calculate_quantity(entry_price, stop_price, config)

    if config.risk.use_kelly_sizing and trade_history:
        kelly_inputs = estimate_kelly_inputs(
            trade_history,
            min_trades=config.risk.kelly_min_trades,
            shrinkage_strength=config.risk.kelly_shrinkage_strength,
        )
        if kelly_inputs is not None:
            kelly_quantity = calculate_kelly_quantity(
                entry_price=entry_price,
                portfolio_value_inr=config.risk.portfolio_value_inr,
                max_single_position_pct=config.risk.max_single_position_pct,
                win_probability=kelly_inputs.win_probability,
                avg_win_pct=kelly_inputs.avg_win_pct,
                avg_loss_pct=kelly_inputs.avg_loss_pct,
                kelly_fraction=config.risk.kelly_fraction,
            )
            return min(kelly_quantity, fixed_fractional)

    return fixed_fractional


def calculate_max_loss(quantity: int, entry_price: float, stop_price: float) -> float:
    """Calculate maximum loss for a position.

    Args:
        quantity: Number of shares.
        entry_price: Entry price per share.
        stop_price: Stop loss price per share.

    Returns:
        Maximum loss amount in INR.
    """
    return quantity * (entry_price - stop_price)
