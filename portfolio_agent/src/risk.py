"""Risk management module."""

import math
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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


@dataclass(frozen=True)
class KellyInputs:
    """Everything Kelly needs about the realized payoff distribution.

    Kelly for a stop-loss trade is a function of three quantities, not two:
    the win probability p, the average *gain* g, and the average *loss* l.
    This used to return only (p, b = g/l), which is sufficient for the binary
    bet where a loss costs the whole stake and insufficient for anything else
    — the allocation fraction needs the scale of l, not just its ratio to g
    (see kelly_allocation_fraction). Both averages are therefore kept.

    Attributes:
        win_probability: Shrunk realized win rate p, in [0, 1].
        avg_win_pct: Mean magnitude of a winning trade's return, in percent.
        avg_loss_pct: Mean magnitude of a losing trade's return, in percent.
    """

    win_probability: float
    avg_win_pct: float
    avg_loss_pct: float

    @property
    def reward_risk_ratio(self) -> float:
        """Payoff ratio b = g / l, the binary-bet parameterization."""
        if self.avg_loss_pct <= 0:
            return 0.0
        return self.avg_win_pct / self.avg_loss_pct


def estimate_kelly_inputs(
    trade_history: List[Dict[str, Any]],
    min_trades: int = 50,
    shrinkage_strength: float = 20.0,
) -> Optional[KellyInputs]:
    """Estimate the realized payoff distribution from trade history.

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

# Ceiling on the fraction of the book a single fractional-Kelly position may
# be sized to *before* the per-name max_single_position_pct cap. Once f* is
# expressed in allocation units it is routinely greater than 1 (see
# kelly_allocation_fraction), and 1.0 here says "fully funded, never
# levered" — this platform trades a cash delivery account and cannot borrow.
# The old code got this constraint for free from a min(1.0, f_star) clamp
# that was really a units bug wearing a risk control's clothes.
MAX_GROSS_EXPOSURE = 1.0


def calculate_kelly_fraction(win_probability: float, reward_risk_ratio: float) -> float:
    """Binary-bet Kelly fraction: f* = p - (1-p)/b.

    **This is not an allocation fraction.** It is the growth-optimal stake in
    the bet where a loss costs the *entire* stake, e.g. a wager. A stop-loss
    equity trade is a different bet: a loss costs the distance to the stop,
    around 5-8% of the position, not 100% of it. Sizing a position at f* of
    the book therefore under-bets by a factor of 1/l — at a 6% average loss,
    roughly 17x — and does so unevenly, because a wide-stop signal (large l)
    gets exactly the same allocation as a tight-stop one when it should get
    less. Use kelly_allocation_fraction() for sizing; this function is kept
    for the diagnostic it is (the sign of the edge, and the classic textbook
    figure) and is what estimate_kelly_inputs' `reward_risk_ratio` pairs with.

    Clamped to [0, 1] — a negative f* (an unprofitable edge) becomes 0 so
    callers fall back to fixed-fractional sizing rather than sizing a
    "negative" position (this platform never shorts).
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
    loses a fraction l with probability 1-p, expected log wealth is maximized
    at

        f* = (p*g - (1-p)*l) / (g*l)  =  (p - (1-p)/b) / l,   b = g/l

    The numerator is the per-rupee-invested edge; dividing by g*l converts it
    from "edge per unit invested" into "units of wealth to invest". The binary
    Kelly formula f* = p - (1-p)/b is the l = 1 special case, which is why it
    can be used directly as a stake fraction there and nowhere else.

    Two consequences worth stating, because they are why this cannot be
    recovered by simply raising kappa on the binary form:

    - The result is routinely > 1 (at p = 0.55, g = 10.8%, l = 6% it is 5.0),
      which is what full Kelly genuinely recommends for a bet that can only
      lose 6%. The caller clamps it — see MAX_GROSS_EXPOSURE and
      max_single_position_pct — rather than the formula pretending otherwise.
    - f* scales as 1/l, so the correction is signal-dependent, not a constant
      shrinkage: it re-sorts the book by stop width instead of leaving the
      ordering intact.

    Args:
        win_probability: Realized win rate p, in [0, 1].
        avg_win_pct: Mean magnitude of a winning trade's return, in percent.
        avg_loss_pct: Mean magnitude of a losing trade's return, in percent.

    Returns:
        Allocation fraction >= 0 (unbounded above; 0 when the edge is
        non-positive or either average is unusable).
    """
    g = abs(float(avg_win_pct)) / 100.0
    l = abs(float(avg_loss_pct)) / 100.0
    if g <= 0.0 or l <= 0.0:
        return 0.0

    p = min(1.0, max(0.0, float(win_probability)))
    edge = p * g - (1.0 - p) * l
    if edge <= 0.0:
        return 0.0
    return edge / (g * l)


def loss_given_stop_pct(entry_price: float, stop_price: float) -> Optional[float]:
    """This trade's downside if the stop fills, as a percent of entry.

    Returns None when the pair is unusable (non-positive entry, or a stop at
    or above it), so the caller falls back to the pooled historical average
    loss rather than sizing off a nonsensical l.
    """
    if entry_price <= 0 or stop_price <= 0 or stop_price >= entry_price:
        return None
    return (entry_price - stop_price) / entry_price * 100.0


def calculate_kelly_quantity(
    entry_price: float,
    portfolio_value_inr: float,
    max_single_position_pct: float,
    win_probability: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    kelly_fraction: float = MAX_KELLY_FRACTION,
    loss_given_stop_pct: Optional[float] = None,
) -> int:
    """Fractional-Kelly position sizing, in allocation units.

    Order of operations, each step a distinct constraint rather than a
    restatement of the previous one:

    1. f* from kelly_allocation_fraction — the growth-optimal allocation.
    2. x kappa (fractional Kelly), capped at MAX_KELLY_FRACTION, because p, g
       and l are estimates and Kelly punishes over-betting asymmetrically.
    3. Capped at MAX_GROSS_EXPOSURE — no leverage in a cash delivery account.
    4. Capped at max_single_position_pct — concentration limit.

    Because f* scales as 1/l, `loss_given_stop_pct` is the parameter that makes
    sizing signal-dependent: two signals with the same edge but stops 3% and
    9% away are two different bets, and only the second one's downside is
    three times the first's. The pooled historical average loss is the
    fallback when the caller has no stop to offer, not the preferred input.

    Args:
        entry_price: Entry price per share.
        portfolio_value_inr: Total portfolio value in INR.
        max_single_position_pct: Hard cap on position value as a fraction of
            portfolio value — Kelly sizing can never exceed this, matching
            the platform's existing fixed-fractional cap.
        win_probability: Realized win rate p (see estimate_kelly_inputs).
        avg_win_pct: Realized average win magnitude, in percent.
        avg_loss_pct: Realized average loss magnitude, in percent. This is the
            loss-given-stop the allocation fraction divides by; passing the
            payoff *ratio* here instead is the units error this signature
            exists to make impossible to write.
        kelly_fraction: Fractional-Kelly multiplier kappa, clamped to
            [0, MAX_KELLY_FRACTION]. The clamp is applied here rather than
            trusted to the caller, so no config, YAML or test fixture can
            route around it.
        loss_given_stop_pct: *This* trade's downside if the stop fills, as a
            percent of the entry price. When supplied it replaces avg_loss_pct
            as l, and the average gain is rescaled to hold the realized payoff
            ratio b = avg_win_pct / avg_loss_pct fixed — b is the part of the
            history that generalizes across trades; the loss scale is not.
            None falls back to the pooled averages.

    Returns:
        Integer quantity >= 0.
    """
    if entry_price <= 0:
        return 0

    if loss_given_stop_pct is not None and loss_given_stop_pct > 0 and avg_loss_pct > 0:
        payoff_ratio = avg_win_pct / avg_loss_pct
        avg_loss_pct = float(loss_given_stop_pct)
        avg_win_pct = avg_loss_pct * payoff_ratio

    f_star = kelly_allocation_fraction(win_probability, avg_win_pct, avg_loss_pct)
    kappa = max(0.0, min(MAX_KELLY_FRACTION, kelly_fraction))
    position_fraction = min(f_star * kappa, MAX_GROSS_EXPOSURE)

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
    if config.risk.use_kelly_sizing and trade_history:
        kelly_inputs = estimate_kelly_inputs(
            trade_history,
            min_trades=config.risk.kelly_min_trades,
            shrinkage_strength=config.risk.kelly_shrinkage_strength,
        )
        if kelly_inputs is not None:
            return calculate_kelly_quantity(
                entry_price=entry_price,
                portfolio_value_inr=config.risk.portfolio_value_inr,
                max_single_position_pct=config.risk.max_single_position_pct,
                win_probability=kelly_inputs.win_probability,
                avg_win_pct=kelly_inputs.avg_win_pct,
                avg_loss_pct=kelly_inputs.avg_loss_pct,
                kelly_fraction=config.risk.kelly_fraction,
                loss_given_stop_pct=loss_given_stop_pct(entry_price, stop_price),
            )

    return calculate_quantity(entry_price, stop_price, config)


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


def calculate_position_size(portfolio_value: float, price: float,
                            risk_per_trade_pct: float, 
                            max_position_pct: float,
                            stop_loss_pct: float) -> int:
    """Calculate optimal position size based on risk parameters.

    Args:
        portfolio_value: Total portfolio value in INR.
        price: Current stock price.
        risk_per_trade_pct: Maximum risk per trade (e.g., 0.01 for 1%).
        max_position_pct: Maximum position as % of portfolio.
        stop_loss_pct: Stop loss percentage.

    Returns:
        Number of shares to buy.
    """
    if price <= 0 or stop_loss_pct <= 0:
        return 0

    # Risk-based sizing
    risk_amount = portfolio_value * risk_per_trade_pct
    risk_per_share = price * stop_loss_pct
    
    if risk_per_share == 0:
        return 0
        
    risk_shares = int(risk_amount / risk_per_share)

    # Position limit sizing
    max_position_value = portfolio_value * max_position_pct
    max_shares = int(max_position_value / price)

    # Take minimum of both constraints
    quantity = min(risk_shares, max_shares)

    return max(0, quantity)


def calculate_stop_loss(entry_price: float, atr: Optional[float] = None,
                        atr_multiplier: float = 2.0,
                        fixed_pct: Optional[float] = None) -> float:
    """Calculate stop loss price.

    Args:
        entry_price: Entry price of the position.
        atr: Average True Range value.
        atr_multiplier: Multiplier for ATR-based stop.
        fixed_pct: Fixed percentage stop loss.

    Returns:
        Stop loss price.
    """
    if fixed_pct is not None:
        return entry_price * (1 - fixed_pct)

    if atr is not None and atr > 0:
        return entry_price - (atr * atr_multiplier)

    # Default 5% stop loss
    return entry_price * 0.95


def calculate_target_price(entry_price: float, risk_reward_ratio: float,
                           stop_loss_price: float) -> float:
    """Calculate target price based on risk-reward ratio.

    Args:
        entry_price: Entry price.
        risk_reward_ratio: Desired reward/risk ratio (e.g., 2.0 for 2:1).
        stop_loss_price: Stop loss price.

    Returns:
        Target price.
    """
    risk = entry_price - stop_loss_price
    reward = risk * risk_reward_ratio
    return entry_price + reward


def calculate_portfolio_risk(positions: list, correlation_matrix: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """Calculate overall portfolio risk metrics.

    Args:
        positions: List of position dictionaries with 'value' and 'volatility'.
        correlation_matrix: Optional correlation matrix between positions.

    Returns:
        Dictionary with portfolio risk metrics.
    """
    if not positions:
        return {'total_value': 0, 'portfolio_volatility': 0, 'var_95': 0}

    total_value = sum(p.get('value', 0) for p in positions)

    if total_value == 0:
        return {'total_value': 0, 'portfolio_volatility': 0, 'var_95': 0}

    # Simple weighted volatility (without correlation)
    weights = [p.get('value', 0) / total_value for p in positions]
    volatilities = [p.get('volatility', 0.02) for p in positions]

    if correlation_matrix is None:
        # Simplified: assume zero correlation
        portfolio_vol = np.sqrt(sum((w * v) ** 2 for w, v in zip(weights, volatilities)))
    else:
        # Full calculation with correlation
        portfolio_vol = 0
        for i, (w_i, v_i) in enumerate(zip(weights, volatilities)):
            for j, (w_j, v_j) in enumerate(zip(weights, volatilities)):
                corr = correlation_matrix.iloc[i, j] if i != j else 1
                portfolio_vol += w_i * v_i * w_j * v_j * corr
        portfolio_vol = np.sqrt(portfolio_vol)

    # Calculate VaR (95% confidence, 1 day)
    var_95 = total_value * portfolio_vol * 1.645

    return {
        'total_value': total_value,
        'portfolio_volatility': portfolio_vol,
        'var_95': var_95
    }


def check_risk_limits(current_price: float, entry_price: float,
                      max_loss_pct: float = 0.05) -> bool:
    """Check if position has exceeded maximum loss limit.

    Args:
        current_price: Current market price.
        entry_price: Original entry price.
        max_loss_pct: Maximum allowed loss percentage.

    Returns:
        True if within limits, False if breach.
    """
    if entry_price <= 0:
        return True

    loss_pct = (entry_price - current_price) / entry_price
    return loss_pct <= max_loss_pct
