"""
Execution Simulator for Portfolio Agent.

Models exact Indian market costs, taxes, and market impact for realistic backtesting.

Two related but distinct things live here:

- `ExecutionSimulator` — the *realized* friction charged to the equity curve
  when a simulated order actually fills (needs price AND quantity).
- `cost_fraction_per_side()` / `round_trip_cost_pct()` — a quantity-free
  *estimate* of the same friction as a fraction of turnover, so the strategy
  layer can gate signals on net-of-cost reward:risk before a quantity has been
  sized. See docs/QUANT_RESEARCH.md section 12.
"""

import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class ExecutionSimulator:
    """
    Simulates realistic trade execution with Indian market friction.

    Includes:
    - Brokerage fees
    - Securities Transaction Tax (STT)
    - Exchange transaction charges
    - SEBI turnover fees
    - GST (Goods and Services Tax)
    - Stamp Duty
    - Slippage and market impact
    - Capital gains tax (STCG/LTCG)

    Rates below are the *delivery* (CNC) schedule — this platform holds
    positions overnight, so the cheaper intraday STT rate never applies.
    """

    # Constants for Indian market
    BROKERAGE_FIXED = 20.0  # ₹20 per order
    BROKERAGE_PERCENT = 0.0003  # 0.03%
    # STT on delivery equity is 0.1% of turnover on BOTH legs (the 0.025%
    # figure quoted for equities is the intraday *sell*-side rate, which a
    # hold-overnight platform never qualifies for).
    STT_RATE = 0.001
    EXCHANGE_TXN_CHARGE_RATE = 0.0000345  # ~0.00345% (NSE equity delivery)
    # SEBI turnover fees: ₹10 per crore of turnover = 0.0001%. Small in
    # absolute terms, but it is part of the GST base and was previously
    # missing from the model entirely.
    SEBI_TURNOVER_FEE_RATE = 0.000001
    GST_RATE = 0.18  # 18%
    STAMP_DUTY_RATE = 0.00015  # 0.015% on BUY only

    # Slippage constants
    SLIPPAGE_ATR_MULTIPLIER = 0.5  # 0.5 * ATR as bid-ask spread penalty
    MARKET_IMPACT_THRESHOLD = 0.01  # 1% of daily volume
    MARKET_IMPACT_QUADRATIC_COEFF = 0.001  # Coefficient for quadratic penalty

    # Tax constants
    STCG_RATE = 0.20  # 20% for Short Term Capital Gains (< 365 days)
    LTCG_RATE = 0.125  # 12.5% for Long Term Capital Gains (>= 365 days)
    LTCG_EXEMPTION = 125000.0  # ₹1.25L annual exemption for LTCG

    def __init__(self):
        """Initialize the ExecutionSimulator."""
        self.ltcg_used = 0.0  # Track LTCG exemption used in the financial year

    def calculate_transaction_costs(
        self,
        side: str,
        price: float,
        quantity: int,
        turnover: Optional[float] = None
    ) -> float:
        """
        Calculate total transaction costs for a trade.

        Args:
            side: 'BUY' or 'SELL'.
            price: Execution price per share.
            quantity: Number of shares.
            turnover: Turnover value (price * quantity). If None, calculated.

        Returns:
            Total transaction cost in INR.
        """
        if turnover is None:
            turnover = price * quantity

        if turnover <= 0:
            return 0.0

        # 1. Brokerage: ₹20 per order or 0.03% (whichever is lower)
        brokerage_percent = turnover * self.BROKERAGE_PERCENT
        brokerage = min(self.BROKERAGE_FIXED, brokerage_percent)

        # 2. STT (Securities Transaction Tax): 0.1% on turnover (both BUY and SELL)
        stt = turnover * self.STT_RATE

        # 3. Exchange Transaction Charges: ~0.00345% on turnover
        exchange_txn_charge = turnover * self.EXCHANGE_TXN_CHARGE_RATE

        # 4. SEBI turnover fees: 0.0001% on turnover (both legs)
        sebi_fee = turnover * self.SEBI_TURNOVER_FEE_RATE

        # 5. GST: 18% on (Brokerage + Txn Charges + SEBI fees)
        gst_base = brokerage + exchange_txn_charge + sebi_fee
        gst = gst_base * self.GST_RATE

        # 6. Stamp Duty: 0.015% on BUY turnover only
        stamp_duty = 0.0
        if side.upper() == 'BUY':
            stamp_duty = turnover * self.STAMP_DUTY_RATE

        # Total cost
        total_cost = brokerage + stt + exchange_txn_charge + sebi_fee + gst + stamp_duty

        logger.debug(
            f"Transaction costs for {side} {quantity}@{price}: "
            f"Brokerage={brokerage:.2f}, STT={stt:.2f}, "
            f"Exchange={exchange_txn_charge:.4f}, SEBI={sebi_fee:.4f}, GST={gst:.2f}, "
            f"StampDuty={stamp_duty:.2f}, Total={total_cost:.2f}"
        )

        return total_cost

    def calculate_slippage_and_impact(
        self, 
        price: float, 
        quantity: int, 
        avg_daily_volume: int, 
        atr: float
    ) -> float:
        """
        Calculate adjusted execution price including slippage and market impact.
        
        Args:
            price: Current market price.
            quantity: Number of shares to trade.
            avg_daily_volume: Average daily trading volume (shares).
            atr: Average True Range (volatility measure).
            
        Returns:
            Adjusted execution price (worse than market price).
        """
        if price <= 0 or quantity <= 0:
            return price
        
        # Base slippage: 0.5 * ATR as bid-ask spread penalty
        base_slippage = self.SLIPPAGE_ATR_MULTIPLIER * atr
        
        # Market Impact: Check if order size > 1% of daily volume
        trade_value = quantity * price
        daily_value = avg_daily_volume * price
        
        market_impact = 0.0
        if daily_value > 0:
            volume_ratio = trade_value / daily_value
            
            if volume_ratio > self.MARKET_IMPACT_THRESHOLD:
                # Quadratic penalty for moving the market
                excess_ratio = volume_ratio - self.MARKET_IMPACT_THRESHOLD
                market_impact = (
                    self.MARKET_IMPACT_QUADRATIC_COEFF * 
                    (excess_ratio ** 2) * 
                    price
                )
        
        total_slippage = base_slippage + market_impact
        
        # For BUY, slippage increases price; for SELL, it decreases price
        # We return the adjusted price that the trader would get (worse)
        adjusted_price_buy = price + total_slippage
        adjusted_price_sell = price - total_slippage
        
        # Return average adjusted price (caller decides direction)
        # Actually, we should return the slippage amount so caller can apply direction
        # Let's return the adjusted price for a BUY order (higher is worse)
        
        logger.debug(
            f"Slippage calculation: price={price}, qty={quantity}, "
            f"ADV={avg_daily_volume}, ATR={atr:.2f}, "
            f"base_slippage={base_slippage:.2f}, market_impact={market_impact:.4f}, "
            f"adjusted_price_buy={adjusted_price_buy:.2f}"
        )
        
        return adjusted_price_buy
    
    def get_adjusted_price(
        self, 
        side: str, 
        price: float, 
        quantity: int, 
        avg_daily_volume: int, 
        atr: float
    ) -> float:
        """
        Get the adjusted execution price for a specific side.
        
        Args:
            side: 'BUY' or 'SELL'.
            price: Current market price.
            quantity: Number of shares.
            avg_daily_volume: Average daily volume.
            atr: Average True Range.
            
        Returns:
            Adjusted execution price.
        """
        adjusted_buy_price = self.calculate_slippage_and_impact(
            price, quantity, avg_daily_volume, atr
        )
        
        # Calculate the total slippage amount
        total_slippage = adjusted_buy_price - price
        
        if side.upper() == 'BUY':
            return price + total_slippage
        else:  # SELL
            return price - total_slippage
    
    def calculate_capital_gains_tax(
        self, 
        entry_price: float, 
        exit_price: float, 
        quantity: int, 
        holding_days: int, 
        ltcg_used: Optional[float] = None
    ) -> float:
        """
        Calculate capital gains tax based on Indian tax rules.
        
        Args:
            entry_price: Purchase price per share.
            exit_price: Sale price per share.
            quantity: Number of shares.
            holding_days: Number of days held.
            ltcg_used: Amount of LTCG exemption already used this financial year.
            
        Returns:
            Tax amount in INR (0 if no profit or loss).
        """
        if entry_price <= 0 or exit_price <= 0 or quantity <= 0:
            return 0.0
        
        # Calculate profit/loss
        profit = (exit_price - entry_price) * quantity
        
        # No tax on losses
        if profit <= 0:
            return 0.0
        
        if ltcg_used is None:
            ltcg_used = self.ltcg_used
        
        if holding_days < 365:
            # STCG: 20% on profits (holding period < 365 days)
            tax = profit * self.STCG_RATE
            logger.debug(
                f"STCG applied: profit={profit:.2f}, tax={tax:.2f}, "
                f"holding_days={holding_days}"
            )
        else:
            # LTCG: 12.5% on profits exceeding ₹1.25L annual exemption
            remaining_exemption = max(0, self.LTCG_EXEMPTION - ltcg_used)
            
            if profit <= remaining_exemption:
                # Entire profit covered by exemption
                tax = 0.0
                # Update used exemption
                self.ltcg_used += profit
            else:
                # Tax only on amount exceeding exemption
                taxable_profit = profit - remaining_exemption
                tax = taxable_profit * self.LTCG_RATE
                # Update used exemption to full
                self.ltcg_used = self.LTCG_EXEMPTION
            
            logger.debug(
                f"LTCG applied: profit={profit:.2f}, remaining_exemption={remaining_exemption:.2f}, "
                f"taxable_profit={taxable_profit if profit > remaining_exemption else 0:.2f}, "
                f"tax={tax:.2f}, holding_days={holding_days}"
            )
        
        return tax
    
    def reset_ltcg_tracker(self) -> None:
        """Reset the LTCG exemption tracker (for new financial year)."""
        self.ltcg_used = 0.0
    
    def should_execute_trade(
        self, 
        side: str, 
        price: float, 
        quantity: int, 
        avg_daily_volume: int, 
        atr: float, 
        expected_reward_atr: float = 1.0
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if a trade should be executed based on cost vs reward analysis.
        
        No trade is executed if expected cost + slippage > expected ATR reward.
        
        Args:
            side: 'BUY' or 'SELL'.
            price: Current market price.
            quantity: Number of shares.
            avg_daily_volume: Average daily volume.
            atr: Average True Range.
            expected_reward_atr: Expected reward in multiples of ATR (default 1.0).
            
        Returns:
            Tuple of (should_execute: bool, info: dict with details).
        """
        turnover = price * quantity
        
        # Calculate transaction costs
        txn_cost = self.calculate_transaction_costs(side, price, quantity, turnover)
        
        # Calculate slippage-adjusted price
        adjusted_price = self.get_adjusted_price(
            side, price, quantity, avg_daily_volume, atr
        )
        
        # Slippage cost (absolute difference)
        slippage_per_share = abs(adjusted_price - price)
        slippage_cost = slippage_per_share * quantity
        
        # Total friction cost
        total_friction = txn_cost + slippage_cost
        
        # Expected reward in INR
        expected_reward_inr = expected_reward_atr * atr * quantity
        
        # Decision: execute if reward > friction
        should_execute = expected_reward_inr > total_friction
        
        info = {
            'should_execute': should_execute,
            'transaction_cost': txn_cost,
            'slippage_cost': slippage_cost,
            'total_friction': total_friction,
            'expected_reward_inr': expected_reward_inr,
            'friction_ratio': total_friction / expected_reward_inr if expected_reward_inr > 0 else float('inf'),
            'adjusted_price': adjusted_price
        }
        
        if not should_execute:
            logger.info(
                f"Trade skipped: friction ({total_friction:.2f}) > reward ({expected_reward_inr:.2f})"
            )

        return should_execute, info


# Default per-side slippage assumption used by the *estimator* below (not by
# ExecutionSimulator's realized fills, which price slippage off each ticker's
# own ATR and traded volume). 25 bps per side is a deliberately conservative
# stand-in for the bid-ask spread plus impact typical of Indian mid/small caps;
# large, liquid NSE names trade materially tighter.
DEFAULT_SLIPPAGE_PCT_PER_SIDE = 0.0025


def cost_fraction_per_side(
    side: str,
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT_PER_SIDE,
    simulator: Optional[ExecutionSimulator] = None,
) -> float:
    """Estimate one leg's friction as a fraction of that leg's turnover.

    Quantity-free by construction, so the strategy layer can charge costs
    against a signal before any position has been sized. Brokerage is taken at
    its percentage rate (0.03%) rather than min(₹20, 0.03%): the flat ₹20 only
    ever *lowers* the effective rate (it binds above ~₹66,667 of turnover), so
    the percentage rate is the upper bound and the conservative choice here.

    Args:
        side: 'BUY' or 'SELL' — stamp duty applies to the buy leg only.
        slippage_pct: Assumed slippage for this leg, as a fraction of turnover.
        simulator: Optional ExecutionSimulator whose rate constants to read
            (lets a caller override the statutory rates); defaults to the
            class-level schedule.

    Returns:
        Friction as a fraction of turnover (e.g. 0.0038 = 38 bps).
    """
    sim = simulator or ExecutionSimulator

    brokerage = sim.BROKERAGE_PERCENT
    stt = sim.STT_RATE
    exchange = sim.EXCHANGE_TXN_CHARGE_RATE
    sebi = sim.SEBI_TURNOVER_FEE_RATE
    gst = (brokerage + exchange + sebi) * sim.GST_RATE
    stamp_duty = sim.STAMP_DUTY_RATE if side.upper() == 'BUY' else 0.0

    return brokerage + stt + exchange + sebi + gst + stamp_duty + max(0.0, slippage_pct)


def round_trip_cost_pct(
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT_PER_SIDE,
    simulator: Optional[ExecutionSimulator] = None,
) -> float:
    """Estimate total buy + sell friction as a fraction of turnover.

    This is the number a strategy has to clear before a trade is worth taking
    at all. With the default 25 bps/side slippage assumption it lands around
    0.8% per round trip — squarely in the 0.5–1.5% range that erodes the
    Indian momentum premium under monthly rebalancing, which is exactly why
    signals are now gated on net-of-cost reward:risk rather than gross.

    Args:
        slippage_pct: Assumed slippage per side, as a fraction of turnover.
        simulator: Optional ExecutionSimulator whose rate constants to read.

    Returns:
        Round-trip friction as a fraction of turnover.
    """
    return (
        cost_fraction_per_side('BUY', slippage_pct, simulator)
        + cost_fraction_per_side('SELL', slippage_pct, simulator)
    )
