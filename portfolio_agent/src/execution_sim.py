"""
Execution Simulator for Portfolio Agent.

Models exact Indian market costs, taxes, and market impact for realistic backtesting.
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
    - GST (Goods and Services Tax)
    - Stamp Duty
    - Slippage and market impact
    - Capital gains tax (STCG/LTCG)
    """
    
    # Constants for Indian market
    BROKERAGE_FIXED = 20.0  # ₹20 per order
    BROKERAGE_PERCENT = 0.0003  # 0.03%
    STT_RATE = 0.001  # 0.1% on both BUY and SELL
    EXCHANGE_TXN_CHARGE_RATE = 0.0000345  # ~0.00345%
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
        
        # 4. GST: 18% on (Brokerage + Txn Charges)
        gst_base = brokerage + exchange_txn_charge
        gst = gst_base * self.GST_RATE
        
        # 5. Stamp Duty: 0.015% on BUY turnover only
        stamp_duty = 0.0
        if side.upper() == 'BUY':
            stamp_duty = turnover * self.STAMP_DUTY_RATE
        
        # Total cost
        total_cost = brokerage + stt + exchange_txn_charge + gst + stamp_duty
        
        logger.debug(
            f"Transaction costs for {side} {quantity}@{price}: "
            f"Brokerage={brokerage:.2f}, STT={stt:.2f}, "
            f"Exchange={exchange_txn_charge:.4f}, GST={gst:.2f}, "
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
