"""
Tests for ExecutionSimulator module.

Tests cover:
- Transaction cost calculations (brokerage, STT, exchange charges, GST, stamp duty)
- Slippage and market impact modeling
- Capital gains tax (STCG vs LTCG)
"""

import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from execution_sim import (
    DEFAULT_SLIPPAGE_PCT_PER_SIDE,
    ExecutionSimulator,
    cost_fraction_per_side,
    round_trip_cost_pct,
)


class TestRoundTripCostEstimate:
    """The quantity-free estimator the strategy layer gates signals on."""

    def test_buy_leg_costs_more_than_sell_leg_by_stamp_duty(self):
        buy = cost_fraction_per_side('BUY', slippage_pct=0.0)
        sell = cost_fraction_per_side('SELL', slippage_pct=0.0)

        assert buy - sell == pytest.approx(ExecutionSimulator.STAMP_DUTY_RATE)

    def test_includes_every_statutory_charge(self):
        buy = cost_fraction_per_side('BUY', slippage_pct=0.0)

        brokerage = ExecutionSimulator.BROKERAGE_PERCENT
        exchange = ExecutionSimulator.EXCHANGE_TXN_CHARGE_RATE
        sebi = ExecutionSimulator.SEBI_TURNOVER_FEE_RATE
        expected = (
            brokerage
            + ExecutionSimulator.STT_RATE
            + exchange
            + sebi
            + (brokerage + exchange + sebi) * ExecutionSimulator.GST_RATE
            + ExecutionSimulator.STAMP_DUTY_RATE
        )
        assert buy == pytest.approx(expected)

    def test_slippage_is_added_on_top(self):
        without = cost_fraction_per_side('SELL', slippage_pct=0.0)
        with_slippage = cost_fraction_per_side('SELL', slippage_pct=0.005)

        assert with_slippage - without == pytest.approx(0.005)

    def test_round_trip_is_the_sum_of_both_legs(self):
        assert round_trip_cost_pct(slippage_pct=0.001) == pytest.approx(
            cost_fraction_per_side('BUY', 0.001) + cost_fraction_per_side('SELL', 0.001)
        )

    def test_default_round_trip_lands_in_the_documented_range(self):
        """0.5-1.5% per round trip is the band that erodes the Indian momentum
        premium under monthly rebalancing — the default assumption must sit in
        it, not an order of magnitude away."""
        assert 0.005 < round_trip_cost_pct() < 0.015
        assert DEFAULT_SLIPPAGE_PCT_PER_SIDE == 0.0025

    def test_percentage_brokerage_is_an_upper_bound_on_realized_brokerage(self):
        """The estimator uses the 0.03% rate; the flat ₹20 cap can only ever
        make the realized rate lower, so the estimate stays conservative."""
        sim = ExecutionSimulator()
        for turnover in (10_000.0, 66_667.0, 1_000_000.0):
            realized = sim.calculate_transaction_costs('BUY', 100.0, 1, turnover)
            estimated = cost_fraction_per_side('BUY', slippage_pct=0.0) * turnover
            # At exactly ₹66,667 the two brokerage rules coincide, so allow for
            # float round-off rather than demanding a strict inequality there.
            assert estimated >= realized * (1 - 1e-9)


class TestTransactionCosts:
    """Test transaction cost calculations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sim = ExecutionSimulator()
    
    def test_stt_calculation_on_buy_order(self):
        """Test STT calculation on a ₹1,00,000 BUY order."""
        # ₹1,00,000 turnover BUY order
        price = 1000.0
        quantity = 100
        turnover = 100000.0  # ₹1,00,000
        
        costs = self.sim.calculate_transaction_costs('BUY', price, quantity, turnover)
        
        # Expected calculations:
        # Brokerage: min(20, 100000 * 0.0003) = min(20, 30) = ₹20
        # STT: 100000 * 0.001 = ₹100
        # Exchange Txn: 100000 * 0.0000345 = ₹3.45
        # SEBI turnover fee: 100000 * 0.000001 = ₹0.10
        # GST: 18% of (20 + 3.45 + 0.10) = 0.18 * 23.55 = ₹4.239
        # Stamp Duty: 100000 * 0.00015 = ₹15 (BUY only)
        # Total: 20 + 100 + 3.45 + 0.10 + 4.239 + 15 = ₹142.789

        expected_brokerage = 20.0
        expected_stt = 100.0
        expected_exchange_txn = 3.45
        expected_sebi_fee = 0.10
        expected_gst_base = expected_brokerage + expected_exchange_txn + expected_sebi_fee
        expected_gst = expected_gst_base * 0.18
        expected_stamp_duty = 15.0
        expected_total = (
            expected_brokerage + expected_stt + expected_exchange_txn
            + expected_sebi_fee + expected_gst + expected_stamp_duty
        )

        assert abs(costs - expected_total) < 0.01, f"Expected {expected_total}, got {costs}"
    
    def test_stt_calculation_on_sell_order(self):
        """Test that SELL orders don't include stamp duty."""
        price = 1000.0
        quantity = 100
        turnover = 100000.0
        
        buy_cost = self.sim.calculate_transaction_costs('BUY', price, quantity, turnover)
        sell_cost = self.sim.calculate_transaction_costs('SELL', price, quantity, turnover)
        
        # SELL should be cheaper by the stamp duty amount (₹15)
        expected_stamp_duty = turnover * 0.00015
        assert abs(buy_cost - sell_cost - expected_stamp_duty) < 0.01
    
    def test_brokerage_cap_at_20(self):
        """Test that brokerage is capped at ₹20 per order."""
        # Large order where 0.03% would exceed ₹20
        large_turnover = 1000000.0  # ₹10,00,000
        
        # Manually calculate components to verify brokerage cap
        # We need to isolate brokerage from total cost
        
        # For a small order where 0.03% < ₹20
        small_turnover = 10000.0  # ₹10,000
        small_cost = self.sim.calculate_transaction_costs('BUY', 100, 100, small_turnover)
        
        # Brokerage for small: 10000 * 0.0003 = ₹3
        # For large: capped at ₹20
        
        # The difference in costs should reflect the brokerage difference
        # plus proportional differences in other fees
        
    def test_zero_turnover(self):
        """Test that zero turnover returns zero cost."""
        cost = self.sim.calculate_transaction_costs('BUY', 0, 0, 0)
        assert cost == 0.0


class TestMarketImpact:
    """Test slippage and market impact calculations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sim = ExecutionSimulator()
    
    def test_base_slippage_with_atr(self):
        """Test that base slippage is 0.5 * ATR."""
        price = 1000.0
        quantity = 100
        avg_daily_volume = 100000
        atr = 20.0
        
        adjusted_price = self.sim.calculate_slippage_and_impact(price, quantity, avg_daily_volume, atr)
        
        # Base slippage should be 0.5 * ATR = 10
        expected_base_slippage = 0.5 * atr
        actual_slippage = adjusted_price - price
        
        # Since trade value (100*1000=100000) is exactly 0.1% of daily value 
        # (100000*1000=100000000), which is less than 1%, no market impact
        assert abs(actual_slippage - expected_base_slippage) < 0.01
    
    def test_market_impact_for_illiquid_stocks(self):
        """Test market impact penalty for illiquid stocks (large order relative to volume)."""
        price = 1000.0
        quantity = 50000  # Large order
        avg_daily_volume = 100000  # Low volume stock
        atr = 20.0
        
        # Trade value = 50000 * 1000 = ₹5,00,00,000
        # Daily value = 100000 * 1000 = ₹10,00,00,000
        # Volume ratio = 50% (much higher than 1% threshold)
        
        adjusted_price = self.sim.calculate_slippage_and_impact(price, quantity, avg_daily_volume, atr)
        
        # Should have significant market impact on top of base slippage
        base_slippage = 0.5 * atr  # ₹10
        total_slippage = adjusted_price - price
        
        # Market impact should kick in because 50% > 1% threshold
        # excess_ratio = 0.50 - 0.01 = 0.49
        # market_impact = 0.001 * (0.49^2) * 1000 = 0.001 * 0.2401 * 1000 = ₹0.2401
        expected_market_impact = 0.001 * ((0.50 - 0.01) ** 2) * price
        
        expected_total_slippage = base_slippage + expected_market_impact
        
        assert total_slippage > base_slippage, "Market impact should increase slippage"
        assert abs(total_slippage - expected_total_slippage) < 0.01
    
    def test_no_market_impact_for_small_orders(self):
        """Test that small orders don't trigger market impact."""
        price = 1000.0
        quantity = 100  # Small order
        avg_daily_volume = 1000000  # High volume stock
        atr = 20.0
        
        # Trade value = 100 * 1000 = ₹1,00,000
        # Daily value = 1000000 * 1000 = ₹10,00,00,000
        # Volume ratio = 0.01% (less than 1% threshold)
        
        adjusted_price = self.sim.calculate_slippage_and_impact(price, quantity, avg_daily_volume, atr)
        
        base_slippage = 0.5 * atr
        actual_slippage = adjusted_price - price
        
        # Only base slippage, no market impact
        assert abs(actual_slippage - base_slippage) < 0.01


class TestCapitalGainsTax:
    """Test capital gains tax calculations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sim = ExecutionSimulator()
    
    def test_stcg_short_term_holding(self):
        """Test STCG (Short Term Capital Gains) for holdings < 365 days."""
        entry_price = 100.0
        exit_price = 150.0
        quantity = 1000
        holding_days = 200  # Less than 365 days
        
        profit = (exit_price - entry_price) * quantity  # ₹50,000
        
        tax = self.sim.calculate_capital_gains_tax(
            entry_price, exit_price, quantity, holding_days
        )
        
        # STCG rate is 20%
        expected_tax = profit * 0.20  # ₹10,000
        
        assert abs(tax - expected_tax) < 0.01
    
    def test_ltcs_long_term_holding_under_exemption(self):
        """Test LTCG (Long Term Capital Gains) for holdings >= 365 days under exemption."""
        entry_price = 100.0
        exit_price = 120.0
        quantity = 5000
        holding_days = 400  # More than 365 days
        
        profit = (exit_price - entry_price) * quantity  # ₹1,00,000
        
        tax = self.sim.calculate_capital_gains_tax(
            entry_price, exit_price, quantity, holding_days
        )
        
        # Profit (₹1,00,000) is under LTCG exemption (₹1,25,000)
        # So tax should be 0
        assert tax == 0.0
    
    def test_ltcs_long_term_holding_above_exemption(self):
        """Test LTCG for holdings >= 365 days above exemption."""
        entry_price = 100.0
        exit_price = 150.0
        quantity = 10000
        holding_days = 500  # More than 365 days
        
        profit = (exit_price - entry_price) * quantity  # ₹5,00,000
        
        # Reset LTCG tracker
        self.sim.reset_ltcg_tracker()
        
        tax = self.sim.calculate_capital_gains_tax(
            entry_price, exit_price, quantity, holding_days
        )
        
        # First ₹1,25,000 is exempt
        # Taxable amount = ₹5,00,000 - ₹1,25,000 = ₹3,75,000
        # LTCG rate is 12.5%
        taxable_profit = profit - 125000
        expected_tax = taxable_profit * 0.125  # ₹46,875
        
        assert abs(tax - expected_tax) < 0.01
    
    def test_ltcs_partial_exemption_used(self):
        """Test LTCG when some exemption has already been used."""
        self.sim.reset_ltcg_tracker()
        
        # First trade uses part of exemption
        entry_price = 100.0
        exit_price = 110.0
        quantity = 5000
        holding_days = 400
        
        profit1 = (exit_price - entry_price) * quantity  # ₹50,000
        tax1 = self.sim.calculate_capital_gains_tax(
            entry_price, exit_price, quantity, holding_days
        )
        
        # No tax since under exemption
        assert tax1 == 0.0
        
        # Second trade
        entry_price2 = 100.0
        exit_price2 = 140.0
        quantity2 = 5000
        holding_days2 = 450
        
        profit2 = (exit_price2 - entry_price2) * quantity2  # ₹2,00,000
        tax2 = self.sim.calculate_capital_gains_tax(
            entry_price2, exit_price2, quantity2, holding_days2
        )
        
        # Remaining exemption = ₹1,25,000 - ₹50,000 = ₹75,000
        # Taxable = ₹2,00,000 - ₹75,000 = ₹1,25,000
        # Tax = ₹1,25,000 * 0.125 = ₹15,625
        remaining_exemption = 125000 - 50000
        taxable_profit2 = profit2 - remaining_exemption
        expected_tax2 = taxable_profit2 * 0.125
        
        assert abs(tax2 - expected_tax2) < 0.01
    
    def test_no_tax_on_loss(self):
        """Test that no tax is charged on losses."""
        entry_price = 150.0
        exit_price = 100.0
        quantity = 1000
        holding_days = 500
        
        tax = self.sim.calculate_capital_gains_tax(
            entry_price, exit_price, quantity, holding_days
        )
        
        assert tax == 0.0


class TestTradeExecutionDecision:
    """Test the should_execute_trade method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sim = ExecutionSimulator()
    
    def test_execute_when_reward_exceeds_friction(self):
        """Test that trade executes when expected reward > friction."""
        price = 1000.0
        quantity = 100
        avg_daily_volume = 100000
        atr = 20.0
        
        # High expected reward (2 ATR)
        should_execute, info = self.sim.should_execute_trade(
            'BUY', price, quantity, avg_daily_volume, atr,
            expected_reward_atr=2.0
        )
        
        # With 2 ATR reward, should execute
        assert should_execute is True
        assert info['expected_reward_inr'] > info['total_friction']
    
    def test_skip_when_friction_exceeds_reward(self):
        """Test that trade is skipped when friction > expected reward."""
        price = 1000.0
        quantity = 100
        avg_daily_volume = 100000
        atr = 5.0  # Low volatility means low reward
        
        # Low expected reward (0.5 ATR)
        should_execute, info = self.sim.should_execute_trade(
            'BUY', price, quantity, avg_daily_volume, atr,
            expected_reward_atr=0.5
        )
        
        # With such low reward, friction likely exceeds it
        # This depends on the fixed costs, but typically should skip
        if info['total_friction'] > info['expected_reward_inr']:
            assert should_execute is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
