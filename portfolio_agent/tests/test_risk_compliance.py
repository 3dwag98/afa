"""Tests for risk and compliance modules."""

import pytest
from portfolio_agent.config.schema import AppConfig
from src.risk import calculate_quantity, calculate_stop_target, calculate_max_loss
from src.compliance import run_compliance_checks, estimate_capital_gains_tax


def _make_config(
    portfolio_value_inr: float = 1000000.0,
    risk_per_trade_pct: float = 0.01,
    max_single_position_pct: float = 0.10,
    min_price_inr: float = 100.0,
    paper_trading_mode: bool = True,
) -> AppConfig:
    """Create a minimal AppConfig for testing."""
    return AppConfig.model_validate({
        "risk": {
            "portfolio_value_inr": portfolio_value_inr,
            "risk_per_trade_pct": risk_per_trade_pct,
            "max_single_position_pct": max_single_position_pct,
        },
        "compliance": {
            "min_price_inr": min_price_inr,
            "target_prob_profit": 0.55,
            "min_reward_risk": 1.5,
            "paper_trading_mode": paper_trading_mode,
        },
        "learning": {"learning_rate": 0.01, "min_trades_for_learning": 10},
        "simulation": {"mc_horizon_days": 20, "mc_simulations": 1000, "random_seed": 42},
        "data": {"tickers": ["RELIANCE"], "min_history_days": 200},
        "paths": {
            "brain_file": "brain.yaml",
            "sqlite_path": ":memory:",
            "excel_output": "output.xlsx",
            "log_file": "test.log",
        },
    })


class TestCalculateQuantity:
    """Test calculate_quantity function."""

    def test_quantity_respects_1pct_risk(self):
        """Test that quantity respects 1% risk per trade."""
        config = _make_config(
            portfolio_value_inr=1000000.0,
            risk_per_trade_pct=0.01,  # 1%
            max_single_position_pct=0.50,  # High enough to not constrain
        )
        entry_price = 100.0
        stop_price = 95.0  # Risk per share = 5

        # Risk amount = 1000000 * 0.01 = 10000
        # Risk per share = 100 - 95 = 5
        # quantity = floor(10000 / 5) = 2000
        quantity = calculate_quantity(entry_price, stop_price, config)

        assert quantity == 2000

    def test_max_position_cap(self):
        """Test that quantity is capped by max position size."""
        config = _make_config(
            portfolio_value_inr=1000000.0,
            risk_per_trade_pct=0.01,
            max_single_position_pct=0.10,  # 10% max position
        )
        entry_price = 100.0
        stop_price = 90.0  # Risk per share = 10

        # Risk amount = 1000000 * 0.01 = 10000
        # Risk per share = 100 - 90 = 10
        # risk-based quantity = floor(10000 / 10) = 1000
        # Max position value = 1000000 * 0.10 = 100000
        # Max shares by position = floor(100000 / 100) = 1000
        # Both constraints give 1000
        quantity = calculate_quantity(entry_price, stop_price, config)

        assert quantity == 1000

    def test_max_position_cap_stricter_than_risk(self):
        """Test when max position cap is stricter than risk constraint."""
        config = _make_config(
            portfolio_value_inr=1000000.0,
            risk_per_trade_pct=0.02,  # 2% risk
            max_single_position_pct=0.05,  # 5% max position
        )
        entry_price = 100.0
        stop_price = 98.0  # Risk per share = 2

        # Risk amount = 1000000 * 0.02 = 20000
        # Risk per share = 100 - 98 = 2
        # risk-based quantity = floor(20000 / 2) = 10000
        # Max position value = 1000000 * 0.05 = 50000
        # Max shares by position = floor(50000 / 100) = 500
        # Position cap is stricter
        quantity = calculate_quantity(entry_price, stop_price, config)

        assert quantity == 500

    def test_zero_quantity_when_stop_ge_entry(self):
        """Test that quantity is 0 when stop >= entry."""
        config = _make_config()
        entry_price = 100.0
        stop_price = 100.0  # No risk

        quantity = calculate_quantity(entry_price, stop_price, config)

        assert quantity == 0

    def test_zero_quantity_when_stop_above_entry(self):
        """Test that quantity is 0 when stop > entry."""
        config = _make_config()
        entry_price = 100.0
        stop_price = 105.0  # Invalid stop

        quantity = calculate_quantity(entry_price, stop_price, config)

        assert quantity == 0


class TestCalculateStopTarget:
    """Test calculate_stop_target function."""

    def test_atr_based_calculation(self):
        """Test ATR-based stop and target calculation."""
        config = _make_config()
        entry_price = 100.0
        atr = 2.0

        # Stop = 100 - 1.5 * 2 = 97.0
        # Target = 100 + 2.0 * 2 = 104.0
        stop, target = calculate_stop_target(entry_price, atr)

        assert stop == 97.0
        assert target == 104.0

    def test_fallback_when_atr_none(self):
        """Test fallback percentages when ATR is None."""
        config = _make_config()
        entry_price = 100.0

        # Stop = 100 * 0.98 = 98.0
        # Target = 100 * 1.03 = 103.0
        stop, target = calculate_stop_target(entry_price, None)

        assert stop == 98.0
        assert target == 103.0

    def test_fallback_when_atr_zero(self):
        """Test fallback percentages when ATR is zero."""
        config = _make_config()
        entry_price = 100.0

        stop, target = calculate_stop_target(entry_price, 0.0)

        assert stop == 98.0
        assert target == 103.0

    def test_fallback_when_atr_negative(self):
        """Test fallback percentages when ATR is negative."""
        config = _make_config()
        entry_price = 100.0

        stop, target = calculate_stop_target(entry_price, -1.0)

        assert stop == 98.0
        assert target == 103.0

    def test_stop_cannot_be_negative(self):
        """Test that stop price cannot be negative."""
        config = _make_config()
        entry_price = 10.0
        atr = 10.0  # Large ATR

        # Stop = 10 - 1.5 * 10 = -5, but should be clamped to 0
        stop, target = calculate_stop_target(entry_price, atr)

        assert stop == 0.0
        assert target == 30.0

    def test_rounded_prices(self):
        """Test that prices are rounded."""
        config = _make_config()
        entry_price = 100.0
        atr = 1.111

        # Stop = 100 - 1.5 * 1.111 = 98.3335 -> 98.33
        # Target = 100 + 2.0 * 1.111 = 102.222 -> 102.22
        stop, target = calculate_stop_target(entry_price, atr)

        assert stop == 98.33
        assert target == 102.22


class TestCalculateMaxLoss:
    """Test calculate_max_loss function."""

    def test_max_loss_calculation(self):
        """Test maximum loss calculation."""
        quantity = 100
        entry_price = 100.0
        stop_price = 95.0

        # Max loss = 100 * (100 - 95) = 500
        max_loss = calculate_max_loss(quantity, entry_price, stop_price)

        assert max_loss == 500.0

    def test_max_loss_zero_quantity(self):
        """Test max loss with zero quantity."""
        max_loss = calculate_max_loss(0, 100.0, 95.0)
        assert max_loss == 0.0


class TestRunComplianceChecks:
    """Test run_compliance_checks function."""

    def test_pass_all_checks(self):
        """Test passing all compliance checks."""
        config = _make_config(
            portfolio_value_inr=1000000.0,
            min_price_inr=100.0,
            max_single_position_pct=0.10,
            paper_trading_mode=True,
        )
        symbol = "RELIANCE"
        close = 150.0
        quantity = 100
        investment_inr = 15000.0  # 100 * 150

        status, reasons = run_compliance_checks(symbol, close, quantity, investment_inr, config)

        assert status == "PASS"
        assert len(reasons) == 0

    def test_penny_stock_fails_compliance(self):
        """Test that penny stock (price below minimum) fails compliance."""
        config = _make_config(
            min_price_inr=100.0,
            paper_trading_mode=True,
        )
        symbol = "PENNY"
        close = 50.0  # Below minimum
        quantity = 100
        investment_inr = 5000.0

        status, reasons = run_compliance_checks(symbol, close, quantity, investment_inr, config)

        assert status == "FAIL"
        assert any("below minimum" in r.lower() for r in reasons)

    def test_empty_symbol_fails(self):
        """Test that empty symbol fails compliance."""
        config = _make_config(paper_trading_mode=True)
        symbol = ""
        close = 150.0
        quantity = 100
        investment_inr = 15000.0

        status, reasons = run_compliance_checks(symbol, close, quantity, investment_inr, config)

        assert status == "FAIL"
        assert any("empty" in r.lower() for r in reasons)

    def test_zero_quantity_fails(self):
        """Test that zero quantity fails compliance."""
        config = _make_config(paper_trading_mode=True)
        symbol = "RELIANCE"
        close = 150.0
        quantity = 0
        investment_inr = 0.0

        status, reasons = run_compliance_checks(symbol, close, quantity, investment_inr, config)

        assert status == "FAIL"
        assert any("quantity" in r.lower() for r in reasons)

    def test_exceeds_max_position_fails(self):
        """Test that exceeding max position fails compliance."""
        config = _make_config(
            portfolio_value_inr=1000000.0,
            max_single_position_pct=0.10,  # 10% = 100000
            paper_trading_mode=True,
        )
        symbol = "RELIANCE"
        close = 150.0
        quantity = 1000
        investment_inr = 150000.0  # Exceeds 100000

        status, reasons = run_compliance_checks(symbol, close, quantity, investment_inr, config)

        assert status == "FAIL"
        assert any("exceeds" in r.lower() or "max position" in r.lower() for r in reasons)

    def test_paper_trading_disabled_fails(self):
        """Test that disabled paper trading mode fails compliance."""
        config = _make_config(paper_trading_mode=False)
        symbol = "RELIANCE"
        close = 150.0
        quantity = 100
        investment_inr = 15000.0

        status, reasons = run_compliance_checks(symbol, close, quantity, investment_inr, config)

        assert status == "FAIL"
        assert any("paper trading" in r.lower() for r in reasons)

    def test_fut_symbol_fails(self):
        """Test that F&O symbol suffix fails compliance."""
        config = _make_config(paper_trading_mode=True)
        symbol = "RELIANCE-FUT"
        close = 150.0
        quantity = 100
        investment_inr = 15000.0

        status, reasons = run_compliance_checks(symbol, close, quantity, investment_inr, config)

        assert status == "FAIL"
        assert any("F&O" in r or "-FUT" in r or "-OPT" in r for r in reasons)

    def test_opt_symbol_fails(self):
        """Test that options symbol suffix fails compliance."""
        config = _make_config(paper_trading_mode=True)
        symbol = "NIFTY-OPT"
        close = 150.0
        quantity = 100
        investment_inr = 15000.0

        status, reasons = run_compliance_checks(symbol, close, quantity, investment_inr, config)

        assert status == "FAIL"


class TestEstimateCapitalGainsTax:
    """Test estimate_capital_gains_tax function."""

    def test_stcg_tax(self):
        """Test short-term capital gains tax (< 365 days)."""
        gain_inr = 100000.0
        holding_days = 180  # Less than 365

        # STCG: 20%
        tax = estimate_capital_gains_tax(gain_inr, holding_days)

        assert tax == 20000.0  # 100000 * 0.20

    def test_stcg_tax_with_holding_days_exactly_364(self):
        """Test STCG with holding days = 364."""
        gain_inr = 50000.0
        holding_days = 364

        tax = estimate_capital_gains_tax(gain_inr, holding_days)

        assert tax == 10000.0  # 50000 * 0.20

    def test_ltcg_tax_under_exemption_limit(self):
        """Test LTCG tax when under exemption limit."""
        gain_inr = 100000.0
        holding_days = 400  # More than 365
        fy_ltcg_used_inr = 0.0

        # Exempt = 125000 - 0 = 125000
        # Taxable = max(0, 100000 - 125000) = 0
        # Tax = 0
        tax = estimate_capital_gains_tax(gain_inr, holding_days, fy_ltcg_used_inr)

        assert tax == 0.0

    def test_ltcg_tax_above_exemption_limit(self):
        """Test LTCG tax when above exemption limit."""
        gain_inr = 200000.0
        holding_days = 400
        fy_ltcg_used_inr = 0.0

        # Exempt = 125000 - 0 = 125000
        # Taxable = max(0, 200000 - 125000) = 75000
        # Tax = 75000 * 0.125 = 9375
        tax = estimate_capital_gains_tax(gain_inr, holding_days, fy_ltcg_used_inr)

        assert tax == 9375.0

    def test_ltcg_tax_with_partial_exemption_used(self):
        """Test LTCG tax when some exemption already used."""
        gain_inr = 100000.0
        holding_days = 400
        fy_ltcg_used_inr = 50000.0

        # Exempt = 125000 - 50000 = 75000
        # Taxable = max(0, 100000 - 75000) = 25000
        # Tax = 25000 * 0.125 = 3125
        tax = estimate_capital_gains_tax(gain_inr, holding_days, fy_ltcg_used_inr)

        assert tax == 3125.0

    def test_ltcg_tax_with_full_exemption_used(self):
        """Test LTCG tax when full exemption already used."""
        gain_inr = 100000.0
        holding_days = 400
        fy_ltcg_used_inr = 125000.0

        # Exempt = 125000 - 125000 = 0
        # Taxable = max(0, 100000 - 0) = 100000
        # Tax = 100000 * 0.125 = 12500
        tax = estimate_capital_gains_tax(gain_inr, holding_days, fy_ltcg_used_inr)

        assert tax == 12500.0

    def test_no_tax_on_loss(self):
        """Test that no tax is applied on losses."""
        gain_inr = -50000.0
        holding_days = 400

        tax = estimate_capital_gains_tax(gain_inr, holding_days)

        assert tax == 0.0

    def test_no_tax_on_zero_gain(self):
        """Test that no tax is applied on zero gain."""
        gain_inr = 0.0
        holding_days = 400

        tax = estimate_capital_gains_tax(gain_inr, holding_days)

        assert tax == 0.0

    def test_ltcg_exactly_at_exemption_threshold(self):
        """Test LTCG exactly at exemption threshold."""
        gain_inr = 125000.0
        holding_days = 400
        fy_ltcg_used_inr = 0.0

        # Exempt = 125000
        # Taxable = 0
        # Tax = 0
        tax = estimate_capital_gains_tax(gain_inr, holding_days, fy_ltcg_used_inr)

        assert tax == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
