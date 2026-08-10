"""Tests for risk and compliance modules."""

import pytest
from portfolio_agent.config.schema import AppConfig
from src.risk import (
    MAX_KELLY_FRACTION,
    calculate_quantity,
    calculate_stop_target,
    calculate_max_loss,
    calculate_kelly_fraction,
    calculate_kelly_quantity,
    calculate_position_quantity,
    estimate_kelly_inputs,
)
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


class TestEstimateKellyInputs:
    """Tests for estimate_kelly_inputs()."""

    def _trades(self, n_wins: int, n_losses: int, win_pct: float = 5.0, loss_pct: float = -2.0):
        trades = [{"outcome": "WIN", "return_pct": win_pct} for _ in range(n_wins)]
        trades += [{"outcome": "LOSS", "return_pct": loss_pct} for _ in range(n_losses)]
        return trades

    def test_none_below_min_trades(self):
        trades = self._trades(3, 2)
        assert estimate_kelly_inputs(trades, min_trades=20) is None

    def test_none_with_no_losses(self):
        trades = self._trades(20, 0)
        assert estimate_kelly_inputs(trades, min_trades=10) is None

    def test_none_with_no_wins(self):
        trades = self._trades(0, 20)
        assert estimate_kelly_inputs(trades, min_trades=10) is None

    def test_computes_win_rate_and_reward_risk(self):
        # 12 wins @ +6%, 8 losses @ -3% => raw p=0.6, b=6/3=2.0
        trades = self._trades(12, 8, win_pct=6.0, loss_pct=-3.0)
        result = estimate_kelly_inputs(trades, min_trades=10, shrinkage_strength=0.0)
        assert result is not None
        win_probability, reward_risk_ratio = result
        assert win_probability == pytest.approx(0.6)
        assert reward_risk_ratio == pytest.approx(2.0)

    def test_shrinks_win_rate_toward_coin_flip_by_default(self):
        # Same 12/8 sample, but with the default 20-pseudo-trade Beta(10, 10)
        # prior: (12 + 10) / (20 + 20) = 0.55, not the raw 0.60. Kelly is far
        # more punishing of over-betting than under-betting, so a 20-trade
        # sample must not be read as a 60% edge.
        trades = self._trades(12, 8, win_pct=6.0, loss_pct=-3.0)
        result = estimate_kelly_inputs(trades, min_trades=10)
        assert result is not None
        win_probability, reward_risk_ratio = result
        assert win_probability == pytest.approx(0.55)
        # The payoff ratio is a magnitude average, not a proportion, so it is
        # reported unshrunk.
        assert reward_risk_ratio == pytest.approx(2.0)

    def test_shrinkage_fades_as_evidence_accumulates(self):
        # The same 60% raw win rate over 20 vs 400 trades: the prior dominates
        # the small sample and barely moves the large one.
        small = estimate_kelly_inputs(self._trades(12, 8), min_trades=10)
        large = estimate_kelly_inputs(self._trades(240, 160), min_trades=10)
        assert small is not None and large is not None
        assert small[0] == pytest.approx(0.55)
        assert large[0] == pytest.approx(0.5952, abs=1e-4)

    def test_default_min_trades_rejects_small_samples(self):
        # 30 realized trades is below the 50-trade floor, so Kelly is not
        # trusted at all and the caller falls back to fixed-fractional sizing.
        assert estimate_kelly_inputs(self._trades(18, 12)) is None
        assert estimate_kelly_inputs(self._trades(30, 25)) is not None

    def test_ignores_pending_trades(self):
        trades = self._trades(10, 10) + [{"outcome": "PENDING", "return_pct": 0.0}] * 100
        result = estimate_kelly_inputs(trades, min_trades=20)
        assert result is not None


class TestCalculateKellyFraction:
    """Tests for calculate_kelly_fraction()."""

    def test_positive_edge(self):
        # p=0.6, b=2.0 => f* = 0.6 - 0.4/2 = 0.4
        assert calculate_kelly_fraction(0.6, 2.0) == pytest.approx(0.4)

    def test_negative_edge_clamped_to_zero(self):
        # p=0.3, b=1.0 => f* = 0.3 - 0.7/1 = -0.4 -> clamped to 0
        assert calculate_kelly_fraction(0.3, 1.0) == 0.0

    def test_zero_reward_risk_returns_zero(self):
        assert calculate_kelly_fraction(0.9, 0.0) == 0.0

    def test_clamped_to_one(self):
        assert calculate_kelly_fraction(0.99, 100.0) <= 1.0


class TestCalculateKellyQuantity:
    """Tests for calculate_kelly_quantity()."""

    def test_basic_sizing(self):
        qty = calculate_kelly_quantity(
            entry_price=100.0,
            portfolio_value_inr=1_000_000.0,
            max_single_position_pct=0.5,
            win_probability=0.6,
            reward_risk_ratio=2.0,
            kelly_fraction=0.25,
        )
        # f* = 0.4, quarter-Kelly = 0.1 -> position value ~= 100,000 -> qty ~= 1000
        # (floor of a floating-point division, so off-by-one from binary rounding is fine)
        assert qty in (999, 1000)

    def test_kappa_is_hard_capped_at_quarter_kelly(self):
        """The cap lives in calculate_kelly_quantity(), not in the config, so
        no YAML, override or test fixture can size above quarter-Kelly."""
        common = dict(
            entry_price=100.0,
            portfolio_value_inr=1_000_000.0,
            max_single_position_pct=0.9,
            win_probability=0.6,
            reward_risk_ratio=2.0,
        )

        at_cap = calculate_kelly_quantity(kelly_fraction=MAX_KELLY_FRACTION, **common)

        assert MAX_KELLY_FRACTION == 0.25
        assert calculate_kelly_quantity(kelly_fraction=0.5, **common) == at_cap
        assert calculate_kelly_quantity(kelly_fraction=1.0, **common) == at_cap
        # Below the cap still scales normally.
        assert calculate_kelly_quantity(kelly_fraction=0.1, **common) < at_cap

    def test_capped_by_max_single_position_pct(self):
        qty = calculate_kelly_quantity(
            entry_price=100.0,
            portfolio_value_inr=1_000_000.0,
            max_single_position_pct=0.05,
            win_probability=0.6,
            reward_risk_ratio=2.0,
            kelly_fraction=0.5,
        )
        # Uncapped would be 2000 shares (200,000 INR); cap is 5% = 50,000 INR -> 500 shares
        assert qty == 500

    def test_zero_quantity_on_negative_edge(self):
        qty = calculate_kelly_quantity(
            entry_price=100.0,
            portfolio_value_inr=1_000_000.0,
            max_single_position_pct=0.5,
            win_probability=0.3,
            reward_risk_ratio=1.0,
            kelly_fraction=0.5,
        )
        assert qty == 0

    def test_zero_entry_price_returns_zero(self):
        qty = calculate_kelly_quantity(
            entry_price=0.0,
            portfolio_value_inr=1_000_000.0,
            max_single_position_pct=0.5,
            win_probability=0.6,
            reward_risk_ratio=2.0,
        )
        assert qty == 0


class TestCalculatePositionQuantity:
    """Tests for calculate_position_quantity() — the single sizing entry point."""

    def test_falls_back_to_fixed_fractional_when_kelly_disabled(self):
        config = _make_config()
        trade_history = [{"outcome": "WIN", "return_pct": 6.0}] * 12 + [{"outcome": "LOSS", "return_pct": -3.0}] * 8
        expected = calculate_quantity(entry_price=150.0, stop_price=145.0, config=config)
        actual = calculate_position_quantity(
            entry_price=150.0, stop_price=145.0, config=config, trade_history=trade_history
        )
        assert actual == expected

    def test_falls_back_when_not_enough_trade_history(self):
        config = _make_config()
        config.risk.use_kelly_sizing = True
        config.risk.kelly_min_trades = 20
        trade_history = [{"outcome": "WIN", "return_pct": 6.0}] * 3
        expected = calculate_quantity(entry_price=150.0, stop_price=145.0, config=config)
        actual = calculate_position_quantity(
            entry_price=150.0, stop_price=145.0, config=config, trade_history=trade_history
        )
        assert actual == expected

    def test_uses_kelly_when_enabled_and_enough_history(self):
        config = _make_config(portfolio_value_inr=1_000_000.0, max_single_position_pct=0.5)
        config.risk.use_kelly_sizing = True
        config.risk.kelly_min_trades = 10
        config.risk.kelly_fraction = 0.25
        config.risk.kelly_shrinkage_strength = 0.0  # raw win rate, no shrinkage
        trade_history = [{"outcome": "WIN", "return_pct": 6.0}] * 12 + [{"outcome": "LOSS", "return_pct": -3.0}] * 8

        actual = calculate_position_quantity(
            entry_price=100.0, stop_price=95.0, config=config, trade_history=trade_history
        )
        # p=0.6, b=2.0 -> f*=0.4, quarter-Kelly=0.1 -> ~100,000 INR / 100 = ~1000 shares
        assert actual in (999, 1000)

    def test_shrinkage_sizes_more_conservatively_than_raw_win_rate(self):
        """Default shrinkage must never size *larger* than the raw estimate."""
        config = _make_config(portfolio_value_inr=1_000_000.0, max_single_position_pct=0.5)
        config.risk.use_kelly_sizing = True
        config.risk.kelly_min_trades = 10
        config.risk.kelly_fraction = 0.25
        trade_history = [{"outcome": "WIN", "return_pct": 6.0}] * 12 + [{"outcome": "LOSS", "return_pct": -3.0}] * 8

        config.risk.kelly_shrinkage_strength = 0.0
        raw = calculate_position_quantity(
            entry_price=100.0, stop_price=95.0, config=config, trade_history=trade_history
        )
        config.risk.kelly_shrinkage_strength = 20.0
        shrunk = calculate_position_quantity(
            entry_price=100.0, stop_price=95.0, config=config, trade_history=trade_history
        )

        # p=0.55, b=2.0 -> f*=0.325, quarter-Kelly=0.08125 -> ~812 shares
        assert shrunk == 812
        assert shrunk < raw


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
