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
    kelly_allocation_fraction,
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
        assert result.win_probability == pytest.approx(0.6)
        assert result.reward_risk_ratio == pytest.approx(2.0)

    def test_reports_the_payoff_magnitudes_not_just_their_ratio(self):
        """The loss magnitude has to survive estimation, not be divided away.

        Sizing a stop-loss trade needs l in its own right (see
        kelly_allocation_fraction); a caller handed only b = g/l cannot
        recover it, which is exactly how the allocation lost its 1/l factor.
        """
        trades = self._trades(12, 8, win_pct=6.0, loss_pct=-3.0)
        result = estimate_kelly_inputs(trades, min_trades=10, shrinkage_strength=0.0)
        assert result is not None
        assert result.avg_win_pct == pytest.approx(6.0)
        assert result.avg_loss_pct == pytest.approx(3.0)
        assert result.reward_risk_ratio == pytest.approx(
            result.avg_win_pct / result.avg_loss_pct
        )

    def test_shrinks_win_rate_toward_coin_flip_by_default(self):
        # Same 12/8 sample, but with the default 20-pseudo-trade Beta(10, 10)
        # prior: (12 + 10) / (20 + 20) = 0.55, not the raw 0.60. Kelly is far
        # more punishing of over-betting than under-betting, so a 20-trade
        # sample must not be read as a 60% edge.
        trades = self._trades(12, 8, win_pct=6.0, loss_pct=-3.0)
        result = estimate_kelly_inputs(trades, min_trades=10)
        assert result is not None
        assert result.win_probability == pytest.approx(0.55)
        # The payoff ratio is a magnitude average, not a proportion, so it is
        # reported unshrunk.
        assert result.reward_risk_ratio == pytest.approx(2.0)

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
    """Tests for calculate_kelly_fraction() — the binary-bet *stake* fraction.

    Kept as its own quantity, and explicitly not a position size: see
    TestKellyAllocationFraction for the conversion sizing actually uses.
    """

    def test_positive_edge(self):
        # p=0.6, b=2.0 => f = 0.6 - 0.4/2 = 0.4
        assert calculate_kelly_fraction(0.6, 2.0) == pytest.approx(0.4)

    def test_negative_edge_clamped_to_zero(self):
        # p=0.3, b=1.0 => f = 0.3 - 0.7/1 = -0.4 -> clamped to 0
        assert calculate_kelly_fraction(0.3, 1.0) == 0.0

    def test_zero_reward_risk_returns_zero(self):
        assert calculate_kelly_fraction(0.9, 0.0) == 0.0

    def test_clamped_to_one(self):
        assert calculate_kelly_fraction(0.99, 100.0) <= 1.0


class TestKellyAllocationFraction:
    """f* = (p*g - (1-p)*l) / (g*l) — the fraction of *wealth* to allocate.

    The binary-bet form f = p - (1-p)/b is the same edge measured per rupee at
    risk. Using it as an allocation silently assumes a loss costs the whole
    position, which for a stop-loss trade it does not.
    """

    def test_equals_the_binary_form_divided_by_loss_given_stop(self):
        # p=0.55, g=10.8%, l=6% -> b=1.8. Binary f = 0.55 - 0.45/1.8 = 0.30.
        # The allocation is that divided by l=0.06, i.e. 5.0 — a factor of
        # 16.7x. This is the defect: sizing a book off 0.30 runs it at ~6% of
        # the quarter-Kelly it documents.
        binary = calculate_kelly_fraction(0.55, 1.8)
        allocation = kelly_allocation_fraction(0.55, avg_win_pct=10.8, avg_loss_pct=6.0)

        assert binary == pytest.approx(0.30, abs=1e-9)
        assert allocation == pytest.approx(5.0, abs=1e-9)
        assert allocation == pytest.approx(binary / 0.06, rel=1e-9)

    def test_is_not_a_constant_rescaling_of_the_binary_form(self):
        """The error the units bug caused is signal-dependent, not a shrink.

        Two signals with the same edge per rupee at risk (same p and b) but
        different stop widths must get *different* allocations — the wide-stop
        one smaller. The binary form gives them the same number, which is why
        raising kappa cannot recover it.
        """
        tight = kelly_allocation_fraction(0.55, avg_win_pct=3.6, avg_loss_pct=2.0)
        wide = kelly_allocation_fraction(0.55, avg_win_pct=18.0, avg_loss_pct=10.0)

        # Same p, same b = 1.8, so the binary form is identical for both.
        assert calculate_kelly_fraction(0.55, 1.8) == pytest.approx(
            calculate_kelly_fraction(0.55, 18.0 / 10.0)
        )
        # The allocation correctly scales inversely with the loss magnitude.
        assert tight == pytest.approx(wide * 5.0, rel=1e-9)
        assert wide < tight

    def test_negative_edge_returns_zero(self):
        # p=0.3, g=l -> edge = 0.3 - 0.7 < 0
        assert kelly_allocation_fraction(0.3, avg_win_pct=5.0, avg_loss_pct=5.0) == 0.0

    def test_unusable_magnitudes_return_zero(self):
        assert kelly_allocation_fraction(0.9, avg_win_pct=0.0, avg_loss_pct=5.0) == 0.0
        assert kelly_allocation_fraction(0.9, avg_win_pct=5.0, avg_loss_pct=0.0) == 0.0

    def test_is_unbounded_above(self):
        """A growth-optimal allocation above 1 is a true statement, not an
        overflow to be clamped away. The leverage limit belongs to the risk
        policy applied by the caller, not to the formula."""
        assert kelly_allocation_fraction(0.6, avg_win_pct=6.0, avg_loss_pct=3.0) > 1.0


class TestCalculateKellyQuantity:
    """Tests for calculate_kelly_quantity()."""

    def test_basic_sizing(self):
        qty = calculate_kelly_quantity(
            entry_price=100.0,
            portfolio_value_inr=1_000_000.0,
            max_single_position_pct=0.05,
            win_probability=0.6,
            avg_win_pct=6.0,
            avg_loss_pct=3.0,
            kelly_fraction=0.25,
        )
        # f* = (0.6*0.06 - 0.4*0.03) / (0.06*0.03) = 0.024/0.0018 = 13.33,
        # quarter-Kelly = 3.33 -> leverage-capped to 1.0 -> concentration cap
        # 5% = 50,000 INR -> 500 shares. With the units right the caps bind,
        # which is the honest consequence, not a bug.
        assert qty == 500

    def test_leverage_constraint_caps_the_allocation_at_the_whole_book(self):
        """No single position may exceed portfolio value: this is a cash book."""
        qty = calculate_kelly_quantity(
            entry_price=100.0,
            portfolio_value_inr=1_000_000.0,
            max_single_position_pct=10.0,  # deliberately non-binding
            win_probability=0.6,
            avg_win_pct=6.0,
            avg_loss_pct=3.0,
            kelly_fraction=MAX_KELLY_FRACTION,
        )
        # kappa*f* = 3.33 would be 333% of the book; the leverage constraint
        # holds it at 100% -> 1,000,000 / 100 = 10,000 shares.
        assert qty == 10_000

    def test_kappa_is_hard_capped_at_quarter_kelly(self):
        """The cap lives in calculate_kelly_quantity(), not in the config, so
        no YAML, override or test fixture can size above quarter-Kelly."""
        common = dict(
            entry_price=100.0,
            portfolio_value_inr=1_000_000.0,
            max_single_position_pct=0.9,
            # A thin edge, so kappa still moves the answer instead of every
            # variant pinning to the concentration cap.
            win_probability=0.4475,
            avg_win_pct=5.0,
            avg_loss_pct=4.0,
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
            avg_win_pct=6.0,
            avg_loss_pct=3.0,
            kelly_fraction=0.5,
        )
        # Cap is 5% = 50,000 INR -> 500 shares.
        assert qty == 500

    def test_sizes_down_continuously_as_the_edge_vanishes(self):
        """Where Kelly still binds after the caps: near zero edge.

        The break-even win rate for g=5%, l=4% is l/(g+l) = 4/9 = 0.444. Just
        above it the allocation is small enough to sit under both the leverage
        and concentration limits, and it collapses to zero at the boundary.
        """
        common = dict(
            entry_price=100.0,
            portfolio_value_inr=1_000_000.0,
            max_single_position_pct=0.9,
            avg_win_pct=5.0,
            avg_loss_pct=4.0,
            kelly_fraction=MAX_KELLY_FRACTION,
        )
        sizes = [
            calculate_kelly_quantity(win_probability=p, **common)
            for p in (4.0 / 9.0, 0.446, 0.450, 0.460)
        ]
        assert sizes[0] == 0
        assert sizes == sorted(sizes)
        assert sizes[-1] > sizes[1] > 0

    def test_zero_quantity_on_negative_edge(self):
        qty = calculate_kelly_quantity(
            entry_price=100.0,
            portfolio_value_inr=1_000_000.0,
            max_single_position_pct=0.5,
            win_probability=0.3,
            avg_win_pct=5.0,
            avg_loss_pct=5.0,
            kelly_fraction=0.5,
        )
        assert qty == 0

    def test_zero_entry_price_returns_zero(self):
        qty = calculate_kelly_quantity(
            entry_price=0.0,
            portfolio_value_inr=1_000_000.0,
            max_single_position_pct=0.5,
            win_probability=0.6,
            avg_win_pct=6.0,
            avg_loss_pct=3.0,
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

    def test_kelly_never_sizes_above_the_fixed_fractional_risk_budget(self):
        """Kelly is a ceiling, not a licence.

        Correcting the allocation units multiplies f* by 1/l (~17x at a 6%
        stop). Left unconstrained that would land as an across-the-board
        increase in position size — a leverage change dressed up as a bug fix.
        The per-trade risk budget, priced off *this* signal's known stop
        distance, remains the binding constraint.
        """
        config = _make_config(portfolio_value_inr=1_000_000.0, max_single_position_pct=0.5)
        config.risk.use_kelly_sizing = True
        config.risk.kelly_min_trades = 10
        config.risk.kelly_fraction = 0.25
        config.risk.kelly_shrinkage_strength = 0.0
        trade_history = [{"outcome": "WIN", "return_pct": 6.0}] * 12 + [{"outcome": "LOSS", "return_pct": -3.0}] * 8

        fixed = calculate_quantity(entry_price=100.0, stop_price=95.0, config=config)
        actual = calculate_position_quantity(
            entry_price=100.0, stop_price=95.0, config=config, trade_history=trade_history
        )

        # 1% of 1,000,000 = 10,000 INR of risk over a 5 INR stop -> 2,000 shares.
        assert fixed == 2000
        assert actual == fixed

    def test_kelly_sizes_down_when_the_estimated_edge_is_thin(self):
        """The case where Kelly is not a no-op: it de-risks toward break-even.

        With g=5%, l=4% the break-even win rate is 4/9 = 0.444. A sample that
        shrinks to just above it must size *below* the fixed-fractional budget.
        """
        config = _make_config(portfolio_value_inr=1_000_000.0, max_single_position_pct=0.5)
        config.risk.use_kelly_sizing = True
        config.risk.kelly_min_trades = 10
        config.risk.kelly_fraction = 0.25
        config.risk.kelly_shrinkage_strength = 0.0
        trade_history = (
            [{"outcome": "WIN", "return_pct": 5.0}] * 45
            + [{"outcome": "LOSS", "return_pct": -4.0}] * 55
        )

        fixed = calculate_quantity(entry_price=100.0, stop_price=95.0, config=config)
        actual = calculate_position_quantity(
            entry_price=100.0, stop_price=95.0, config=config, trade_history=trade_history
        )

        # p=0.45 -> edge = 0.45*0.05 - 0.55*0.04 = 0.0005, f* = 0.25,
        # quarter-Kelly = 0.0625 -> 62,500 INR / 100 = 625 shares.
        assert actual == 625
        assert 0 < actual < fixed

    def test_shrinkage_sizes_more_conservatively_than_raw_win_rate(self):
        """Default shrinkage must never size *larger* than the raw estimate."""
        config = _make_config(portfolio_value_inr=1_000_000.0, max_single_position_pct=0.5)
        config.risk.use_kelly_sizing = True
        config.risk.kelly_min_trades = 10
        config.risk.kelly_fraction = 0.25
        # A thin edge, so both variants sit below the caps and the comparison
        # measures shrinkage rather than the constraint stack. g=4%, l=5% puts
        # break-even at 5/9 = 0.556, so a raw 0.57 win rate is a real but small
        # edge and shrinking it toward 0.5 correctly shrinks the position.
        trade_history = (
            [{"outcome": "WIN", "return_pct": 4.0}] * 57
            + [{"outcome": "LOSS", "return_pct": -5.0}] * 43
        )

        config.risk.kelly_shrinkage_strength = 0.0
        raw = calculate_position_quantity(
            entry_price=100.0, stop_price=95.0, config=config, trade_history=trade_history
        )
        config.risk.kelly_shrinkage_strength = 20.0
        shrunk = calculate_position_quantity(
            entry_price=100.0, stop_price=95.0, config=config, trade_history=trade_history
        )

        assert shrunk < raw


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestNetRealizedReturn:
    """Kelly's two inputs both have to be measured after friction. On an Indian
    delivery round trip that is ~0.8% of turnover — enough to shrink b and to
    flip marginal winners into losers."""

    def test_costs_are_charged_to_both_legs(self):
        from src.risk import net_realized_return_pct

        gross = net_realized_return_pct(100.0, 110.0, buy_cost_pct=0.0, sell_cost_pct=0.0)
        net = net_realized_return_pct(100.0, 110.0, buy_cost_pct=0.004, sell_cost_pct=0.004)

        assert gross == pytest.approx(10.0)
        # basis 100.4, proceeds 109.56 -> 9.12%
        assert net == pytest.approx(9.12, abs=0.01)
        assert net < gross

    def test_an_unusable_entry_price_returns_zero(self):
        from src.risk import net_realized_return_pct

        assert net_realized_return_pct(0.0, 110.0, 0.004, 0.004) == 0.0

    def test_a_marginal_gross_win_is_reclassified_as_a_net_loss(self):
        """The bias that matters: counting this as a WIN adds a phantom win to
        p *and* drags the average win magnitude down, inflating f* twice."""
        from src.risk import to_net_realized_trades

        restated = to_net_realized_trades(
            [{"entry_price": 100.0, "exit_price": 100.3, "outcome": "WIN", "return_pct": 0.3}],
            buy_cost_pct=0.004,
            sell_cost_pct=0.004,
        )

        assert restated[0]["outcome"] == "LOSS"
        assert restated[0]["return_pct"] < 0

    def test_other_keys_survive_the_restatement(self):
        from src.risk import to_net_realized_trades

        restated = to_net_realized_trades(
            [{"entry_price": 100.0, "exit_price": 120.0, "signal_trigger": "Trend",
              "symbol": "ACME", "outcome": "WIN", "return_pct": 20.0}],
            buy_cost_pct=0.004, sell_cost_pct=0.004,
        )

        assert restated[0]["signal_trigger"] == "Trend"
        assert restated[0]["symbol"] == "ACME"
        assert restated[0]["outcome"] == "WIN"

    def test_open_and_unpriced_trades_are_dropped_not_guessed_at(self):
        from src.risk import to_net_realized_trades

        restated = to_net_realized_trades(
            [
                {"entry_price": 100.0, "exit_price": None, "outcome": "PENDING"},
                {"entry_price": None, "exit_price": 120.0, "outcome": "WIN"},
                {"entry_price": 100.0, "exit_price": "n/a", "outcome": "WIN"},
            ],
            buy_cost_pct=0.004, sell_cost_pct=0.004,
        )

        assert restated == []

    def test_kelly_sizes_smaller_off_net_history_than_gross(self):
        """The DoD for the calibration task: identical trades, costed, produce
        a smaller position."""
        from src.risk import calculate_kelly_quantity, estimate_kelly_inputs, to_net_realized_trades

        gross_history = [
            {
                "entry_price": 100.0,
                "exit_price": 106.0 if i % 5 else 96.0,
                "outcome": "WIN" if i % 5 else "LOSS",
                "return_pct": 6.0 if i % 5 else -4.0,
            }
            for i in range(60)
        ]
        net_history = to_net_realized_trades(gross_history, buy_cost_pct=0.004, sell_cost_pct=0.004)

        def _allocation(history):
            inputs = estimate_kelly_inputs(history, min_trades=50)
            assert inputs is not None
            return kelly_allocation_fraction(
                inputs.win_probability, inputs.avg_win_pct, inputs.avg_loss_pct
            )

        # Measured on the allocation rather than the share count: this sample
        # carries a large edge, so both variants would pin to the leverage and
        # concentration caps and the comparison would read equal. The caps
        # binding is the documented behaviour (see calculate_kelly_quantity);
        # what this test is about is that friction reaches f* at all.
        assert _allocation(net_history) < _allocation(gross_history)
