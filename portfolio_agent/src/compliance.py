"""Compliance and validation module."""

from typing import Dict, Any, List, Optional, Tuple

from portfolio_agent.config.schema import AppConfig


def run_compliance_checks(
    symbol: str,
    close: float,
    quantity: int,
    investment_inr: float,
    config: AppConfig
) -> Tuple[str, List[str]]:
    """Run compliance checks on a trade.

    Args:
        symbol: Stock symbol.
        close: Closing price.
        quantity: Number of shares.
        investment_inr: Total investment amount in INR.
        config: Application configuration.

    Returns:
        Tuple of (status, failed_reasons) where status is 'PASS' or 'FAIL'.
    """
    failed_reasons = []

    # Check: symbol must not be empty
    if not symbol or symbol.strip() == "":
        failed_reasons.append("Symbol is empty")

    # Check: close >= config.compliance.min_price_inr
    if close < config.compliance.min_price_inr:
        failed_reasons.append(f"Price {close} below minimum {config.compliance.min_price_inr}")

    # Check: quantity > 0
    if quantity <= 0:
        failed_reasons.append("Quantity must be greater than 0")

    # Check: investment_inr <= portfolio_value_inr * max_single_position_pct
    max_position_value = config.risk.portfolio_value_inr * config.risk.max_single_position_pct
    if investment_inr > max_position_value:
        failed_reasons.append(f"Investment {investment_inr} exceeds max position {max_position_value}")

    # Check: paper_trading_mode must be true for now
    if not config.compliance.paper_trading_mode:
        failed_reasons.append("Paper trading mode must be enabled")

    # Check: no F&O symbol suffix like "-FUT" or "-OPT"
    if symbol.endswith("-FUT") or symbol.endswith("-OPT"):
        failed_reasons.append("F&O symbols not allowed")

    status = "PASS" if len(failed_reasons) == 0 else "FAIL"
    return (status, failed_reasons)


def estimate_capital_gains_tax(
    gain_inr: float,
    holding_days: int,
    fy_ltcg_used_inr: float = 0.0
) -> float:
    """Estimate capital gains tax.

    Args:
        gain_inr: Capital gain amount (negative for losses).
        holding_days: Number of days held.
        fy_ltcg_used_inr: LTCG exemption already used in FY.

    Returns:
        Tax amount in INR.
    """
    # If gain <= 0, return 0
    if gain_inr <= 0:
        return 0.0

    if holding_days < 365:
        # STCG: tax = gain * 20%
        return gain_inr * 0.20
    else:
        # LTCG: exempt = max(0, 125000 - fy_ltcg_used_inr)
        exempt = max(0.0, 125000 - fy_ltcg_used_inr)
        taxable_ltcg = max(0.0, gain_inr - exempt)
        tax = taxable_ltcg * 0.125
        return tax


def check_price_minimum(price: float, min_price: float) -> bool:
    """Check if stock price meets minimum threshold.

    Args:
        price: Current stock price.
        min_price: Minimum allowed price.

    Returns:
        True if price is acceptable.
    """
    return price >= min_price


def check_position_concentration(current_value: float, portfolio_value: float,
                                  max_pct: float) -> bool:
    """Check if position exceeds concentration limits.

    Args:
        current_value: Current position value.
        portfolio_value: Total portfolio value.
        max_pct: Maximum allowed concentration.

    Returns:
        True if within limits.
    """
    if portfolio_value <= 0:
        return True
    concentration = current_value / portfolio_value
    return concentration <= max_pct


def validate_ticker(ticker: str, allowed_tickers: List[str]) -> bool:
    """Validate that ticker is in allowed list.

    Args:
        ticker: Ticker symbol to validate.
        allowed_tickers: List of allowed tickers.

    Returns:
        True if ticker is allowed.
    """
    return ticker in allowed_tickers


def check_risk_reward_ratio(entry_price: float, target_price: float,
                            stop_loss_price: float, min_ratio: float,
                            buy_cost_pct: Optional[float] = None,
                            sell_cost_pct: Optional[float] = None) -> bool:
    """Check if trade meets the minimum risk-reward ratio, net of costs.

    The strategy layer reports `reward_risk` net of estimated round-trip
    friction (src/risk.py::net_reward_risk). Compliance must measure the same
    quantity against the same `min_reward_risk` threshold, or the two gates
    disagree: a trade the strategy graded cost-negative at 0.84 would be
    recomputed here at a gross 1.33 and wave through, while the exported
    report shows the net figure compliance never actually tested.

    Args:
        entry_price: Entry price.
        target_price: Target price.
        stop_loss_price: Stop loss price.
        min_ratio: Minimum required reward/risk ratio.
        buy_cost_pct: Buy-leg friction as a fraction of turnover. Defaults to
            the platform's standard estimate.
        sell_cost_pct: Sell-leg friction as a fraction of turnover.

    Returns:
        True if the net ratio is acceptable.
    """
    if stop_loss_price >= entry_price:
        return False

    from .execution_sim import cost_fraction_per_side
    from .risk import net_reward_risk

    if buy_cost_pct is None:
        buy_cost_pct = cost_fraction_per_side('BUY')
    if sell_cost_pct is None:
        sell_cost_pct = cost_fraction_per_side('SELL')

    ratio = net_reward_risk(
        entry_price=entry_price,
        stop_price=stop_loss_price,
        target_price=target_price,
        buy_cost_pct=buy_cost_pct,
        sell_cost_pct=sell_cost_pct,
    )
    return ratio >= min_ratio


def compliance_check(recommendation: Dict[str, Any], 
                     config: Any,
                     current_positions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run full compliance check on a recommendation.

    Args:
        recommendation: Recommendation dictionary.
        config: Configuration object with limits.
        current_positions: Current portfolio positions.

    Returns:
        Dictionary with compliance results.
    """
    results = {
        'passed': True,
        'checks': {},
        'warnings': []
    }

    ticker = recommendation.get('ticker', '')
    action = recommendation.get('action', '')
    quantity = recommendation.get('quantity', 0)
    price = recommendation.get('entry_price', 0)

    # Check minimum price
    price_ok = check_price_minimum(price, config.min_price_inr)
    results['checks']['min_price'] = price_ok
    if not price_ok:
        results['passed'] = False
        results['warnings'].append(f"Price below minimum: {price} < {config.min_price_inr}")

    # Check ticker validity
    ticker_ok = validate_ticker(ticker, config.tickers)
    results['checks']['valid_ticker'] = ticker_ok
    if not ticker_ok:
        results['passed'] = False
        results['warnings'].append(f"Ticker not in allowed list: {ticker}")

    # Check position concentration (for BUY actions)
    if action == 'BUY' and current_positions:
        existing = current_positions.get(ticker, {}).get('value', 0)
        new_value = existing + (quantity * price)
        concentration_ok = check_position_concentration(
            new_value, config.portfolio_value_inr, config.max_single_position_pct
        )
        results['checks']['concentration'] = concentration_ok
        if not concentration_ok:
            results['passed'] = False
            results['warnings'].append("Position would exceed concentration limit")
    else:
        results['checks']['concentration'] = True

    # Check risk-reward ratio
    if action == 'BUY':
        target = recommendation.get('target_price', price * 1.05)
        stop = recommendation.get('stop_loss', price * 0.95)
        rr_ok = check_risk_reward_ratio(price, target, stop, config.min_reward_risk)
        results['checks']['risk_reward'] = rr_ok
        if not rr_ok:
            results['warnings'].append("Risk-reward ratio below minimum")
    else:
        results['checks']['risk_reward'] = True

    return results


def generate_compliance_report(compliance_results: List[Dict[str, Any]]) -> str:
    """Generate text report from compliance checks.

    Args:
        compliance_results: List of compliance check results.

    Returns:
        Formatted report string.
    """
    total = len(compliance_results)
    passed = sum(1 for r in compliance_results if r.get('passed', False))

    report = [
        "=" * 50,
        "COMPLIANCE REPORT",
        "=" * 50,
        f"Total Recommendations: {total}",
        f"Passed: {passed}",
        f"Failed: {total - passed}",
        f"Pass Rate: {(passed/total*100) if total > 0 else 0:.1f}%",
        "-" * 50
    ]

    for i, result in enumerate(compliance_results):
        status = "PASS" if result.get('passed') else "FAIL"
        ticker = result.get('ticker', 'N/A')
        warnings = result.get('warnings', [])

        report.append(f"\n[{i+1}] {ticker}: {status}")
        for check_name, check_result in result.get('checks', {}).items():
            symbol = "✓" if check_result else "✗"
            report.append(f"    {symbol} {check_name}")

        if warnings:
            report.append("    Warnings:")
            for warning in warnings:
                report.append(f"      - {warning}")

    report.append("=" * 50)
    return "\n".join(report)
