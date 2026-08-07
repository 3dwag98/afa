"""Compliance and validation module."""

from typing import Dict, Any, List, Optional


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
                            stop_loss_price: float, min_ratio: float) -> bool:
    """Check if trade meets minimum risk-reward ratio.

    Args:
        entry_price: Entry price.
        target_price: Target price.
        stop_loss_price: Stop loss price.
        min_ratio: Minimum required reward/risk ratio.

    Returns:
        True if ratio is acceptable.
    """
    if stop_loss_price >= entry_price:
        return False

    risk = entry_price - stop_loss_price
    reward = target_price - entry_price

    if risk <= 0:
        return False

    ratio = reward / risk
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
