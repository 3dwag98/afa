"""Risk management module."""

import math
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

from .config import AppConfig


def calculate_stop_target(entry_price: float, atr: Optional[float], config: AppConfig) -> Tuple[float, float]:
    """Calculate stop loss and target prices based on ATR.

    Args:
        entry_price: Entry price of the position.
        atr: Average True Range value, or None.
        config: Application configuration.

    Returns:
        Tuple of (stop_price, target_price), both rounded.
    """
    # Use fallback if atr is None or <= 0
    if atr is None or atr <= 0:
        stop = entry_price * 0.98  # 2% fallback stop
        target = entry_price * 1.03  # 3% fallback target
    else:
        stop = entry_price - 1.5 * atr
        target = entry_price + 2.0 * atr

    # Stop cannot be negative
    stop = max(0.0, stop)

    return (round(stop, 2), round(target, 2))


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
    risk_amount = config.portfolio_value_inr * config.risk_per_trade_pct

    # Risk per share = entry_price - stop_price
    risk_per_share = entry_price - stop_price

    # If risk per share <= 0, return 0
    if risk_per_share <= 0:
        return 0

    # quantity = floor(risk_amount / risk_per_share)
    quantity = math.floor(risk_amount / risk_per_share)

    # Max position value = portfolio_value_inr * max_single_position_pct
    max_position_value = config.portfolio_value_inr * config.max_single_position_pct

    # Reduce quantity if quantity * entry_price > max position value
    if quantity * entry_price > max_position_value:
        quantity = math.floor(max_position_value / entry_price)

    return max(0, quantity)


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
