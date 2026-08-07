"""Monte Carlo simulation module for risk analysis."""

import numpy as np
import pandas as pd
from typing import Dict, Any


def run_monte_carlo(returns: pd.Series, horizon_days: int = 20, 
                    simulations: int = 1000, seed: int = 42) -> Dict[str, Any]:
    """Run Monte Carlo simulation on historical returns.

    Args:
        returns: Series of daily returns.
        horizon_days: Number of days to simulate forward.
        simulations: Number of simulation runs.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with simulation results.
    """
    np.random.seed(seed)

    if len(returns) < 30:
        return {
            'mean_return': 0.0,
            'std_return': 0.0,
            'percentile_5': 0.0,
            'percentile_95': 0.0,
            'probability_profit': 0.5,
            'simulations_count': 0,
            'horizon_days': horizon_days,
            'error': 'Insufficient data for simulation'
        }

    # Calculate parameters
    mu = returns.mean()
    sigma = returns.std()

    # Simulate future paths
    final_values = []
    for _ in range(simulations):
        random_returns = np.random.normal(mu, sigma, horizon_days)
        cumulative_return = np.prod(1 + random_returns) - 1
        final_values.append(cumulative_return)

    final_values = np.array(final_values)

    # Calculate statistics
    mean_return = np.mean(final_values)
    std_return = np.std(final_values)
    percentile_5 = np.percentile(final_values, 5)
    percentile_95 = np.percentile(final_values, 95)
    probability_profit = np.mean(final_values > 0)

    return {
        'mean_return': mean_return,
        'std_return': std_return,
        'percentile_5': percentile_5,
        'percentile_95': percentile_95,
        'probability_profit': probability_profit,
        'simulations_count': simulations,
        'horizon_days': horizon_days
    }


def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Calculate Value at Risk.

    Args:
        returns: Series of daily returns.
        confidence: Confidence level (e.g., 0.95 for 95%).

    Returns:
        VaR at specified confidence level.
    """
    return np.percentile(returns, (1 - confidence) * 100)


def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall).

    Args:
        returns: Series of daily returns.
        confidence: Confidence level.

    Returns:
        CVaR at specified confidence level.
    """
    var = calculate_var(returns, confidence)
    return returns[returns <= var].mean()
