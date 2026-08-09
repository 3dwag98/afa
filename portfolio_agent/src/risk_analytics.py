"""
Risk Analytics Module for Portfolio Agent.

Provides institutional-grade risk metrics for evaluating a completed backtest's
PORTFOLIO-LEVEL performance: CAGR/Sharpe/Sortino/Calmar, drawdown analysis, and
bootstrap-resampling Monte Carlo simulation of the realized trade distribution
(probability of ruin). This is a distinct concern from src/monte_carlo.py,
which runs a forward-looking, per-symbol lognormal simulation that feeds
strategy *scoring* decisions during a run — not a backtest report metric.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


class RiskAnalyzer:
    """
    Institutional-grade risk analytics for portfolio evaluation.
    
    Calculates advanced metrics including CAGR, Sharpe, Sortino, Calmar ratios,
    drawdown analysis, and Monte Carlo simulations for strategy assessment.
    
    Attributes:
        daily_equity_curve: pd.Series of daily portfolio values.
        trade_log: List of trade dictionaries with pnl information.
        risk_free_rate: Annual risk-free rate (default 6.5% for India).
    """
    
    def __init__(
        self,
        daily_equity_curve: pd.Series,
        trade_log: List[Dict[str, Any]],
        risk_free_rate: float = 0.065
    ):
        """
        Initialize the RiskAnalyzer.
        
        Args:
            daily_equity_curve: Time series of daily portfolio values.
            trade_log: List of trade records with 'pnl' or 'return' fields.
            risk_free_rate: Annual risk-free rate (default 6.5% for India).
        """
        self.daily_equity_curve = daily_equity_curve.copy()
        self.trade_log = trade_log
        self.risk_free_rate = risk_free_rate
        
        # Ensure index is datetime for proper resampling
        if not isinstance(self.daily_equity_curve.index, pd.DatetimeIndex):
            self.daily_equity_curve.index = pd.to_datetime(self.daily_equity_curve.index)
        
        # Calculate daily returns once
        self.daily_returns = self.daily_equity_curve.pct_change().dropna()
        
        # Cache for computed metrics
        self._metrics_cache: Optional[Dict[str, Any]] = None
    
    def calculate_cagr(self) -> float:
        """
        Calculate Compound Annual Growth Rate.
        
        CAGR = (Ending Value / Beginning Value)^(1/Years) - 1
        
        Returns:
            CAGR as a decimal (e.g., 0.15 for 15%).
        """
        if len(self.daily_equity_curve) < 2:
            return 0.0
        
        start_value = self.daily_equity_curve.iloc[0]
        end_value = self.daily_equity_curve.iloc[-1]
        
        if start_value <= 0:
            return 0.0
        
        # Calculate number of years
        start_date = self.daily_equity_curve.index[0]
        end_date = self.daily_equity_curve.index[-1]
        years = (end_date - start_date).days / 365.25
        
        if years <= 0:
            return 0.0
        
        cagr = (end_value / start_value) ** (1 / years) - 1
        return cagr
    
    def calculate_annualized_volatility(self) -> float:
        """
        Calculate annualized volatility (standard deviation of returns).
        
        Volatility = Std(Daily Returns) * sqrt(252)
        
        Returns:
            Annualized volatility as a decimal.
        """
        if len(self.daily_returns) < 2:
            return 0.0
        
        daily_std = self.daily_returns.std()
        annualized_vol = daily_std * np.sqrt(252)
        return annualized_vol
    
    def calculate_sharpe_ratio(self) -> float:
        """
        Calculate Sharpe Ratio.
        
        Sharpe = (CAGR - Risk Free Rate) / Annualized Volatility
        
        Uses India risk-free rate of 6.5% by default.
        
        Returns:
            Sharpe ratio (dimensionless).
        """
        cagr = self.calculate_cagr()
        volatility = self.calculate_annualized_volatility()
        
        if volatility == 0:
            return 0.0
        
        sharpe = (cagr - self.risk_free_rate) / volatility
        return sharpe
    
    def calculate_sortino_ratio(self) -> float:
        """
        Calculate Sortino Ratio using Downside Deviation.
        
        Sortino = (CAGR - Risk Free Rate) / Downside Deviation
        
        Downside Deviation only considers negative returns (penalizes downside volatility).
        Handles zero downside deviation gracefully.
        
        Returns:
            Sortino ratio (dimensionless).
        """
        cagr = self.calculate_cagr()
        
        # Calculate downside returns (only negative returns)
        downside_returns = self.daily_returns[self.daily_returns < 0]
        
        if len(downside_returns) == 0:
            # No negative returns - infinite sortino, but return large finite value
            return float('inf') if cagr > self.risk_free_rate else 0.0
        
        # Calculate downside deviation (annualized)
        downside_deviation = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(252)
        
        if downside_deviation == 0:
            return float('inf') if cagr > self.risk_free_rate else 0.0
        
        sortino = (cagr - self.risk_free_rate) / downside_deviation
        return sortino
    
    def calculate_max_drawdown(self) -> float:
        """
        Calculate Maximum Drawdown (MDD).
        
        MDD = max((Peak - Trough) / Peak) over all time periods.
        
        Returns:
            Maximum drawdown as a decimal (e.g., 0.20 for 20%).
        """
        if len(self.daily_equity_curve) < 2:
            return 0.0
        
        # Calculate running maximum
        running_max = self.daily_equity_curve.expanding().max()
        
        # Calculate drawdown at each point
        drawdown = (running_max - self.daily_equity_curve) / running_max
        
        # Handle division by zero
        drawdown = drawdown.replace([np.inf, -np.inf], 0).fillna(0)
        
        max_drawdown = drawdown.max()
        return max_drawdown
    
    def calculate_calmar_ratio(self) -> float:
        """
        Calculate Calmar Ratio.
        
        Calmar = CAGR / Max Drawdown
        
        Returns:
            Calmar ratio (dimensionless).
        """
        cagr = self.calculate_cagr()
        mdd = self.calculate_max_drawdown()
        
        if mdd == 0:
            return float('inf') if cagr > 0 else 0.0
        
        calmar = cagr / mdd
        return calmar
    
    def calculate_profit_factor(self) -> float:
        """
        Calculate Profit Factor.
        
        Profit Factor = Gross Profits / Gross Losses
        
        Returns:
            Profit factor (dimensionless). Returns inf if no losses.
        """
        if not self.trade_log:
            return 1.0
        
        gross_profits = 0.0
        gross_losses = 0.0
        
        for trade in self.trade_log:
            pnl = trade.get('pnl', trade.get('return', 0))
            if pnl > 0:
                gross_profits += pnl
            elif pnl < 0:
                gross_losses += abs(pnl)
        
        if gross_losses == 0:
            return float('inf') if gross_profits > 0 else 1.0
        
        profit_factor = gross_profits / gross_losses
        return profit_factor
    
    def calculate_win_rate(self) -> float:
        """
        Calculate Win Rate.
        
        Win Rate = Number of Winning Trades / Total Trades
        
        Returns:
            Win rate as a decimal (e.g., 0.60 for 60%).
        """
        if not self.trade_log:
            return 0.0
        
        winning_trades = sum(1 for t in self.trade_log if t.get('pnl', t.get('return', 0)) > 0)
        total_trades = len(self.trade_log)
        
        if total_trades == 0:
            return 0.0
        
        win_rate = winning_trades / total_trades
        return win_rate
    
    def calculate_expectancy(self) -> float:
        """
        Calculate Expectancy (average profit per trade).
        
        Expectancy = Total PnL / Number of Trades
        
        Returns:
            Average profit per trade in currency units.
        """
        if not self.trade_log:
            return 0.0
        
        total_pnl = sum(t.get('pnl', t.get('return', 0)) for t in self.trade_log)
        expectancy = total_pnl / len(self.trade_log)
        return expectancy
    
    def calculate_drawdown_duration(self) -> int:
        """
        Calculate the duration of the maximum drawdown period.
        
        This is the number of days from the peak before MDD to the recovery point.
        
        Returns:
            Number of days in the maximum drawdown period.
        """
        if len(self.daily_equity_curve) < 2:
            return 0
        
        # Calculate running maximum
        running_max = self.daily_equity_curve.expanding().max()
        
        # Calculate drawdown at each point
        drawdown = (running_max - self.daily_equity_curve) / running_max
        drawdown = drawdown.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Find the index of maximum drawdown
        mdd_idx = drawdown.idxmax()
        mdd_value = drawdown.max()
        
        if mdd_value == 0:
            return 0
        
        # Find the peak date (last peak before the MDD)
        peak_idx = None
        for idx in reversed(drawdown[:mdd_idx].index):
            if self.daily_equity_curve.loc[idx] == running_max.loc[idx]:
                peak_idx = idx
                break
        
        if peak_idx is None:
            peak_idx = drawdown.index[0]
        
        # Find the recovery date (when equity exceeds the previous peak)
        peak_value = self.daily_equity_curve.loc[peak_idx]
        recovery_idx = None
        
        for idx in drawdown.index[drawdown.index >= mdd_idx]:
            if self.daily_equity_curve.loc[idx] >= peak_value:
                recovery_idx = idx
                break
        
        # If no recovery found, use the last date
        if recovery_idx is None:
            recovery_idx = drawdown.index[-1]
        
        # Calculate duration in days
        duration = (recovery_idx - peak_idx).days
        return max(0, duration)
    
    def get_underwater_equity_curve(self) -> pd.Series:
        """
        Calculate the underwater equity curve.
        
        The underwater curve shows the percentage drawdown from the running maximum
        at each point in time.
        
        Returns:
            pd.Series of drawdown percentages (negative values).
        """
        if len(self.daily_equity_curve) < 2:
            return pd.Series(dtype=float, index=self.daily_equity_curve.index)
        
        # Calculate running maximum
        running_max = self.daily_equity_curve.expanding().max()
        
        # Calculate underwater (drawdown as negative percentage)
        underwater = -((running_max - self.daily_equity_curve) / running_max * 100)
        underwater = underwater.replace([np.inf, -np.inf], 0).fillna(0)
        
        return underwater
    
    def run_monte_carlo_simulation(
        self,
        n_simulations: int = 10000,
        ruin_threshold: float = 0.50,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation using bootstrap resampling of trade returns.
        
        This simulates 10,000 possible outcomes by randomly shuffling the order
        of historical trades (Bootstrap Resampling).
        
        Args:
            n_simulations: Number of Monte Carlo simulations (default 10,000).
            ruin_threshold: Portfolio drop threshold for "ruin" (default 50%).
            seed: Random seed for reproducibility.
        
        Returns:
            Dictionary with:
                - probability_of_ruin: Chance of portfolio dropping below threshold.
                - percentile_5: 5th percentile terminal wealth.
                - percentile_95: 95th percentile terminal wealth.
                - median_terminal_wealth: Median terminal wealth.
                - mean_terminal_wealth: Mean terminal wealth.
        """
        if not self.trade_log or len(self.trade_log) < 2:
            return {
                'probability_of_ruin': 0.0,
                'percentile_5': self.daily_equity_curve.iloc[-1] if len(self.daily_equity_curve) > 0 else 0,
                'percentile_95': self.daily_equity_curve.iloc[-1] if len(self.daily_equity_curve) > 0 else 0,
                'median_terminal_wealth': self.daily_equity_curve.iloc[-1] if len(self.daily_equity_curve) > 0 else 0,
                'mean_terminal_wealth': self.daily_equity_curve.iloc[-1] if len(self.daily_equity_curve) > 0 else 0,
                'simulations_run': 0
            }
        
        # Extract trade returns (as decimals, e.g., 0.05 for +5%)
        trade_returns = []
        for trade in self.trade_log:
            pnl = trade.get('pnl', 0)
            entry_price = trade.get('entry_price', 1)
            quantity = trade.get('quantity', 1)
            
            # Calculate return as percentage
            if entry_price > 0 and quantity > 0:
                trade_return = pnl / (entry_price * quantity)
            else:
                # Fallback: try to get return directly
                trade_return = trade.get('return', 0)
                if isinstance(trade_return, (int, float)):
                    # If it looks like an absolute value, convert to decimal
                    if abs(trade_return) > 1:
                        trade_return = trade_return / 100.0
                else:
                    trade_return = 0
            
            trade_returns.append(trade_return)
        
        if not trade_returns:
            return {
                'probability_of_ruin': 0.0,
                'percentile_5': 0,
                'percentile_95': 0,
                'median_terminal_wealth': 0,
                'mean_terminal_wealth': 0,
                'simulations_run': 0
            }
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        initial_capital = self.daily_equity_curve.iloc[0] if len(self.daily_equity_curve) > 0 else 100000
        ruin_level = initial_capital * ruin_threshold
        
        terminal_wealths = []
        ruined_count = 0
        
        trade_array = np.array(trade_returns)
        n_trades = len(trade_array)
        
        for _ in range(n_simulations):
            # Bootstrap resampling: randomly shuffle trade order
            shuffled_returns = np.random.choice(trade_array, size=n_trades, replace=True)
            
            # Simulate cumulative returns
            cumulative_returns = np.cumprod(1 + shuffled_returns)
            
            # Track minimum portfolio value during simulation (for ruin check)
            portfolio_values = initial_capital * cumulative_returns
            min_portfolio = min(portfolio_values.min(), initial_capital)
            
            # Check for ruin
            if min_portfolio < ruin_level:
                ruined_count += 1
            
            # Record terminal wealth
            terminal_wealth = initial_capital * cumulative_returns[-1]
            terminal_wealths.append(terminal_wealth)
        
        terminal_wealths = np.array(terminal_wealths)
        
        # Calculate statistics
        probability_of_ruin = ruined_count / n_simulations
        percentile_5 = np.percentile(terminal_wealths, 5)
        percentile_95 = np.percentile(terminal_wealths, 95)
        median_wealth = np.median(terminal_wealths)
        mean_wealth = np.mean(terminal_wealths)
        
        return {
            'probability_of_ruin': probability_of_ruin,
            'percentile_5': percentile_5,
            'percentile_95': percentile_95,
            'median_terminal_wealth': median_wealth,
            'mean_terminal_wealth': mean_wealth,
            'simulations_run': n_simulations
        }
    
    def generate_analytics_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive risk analytics report.
        
        Returns:
            Structured dictionary containing all metrics for Excel reporting.
        """
        # Run Monte Carlo simulation
        mc_results = self.run_monte_carlo_simulation()
        
        # Compile all metrics
        report = {
            # Core Metrics
            'cagr': self.calculate_cagr(),
            'cagr_pct': self.calculate_cagr() * 100,
            'annualized_volatility': self.calculate_annualized_volatility(),
            'annualized_volatility_pct': self.calculate_annualized_volatility() * 100,
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'sortino_ratio': self.calculate_sortino_ratio(),
            'calmar_ratio': self.calculate_calmar_ratio(),
            'profit_factor': self.calculate_profit_factor(),
            'win_rate': self.calculate_win_rate(),
            'win_rate_pct': self.calculate_win_rate() * 100,
            'expectancy': self.calculate_expectancy(),
            
            # Drawdown Analysis
            'max_drawdown': self.calculate_max_drawdown(),
            'max_drawdown_pct': self.calculate_max_drawdown() * 100,
            'drawdown_duration_days': self.calculate_drawdown_duration(),
            'underwater_equity_curve': self.get_underwater_equity_curve(),
            
            # Monte Carlo Simulation
            'mc_probability_of_ruin': mc_results['probability_of_ruin'],
            'mc_probability_of_ruin_pct': mc_results['probability_of_ruin'] * 100,
            'mc_percentile_5': mc_results['percentile_5'],
            'mc_percentile_95': mc_results['percentile_95'],
            'mc_median_terminal_wealth': mc_results['median_terminal_wealth'],
            'mc_mean_terminal_wealth': mc_results['mean_terminal_wealth'],
            'mc_simulations_run': mc_results['simulations_run'],
            
            # Summary Statistics
            'total_trades': len(self.trade_log),
            'initial_capital': self.daily_equity_curve.iloc[0] if len(self.daily_equity_curve) > 0 else 0,
            'final_capital': self.daily_equity_curve.iloc[-1] if len(self.daily_equity_curve) > 0 else 0,
            'total_return': (self.daily_equity_curve.iloc[-1] / self.daily_equity_curve.iloc[0] - 1) if len(self.daily_equity_curve) > 1 else 0,
            'total_return_pct': (self.daily_equity_curve.iloc[-1] / self.daily_equity_curve.iloc[0] - 1) * 100 if len(self.daily_equity_curve) > 1 else 0,
            'risk_free_rate': self.risk_free_rate,
            'analysis_period_days': len(self.daily_equity_curve),
        }
        
        # Handle infinite values for JSON/Excel compatibility
        for key, value in report.items():
            if isinstance(value, float):
                if np.isinf(value):
                    report[key] = 999999.99  # Large finite value for display
                elif np.isnan(value):
                    report[key] = 0.0
        
        return report
