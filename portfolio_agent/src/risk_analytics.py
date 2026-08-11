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
from typing import Dict, Any, List, Optional, Tuple, Union

try:
    from .performance_stats import (
        TRADING_DAYS_PER_YEAR,
        annual_rate_to_period,
        deflated_sharpe_ratio,
        expected_maximum_sharpe,
        probabilistic_sharpe_ratio,
    )
    from .trial_log import DEFAULT_TRIAL_LOG, trial_statistics
except ImportError:  # pragma: no cover - flat-path import used by the tests
    from performance_stats import (
        TRADING_DAYS_PER_YEAR,
        annual_rate_to_period,
        deflated_sharpe_ratio,
        expected_maximum_sharpe,
        probabilistic_sharpe_ratio,
    )
    from trial_log import DEFAULT_TRIAL_LOG, trial_statistics

# Keys that may carry a trade's realized profit, most specific first.
# BacktestEngine writes 'net_pnl' (P&L after costs and taxes); older/simpler
# trade logs — and the test fixtures — use a plain 'pnl'. Reading only 'pnl',
# as this module used to, silently scored every real backtest as zero profit
# and zero loss, which made win rate, profit factor, expectancy and the whole
# Monte Carlo risk-of-ruin block meaningless in the exported report.
_PNL_KEYS = ('net_pnl', 'pnl', 'gross_pnl')


def _trade_pnl(trade: Dict[str, Any]) -> float:
    """Realized profit for one trade, in currency units."""
    for key in _PNL_KEYS:
        value = trade.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

    value = trade.get('return', 0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _closed_trades(trade_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only round trips that actually closed.

    A BUY leg is recorded with `exit_date: None` and a negative net P&L (its
    transaction cost); counting those as trades inflates the trade count and
    reports every open position as a loss. Records with no `exit_date` key at
    all are kept — those come from simpler trade logs where every entry is
    already a completed trade.
    """
    if not trade_log:
        return []
    return [t for t in trade_log if 'exit_date' not in t or t.get('exit_date') is not None]


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
        risk_free_rate: Union[float, pd.Series] = 0.065,
        random_seed: Optional[int] = 42,
        trial_log_path: Optional[str] = DEFAULT_TRIAL_LOG,
    ):
        """
        Initialize the RiskAnalyzer.

        Args:
            daily_equity_curve: Time series of daily portfolio values.
            trade_log: List of trade records with 'net_pnl' (or 'pnl') fields.
            risk_free_rate: Annual risk-free rate. A float is a constant rate
                (default 6.5%); a pd.Series indexed by date is the
                period-by-period rate — the 91-day T-bill series, say — which
                is what the Sharpe denominator's benchmark should be over a
                window in which the policy rate moved as much as India's has.
                A Series is reindexed onto the equity curve and forward-filled.
            random_seed: Seed for the bootstrap Monte Carlo. Defaults to a
                fixed seed so re-running the same backtest reproduces the same
                risk-of-ruin and terminal-wealth percentiles — unseeded, those
                report cells changed on every single run. Pass None for a
                genuinely random draw.
        """
        self.daily_equity_curve = daily_equity_curve.copy()
        self.trade_log = trade_log
        self.risk_free_rate = risk_free_rate
        self.random_seed = random_seed
        self.trial_log_path = trial_log_path
        self.closed_trades = _closed_trades(trade_log)
        
        # Ensure index is datetime for proper resampling
        if not isinstance(self.daily_equity_curve.index, pd.DatetimeIndex):
            self.daily_equity_curve.index = pd.to_datetime(self.daily_equity_curve.index)
        
        # Calculate daily returns once
        self.daily_returns = self.daily_equity_curve.pct_change().dropna()

        # Daily excess returns over the risk-free rate, computed once. Every
        # risk-adjusted ratio below is built from this rather than from CAGR:
        # a geometric numerator over an arithmetic denominator is biased low
        # by roughly sigma/2 (CAGR ~= mu - sigma^2/2), a -0.10 constant at 20%
        # volatility that grows with volatility and so penalizes volatile
        # strategies twice.
        self.daily_excess_returns = self.daily_returns - self._daily_risk_free()
        
        # Cache for computed metrics
        self._metrics_cache: Optional[Dict[str, Any]] = None
    
    def _daily_risk_free(self) -> "Union[float, pd.Series]":
        """Per-day risk-free rate aligned to the equity curve.

        A constant 6.5% was hardcoded here, which makes the Sharpe
        denominator's benchmark a guess about a multi-year window over which
        India's policy rate moved materially. A Series (the 91-day T-bill, for
        instance) is reindexed onto the return dates and forward-filled, so
        each day is compared against the rate that actually prevailed.
        """
        if isinstance(self.risk_free_rate, pd.Series):
            annual = self.risk_free_rate.copy()
            if not isinstance(annual.index, pd.DatetimeIndex):
                annual.index = pd.to_datetime(annual.index)
            annual = annual.sort_index().reindex(
                self.daily_returns.index, method="ffill"
            )
            # Days before the first quoted rate have no benchmark; back-filling
            # is the least-wrong choice and beats silently scoring them at zero.
            annual = annual.bfill().fillna(0.0)
            return (1.0 + annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
        return annual_rate_to_period(float(self.risk_free_rate), TRADING_DAYS_PER_YEAR)

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
        Annualized Sharpe ratio, both moments measured arithmetically.

            SR = mean(r_t - rf_t) / sd(r_t - rf_t) * sqrt(252)

        This used to be (CAGR - rf) / annualized_sigma, which divides a
        geometric mean by an arithmetic standard deviation. Since
        CAGR ~= mu - sigma^2/2, that expression returns approximately
        (mu - rf)/sigma - sigma/2: biased low by sigma/2, which is a fixed
        -0.10 of Sharpe at 20% volatility and worse as volatility rises. It
        also made the reported figure incomparable with the conventional 1.2
        target it was being judged against.

        Returns:
            Sharpe ratio (dimensionless).
        """
        excess = self.daily_excess_returns
        if len(excess) < 2:
            return 0.0

        sigma = float(excess.std(ddof=1))
        if sigma <= 0 or not np.isfinite(sigma):
            return 0.0

        return float(excess.mean()) / sigma * np.sqrt(TRADING_DAYS_PER_YEAR)

    def calculate_sharpe_ratio_per_period(self) -> float:
        """Daily (non-annualized) Sharpe — the frequency PSR and DSR expect.

        Feeding an annualized Sharpe to probabilistic_sharpe_ratio alongside a
        daily observation count overstates significance by sqrt(252), so the
        two frequencies are kept as separate, separately-named methods rather
        than as one number the caller has to remember to rescale.
        """
        annualized = self.calculate_sharpe_ratio()
        return annualized / np.sqrt(TRADING_DAYS_PER_YEAR)

    def calculate_probabilistic_sharpe_ratio(self, benchmark_sharpe: float = 0.0) -> float:
        """P(true Sharpe > benchmark), adjusted for skewness and kurtosis.

        Args:
            benchmark_sharpe: Annualized Sharpe to beat; 0.0 asks only whether
                the strategy has any edge at all.
        """
        excess = self.daily_excess_returns
        if len(excess) < 3:
            return 0.0

        return probabilistic_sharpe_ratio(
            observed_sharpe=self.calculate_sharpe_ratio_per_period(),
            n_observations=len(excess),
            skewness=float(excess.skew()),
            # pandas reports EXCESS kurtosis; PSR wants the raw fourth moment,
            # which is 3.0 for a normal.
            kurtosis=float(excess.kurtosis()) + 3.0,
            benchmark_sharpe=benchmark_sharpe / np.sqrt(TRADING_DAYS_PER_YEAR),
        )

    def calculate_deflated_sharpe_ratio(
        self,
        n_trials: Optional[int] = None,
        sharpe_variance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """PSR against the expected maximum Sharpe of N trials.

        N and V[SR] come from the trial log (src/trial_log.py) unless passed
        explicitly. With no log there is no N, and the honest answer is that
        the DSR is not computable — not that there was one trial. A reported
        Sharpe with an unknown trial count cannot be deflated, and saying so
        is the point of the statistic.

        Returns:
            Dict with dsr, n_trials, sharpe_variance and expected_max_sharpe;
            `computable` is False when the trial log cannot supply N.
        """
        if n_trials is None or sharpe_variance is None:
            stats_from_log = (
                trial_statistics(self.trial_log_path) if self.trial_log_path else {}
            )
            n_trials = n_trials if n_trials is not None else stats_from_log.get("n_trials", 0)
            if sharpe_variance is None:
                sharpe_variance = stats_from_log.get("sharpe_variance", 0.0)

        if not n_trials or n_trials < 2 or not sharpe_variance or sharpe_variance <= 0:
            return {
                "dsr": 0.0,
                "n_trials": int(n_trials or 0),
                "sharpe_variance": float(sharpe_variance or 0.0),
                "expected_max_sharpe": 0.0,
                "computable": False,
            }

        excess = self.daily_excess_returns
        # V[SR] in the log is over ANNUALIZED Sharpes; PSR works per-period.
        variance_per_period = float(sharpe_variance) / TRADING_DAYS_PER_YEAR
        dsr = deflated_sharpe_ratio(
            observed_sharpe=self.calculate_sharpe_ratio_per_period(),
            n_observations=len(excess),
            skewness=float(excess.skew()),
            kurtosis=float(excess.kurtosis()) + 3.0,
            n_trials=int(n_trials),
            sharpe_variance=variance_per_period,
        )
        from_stats = expected_maximum_sharpe(int(n_trials), variance_per_period)
        return {
            "dsr": dsr,
            "n_trials": int(n_trials),
            "sharpe_variance": float(sharpe_variance),
            # Reported annualized, to be comparable with the headline Sharpe.
            "expected_max_sharpe": from_stats * float(np.sqrt(TRADING_DAYS_PER_YEAR)),
            "computable": True,
        }

    
    def calculate_sortino_ratio(self) -> float:
        """
        Calculate Sortino Ratio using Downside Deviation.
        
        Sortino = mean(excess) / downside deviation * sqrt(252)

        Same measurement-space correction as calculate_sharpe_ratio: the
        numerator is the arithmetic mean daily excess return, not CAGR.
        Downside deviation only considers negative excess returns, so the
        threshold is the risk-free rate rather than zero.

        Returns:
            Sortino ratio (dimensionless).
        """
        excess = self.daily_excess_returns
        if len(excess) < 2:
            return 0.0

        mean_excess = float(excess.mean())
        downside_returns = excess[excess < 0]

        if len(downside_returns) == 0:
            # No down days at all - infinite Sortino, reported as such and
            # clamped for display by generate_analytics_report().
            return float('inf') if mean_excess > 0 else 0.0

        downside_deviation = float(np.sqrt((downside_returns ** 2).mean()))

        if downside_deviation <= 0:
            return float('inf') if mean_excess > 0 else 0.0

        return mean_excess / downside_deviation * float(np.sqrt(TRADING_DAYS_PER_YEAR))
    
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
        if not self.closed_trades:
            return 1.0

        gross_profits = 0.0
        gross_losses = 0.0

        for trade in self.closed_trades:
            pnl = _trade_pnl(trade)
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
        if not self.closed_trades:
            return 0.0

        winning_trades = sum(1 for t in self.closed_trades if _trade_pnl(t) > 0)
        total_trades = len(self.closed_trades)

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
        if not self.closed_trades:
            return 0.0

        total_pnl = sum(_trade_pnl(t) for t in self.closed_trades)
        expectancy = total_pnl / len(self.closed_trades)
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
            seed: Random seed for reproducibility. Defaults to the analyzer's
                random_seed.
        
        Returns:
            Dictionary with:
                - probability_of_ruin: Chance of portfolio dropping below threshold.
                - percentile_5: 5th percentile terminal wealth.
                - percentile_95: 95th percentile terminal wealth.
                - median_terminal_wealth: Median terminal wealth.
                - mean_terminal_wealth: Mean terminal wealth.
        """
        if len(self.closed_trades) < 2:
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
        for trade in self.closed_trades:
            pnl = _trade_pnl(trade)
            entry_price = trade.get('entry_price') or 0
            quantity = trade.get('quantity') or 0

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
        
        # Draw from a local generator rather than seeding NumPy's global RNG,
        # so reproducibility here does not perturb (or depend on) any other
        # random draw in the process.
        rng = np.random.default_rng(self.random_seed if seed is None else seed)

        initial_capital = self.daily_equity_curve.iloc[0] if len(self.daily_equity_curve) > 0 else 100000
        ruin_level = initial_capital * ruin_threshold
        
        terminal_wealths = []
        ruined_count = 0
        
        trade_array = np.array(trade_returns)
        n_trades = len(trade_array)
        
        for _ in range(n_simulations):
            # Bootstrap resampling: randomly shuffle trade order
            shuffled_returns = rng.choice(trade_array, size=n_trades, replace=True)
            
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

        psr = self.calculate_probabilistic_sharpe_ratio()
        dsr_block = self.calculate_deflated_sharpe_ratio()
        
        # Compile all metrics
        report = {
            # Core Metrics
            'cagr': self.calculate_cagr(),
            'cagr_pct': self.calculate_cagr() * 100,
            'annualized_volatility': self.calculate_annualized_volatility(),
            'annualized_volatility_pct': self.calculate_annualized_volatility() * 100,
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            # A raw Sharpe says nothing about how many configurations were
            # tried to find it. These two do (docs/QUANT_RESEARCH.md, and
            # Bailey & Lopez de Prado's Deflated Sharpe Ratio).
            'probabilistic_sharpe_ratio': psr,
            'deflated_sharpe_ratio': dsr_block['dsr'],
            'dsr_computable': dsr_block['computable'],
            'dsr_n_trials': dsr_block['n_trials'],
            'dsr_expected_max_sharpe': dsr_block['expected_max_sharpe'],
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
            'total_trades': len(self.closed_trades),
            'total_trade_records': len(self.trade_log),
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
