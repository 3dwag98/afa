"""
Risk Analytics Module for Portfolio Agent.

Provides institutional-grade risk metrics for evaluating a completed backtest's
PORTFOLIO-LEVEL performance: CAGR/Sharpe/Sortino/Calmar, drawdown analysis, and
bootstrap-resampling Monte Carlo simulation of the realized trade distribution
(probability of ruin). This is a distinct concern from src/monte_carlo.py,
which runs a forward-looking, per-symbol lognormal simulation that feeds
strategy *scoring* decisions during a run — not a backtest report metric.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# A published yield above this is being quoted in percent (6.8) rather than as
# a decimal (0.068). No plausible 91-day T-bill yield reaches 100% a year, and
# reading 6.8 raw would report a 680% risk-free rate — which silently drives
# every Sharpe deeply negative rather than failing.
_PERCENT_UNIT_THRESHOLD = 1.0

from .performance_stats import (
    TRADING_DAYS_PER_YEAR,
    evaluate_sharpe,
    excess_returns,
    sharpe_ratio as arithmetic_sharpe_ratio,
)

# Keys that may carry a trade's realized profit, most specific first.
# BacktestEngine writes 'net_pnl' (P&L after costs and taxes); older/simpler
# trade logs — and the test fixtures — use a plain 'pnl'. Reading only 'pnl',
# as this module used to, silently scored every real backtest as zero profit
# and zero loss, which made win rate, profit factor, expectancy and the whole
# Monte Carlo risk-of-ruin block meaningless in the exported report.
_PNL_KEYS = ('net_pnl', 'pnl', 'gross_pnl')


def load_risk_free_series(path: str | Path) -> Optional[pd.Series]:
    """Read a dated risk-free rate series, e.g. the 91-day T-bill.

    Expects two columns, `date` and `annualized_yield`. The Sharpe wants the
    excess return computed day by day against the rate that actually prevailed,
    not against a single number chosen for the whole window — over a five-year
    Indian backtest a constant is off by hundreds of basis points at both ends.

    Units are normalized rather than assumed: rate series are published both as
    decimals (0.068) and as percent (6.8), and reading the latter raw would
    subtract a 680% annual hurdle from every daily return.

    Returns:
        A date-indexed Series of annualized decimal yields, sorted, or None
        when the file does not exist — the caller then falls back to the
        configured constant and says so.
    """
    source = Path(path)
    if not source.exists():
        return None

    frame = pd.read_csv(source)
    missing = {"date", "annualized_yield"} - set(frame.columns)
    if missing:
        raise ValueError(
            f"{source}: risk-free rate CSV needs columns date,annualized_yield "
            f"(missing {sorted(missing)})"
        )

    index = pd.to_datetime(frame["date"], errors="coerce")
    values = pd.to_numeric(frame["annualized_yield"], errors="coerce")
    series = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(index)).dropna()
    series = series[~series.index.isna()].sort_index()
    if series.empty:
        return None

    if float(series.abs().max()) > _PERCENT_UNIT_THRESHOLD:
        series = series / 100.0

    return series


def resolve_risk_free_rate(
    csv_path: str | Path, fallback_annual_rate: float
) -> float | pd.Series:
    """The dated series when it exists, otherwise the configured constant.

    Logs which one it used. A Sharpe measured against a 6.5% hurdle and one
    measured against a moving 4-7% hurdle are different statistics, and which
    was computed should never have to be inferred from the absence of a file.
    """
    series = load_risk_free_series(csv_path)
    if series is not None:
        logger.info(
            "Risk-free rate: %d dated observations from %s (%s to %s)",
            series.size, csv_path, series.index.min().date(), series.index.max().date(),
        )
        return series

    logger.info(
        "Risk-free rate: no series at %s; using the constant %.4f from "
        "risk.risk_free_rate across the whole window",
        csv_path, fallback_annual_rate,
    )
    return fallback_annual_rate


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
        risk_free_rate: Annual risk-free rate, or a dated Series of them. No
            default — see __init__.
    """
    
    def __init__(
        self,
        daily_equity_curve: pd.Series,
        trade_log: List[Dict[str, Any]],
        risk_free_rate: float | pd.Series,
        random_seed: Optional[int] = 42,
    ):
        """
        Initialize the RiskAnalyzer.

        Args:
            daily_equity_curve: Time series of daily portfolio values.
            trade_log: List of trade records with 'net_pnl' (or 'pnl') fields.
            risk_free_rate: Annual risk-free rate as a decimal, or a
                date-indexed Series of annualized rates — e.g. the 91-day
                T-bill yield (see load_risk_free_series).

                Deliberately has no default. It used to default to 0.065, which
                meant every Sharpe this class reported was quietly measured
                against a 6.5% hurdle that no call site had to acknowledge —
                and the backtester was in fact constructing it without the
                argument at all. A constant is also a guess about the entire
                window: India's policy rate moved materially over 2021-2025, so
                the Series form is the correct input wherever that data exists.
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
        self.closed_trades = _closed_trades(trade_log)
        
        # Ensure index is datetime for proper resampling
        if not isinstance(self.daily_equity_curve.index, pd.DatetimeIndex):
            self.daily_equity_curve.index = pd.to_datetime(self.daily_equity_curve.index)
        
        # Calculate daily returns once
        self.daily_returns = self.daily_equity_curve.pct_change().dropna()

        # Cache for computed metrics
        self._metrics_cache: Optional[Dict[str, Any]] = None

    @property
    def risk_free_rate(self) -> float | np.ndarray:
        """The risk-free input, as the metric functions want to consume it.

        A scalar passes through as an annual rate. A Series is aligned to the
        return index, forward-filled across non-trading days and converted from
        annualized to per-period, so a rate series sampled weekly or monthly
        can be handed in directly.
        """
        if not isinstance(self._risk_free_rate, pd.Series):
            return self._risk_free_rate

        annualized = self._risk_free_rate.reindex(
            self.daily_returns.index, method="ffill"
        ).bfill()
        return ((1.0 + annualized.to_numpy()) ** (1.0 / TRADING_DAYS_PER_YEAR)) - 1.0

    @risk_free_rate.setter
    def risk_free_rate(self, value: float | pd.Series) -> None:
        self._risk_free_rate = value

    @property
    def mean_annual_risk_free_rate(self) -> float:
        """A single annualized rate, for the report cell and the ratios that
        genuinely need a scalar. Averaged over the window when a series was
        supplied, so the reported figure describes the period it covers."""
        if not isinstance(self._risk_free_rate, pd.Series):
            return float(self._risk_free_rate)
        if self._risk_free_rate.empty:
            return 0.0
        return float(self._risk_free_rate.mean())
    
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
        Calculate Sharpe Ratio: mean daily excess return / its standard
        deviation, annualized by sqrt(252).

        **Both terms are arithmetic.** This used to divide CAGR — a geometric
        mean — by an arithmetic annualized standard deviation. Because
        CAGR ~= mu - sigma^2/2, that hybrid returns approximately

            (mu - rf)/sigma - sigma/2

        so it was biased low by sigma/2: a flat -0.10 of Sharpe at 20%
        annualized volatility, and worse for more volatile strategies, which
        means it charged volatility twice and was not comparable to the
        conventional 1.2 target the platform is aiming at.

        The risk-free rate is subtracted per period rather than once from the
        annualized return, and compounded rather than divided by 252 (see
        performance_stats.to_daily_risk_free).

        Returns:
            Annualized Sharpe ratio (dimensionless).
        """
        return arithmetic_sharpe_ratio(
            self.daily_returns.to_numpy(),
            risk_free_rate=self.risk_free_rate,
            periods_per_year=TRADING_DAYS_PER_YEAR,
        )

    def calculate_geometric_sharpe_ratio(self) -> float:
        """The old (CAGR - rf) / sigma figure, kept for continuity.

        Reported alongside the Sharpe rather than as it, so the gap between the
        two is visible in any report that carries both. It is a legitimate
        quantity — growth rate per unit of volatility — but it is not the
        Sharpe ratio and should not be compared to a Sharpe threshold.
        """
        volatility = self.calculate_annualized_volatility()
        if volatility == 0:
            return 0.0
        return (self.calculate_cagr() - self.mean_annual_risk_free_rate) / volatility

    def calculate_sharpe_statistics(
        self,
        n_trials: int = 1,
        sharpe_variance: Optional[float] = None,
        benchmark_sharpe: float = 0.0,
    ) -> Dict[str, float]:
        """Sharpe with its selection-bias adjustments (PSR and DSR).

        A Sharpe ratio reported without the number of configurations tried is
        the maximum of an unrecorded number of draws. See
        src/performance_stats.py for the statistics; `n_trials` should come
        from the trial log, not from memory.

        Args:
            n_trials: Configurations tried to arrive at this result.
            sharpe_variance: Variance of Sharpe across those trials; estimated
                from this sample when omitted.
            benchmark_sharpe: SR* for the undeflated PSR.

        Returns:
            Dict of sharpe_ratio, psr, dsr, deflation_threshold_sharpe and the
            return moments behind them.
        """
        return evaluate_sharpe(
            self.daily_returns.to_numpy(),
            risk_free_rate=self.risk_free_rate,
            n_trials=n_trials,
            sharpe_variance=sharpe_variance,
            periods_per_year=TRADING_DAYS_PER_YEAR,
            benchmark_sharpe=benchmark_sharpe,
        )


    def calculate_sortino_ratio(self) -> float:
        """
        Calculate Sortino Ratio using Downside Deviation.

        Sortino = annualized mean excess return / annualized downside deviation

        Downside deviation only considers returns below the target (here, the
        risk-free rate), penalizing downside volatility while leaving upside
        volatility uncharged.

        Both terms are arithmetic and measured on the same excess-return
        series, for the same reason as calculate_sharpe_ratio: dividing a
        geometric numerator by an arithmetic denominator biases the ratio low
        by roughly sigma/2 and does so more for volatile strategies.

        Returns:
            Sortino ratio (dimensionless).
        """
        excess = excess_returns(
            self.daily_returns.to_numpy(),
            risk_free_rate=self.risk_free_rate,
            periods_per_year=TRADING_DAYS_PER_YEAR,
        )
        if excess.size < 2:
            return 0.0

        annualized_excess = float(np.mean(excess)) * TRADING_DAYS_PER_YEAR
        downside = excess[excess < 0]

        if downside.size == 0:
            # No period underperformed the risk-free rate.
            return float('inf') if annualized_excess > 0 else 0.0

        downside_deviation = float(
            np.sqrt(np.mean(downside**2)) * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
        if downside_deviation == 0:
            return float('inf') if annualized_excess > 0 else 0.0

        return annualized_excess / downside_deviation
    
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
    
    def generate_analytics_report(
        self,
        n_trials: int = 1,
        sharpe_variance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive risk analytics report.

        Args:
            n_trials: Number of configurations tried to arrive at this result,
                used to deflate the Sharpe ratio. Defaults to 1, which applies
                no deflation — honest only for a single pre-registered run.
                Pass the count from the trial log for anything exploratory.
            sharpe_variance: Variance of Sharpe across those trials; estimated
                from this sample when omitted.

        Returns:
            Structured dictionary containing all metrics for Excel reporting.
        """
        # Run Monte Carlo simulation
        mc_results = self.run_monte_carlo_simulation()
        sharpe_stats = self.calculate_sharpe_statistics(
            n_trials=n_trials, sharpe_variance=sharpe_variance
        )

        # Compile all metrics
        report = {
            # Core Metrics
            'cagr': self.calculate_cagr(),
            'cagr_pct': self.calculate_cagr() * 100,
            'annualized_volatility': self.calculate_annualized_volatility(),
            'annualized_volatility_pct': self.calculate_annualized_volatility() * 100,
            'sharpe_ratio': self.calculate_sharpe_ratio(),

            # Selection-bias-aware statistics. A Sharpe reported without the
            # number of trials behind it is the maximum of an unrecorded number
            # of draws; DSR below 0.95 means it is not distinguishable from
            # what the search alone would have produced.
            'geometric_sharpe_ratio': self.calculate_geometric_sharpe_ratio(),
            'probabilistic_sharpe_ratio': sharpe_stats['psr'],
            'deflated_sharpe_ratio': sharpe_stats['dsr'],
            'deflation_threshold_sharpe': sharpe_stats['deflation_threshold_sharpe'],
            'return_skewness': sharpe_stats['skewness'],
            'return_kurtosis': sharpe_stats['kurtosis'],
            'n_trials': sharpe_stats['n_trials'],
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
            'risk_free_rate': self.mean_annual_risk_free_rate,
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
