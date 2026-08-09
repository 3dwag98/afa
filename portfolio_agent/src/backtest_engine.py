"""
Event-Driven Backtesting Engine for Portfolio Agent.

This engine simulates market data over a 5-year period, strictly avoiding look-ahead bias,
and allowing the Agent's Brain to learn over time.
"""

import copy
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

from portfolio_agent.config.schema import StrategyConfig
from portfolio_agent.features.pipeline import build_features
from portfolio_agent.strategies.base import BaseStrategy
from portfolio_agent.strategies.registry import load_strategy
from portfolio_agent.strategies.types import RiskParams, StrategyContext, StrategySignal
from portfolio_agent.strategies.weighting import evaluate_and_learn

# Import from src module
try:
    from .data_store import load_ticker_data
    from .models import AgentBrain
    from .execution_sim import ExecutionSimulator
    from .monte_carlo import run_monte_carlo, run_monte_carlo_garch
    from .risk import calculate_kelly_quantity, estimate_kelly_inputs
except ImportError:
    from data_store import load_ticker_data
    from models import AgentBrain
    from execution_sim import ExecutionSimulator
    from monte_carlo import run_monte_carlo, run_monte_carlo_garch
    from risk import calculate_kelly_quantity, estimate_kelly_inputs


logger = logging.getLogger(__name__)


def _score_one_ticker(
    strategy: BaseStrategy,
    ticker: str,
    hist_data: pd.DataFrame,
    required_features: List[str],
    risk_params: RiskParams,
    weights: Dict[str, float],
    mc_params: Tuple[int, int, int, bool],
) -> Optional[StrategySignal]:
    """Score a single ticker: run Monte Carlo + build features + call strategy.score().

    Module-level (not a method) so it is picklable for ProcessPoolExecutor dispatch.
    """
    try:
        horizon_days, simulations, seed, use_garch = mc_params
        daily_returns = hist_data['close'].pct_change().dropna().tolist()
        mc_fn = run_monte_carlo_garch if use_garch else run_monte_carlo
        mc_result = mc_fn(
            symbol=ticker,
            daily_returns=daily_returns,
            horizon_days=horizon_days,
            simulations=simulations,
            seed=seed,
        )
        features = build_features(hist_data, required_features)
        context = StrategyContext(risk=risk_params, weights=weights, mc_result=mc_result)
        return strategy.score(ticker, features, context)
    except Exception:
        logger.debug(f"Signal generation failed for {ticker}", exc_info=True)
        return None


# Per-worker constants, installed once by _init_scoring_worker() instead of
# being re-pickled with every task. A 5-year backtest scores each ticker on
# ~1,250 days; shipping the strategy object, risk params and feature list with
# each of those tasks was the dominant cost of the parallel path.
_WORKER_STRATEGY: Optional[BaseStrategy] = None
_WORKER_FEATURES: List[str] = []
_WORKER_RISK: Optional[RiskParams] = None
_WORKER_MC_PARAMS: Tuple[int, int, int, bool] = (0, 0, 0, False)


def _init_scoring_worker(
    strategy: BaseStrategy,
    required_features: List[str],
    risk_params: RiskParams,
    mc_params: Tuple[int, int, int, bool],
) -> None:
    """ProcessPoolExecutor initializer: pin the run-constant scoring inputs."""
    global _WORKER_STRATEGY, _WORKER_FEATURES, _WORKER_RISK, _WORKER_MC_PARAMS
    _WORKER_STRATEGY = strategy
    _WORKER_FEATURES = required_features
    _WORKER_RISK = risk_params
    _WORKER_MC_PARAMS = mc_params


def _score_one_ticker_in_worker(
    ticker: str, hist_data: pd.DataFrame, weights: Dict[str, float]
) -> Optional[StrategySignal]:
    """Worker-side entry point: only the per-day varying arguments travel."""
    if _WORKER_STRATEGY is None or _WORKER_RISK is None:
        raise RuntimeError("Scoring worker was not initialized")
    return _score_one_ticker(
        _WORKER_STRATEGY, ticker, hist_data, _WORKER_FEATURES,
        _WORKER_RISK, weights, _WORKER_MC_PARAMS,
    )


class BacktestEngine:
    """
    Event-Driven Backtesting Engine.
    
    Simulates historical trading with strict look-ahead bias prevention.
    At date T, the agent can ONLY see data up to T-1.
    """
    
    def __init__(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float,
        universe_tickers: List[str],
        initial_brain: Optional[Dict[str, Any]] = None,
        strategy: Optional["BaseStrategy"] = None,
        risk_params: Optional["RiskParams"] = None,
        parallel: bool = False,
        max_workers: Optional[int] = None,
        mc_horizon_days: int = 20,
        mc_simulations: int = 1000,
        mc_seed: int = 42,
        use_garch_volatility: bool = False,
        use_kelly_sizing: bool = False,
        kelly_fraction: float = 0.5,
        kelly_min_trades: int = 20,
    ):
        """
        Initialize the BacktestEngine.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            initial_capital: Initial cash in INR.
            universe_tickers: List of ticker symbols to trade.
            initial_brain: Optional initial brain state. If None, uses default weights.
            strategy: Strategy used for signal generation. Defaults to the
                registered "rule_based" strategy if not provided.
            risk_params: Risk/eligibility parameters passed to the strategy.
                Defaults to a conservative built-in set if not provided.
            parallel: If True and the strategy does not support GPU batching,
                dispatch per-ticker signal generation across a CPU process pool.
                If the strategy does support GPU batching (e.g. an ML strategy),
                all eligible tickers are scored in a single batched call instead.
            max_workers: Max worker processes when parallel=True (default: CPU count).
            use_garch_volatility: If True, forecast each ticker's forward
                volatility with GJR-GARCH(1,1) instead of a flat historical
                standard deviation (see src/volatility_models.py).
            use_kelly_sizing: If True, size positions with fractional-Kelly
                once enough realized trades exist in this run's trade_log,
                falling back to fixed-fractional sizing otherwise.
            kelly_fraction: Fractional-Kelly multiplier kappa (default 0.5 = half-Kelly).
            kelly_min_trades: Minimum realized trades required before Kelly sizing is trusted.
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.universe_tickers = [t.upper() if not t.endswith('.NS') else t for t in universe_tickers]

        if strategy is None:
            strategy_config = StrategyConfig()
            strategy = load_strategy(strategy_config)
        self.strategy = strategy
        self.risk_params = risk_params or RiskParams(
            target_prob_profit=0.55,
            min_reward_risk=1.5,
            min_price_inr=20.0,
            portfolio_value_inr=initial_capital,
            risk_per_trade_pct=0.01,
            max_single_position_pct=0.03,
        )
        self.parallel = parallel
        self.max_workers = max_workers
        # Created lazily on the first parallel scoring round and reused for the
        # whole run (see _get_scoring_executor).
        self._scoring_executor: Optional[ProcessPoolExecutor] = None
        self.mc_horizon_days = mc_horizon_days
        self.mc_simulations = mc_simulations
        self.mc_seed = mc_seed
        self.use_garch_volatility = use_garch_volatility
        self.use_kelly_sizing = use_kelly_sizing
        self.kelly_fraction = kelly_fraction
        self.kelly_min_trades = kelly_min_trades


        # Track untradeable tickers (delisted or NaN issues) - initialize BEFORE loading data
        self.untradeable_tickers: set = set()
        
        # Load all ticker data into memory
        self.ticker_data: Dict[str, pd.DataFrame] = {}
        self._load_all_data()
        
        # Create unified master_date_index (all trading days in the 5-year period)
        self.master_date_index = self._build_master_date_index()
        
        # State Management
        self.cash = initial_capital
        self.holdings: Dict[str, int] = {}  # ticker -> quantity
        # ticker -> {'entry_price', 'entry_date', 'quantity'} for every open
        # position. The cost basis has to be tracked explicitly: reconstructing
        # it by scanning the trade log (as this engine used to) silently failed,
        # because trade records have no 'action' key to match on — which left
        # every realized P&L measured from the wrong price and every
        # holding_days reported as 0.
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self.portfolio_value = initial_capital
        
        # Initialize agent brain (isolated copy for backtesting)
        if initial_brain is None:
            self.agent_brain = AgentBrain()
        else:
            self.agent_brain = AgentBrain(
                weights=initial_brain.get('weights', {
                    "Trend": 25.0,
                    "Breakout": 25.0,
                    "Volume": 20.0,
                    "MC_Prob": 30.0
                }),
                trade_history=initial_brain.get('trade_history', []),
                learning_log=initial_brain.get('learning_log', []),
                updated_at=initial_brain.get('updated_at')
            )
        
        # Results storage
        self.daily_equity_curve: pd.Series = pd.Series(dtype=float)
        self.trade_log: List[Dict[str, Any]] = []
        self.brain_evolution: List[Dict[str, Any]] = []
        self.daily_activity_log: List[Dict[str, Any]] = []
        
        # Stop-loss and take-profit tracking (ticker -> {stop_price, target_price, entry_price})
        self.stop_loss_levels: Dict[str, float] = {}
        self.take_profit_levels: Dict[str, float] = {}
        
        # Pending orders for T+1 execution (list of order dicts)
        self.pending_orders: List[Dict[str, Any]] = []
        
        # Trading day counter for learning triggers
        self.trading_day_count = 0
        
        # Execution simulator for realistic friction modeling
        self.execution_sim = ExecutionSimulator()
    
    def _load_all_data(self) -> None:
        """Load all ticker data into memory using data_store.load_ticker_data."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Log progress for large universes
        total_tickers = len(self.universe_tickers)
        if total_tickers > 100:
            logger.info(f"Loading data for {total_tickers} tickers (this may take a while)...")
        
        loaded_count = 0
        for i, ticker in enumerate(self.universe_tickers):
            df = load_ticker_data(ticker, 
                                  start_date=self.start_date.strftime('%Y-%m-%d'),
                                  end_date=self.end_date.strftime('%Y-%m-%d'))
            if df is not None and len(df) > 0:
                # Ensure proper column names
                df = df.copy()
                df.columns = [c.lower() for c in df.columns]
                
                # Ensure index is datetime
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                
                df = df.sort_index()
                self.ticker_data[ticker] = df
                loaded_count += 1
            else:
                # Mark as untradeable if no data
                self.untradeable_tickers.add(ticker)
                logger.warning(f"No data available for {ticker}, marking as untradeable")
            
            # Log progress every 50 tickers for large universes
            if total_tickers > 100 and (i + 1) % 50 == 0:
                logger.info(f"Loaded {i + 1}/{total_tickers} tickers...")
        
        logger.info(f"Successfully loaded {loaded_count}/{total_tickers} tickers")
    
    def _build_master_date_index(self) -> pd.DatetimeIndex:
        """
        Build a unified master date index from all available ticker data.
        
        Returns:
            DatetimeIndex of all trading days in the period.
        """
        all_dates = set()
        for ticker, df in self.ticker_data.items():
            if ticker not in self.untradeable_tickers:
                all_dates.update(df.index.tolist())
        
        if not all_dates:
            # Fallback to business days if no data
            return pd.bdate_range(start=self.start_date, end=self.end_date)
        
        date_index = pd.DatetimeIndex(sorted(all_dates))
        # Filter to within our date range
        date_index = date_index[(date_index >= self.start_date) & (date_index <= self.end_date)]
        return date_index
    
    def _get_price_at_date(self, ticker: str, date: pd.Timestamp, price_type: str = 'close') -> Optional[float]:
        """
        Get price for a ticker at a specific date.
        
        Args:
            ticker: Ticker symbol.
            date: Date timestamp.
            price_type: 'open', 'high', 'low', 'close', 'adj_close'.
            
        Returns:
            Price value or None if not available.
        """
        if ticker in self.untradeable_tickers:
            return None
        
        if ticker not in self.ticker_data:
            return None
        
        df = self.ticker_data[ticker]
        
        # Handle different column name variations
        col_map = {
            'open': ['open', 'Open'],
            'high': ['high', 'High'],
            'low': ['low', 'Low'],
            'close': ['close', 'Close', 'adj_close', 'Adj Close'],
            'adj_close': ['adj_close', 'Adj Close', 'close', 'Close']
        }
        
        columns_to_try = col_map.get(price_type, [price_type])
        
        for col in columns_to_try:
            if col in df.columns:
                try:
                    # Try exact match first
                    return df.loc[date, col]
                except KeyError:
                    # Try nearest previous date (forward fill behavior)
                    available_dates = df[df.index <= date].index
                    if len(available_dates) > 0:
                        nearest_date = available_dates[-1]
                        return df.loc[nearest_date, col]
        
        return None
    
    def _get_historical_data_up_to(self, ticker: str, up_to_date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """
        Get historical data for a ticker up to (but not including) a specific date.
        
        This is CRITICAL for avoiding look-ahead bias.
        
        Args:
            ticker: Ticker symbol.
            up_to_date: The date up to which data should be returned (exclusive).
            
        Returns:
            DataFrame with data up to up_to_date - 1 day.
        """
        if ticker in self.untradeable_tickers:
            return None
        
        if ticker not in self.ticker_data:
            return None
        
        df = self.ticker_data[ticker].copy()
        
        # Filter to dates strictly before up_to_date
        mask = df.index < up_to_date
        return df[mask].copy()
    
    def _mark_to_market(self, current_date: pd.Timestamp) -> float:
        """
        Mark-to-Market portfolio using current date's closing prices.
        
        Args:
            current_date: Current date timestamp.
            
        Returns:
            Total portfolio value.
        """
        holdings_value = 0.0
        
        for ticker, quantity in self.holdings.items():
            if quantity > 0:
                close_price = self._get_price_at_date(ticker, current_date, 'close')
                if close_price is not None and not pd.isna(close_price):
                    holdings_value += quantity * close_price
                else:
                    # Force liquidation at last known price if NaN
                    last_known_price = self._get_last_valid_price(ticker, current_date)
                    if last_known_price is not None:
                        holdings_value += quantity * last_known_price
                        # Mark as untradeable
                        self.untradeable_tickers.add(ticker)
        
        self.portfolio_value = self.cash + holdings_value
        return self.portfolio_value
    
    def _get_last_valid_price(self, ticker: str, before_date: pd.Timestamp) -> Optional[float]:
        """Get the last valid price before a given date."""
        if ticker not in self.ticker_data:
            return None
        
        df = self.ticker_data[ticker]
        available = df[df.index < before_date]
        
        if len(available) == 0:
            return None
        
        last_row = available.iloc[-1]
        for col in ['close', 'Close', 'adj_close', 'Adj Close']:
            if col in last_row.index and not pd.isna(last_row[col]):
                return last_row[col]
        
        return None
    
    def _open_position(
        self, ticker: str, quantity: int, price: float, entry_date: pd.Timestamp
    ) -> None:
        """Record (or add to) an open position, keeping a weighted cost basis."""
        existing = self.open_positions.get(ticker)
        if existing and existing['quantity'] > 0:
            total_quantity = existing['quantity'] + quantity
            existing['entry_price'] = (
                existing['entry_price'] * existing['quantity'] + price * quantity
            ) / total_quantity
            existing['quantity'] = total_quantity
            # Entry date stays at the *first* entry, so holding period (and
            # therefore STCG vs LTCG) is measured from when risk was first taken.
        else:
            self.open_positions[ticker] = {
                'entry_price': price,
                'entry_date': entry_date,
                'quantity': quantity,
            }

    def _close_position(self, ticker: str, quantity: int) -> None:
        """Reduce (or clear) an open position after a sale."""
        position = self.open_positions.get(ticker)
        if position is None:
            return
        position['quantity'] -= quantity
        if position['quantity'] <= 0:
            del self.open_positions[ticker]

    def _get_entry_price_for_tax(self, ticker: str) -> float:
        """
        Get the cost basis for an open holding (for P&L and tax calculation).

        Args:
            ticker: Ticker symbol.

        Returns:
            Entry price per share, or 0.0 if there is no open position.
        """
        position = self.open_positions.get(ticker)
        return float(position['entry_price']) if position else 0.0

    def _get_entry_date_for_ticker(self, ticker: str) -> Optional[str]:
        """
        Get the entry date for an open holding (for the trade log).

        Args:
            ticker: Ticker symbol.

        Returns:
            Entry date as YYYY-MM-DD, or None if there is no open position.
        """
        position = self.open_positions.get(ticker)
        if not position:
            return None
        return pd.Timestamp(position['entry_date']).strftime('%Y-%m-%d')

    def _get_holding_days(self, ticker: str, current_date: pd.Timestamp) -> int:
        """
        Get the number of calendar days an open position has been held.

        Args:
            ticker: Ticker symbol.
            current_date: Current date.

        Returns:
            Number of days held (0 if there is no open position).
        """
        position = self.open_positions.get(ticker)
        if not position:
            return 0
        return max(0, (current_date - pd.Timestamp(position['entry_date'])).days)


    def _check_stop_loss_take_profit(self, current_date: pd.Timestamp) -> List[Dict[str, Any]]:
        """
        Check for stop-losses and take-profits based on current date's intraday High/Low.
        
        Args:
            current_date: Current date timestamp.
            
        Returns:
            List of executed stop-loss/take-profit trades.
        """
        executed_trades = []
        tickers_to_remove = []
        
        for ticker in list(self.holdings.keys()):
            quantity = self.holdings[ticker]
            if quantity <= 0:
                continue
            
            if ticker in self.untradeable_tickers:
                continue
            
            df = self.ticker_data.get(ticker)
            if df is None or current_date not in df.index:
                continue
            
            row = df.loc[current_date]
            
            # Get high and low for the day
            high = None
            low = None
            for h_col in ['high', 'High']:
                if h_col in row.index:
                    high = row[h_col]
                    break
            for l_col in ['low', 'Low']:
                if l_col in row.index:
                    low = row[l_col]
                    break
            
            if high is None or low is None:
                continue
            
            stop_price = self.stop_loss_levels.get(ticker)
            target_price = self.take_profit_levels.get(ticker)
            # The position's actual cost basis — NOT the previous close, which
            # is what this used to use and which made every stop/target exit
            # report a one-day P&L instead of the trade's real P&L.
            entry_price = self._get_entry_price_for_tax(ticker)
            holding_days = self._get_holding_days(ticker, current_date)

            triggered = False
            trigger_price = None
            trigger_type = None
            
            # Check stop-loss (price hit or went below stop)
            if stop_price is not None and low <= stop_price:
                triggered = True
                trigger_price = stop_price
                trigger_type = 'STOP_LOSS'
            
            # Check take-profit (price hit or went above target)
            elif target_price is not None and high >= target_price:
                triggered = True
                trigger_price = target_price
                trigger_type = 'TAKE_PROFIT'
            
            if triggered:
                # Exits pay the same friction as any other sale — brokerage,
                # STT, exchange charges, GST and capital gains tax. Booking
                # them at zero cost (as this path used to) inflated reported
                # net P&L on every stop and target hit.
                sale_value = quantity * trigger_price
                txn_cost = self.execution_sim.calculate_transaction_costs(
                    side='SELL', price=trigger_price, quantity=quantity
                )
                cap_gains_tax = self.execution_sim.calculate_capital_gains_tax(
                    entry_price=entry_price,
                    exit_price=trigger_price,
                    quantity=quantity,
                    holding_days=holding_days,
                )
                self.cash += sale_value - txn_cost - cap_gains_tax

                # Calculate return percentage
                entry_val = (entry_price or 0) * quantity
                gross_pnl = (trigger_price - (entry_price or 0)) * quantity
                net_pnl = gross_pnl - txn_cost - cap_gains_tax
                return_pct = (gross_pnl / entry_val * 100) if entry_val > 0 else 0.0

                # Determine exit reason
                exit_reason = 'stop_loss' if trigger_type == 'STOP_LOSS' else 'target'

                # Record trade with all 16 required fields
                trade_record = {
                    'trade_id': f"T{len(self.trade_log) + 1:06d}",
                    'ticker': ticker,
                    'entry_date': self._get_entry_date_for_ticker(ticker),
                    'entry_price': entry_price,
                    'exit_date': current_date.strftime('%Y-%m-%d'),
                    'exit_price': trigger_price,
                    'quantity': quantity,
                    'side': 'LONG',
                    'signal_trigger': 'STOP_LOSS' if trigger_type == 'STOP_LOSS' else 'TAKE_PROFIT',
                    'gross_pnl': gross_pnl,
                    'transaction_costs': txn_cost,
                    'taxes': cap_gains_tax,
                    'net_pnl': net_pnl,
                    'return_pct': return_pct,
                    'holding_days': holding_days,
                    'exit_reason': exit_reason
                }
                self.trade_log.append(trade_record)
                executed_trades.append(trade_record)

                # Remove holding
                del self.holdings[ticker]
                self._close_position(ticker, quantity)
                tickers_to_remove.append(ticker)

                # Any still-pending order for this ticker is stale now that the
                # position is closed — drop it so a queued SELL cannot execute
                # against a position that no longer exists.
                self.pending_orders = [
                    o for o in self.pending_orders
                    if not (o['ticker'] == ticker and o['action'] == 'SELL')
                ]

                # Clear stop/target levels
                if ticker in self.stop_loss_levels:
                    del self.stop_loss_levels[ticker]
                if ticker in self.take_profit_levels:
                    del self.take_profit_levels[ticker]

        return executed_trades
    
    def _generate_signals(self, current_date: pd.Timestamp) -> Dict[str, StrategySignal]:
        """
        Run the configured strategy's signal generation using data up to T-1.

        CRITICAL: Only uses data up to current_date - 1 day to avoid look-ahead bias.

        Uses the SAME strategy code path as the live orchestrator (via
        strategies/registry.py), so backtest and live decisions can never
        drift apart. Rule-based strategies are scored per-ticker (optionally
        parallelized across CPU workers); strategies that support GPU batching
        (e.g. an ML strategy) are scored in a single stacked forward pass.

        Args:
            current_date: Current date timestamp.

        Returns:
            Dictionary of ticker -> StrategySignal.
        """
        eligible: Dict[str, pd.DataFrame] = {}
        for ticker in self.universe_tickers:
            if ticker in self.untradeable_tickers:
                continue
            hist_data = self._get_historical_data_up_to(ticker, current_date)
            if hist_data is None or len(hist_data) < 20:
                continue
            eligible[ticker] = hist_data

        if not eligible:
            return {}

        weights = dict(self.agent_brain.weights)

        if self.strategy.supports_gpu_batch or self.strategy.requires_full_batch:
            features_by_symbol = {
                ticker: build_features(hist_data, self.strategy.required_features())
                for ticker, hist_data in eligible.items()
            }
            context = StrategyContext(risk=self.risk_params, weights=weights)
            return self.strategy.score_batch(features_by_symbol, context)

        if self.parallel and len(eligible) > 1:
            parallel_signals = self._score_tickers_parallel(eligible, weights)
            if parallel_signals is not None:
                return parallel_signals

        required_features = self.strategy.required_features()
        mc_params = self._mc_params()

        signals: Dict[str, StrategySignal] = {}
        for ticker, hist_data in eligible.items():
            result = _score_one_ticker(
                self.strategy, ticker, hist_data, required_features, self.risk_params, weights, mc_params
            )
            if result is not None:
                signals[ticker] = result
        return signals

    def _mc_params(self) -> Tuple[int, int, int, bool]:
        """Monte Carlo settings bundled for dispatch to scoring workers."""
        return (self.mc_horizon_days, self.mc_simulations, self.mc_seed, self.use_garch_volatility)

    def _get_scoring_executor(self) -> ProcessPoolExecutor:
        """Return this run's scoring process pool, creating it on first use.

        The pool is created once per backtest, not once per trading day: a
        5-year run has ~1,250 scoring rounds, and spawning (and tearing down)
        a full set of worker processes for each of them cost far more than the
        scoring itself — especially on Windows, where every worker re-imports
        the package from scratch.
        """
        if self._scoring_executor is None:
            self._scoring_executor = ProcessPoolExecutor(
                max_workers=self.max_workers,
                initializer=_init_scoring_worker,
                initargs=(
                    self.strategy,
                    self.strategy.required_features(),
                    self.risk_params,
                    self._mc_params(),
                ),
            )
        return self._scoring_executor

    def shutdown_workers(self) -> None:
        """Tear down the scoring process pool, if one was started."""
        if self._scoring_executor is not None:
            self._scoring_executor.shutdown(wait=True)
            self._scoring_executor = None

    def _score_tickers_parallel(
        self, eligible: Dict[str, pd.DataFrame], weights: Dict[str, float]
    ) -> Optional[Dict[str, StrategySignal]]:
        """Score every eligible ticker across the run's CPU process pool.

        Results are reassembled in `eligible` order (i.e. universe order),
        never in worker-completion order. That matters beyond tidiness:
        downstream order creation walks this dict, and with finite cash the
        order in which BUY candidates are considered decides which ones
        actually get filled — so a completion-ordered dict would make the same
        backtest produce different trades, different equity curves and
        different Excel reports on every run.

        Returns None if the pool could not be used at all, so the caller can
        fall back to serial scoring rather than losing the run.
        """
        try:
            executor = self._get_scoring_executor()
            futures = {
                ticker: executor.submit(_score_one_ticker_in_worker, ticker, hist_data, weights)
                for ticker, hist_data in eligible.items()
            }
        except Exception:
            logger.warning(
                "Could not start parallel scoring workers; falling back to serial scoring",
                exc_info=True,
            )
            self.parallel = False
            self.shutdown_workers()
            return None

        signals: Dict[str, StrategySignal] = {}
        for ticker in eligible:  # deterministic: universe order, not completion order
            try:
                result = futures[ticker].result()
            except Exception:
                # A single failed ticker is skipped exactly as in the serial
                # path (where _score_one_ticker returns None), instead of
                # aborting the whole backtest.
                logger.warning(f"Parallel signal generation failed for {ticker}", exc_info=True)
                continue
            if result is not None:
                signals[ticker] = result
        return signals
    
    def _execute_pending_orders(self, execution_date: pd.Timestamp) -> List[Dict[str, Any]]:
        """
        Execute simulated trades for T+1 Open.
        
        This avoids look-ahead bias by executing at the next day's open price.
        Uses ExecutionSimulator to model realistic friction (costs, slippage, taxes).
        
        Args:
            execution_date: The date to execute trades (T+1).
            
        Returns:
            List of executed trades.
        """
        executed_trades = []
        orders_to_remove = []

        for i, order in enumerate(self.pending_orders):
            if order['execution_date'] > execution_date:
                continue

            # Anything due today or earlier is handled on this pass and leaves
            # the book, filled or not.
            #
            # "or earlier" matters twice. Orders are scheduled for the next
            # calendar weekday, which lands on a market holiday often enough to
            # matter; those now fill at the next session's open instead of
            # sitting in the book with a date that could never come round
            # again. And an order that could not be funded is dropped rather
            # than left queued forever — pending_orders used to grow without
            # bound for the whole run.
            orders_to_remove.append(i)

            ticker = order['ticker']
            action = order['action']
            quantity = order['quantity']

            if ticker in self.untradeable_tickers:
                continue

            # Get open price for execution
            open_price = self._get_price_at_date(ticker, execution_date, 'open')
            
            if open_price is None or pd.isna(open_price):
                # Try to use close price from previous day as fallback
                open_price = self._get_last_valid_price(ticker, execution_date)
            
            if open_price is None:
                continue

            # Get market data for slippage calculation
            df = self.ticker_data.get(ticker)
            avg_daily_volume = 0
            atr = open_price * 0.02  # Default ATR estimate (2% of price)
            
            if df is not None and execution_date in df.index:
                # Calculate average daily volume (last 20 days)
                prev_dates = df[df.index < execution_date].tail(20)
                if 'volume' in prev_dates.columns and len(prev_dates) > 0:
                    avg_daily_volume = int(prev_dates['volume'].mean())
                
                # Calculate ATR (14-day)
                if len(prev_dates) >= 14:
                    highs = prev_dates['high'] if 'high' in prev_dates.columns else prev_dates.iloc[:, 1]
                    lows = prev_dates['low'] if 'low' in prev_dates.columns else prev_dates.iloc[:, 2]
                    closes = prev_dates['close'] if 'close' in prev_dates.columns else prev_dates.iloc[:, 3]
                    
                    tr_list = []
                    for j in range(1, len(prev_dates)):
                        high_low = highs.iloc[j] - lows.iloc[j]
                        high_close_prev = abs(highs.iloc[j] - closes.iloc[j-1])
                        low_close_prev = abs(lows.iloc[j] - closes.iloc[j-1])
                        tr = max(high_low, high_close_prev, low_close_prev)
                        tr_list.append(tr)
                    
                    if tr_list:
                        atr = sum(tr_list[-14:]) / min(14, len(tr_list[-14:]))
            
            # Check if trade should be executed based on cost vs reward
            should_execute, exec_info = self.execution_sim.should_execute_trade(
                side=action,
                price=open_price,
                quantity=quantity,
                avg_daily_volume=max(avg_daily_volume, 1),  # Avoid division by zero
                atr=atr,
                expected_reward_atr=1.0  # Expecting 1 ATR reward
            )
            
            if not should_execute:
                logger.info(f"Skipping {action} order for {ticker}: friction exceeds expected reward")
                continue


            # Get adjusted execution price with slippage
            adjusted_price = exec_info['adjusted_price']
            trade_value = quantity * adjusted_price
            txn_cost = exec_info['transaction_cost']
            
            if action == 'BUY':
                total_cost = trade_value + txn_cost
                if self.cash >= total_cost:
                    self.cash -= total_cost
                    self.holdings[ticker] = self.holdings.get(ticker, 0) + quantity
                    self._open_position(ticker, quantity, adjusted_price, execution_date)

                    # Set stop-loss and take-profit levels
                    # Stop-loss at 5% below entry
                    self.stop_loss_levels[ticker] = adjusted_price * 0.95
                    # Take-profit at 10% above entry
                    self.take_profit_levels[ticker] = adjusted_price * 1.10
                    
                    # Record BUY trade with all 16 required fields
                    trade_record = {
                        'trade_id': f"T{len(self.trade_log) + 1:06d}",
                        'ticker': ticker,
                        'entry_date': execution_date.strftime('%Y-%m-%d'),
                        'entry_price': adjusted_price,
                        'exit_date': None,
                        'exit_price': None,
                        'quantity': quantity,
                        'side': 'LONG',
                        'signal_trigger': order.get('trigger', 'SIGNAL'),
                        'gross_pnl': 0.0,
                        'transaction_costs': txn_cost,
                        'taxes': 0.0,
                        'net_pnl': -txn_cost,
                        'return_pct': 0.0,
                        'holding_days': 0,
                        'exit_reason': None
                    }
                    self.trade_log.append(trade_record)
                    executed_trades.append(trade_record)
                else:
                    logger.debug(
                        f"Insufficient cash to fill BUY {quantity} {ticker} "
                        f"@ {adjusted_price:.2f} (need {total_cost:.2f}, have {self.cash:.2f})"
                    )

            elif action == 'SELL':
                if ticker in self.holdings and self.holdings[ticker] >= quantity:
                    # Get holding info for capital gains tax calculation
                    entry_price = self._get_entry_price_for_tax(ticker)
                    holding_days = self._get_holding_days(ticker, execution_date)
                    entry_date_str = self._get_entry_date_for_ticker(ticker)

                    # Calculate capital gains tax
                    cap_gains_tax = self.execution_sim.calculate_capital_gains_tax(
                        entry_price=entry_price,
                        exit_price=adjusted_price,
                        quantity=quantity,
                        holding_days=holding_days
                    )

                    self.holdings[ticker] -= quantity
                    self._close_position(ticker, quantity)
                    # Sale proceeds net of tax AND transaction costs; the
                    # brokerage/STT/GST on the sell leg was previously reported
                    # in the trade log but never deducted from cash, so the
                    # equity curve overstated every exit.
                    self.cash += trade_value - cap_gains_tax - txn_cost

                    # Clear stop/target levels
                    if ticker in self.stop_loss_levels:
                        del self.stop_loss_levels[ticker]
                    if ticker in self.take_profit_levels:
                        del self.take_profit_levels[ticker]
                    
                    # Calculate return percentage
                    entry_val = (entry_price or 0) * quantity
                    gross_pnl = (adjusted_price - (entry_price or 0)) * quantity
                    net_pnl = gross_pnl - cap_gains_tax - txn_cost
                    return_pct = (gross_pnl / entry_val * 100) if entry_val > 0 else 0.0
                    
                    # Determine exit reason
                    trigger_type = order.get('trigger', 'SIGNAL')
                    if trigger_type == 'STOP_LOSS':
                        exit_reason = 'stop_loss'
                    elif trigger_type == 'TAKE_PROFIT':
                        exit_reason = 'target'
                    else:
                        exit_reason = 'end_of_backtest'
                    
                    # Record SELL trade with all 16 required fields
                    trade_record = {
                        'trade_id': f"T{len(self.trade_log) + 1:06d}",
                        'ticker': ticker,
                        'entry_date': entry_date_str,
                        'entry_price': entry_price,
                        'exit_date': execution_date.strftime('%Y-%m-%d'),
                        'exit_price': adjusted_price,
                        'quantity': quantity,
                        'side': 'LONG',
                        'signal_trigger': trigger_type,
                        'gross_pnl': gross_pnl,
                        'transaction_costs': txn_cost,
                        'taxes': cap_gains_tax,
                        'net_pnl': net_pnl,
                        'return_pct': return_pct,
                        'holding_days': holding_days,
                        'exit_reason': exit_reason
                    }
                    self.trade_log.append(trade_record)
                    executed_trades.append(trade_record)

                    # Remove holding if zero
                    if self.holdings[ticker] == 0:
                        del self.holdings[ticker]

        # Remove processed orders (indices are unique and ascending, so popping
        # from the back keeps the remaining indices valid)
        for i in sorted(orders_to_remove, reverse=True):
            self.pending_orders.pop(i)

        return executed_trades
    
    def _kelly_quantity(self, entry_price: float) -> int:
        """Fractional-Kelly position size from this run's realized trade_log so far.

        Returns 0 (triggering the caller's fixed-fractional fallback) when
        there aren't yet enough realized trades to estimate Kelly inputs
        reliably (see risk.py::estimate_kelly_inputs).
        """
        # Only *closed* round trips are realized outcomes. Open BUY legs carry
        # net_pnl = -transaction_costs, so counting them here classified every
        # open position as a loss and dragged the Kelly win probability down.
        realized = [
            {"outcome": "WIN" if t.get("net_pnl", 0.0) > 0 else "LOSS", "return_pct": t.get("return_pct", 0.0)}
            for t in self.trade_log
            if t.get("exit_date") is not None
        ]
        kelly_inputs = estimate_kelly_inputs(realized, min_trades=self.kelly_min_trades)
        if kelly_inputs is None:
            return 0
        win_probability, reward_risk_ratio = kelly_inputs
        return calculate_kelly_quantity(
            entry_price=entry_price,
            portfolio_value_inr=self.portfolio_value,
            max_single_position_pct=self.risk_params.max_single_position_pct,
            win_probability=win_probability,
            reward_risk_ratio=reward_risk_ratio,
            kelly_fraction=self.kelly_fraction,
        )

    def _create_pending_orders(self, signals: Dict[str, StrategySignal], current_date: pd.Timestamp) -> None:
        """
        Create pending orders for T+1 execution based on signals.

        Orders are queued in a deterministic, economically sensible order:
        every SELL first (freeing capital for the same execution date), then
        BUYs by descending signal score with the ticker as tie-breaker. Cash
        is finite, so whichever BUY is queued first is the one that gets
        filled — leaving that to dict iteration order made results depend on
        how the strategy happened to be scored (serial vs. parallel, and which
        worker finished first). Ranking by conviction is both reproducible and
        the behaviour a capital-constrained portfolio actually wants.

        Args:
            signals: Dictionary of ticker -> StrategySignal.
            current_date: Current date (orders will execute at T+1 open).
        """
        execution_date = current_date + pd.Timedelta(days=1)

        # Skip weekends
        while execution_date.weekday() >= 5:
            execution_date += pd.Timedelta(days=1)

        sells = sorted(
            (t for t, s in signals.items() if s.signal == 'SELL' and t in self.holdings)
        )
        for ticker in sells:
            quantity = self.holdings[ticker]
            if quantity > 0:
                self.pending_orders.append({
                    'ticker': ticker,
                    'action': 'SELL',
                    'quantity': quantity,
                    'execution_date': execution_date,
                    'trigger': signals[ticker].trigger or 'SIGNAL'
                })

        buys = sorted(
            (t for t, s in signals.items() if s.signal == 'BUY' and t not in self.holdings),
            key=lambda t: (-signals[t].score, t),
        )
        for ticker in buys:
            sig = signals[ticker]
            price = sig.entry_price or 100
            quantity = self._kelly_quantity(price) if self.use_kelly_sizing else 0

            if quantity <= 0:
                # Default / Kelly-unavailable fallback: 10% of portfolio per position.
                position_value = self.portfolio_value * 0.10
                quantity = int(position_value / price)

            if quantity > 0:
                self.pending_orders.append({
                    'ticker': ticker,
                    'action': 'BUY',
                    'quantity': quantity,
                    'execution_date': execution_date,
                    'trigger': sig.trigger or 'SIGNAL'
                })


    def _evaluate_and_learn(self, learning_rate: float = 0.15, min_trades_for_learning: int = 3) -> None:
        """
        Trigger agent learning every N trading days.

        Uses the same pure weight-adaptation function (strategies/weighting.py)
        as the live orchestrator, so backtest and live learning stay identical.
        """
        # Snapshot brain state before learning
        brain_snapshot = {
            'trading_day': self.trading_day_count,
            'weights': dict(self.agent_brain.weights),
            'trade_count': len(self.agent_brain.trade_history)
        }
        self.brain_evolution.append(brain_snapshot)

        try:
            new_weights, message = evaluate_and_learn(
                weights=self.agent_brain.weights,
                trade_history=self.agent_brain.trade_history,
                learning_rate=learning_rate,
                min_trades_for_learning=min_trades_for_learning,
            )
            self.agent_brain.weights = new_weights
            if message is not None:
                self.agent_brain.learning_log.append({
                    'trading_day': self.trading_day_count,
                    'entry': message,
                })
        except Exception:
            # Log error but continue
            pass
    
    def _handle_delisted_tickers(self, current_date: pd.Timestamp) -> None:
        """
        Handle delisted tickers or those with NaN issues.

        Force liquidation at last known price. A holding already marked
        untradeable is liquidated too: skipping those (as this used to) left
        the position on the books forever, marked at a stale price that could
        never move again.
        """
        for ticker in list(self.holdings.keys()):
            df = self.ticker_data.get(ticker)
            if df is None:
                self.untradeable_tickers.add(ticker)
                continue

            # Liquidate when the ticker has stopped trading: either it is
            # already flagged untradeable, or today is past its last bar.
            last_date = df.index.max()
            delisted = ticker in self.untradeable_tickers or (
                current_date not in df.index and current_date > last_date
            )
            if delisted:
                # Ticker appears to be delisted
                self.untradeable_tickers.add(ticker)

                # Force liquidation
                quantity = self.holdings[ticker]
                last_price = self._get_last_valid_price(ticker, current_date)
                if last_price is None:
                    last_price = self._get_price_at_date(ticker, last_date, 'close')
                entry_price = self._get_entry_price_for_tax(ticker)
                holding_days = self._get_holding_days(ticker, current_date)

                if last_price is not None and quantity > 0:
                    sale_value = quantity * last_price
                    txn_cost = self.execution_sim.calculate_transaction_costs(
                        side='SELL', price=last_price, quantity=quantity
                    )
                    cap_gains_tax = self.execution_sim.calculate_capital_gains_tax(
                        entry_price=entry_price,
                        exit_price=last_price,
                        quantity=quantity,
                        holding_days=holding_days,
                    )
                    self.cash += sale_value - txn_cost - cap_gains_tax

                    # Calculate return percentage
                    entry_val = (entry_price or 0) * quantity
                    gross_pnl = (last_price - (entry_price or 0)) * quantity
                    net_pnl = gross_pnl - txn_cost - cap_gains_tax
                    return_pct = (gross_pnl / entry_val * 100) if entry_val > 0 else 0.0

                    # Record SELL trade with all 16 required fields
                    trade_record = {
                        'trade_id': f"T{len(self.trade_log) + 1:06d}",
                        'ticker': ticker,
                        'entry_date': self._get_entry_date_for_ticker(ticker),
                        'entry_price': entry_price,
                        'exit_date': current_date.strftime('%Y-%m-%d'),
                        'exit_price': last_price,
                        'quantity': quantity,
                        'side': 'LONG',
                        'signal_trigger': 'DELISTED',
                        'gross_pnl': gross_pnl,
                        'transaction_costs': txn_cost,
                        'taxes': cap_gains_tax,
                        'net_pnl': net_pnl,
                        'return_pct': return_pct,
                        'holding_days': holding_days,
                        'exit_reason': 'delisted'
                    }
                    self.trade_log.append(trade_record)

                    del self.holdings[ticker]
                    self._close_position(ticker, quantity)
                    self.stop_loss_levels.pop(ticker, None)
                    self.take_profit_levels.pop(ticker, None)


    def run_backtest(self) -> Dict[str, Any]:
        """
        Run the complete backtest through the time-travel loop.

        Each trading day T is replayed in real market order:

        1. Fill orders queued yesterday, at T's open.
        2. Check stops/targets against T's intraday high/low.
        3. Liquidate anything that stopped trading.
        4. Mark to market at T's close -> that day's equity point.
        5. Score the universe using data strictly before T.
        6. Queue tomorrow's orders, sized off T's end-of-day equity.

        The mark-to-market deliberately comes *after* the day's fills: it used
        to run first, so the equity curve (and every metric derived from it)
        ignored the trades that happened that same day.

        Returns:
            Dictionary containing:
            - daily_equity_curve: pd.Series
            - trade_log: list of dicts
            - brain_evolution: list of weight snapshots
            - daily_activity_log: list of dicts with day-by-day activity
        """
        equity_curve = {}

        try:
            for i, current_date in enumerate(self.master_date_index):
                self.trading_day_count = i + 1
                date_str = current_date.strftime('%Y-%m-%d')

                # Step A: Fill orders queued on T-1, at T's open.
                executed_orders = self._execute_pending_orders(current_date)

                for trade in executed_orders:
                    ticker = trade['ticker']
                    action = 'BUY' if trade.get('exit_date') is None else 'SELL'
                    notes = f"Order executed at {trade.get('entry_price') or trade.get('exit_price')}"
                    if trade.get('transaction_costs', 0) > 0:
                        notes += f", cost: {trade['transaction_costs']:.2f}"

                    self.daily_activity_log.append({
                        'date': date_str,
                        'ticker': ticker,
                        'action': action,
                        'price': trade.get('entry_price') if action == 'BUY' else trade.get('exit_price'),
                        'quantity': trade['quantity'],
                        'position_value': self._position_value(ticker, current_date),
                        'cash_balance': round(self.cash, 2),
                        'total_portfolio_value': round(self.portfolio_value, 2),
                        'score': None,
                        'signal': trade.get('signal_trigger'),
                        'notes': notes
                    })

                # Step B: Check for stop-losses and take-profits based on T's intraday High/Low
                executed_sl_tp = self._check_stop_loss_take_profit(current_date)

                for trade in executed_sl_tp:
                    ticker = trade['ticker']
                    action = 'STOP_LOSS_HIT' if trade.get('exit_reason') == 'stop_loss' else 'TARGET_HIT'

                    self.daily_activity_log.append({
                        'date': date_str,
                        'ticker': ticker,
                        'action': action,
                        'price': trade['exit_price'],
                        'quantity': trade['quantity'],
                        'position_value': self._position_value(ticker, current_date),
                        'cash_balance': round(self.cash, 2),
                        'total_portfolio_value': round(self.portfolio_value, 2),
                        'score': None,
                        'signal': None,
                        'notes': f"{action.replace('_', ' ')} triggered"
                    })

                # Step C: Liquidate holdings that have stopped trading
                self._handle_delisted_tickers(current_date)

                # Step D: Mark to market on T's close — this is the day's
                # equity point, and it now includes everything above.
                self._mark_to_market(current_date)
                equity_curve[current_date] = self.portfolio_value

                self.daily_activity_log.append({
                    'date': date_str,
                    'ticker': 'PORTFOLIO',
                    'action': 'MARK_TO_MARKET',
                    'price': None,
                    'quantity': None,
                    'position_value': None,
                    'cash_balance': round(self.cash, 2),
                    'total_portfolio_value': round(self.portfolio_value, 2),
                    'score': None,
                    'signal': None,
                    'notes': 'EOD valuation'
                })

                # Step E: Run the agent's signal generation using data up to T-1
                signals = self._generate_signals(current_date)

                for ticker in sorted(signals):
                    sig = signals[ticker]
                    self.daily_activity_log.append({
                        'date': date_str,
                        'ticker': ticker,
                        'action': 'HOLD',  # Signal evaluation, not an actual trade yet
                        'price': sig.entry_price,
                        'quantity': None,
                        'position_value': self._position_value(ticker, current_date),
                        'cash_balance': round(self.cash, 2),
                        'total_portfolio_value': round(self.portfolio_value, 2),
                        'score': sig.score,
                        'signal': sig.signal,
                        'notes': f"Signal evaluated: {sig.signal}"
                    })

                # Step F: Queue orders for T+1 execution
                self._create_pending_orders(signals, current_date)

                # Step G: Every 20 trading days, trigger evaluate_and_learn
                if self.trading_day_count % 20 == 0:
                    self._evaluate_and_learn()
        finally:
            # Workers are per-run, so they are released here whether the run
            # finished or raised.
            self.shutdown_workers()

        # Final brain snapshot
        final_snapshot = {
            'trading_day': self.trading_day_count,
            'weights': dict(self.agent_brain.weights),
            'trade_count': len(self.agent_brain.trade_history)
        }
        self.brain_evolution.append(final_snapshot)

        # Store results
        self.daily_equity_curve = pd.Series(equity_curve)

        return {
            'daily_equity_curve': self.daily_equity_curve,
            'trade_log': self.trade_log,
            'brain_evolution': self.brain_evolution,
            'daily_activity_log': self.daily_activity_log
        }

    def _position_value(self, ticker: str, current_date: pd.Timestamp) -> Optional[float]:
        """Mark an open position to T's close for the daily activity log."""
        quantity = self.holdings.get(ticker, 0)
        if quantity <= 0:
            return None
        close_price = self._get_price_at_date(ticker, current_date, 'close')
        if close_price is None or pd.isna(close_price):
            return None
        return round(quantity * close_price, 2)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics from the backtest results.
        
        Returns:
            Dictionary of performance metrics.
        """
        if len(self.daily_equity_curve) == 0:
            return {}
        
        equity = self.daily_equity_curve
        
        # Total return
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
        
        # Daily returns
        daily_returns = equity.pct_change().dropna()
        
        # Annualized return (assuming 252 trading days)
        n_days = len(equity)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1
        
        # Volatility
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Max drawdown
        rolling_max = equity.expanding().max()
        drawdowns = (equity - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Win rate over closed round trips (open BUY legs have no realized P&L).
        # Reads 'net_pnl', the key this engine actually writes — looking for a
        # 'pnl' key that never existed reported every run as zero trades.
        closed_trades = [t for t in self.trade_log if t.get('exit_date') is not None]
        winning_trades = sum(1 for t in closed_trades if t.get('net_pnl', 0) > 0)
        total_trades = len(closed_trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'final_portfolio_value': self.portfolio_value,
            'final_cash': self.cash,
            'final_holdings': self.holdings
        }
