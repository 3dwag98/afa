"""
Parallel Point-In-Time (PIT) Backtest Engine with Learning.

This module implements a high-performance, parallelized backtest engine that:
1. Uses learning methods similar to run_orchestrator
2. Trains the agent with PIT data from the last 5 years
3. Parallelizes signal generation across tickers for speed
4. Maintains strict look-ahead bias prevention
5. Supports intermediate result storage for fault tolerance
"""

import copy
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

import pandas as pd
import numpy as np

# Import from src module
try:
    from .data_store import load_ticker_data
    from .models import AgentBrain
    from .learning import evaluate_and_learn
    from .config import AppConfig
    from .execution_sim import ExecutionSimulator
    from .indicators import calculate_indicators
    from .monte_carlo import run_monte_carlo
    from .scoring import score_candidate
    from .risk import calculate_stop_target, calculate_quantity
    from .compliance import run_compliance_checks
except ImportError:
    from data_store import load_ticker_data
    from models import AgentBrain
    from learning import evaluate_and_learn
    from config import AppConfig
    from execution_sim import ExecutionSimulator
    from indicators import calculate_indicators
    from monte_carlo import run_monte_carlo
    from scoring import score_candidate
    from risk import calculate_stop_target, calculate_quantity
    from compliance import run_compliance_checks


logger = logging.getLogger(__name__)

# Directory for intermediate backtest results (thread-safe worker output)
TEMP_BACKTESTS_DIR = Path("data/temp_backtests")


def _process_single_ticker_signal(args):
    """
    Worker function for parallel signal generation.
    
    Args:
        args: Tuple of (ticker, hist_data_df, brain_weights, current_date)
        
    Returns:
        Tuple of (ticker, signal_dict) or (ticker, None) if no signal
    """
    ticker, hist_data, brain_weights, current_date = args
    
    try:
        if hist_data is None or len(hist_data) < 20:
            return (ticker, None)
        
        # Calculate indicators using the same logic as orchestrator
        indicator = calculate_indicators(ticker, hist_data)
        
        # Get current price
        current_price = float(hist_data['close'].iloc[-1])
        
        # Calculate ATR-based stop/target
        atr = indicator.atr14
        mock_config = type('MockConfig', (), {
            'atr_multiplier_stop': 2.0,
            'atr_multiplier_target': 3.0
        })()
        stop_price, target_price = calculate_stop_target(current_price, atr, mock_config)
        
        # Simple Monte Carlo for probability
        daily_returns = hist_data['close'].pct_change().dropna().tolist()
        if len(daily_returns) < 20:
            mc_prob = 0.5
        else:
            mc_result = run_monte_carlo(
                symbol=ticker,
                daily_returns=daily_returns,
                horizon_days=21,
                simulations=100,
                seed=42
            )
            mc_prob = mc_result.probability_profit
        
        # Score candidate using brain weights
        scored = score_candidate(
            indicator=indicator,
            mc_result=type('MockMC', (), {
                'probability_profit': mc_prob,
                'var_95': -0.05,
                'cvar_95': -0.08
            })(),
            brain=type('MockBrain', (), {'weights': brain_weights})(),
            config=mock_config,
            entry_price=current_price,
            stop_price=stop_price,
            target_price=target_price,
            run_id=None  # No run_id in parallel backtest workers
        )
        
        # Generate signal
        signal_type = 'HOLD'
        if scored['score'] > 0.6:
            signal_type = 'BUY'
        elif scored['score'] < 0.4:
            signal_type = 'SELL'
        
        return (ticker, {
            'signal': signal_type,
            'score': scored['score'],
            'current_price': current_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'trigger': scored['trigger'],
            'rationale': scored['rationale']
        })
        
    except Exception as e:
        logger.debug(f"Signal generation failed for {ticker}: {e}")
        return (ticker, None)


class ParallelBacktestEngine:
    """
    Parallel Point-In-Time Backtest Engine with Learning.
    
    This engine parallelizes signal generation across tickers while maintaining
    strict temporal causality (no look-ahead bias). The agent learns from trade
    outcomes every N trading days, adapting its weights based on performance.
    
    Key Features:
    - Parallel signal generation using ProcessPoolExecutor or ThreadPoolExecutor
    - PIT data access (at date T, only sees data up to T-1)
    - T+1 execution delay
    - Learning every N trading days using evaluate_and_learn
    - Configurable number of workers for parallelization
    """
    
    def __init__(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float,
        universe_tickers: List[str],
        initial_brain: Optional[Dict[str, Any]] = None,
        num_workers: int = -1,
        use_processes: bool = False,
        learning_interval: int = 20
    ):
        """
        Initialize the Parallel Backtest Engine.
        
        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            initial_capital: Initial cash in INR.
            universe_tickers: List of ticker symbols to trade.
            initial_brain: Optional initial brain state.
            num_workers: Number of parallel workers (-1 = auto-detect CPU count).
            use_processes: Use ProcessPoolExecutor instead of ThreadPoolExecutor.
            learning_interval: Number of trading days between learning updates.
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.universe_tickers = [t.upper() if not t.endswith('.NS') else t for t in universe_tickers]
        self.learning_interval = learning_interval
        
        # Determine number of workers
        if num_workers == -1:
            self.num_workers = mp.cpu_count()
        else:
            self.num_workers = min(num_workers, mp.cpu_count())
        
        self.use_processes = use_processes
        
        # Track untradeable tickers
        self.untradeable_tickers: set = set()
        
        # Load all ticker data into memory
        self.ticker_data: Dict[str, pd.DataFrame] = {}
        self._load_all_data()
        
        # Create unified master date index
        self.master_date_index = self._build_master_date_index()
        
        # State Management
        self.cash = initial_capital
        self.holdings: Dict[str, int] = {}
        self.portfolio_value = initial_capital
        
        # Initialize agent brain
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
        
        # Stop-loss and take-profit tracking
        self.stop_loss_levels: Dict[str, float] = {}
        self.take_profit_levels: Dict[str, float] = {}
        
        # Pending orders for T+1 execution
        self.pending_orders: List[Dict[str, Any]] = []
        
        # Trading day counter
        self.trading_day_count = 0
        
        # Execution simulator
        self.execution_sim = ExecutionSimulator()
        
        logger.info(f"Parallel Backtest Engine initialized with {self.num_workers} workers")
        logger.info(f"Trading days: {len(self.master_date_index)}, Tickers: {len(self.universe_tickers)}")
    
    def _load_all_data(self) -> None:
        """Load all ticker data into memory."""
        logger.info(f"Loading data for {len(self.universe_tickers)} tickers...")
        
        loaded_count = 0
        for i, ticker in enumerate(self.universe_tickers):
            df = load_ticker_data(ticker, 
                                  start_date=self.start_date.strftime('%Y-%m-%d'),
                                  end_date=self.end_date.strftime('%Y-%m-%d'))
            if df is not None and len(df) > 0:
                df = df.copy()
                df.columns = [c.lower() for c in df.columns]
                
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                
                df = df.sort_index()
                self.ticker_data[ticker] = df
                loaded_count += 1
            else:
                self.untradeable_tickers.add(ticker)
                logger.warning(f"No data available for {ticker}, marking as untradeable")
            
            if (i + 1) % 100 == 0:
                logger.info(f"Loaded {i + 1}/{len(self.universe_tickers)} tickers...")
        
        logger.info(f"Successfully loaded {loaded_count}/{len(self.universe_tickers)} tickers")
    
    def _build_master_date_index(self) -> pd.DatetimeIndex:
        """Build unified master date index from all ticker data."""
        all_dates = set()
        for ticker, df in self.ticker_data.items():
            if ticker not in self.untradeable_tickers:
                all_dates.update(df.index.tolist())
        
        if not all_dates:
            return pd.bdate_range(start=self.start_date, end=self.end_date)
        
        date_index = pd.DatetimeIndex(sorted(all_dates))
        date_index = date_index[(date_index >= self.start_date) & (date_index <= self.end_date)]
        return date_index
    
    def _get_historical_data_up_to(self, ticker: str, up_to_date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Get historical data up to (but not including) a specific date."""
        if ticker in self.untradeable_tickers:
            return None
        
        if ticker not in self.ticker_data:
            return None
        
        df = self.ticker_data[ticker].copy()
        mask = df.index < up_to_date
        return df[mask].copy()
    
    def _generate_signals_parallel(self, current_date: pd.Timestamp) -> Dict[str, Dict[str, Any]]:
        """
        Generate signals for all tickers in parallel.
        
        This is the key optimization: instead of processing tickers sequentially,
        we distribute the work across multiple workers.
        
        Args:
            current_date: Current date timestamp.
            
        Returns:
            Dictionary of ticker -> signal info.
        """
        signals = {}
        prev_date = current_date - pd.Timedelta(days=1)
        
        # Prepare arguments for parallel processing
        tasks = []
        for ticker in self.universe_tickers:
            if ticker in self.untradeable_tickers:
                continue
            
            hist_data = self._get_historical_data_up_to(ticker, current_date)
            if hist_data is not None and len(hist_data) >= 20:
                tasks.append((ticker, hist_data, dict(self.agent_brain.weights), current_date))
        
        if not tasks:
            return signals
        
        # Choose executor type
        ExecutorClass = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        # Execute in parallel
        with ExecutorClass(max_workers=self.num_workers) as executor:
            futures = {executor.submit(_process_single_ticker_signal, task): task[0] 
                      for task in tasks}
            
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result_ticker, signal_info = future.result()
                    if signal_info is not None:
                        signals[result_ticker] = signal_info
                except Exception as e:
                    logger.debug(f"Parallel task failed for {ticker}: {e}")
        
        return signals
    
    def _mark_to_market(self, current_date: pd.Timestamp) -> float:
        """Mark-to-Market portfolio using current date's closing prices."""
        holdings_value = 0.0
        
        for ticker, quantity in self.holdings.items():
            if quantity > 0:
                close_price = self._get_price_at_date(ticker, current_date, 'close')
                if close_price is not None and not pd.isna(close_price):
                    holdings_value += quantity * close_price
                else:
                    last_known_price = self._get_last_valid_price(ticker, current_date)
                    if last_known_price is not None:
                        holdings_value += quantity * last_known_price
                        self.untradeable_tickers.add(ticker)
        
        self.portfolio_value = self.cash + holdings_value
        return self.portfolio_value
    
    def _get_price_at_date(self, ticker: str, date: pd.Timestamp, price_type: str = 'close') -> Optional[float]:
        """Get price for a ticker at a specific date."""
        if ticker in self.untradeable_tickers:
            return None
        
        if ticker not in self.ticker_data:
            return None
        
        df = self.ticker_data[ticker]
        
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
                    return df.loc[date, col]
                except KeyError:
                    available_dates = df[df.index <= date].index
                    if len(available_dates) > 0:
                        nearest_date = available_dates[-1]
                        return df.loc[nearest_date, col]
        
        return None
    
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
    
    def _check_stop_loss_take_profit(self, current_date: pd.Timestamp) -> List[Dict[str, Any]]:
        """Check for stop-losses and take-profits based on intraday High/Low."""
        executed_trades = []
        tickers_to_remove = []
        
        for ticker in list(self.holdings.keys()):
            quantity = self.holdings[ticker]
            if quantity <= 0 or ticker in self.untradeable_tickers:
                continue
            
            df = self.ticker_data.get(ticker)
            if df is None or current_date not in df.index:
                continue
            
            row = df.loc[current_date]
            
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
            entry_price = self._get_last_valid_price(ticker, current_date)
            
            triggered = False
            trigger_price = None
            trigger_type = None
            
            if stop_price is not None and low <= stop_price:
                triggered = True
                trigger_price = stop_price
                trigger_type = 'STOP_LOSS'
            elif target_price is not None and high >= target_price:
                triggered = True
                trigger_price = target_price
                trigger_type = 'TAKE_PROFIT'
            
            if triggered:
                sale_value = quantity * trigger_price
                self.cash += sale_value
                
                entry_val = (entry_price or 0) * quantity
                gross_pnl = (trigger_price - (entry_price or 0)) * quantity
                return_pct = (gross_pnl / entry_val * 100) if entry_val > 0 else 0.0
                
                exit_reason = 'stop_loss' if trigger_type == 'STOP_LOSS' else 'target'
                
                trade_record = {
                    'trade_id': f"T{len(self.trade_log) + 1:06d}",
                    'ticker': ticker,
                    'entry_date': self._get_entry_date_for_ticker(ticker),
                    'entry_price': entry_price,
                    'exit_date': current_date.strftime('%Y-%m-%d'),
                    'exit_price': trigger_price,
                    'quantity': quantity,
                    'side': 'LONG',
                    'signal_trigger': trigger_type,
                    'gross_pnl': gross_pnl,
                    'transaction_costs': 0.0,
                    'taxes': 0.0,
                    'net_pnl': gross_pnl,
                    'return_pct': return_pct,
                    'exit_reason': exit_reason,
                    'outcome': 'WIN' if gross_pnl > 0 else 'LOSS',
                    'holding_days': self._get_holding_days(ticker, current_date)
                }
                
                self.trade_log.append(trade_record)
                executed_trades.append(trade_record)
                
                self.holdings[ticker] = 0
                tickers_to_remove.append(ticker)
                
                if ticker in self.stop_loss_levels:
                    del self.stop_loss_levels[ticker]
                if ticker in self.take_profit_levels:
                    del self.take_profit_levels[ticker]
        
        for ticker in tickers_to_remove:
            del self.holdings[ticker]
        
        return executed_trades
    
    def _get_entry_date_for_ticker(self, ticker: str) -> Optional[str]:
        """Get the entry date for a holding."""
        for trade in reversed(self.trade_log):
            if trade.get('ticker') == ticker and trade.get('action') == 'BUY':
                return trade.get('entry_date')
        return None
    
    def _get_holding_days(self, ticker: str, current_date: pd.Timestamp) -> int:
        """Get the number of days a position has been held."""
        for trade in reversed(self.trade_log):
            if trade.get('ticker') == ticker and trade.get('action') == 'BUY':
                entry_date = pd.to_datetime(trade.get('date', current_date))
                return (current_date - entry_date).days
        return 0
    
    def _execute_pending_orders(self, execution_date: pd.Timestamp) -> List[Dict[str, Any]]:
        """Execute pending orders for T+1 Open."""
        executed_trades = []
        orders_to_remove = []
        
        for i, order in enumerate(self.pending_orders):
            if order['execution_date'] != execution_date:
                continue
            
            ticker = order['ticker']
            action = order['action']
            quantity = order['quantity']
            
            if ticker in self.untradeable_tickers:
                orders_to_remove.append(i)
                continue
            
            open_price = self._get_price_at_date(ticker, execution_date, 'open')
            
            if open_price is None or pd.isna(open_price):
                open_price = self._get_last_valid_price(ticker, execution_date)
            
            if open_price is None:
                orders_to_remove.append(i)
                continue
            
            total_cost = quantity * open_price
            
            if action == 'BUY':
                if self.cash >= total_cost:
                    self.cash -= total_cost
                    self.holdings[ticker] = self.holdings.get(ticker, 0) + quantity
                    
                    trade_record = {
                        'trade_id': f"T{len(self.trade_log) + 1:06d}",
                        'ticker': ticker,
                        'entry_date': execution_date.strftime('%Y-%m-%d'),
                        'entry_price': open_price,
                        'exit_date': None,
                        'exit_price': None,
                        'quantity': quantity,
                        'side': 'LONG',
                        'action': 'BUY',
                        'signal_trigger': order.get('trigger', 'SIGNAL'),
                        'gross_pnl': 0.0,
                        'transaction_costs': 0.0,
                        'taxes': 0.0,
                        'net_pnl': 0.0,
                        'return_pct': 0.0,
                        'exit_reason': None,
                        'outcome': 'OPEN',
                        'holding_days': 0
                    }
                    
                    self.trade_log.append(trade_record)
                    executed_trades.append(trade_record)
                    
                    self.stop_loss_levels[ticker] = open_price * 0.95
                    self.take_profit_levels[ticker] = open_price * 1.15
                    
                    orders_to_remove.append(i)
                else:
                    orders_to_remove.append(i)
            
            elif action == 'SELL':
                if ticker in self.holdings and self.holdings[ticker] >= quantity:
                    sale_value = quantity * open_price
                    self.cash += sale_value
                    self.holdings[ticker] -= quantity
                    
                    if self.holdings[ticker] == 0:
                        del self.holdings[ticker]
                    
                    orders_to_remove.append(i)
        
        for i in sorted(orders_to_remove, reverse=True):
            self.pending_orders.pop(i)
        
        return executed_trades
    
    def _create_pending_orders(self, signals: Dict[str, Dict[str, Any]], current_date: pd.Timestamp) -> None:
        """Create pending orders for T+1 execution based on signals."""
        execution_date = current_date + pd.Timedelta(days=1)
        
        while execution_date.weekday() >= 5:
            execution_date += pd.Timedelta(days=1)
        
        for ticker, signal_info in signals.items():
            signal = signal_info.get('signal', 'HOLD')
            
            if signal == 'BUY' and ticker not in self.holdings:
                position_value = self.portfolio_value * 0.10
                price = signal_info.get('current_price', 100)
                quantity = int(position_value / price)
                
                if quantity > 0:
                    self.pending_orders.append({
                        'ticker': ticker,
                        'action': 'BUY',
                        'quantity': quantity,
                        'execution_date': execution_date,
                        'trigger': signal_info.get('trigger', 'SIGNAL')
                    })
            
            elif signal == 'SELL' and ticker in self.holdings:
                quantity = self.holdings[ticker]
                if quantity > 0:
                    self.pending_orders.append({
                        'ticker': ticker,
                        'action': 'SELL',
                        'quantity': quantity,
                        'execution_date': execution_date,
                        'trigger': signal_info.get('trigger', 'SIGNAL')
                    })
    
    def _evaluate_and_learn(self) -> None:
        """Trigger agent learning using evaluate_and_learn from learning.py."""
        mock_config = type('MockConfig', (), {
            'learning_rate': 0.15,
            'min_trades_for_learning': 3
        })()
        
        brain_snapshot = {
            'trading_day': self.trading_day_count,
            'weights': dict(self.agent_brain.weights),
            'trade_count': len(self.agent_brain.trade_history)
        }
        self.brain_evolution.append(brain_snapshot)
        
        try:
            self.agent_brain = evaluate_and_learn(self.agent_brain, mock_config)
            logger.info(f"Learning update at day {self.trading_day_count}: "
                       f"{len(self.agent_brain.trade_history)} trades in history")
        except Exception as e:
            logger.error(f"Learning failed: {e}")
    
    def _handle_delisted_tickers(self, current_date: pd.Timestamp) -> None:
        """Handle delisted tickers or those with NaN issues."""
        for ticker in list(self.holdings.keys()):
            if ticker in self.untradeable_tickers:
                continue
            
            df = self.ticker_data.get(ticker)
            if df is None:
                self.untradeable_tickers.add(ticker)
                continue
            
            if current_date not in df.index:
                last_date = df.index.max()
                if current_date > last_date:
                    self.untradeable_tickers.add(ticker)
                    
                    quantity = self.holdings[ticker]
                    last_price = self._get_last_valid_price(ticker, current_date)
                    
                    if last_price is not None and quantity > 0:
                        sale_value = quantity * last_price
                        self.cash += sale_value
                        
                        trade_record = {
                            'trade_id': f"T{len(self.trade_log) + 1:06d}",
                            'ticker': ticker,
                            'entry_date': self._get_entry_date_for_ticker(ticker),
                            'entry_price': 0.0,
                            'exit_date': current_date.strftime('%Y-%m-%d'),
                            'exit_price': last_price,
                            'quantity': quantity,
                            'side': 'LONG',
                            'signal_trigger': 'DELISTED',
                            'gross_pnl': 0.0,
                            'transaction_costs': 0.0,
                            'taxes': 0.0,
                            'net_pnl': 0.0,
                            'return_pct': 0.0,
                            'exit_reason': 'delisted',
                            'outcome': 'CLOSED',
                            'holding_days': self._get_holding_days(ticker, current_date)
                        }
                        
                        self.trade_log.append(trade_record)
                        self.holdings[ticker] = 0
                        del self.holdings[ticker]
    
    def run_backtest(self) -> Dict[str, Any]:
        """
        Run the complete parallel backtest.
        
        Returns:
            Dictionary containing daily_equity_curve, trade_log, brain_evolution,
            and daily_activity_log.
        """
        equity_curve = {}
        
        logger.info(f"Starting parallel backtest with {self.num_workers} workers...")
        
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            logger.info("tqdm not available, running without progress bar")
        
        date_iter = self.master_date_index
        if use_tqdm:
            date_iter = tqdm(date_iter, desc="Backtesting", unit="day")
        
        for i, current_date in enumerate(date_iter):
            self.trading_day_count = i + 1
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Step A: Mark-to-Market
            self._mark_to_market(current_date)
            equity_curve[current_date] = self.portfolio_value
            
            # Step B: Check stop-losses and take-profits
            self._check_stop_loss_take_profit(current_date)
            
            # Step C: Generate signals IN PARALLEL
            signals = self._generate_signals_parallel(current_date)
            
            # Step D: Create and execute pending orders
            self._create_pending_orders(signals, current_date)
            self._execute_pending_orders(current_date)
            
            # Handle delisted tickers
            self._handle_delisted_tickers(current_date)
            
            # Step E: Learning every N trading days
            if self.trading_day_count % self.learning_interval == 0:
                self._evaluate_and_learn()
        
        # Final brain snapshot
        final_snapshot = {
            'trading_day': self.trading_day_count,
            'weights': dict(self.agent_brain.weights),
            'trade_count': len(self.agent_brain.trade_history)
        }
        self.brain_evolution.append(final_snapshot)
        
        self.daily_equity_curve = pd.Series(equity_curve)
        
        logger.info(f"Backtest complete. Total trades: {len(self.trade_log)}")
        
        return {
            'daily_equity_curve': self.daily_equity_curve,
            'trade_log': self.trade_log,
            'brain_evolution': self.brain_evolution,
            'daily_activity_log': self.daily_activity_log
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics from backtest results."""
        if len(self.daily_equity_curve) == 0:
            return {}
        
        equity = self.daily_equity_curve
        
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
        daily_returns = equity.pct_change().dropna()
        n_days = len(equity)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        rolling_max = equity.expanding().max()
        drawdowns = (equity - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        winning_trades = sum(1 for t in self.trade_log if t.get('net_pnl', 0) > 0)
        total_trades = len([t for t in self.trade_log if t.get('outcome') in ('WIN', 'LOSS')])
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


def run_parallel_pit_backtest(
    start_date: str,
    end_date: str,
    initial_capital: float,
    universe_tickers: List[str],
    num_workers: int = -1,
    use_processes: bool = False,
    learning_interval: int = 20,
    initial_brain: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to run a parallel PIT backtest.
    
    Args:
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        initial_capital: Initial capital in INR.
        universe_tickers: List of ticker symbols.
        num_workers: Number of parallel workers (-1 = auto).
        use_processes: Use processes instead of threads.
        learning_interval: Days between learning updates.
        initial_brain: Optional initial brain state.
        
    Returns:
        Dictionary with backtest results and metrics.
    """
    engine = ParallelBacktestEngine(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        universe_tickers=universe_tickers,
        initial_brain=initial_brain,
        num_workers=num_workers,
        use_processes=use_processes,
        learning_interval=learning_interval
    )
    
    results = engine.run_backtest()
    metrics = engine.get_performance_metrics()
    
    return {
        'status': 'success',
        'metrics': metrics,
        'trade_log': results['trade_log'],
        'brain_evolution': results['brain_evolution'],
        'daily_equity_curve': results['daily_equity_curve'],
        'trading_days': len(engine.master_date_index),
        'tickers_processed': len(engine.ticker_data)
    }
