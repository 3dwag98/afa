"""
Event-Driven Backtesting Engine for Portfolio Agent.

This engine simulates market data over a 5-year period, strictly avoiding look-ahead bias,
and allowing the Agent's Brain to learn over time.
"""

import copy
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

# Import from src module
try:
    from .data_store import load_ticker_data
    from .models import AgentBrain
    from .learning import evaluate_and_learn
    from .config import AppConfig
    from .execution_sim import ExecutionSimulator
except ImportError:
    from data_store import load_ticker_data
    from models import AgentBrain
    from learning import evaluate_and_learn
    from config import AppConfig
    from execution_sim import ExecutionSimulator


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
        initial_brain: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the BacktestEngine.
        
        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            initial_capital: Initial cash in INR.
            universe_tickers: List of ticker symbols to trade.
            initial_brain: Optional initial brain state. If None, uses default weights.
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.universe_tickers = [t.upper() if not t.endswith('.NS') else t for t in universe_tickers]
        
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
    
    def _get_entry_price_for_tax(self, ticker: str) -> float:
        """
        Get the entry price for a holding (for tax calculation).
        
        Args:
            ticker: Ticker symbol.
            
        Returns:
            Entry price or 0 if not found.
        """
        # Look up the entry price from trade log
        for trade in reversed(self.trade_log):
            if trade.get('ticker') == ticker and trade.get('action') == 'BUY':
                return trade.get('entry_price', trade.get('price', 0))
        
        # Fallback: use stop_loss level as proxy (set at 95% of entry)
        if ticker in self.stop_loss_levels:
            return self.stop_loss_levels[ticker] / 0.95
        
        return 0.0
    
    def _get_holding_days(self, ticker: str, current_date: pd.Timestamp) -> int:
        """
        Get the number of days a position has been held.
        
        Args:
            ticker: Ticker symbol.
            current_date: Current date.
            
        Returns:
            Number of days held.
        """
        # Look up the entry date from trade log
        for trade in reversed(self.trade_log):
            if trade.get('ticker') == ticker and trade.get('action') == 'BUY':
                entry_date = pd.to_datetime(trade.get('date', current_date))
                return (current_date - entry_date).days
        
        return 0
    
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
            entry_price = self._get_last_valid_price(ticker, current_date)
            
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
                # Execute sale at trigger price
                sale_value = quantity * trigger_price
                self.cash += sale_value
                
                # Record trade
                trade_record = {
                    'date': current_date.strftime('%Y-%m-%d'),
                    'ticker': ticker,
                    'action': 'SELL',
                    'quantity': quantity,
                    'price': trigger_price,
                    'value': sale_value,
                    'trigger': trigger_type,
                    'entry_price': entry_price,
                    'exit_price': trigger_price,
                    'pnl': (trigger_price - (entry_price or 0)) * quantity
                }
                self.trade_log.append(trade_record)
                executed_trades.append(trade_record)
                
                # Remove holding
                del self.holdings[ticker]
                tickers_to_remove.append(ticker)
                
                # Clear stop/target levels
                if ticker in self.stop_loss_levels:
                    del self.stop_loss_levels[ticker]
                if ticker in self.take_profit_levels:
                    del self.take_profit_levels[ticker]
        
        return executed_trades
    
    def _generate_signals(self, current_date: pd.Timestamp) -> Dict[str, Dict[str, Any]]:
        """
        Run the Agent's signal generation using data up to T-1.
        
        CRITICAL: Only uses data up to current_date - 1 day to avoid look-ahead bias.
        
        Args:
            current_date: Current date timestamp.
            
        Returns:
            Dictionary of ticker -> signal info.
        """
        signals = {}
        prev_date = current_date - pd.Timedelta(days=1)
        
        # Use tqdm for progress tracking if available and universe is large
        try:
            from tqdm import tqdm
            use_tqdm = len(self.universe_tickers) > 100
        except ImportError:
            use_tqdm = False
        
        ticker_iter = self.universe_tickers
        if use_tqdm:
            ticker_iter = tqdm(ticker_iter, desc=f"Scanning tickers ({current_date.date()})", unit="ticker", leave=False)
        
        for ticker in ticker_iter:
            if ticker in self.untradeable_tickers:
                continue
            
            # Get historical data up to T-1
            hist_data = self._get_historical_data_up_to(ticker, current_date)
            
            if hist_data is None or len(hist_data) < 20:
                # Not enough data for signals
                continue
            
            # Simple signal generation logic (mimicking fetch_and_score)
            # In a real implementation, this would call the agent's scoring logic
            
            close_prices = hist_data['close'] if 'close' in hist_data.columns else hist_data.get('Close', hist_data.iloc[:, 0])
            
            # Calculate simple momentum signal
            if len(close_prices) >= 20:
                sma_20 = close_prices.rolling(window=20).mean().iloc[-1]
                current_price = close_prices.iloc[-1]
                
                # Trend signal
                trend_signal = 1 if current_price > sma_20 else -1
                
                # Volume signal (if available)
                volume_signal = 0
                if 'volume' in hist_data.columns:
                    avg_vol = hist_data['volume'].rolling(window=20).mean().iloc[-1]
                    current_vol = hist_data['volume'].iloc[-1]
                    if current_vol > avg_vol * 1.5:
                        volume_signal = 1
                    elif current_vol < avg_vol * 0.5:
                        volume_signal = -1
                
                # Combine signals using brain weights
                weights = self.agent_brain.weights
                combined_score = (
                    weights.get('Trend', 25.0) * trend_signal +
                    weights.get('Volume', 20.0) * volume_signal
                ) / 100.0
                
                # Generate signal
                if combined_score > 0.3:
                    signal_type = 'BUY'
                elif combined_score < -0.3:
                    signal_type = 'SELL'
                else:
                    signal_type = 'HOLD'
                
                signals[ticker] = {
                    'signal': signal_type,
                    'score': combined_score,
                    'current_price': current_price,
                    'sma_20': sma_20,
                    'trigger': 'Trend' if abs(trend_signal) > 0 else 'Volume'
                }
        
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
            if order['execution_date'] != execution_date:
                continue
            
            ticker = order['ticker']
            action = order['action']
            quantity = order['quantity']
            
            if ticker in self.untradeable_tickers:
                orders_to_remove.append(i)
                continue
            
            # Get open price for execution
            open_price = self._get_price_at_date(ticker, execution_date, 'open')
            
            if open_price is None or pd.isna(open_price):
                # Try to use close price from previous day as fallback
                open_price = self._get_last_valid_price(ticker, execution_date)
            
            if open_price is None:
                orders_to_remove.append(i)
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
                orders_to_remove.append(i)
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
                    
                    # Set stop-loss and take-profit levels
                    # Stop-loss at 5% below entry
                    self.stop_loss_levels[ticker] = adjusted_price * 0.95
                    # Take-profit at 10% above entry
                    self.take_profit_levels[ticker] = adjusted_price * 1.10
                    
                    trade_record = {
                        'date': execution_date.strftime('%Y-%m-%d'),
                        'ticker': ticker,
                        'action': 'BUY',
                        'quantity': quantity,
                        'price': adjusted_price,
                        'value': trade_value,
                        'txn_cost': txn_cost,
                        'trigger': order.get('trigger', 'SIGNAL'),
                        'entry_price': adjusted_price
                    }
                    self.trade_log.append(trade_record)
                    executed_trades.append(trade_record)
                    orders_to_remove.append(i)
                    
            elif action == 'SELL':
                if ticker in self.holdings and self.holdings[ticker] >= quantity:
                    # Get holding info for capital gains tax calculation
                    entry_price = self._get_entry_price_for_tax(ticker)
                    holding_days = self._get_holding_days(ticker, execution_date)
                    
                    # Calculate capital gains tax
                    cap_gains_tax = self.execution_sim.calculate_capital_gains_tax(
                        entry_price=entry_price,
                        exit_price=adjusted_price,
                        quantity=quantity,
                        holding_days=holding_days
                    )
                    
                    self.holdings[ticker] -= quantity
                    self.cash += trade_value - cap_gains_tax  # Deduct tax from cash
                    
                    # Clear stop/target levels
                    if ticker in self.stop_loss_levels:
                        del self.stop_loss_levels[ticker]
                    if ticker in self.take_profit_levels:
                        del self.take_profit_levels[ticker]
                    
                    trade_record = {
                        'date': execution_date.strftime('%Y-%m-%d'),
                        'ticker': ticker,
                        'action': 'SELL',
                        'quantity': quantity,
                        'price': adjusted_price,
                        'value': trade_value,
                        'txn_cost': txn_cost,
                        'cap_gains_tax': cap_gains_tax,
                        'trigger': order.get('trigger', 'SIGNAL'),
                        'exit_price': adjusted_price,
                        'entry_price': entry_price,
                        'holding_days': holding_days,
                        'pnl': (adjusted_price - entry_price) * quantity - cap_gains_tax - txn_cost
                    }
                    self.trade_log.append(trade_record)
                    executed_trades.append(trade_record)
                    orders_to_remove.append(i)
                    
                    # Remove holding if zero
                    if self.holdings[ticker] == 0:
                        del self.holdings[ticker]
        
        # Remove executed orders
        for i in sorted(orders_to_remove, reverse=True):
            self.pending_orders.pop(i)
        
        return executed_trades
    
    def _create_pending_orders(self, signals: Dict[str, Dict[str, Any]], current_date: pd.Timestamp) -> None:
        """
        Create pending orders for T+1 execution based on signals.
        
        Args:
            signals: Dictionary of ticker -> signal info.
            current_date: Current date (orders will execute at T+1 open).
        """
        execution_date = current_date + pd.Timedelta(days=1)
        
        # Skip weekends
        while execution_date.weekday() >= 5:
            execution_date += pd.Timedelta(days=1)
        
        for ticker, signal_info in signals.items():
            signal = signal_info.get('signal', 'HOLD')
            
            if signal == 'BUY' and ticker not in self.holdings:
                # Calculate position size (10% of portfolio per position)
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
        """
        Trigger agent learning every 20 trading days.
        
        Creates a mock config for the learning function.
        """
        # Create a mock config for learning
        mock_config = type('MockConfig', (), {
            'learning_rate': 0.15,
            'min_trades_for_learning': 3
        })()
        
        # Snapshot brain state before learning
        brain_snapshot = {
            'trading_day': self.trading_day_count,
            'weights': dict(self.agent_brain.weights),
            'trade_count': len(self.agent_brain.trade_history)
        }
        self.brain_evolution.append(brain_snapshot)
        
        # Run learning
        try:
            self.agent_brain = evaluate_and_learn(self.agent_brain, mock_config)
        except Exception as e:
            # Log error but continue
            pass
    
    def _handle_delisted_tickers(self, current_date: pd.Timestamp) -> None:
        """
        Handle delisted tickers or those with NaN issues.
        
        Force liquidation at last known price.
        """
        for ticker in list(self.holdings.keys()):
            if ticker in self.untradeable_tickers:
                continue
            
            df = self.ticker_data.get(ticker)
            if df is None:
                self.untradeable_tickers.add(ticker)
                continue
            
            # Check if current date is in data
            if current_date not in df.index:
                # Check if we've passed the last available date
                last_date = df.index.max()
                if current_date > last_date:
                    # Ticker appears to be delisted
                    self.untradeable_tickers.add(ticker)
                    
                    # Force liquidation
                    quantity = self.holdings[ticker]
                    last_price = self._get_last_valid_price(ticker, current_date)
                    
                    if last_price is not None and quantity > 0:
                        sale_value = quantity * last_price
                        self.cash += sale_value
                        
                        trade_record = {
                            'date': current_date.strftime('%Y-%m-%d'),
                            'ticker': ticker,
                            'action': 'SELL',
                            'quantity': quantity,
                            'price': last_price,
                            'value': sale_value,
                            'trigger': 'DELISTED',
                            'exit_price': last_price
                        }
                        self.trade_log.append(trade_record)
                        
                        del self.holdings[ticker]
    
    def run_backtest(self) -> Dict[str, Any]:
        """
        Run the complete backtest through the time-travel loop.
        
        Returns:
            Dictionary containing:
            - daily_equity_curve: pd.Series
            - trade_log: list of dicts
            - brain_evolution: list of weight snapshots
        """
        equity_curve = {}
        
        for i, current_date in enumerate(self.master_date_index):
            self.trading_day_count = i + 1
            
            # Step A: Mark-to-Market portfolio using T's closing prices
            self._mark_to_market(current_date)
            equity_curve[current_date] = self.portfolio_value
            
            # Step B: Check for stop-losses and take-profits based on T's intraday High/Low
            self._check_stop_loss_take_profit(current_date)
            
            # Step C: Run Agent's signal generation using data up to T-1
            signals = self._generate_signals(current_date)
            
            # Step D: Create pending orders for T+1 execution
            self._create_pending_orders(signals, current_date)
            
            # Execute any pending orders for today (from previous day's signals)
            self._execute_pending_orders(current_date)
            
            # Handle delisted tickers
            self._handle_delisted_tickers(current_date)
            
            # Step E: Every 20 trading days, trigger evaluate_and_learn
            if self.trading_day_count % 20 == 0:
                self._evaluate_and_learn()
        
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
            'brain_evolution': self.brain_evolution
        }
    
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
        
        # Win rate
        winning_trades = sum(1 for t in self.trade_log if t.get('pnl', 0) > 0)
        total_trades = len([t for t in self.trade_log if 'pnl' in t])
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


# Module-level logger
import logging
logger = logging.getLogger(__name__)
