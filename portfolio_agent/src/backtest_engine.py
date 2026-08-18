"""
Event-Driven Backtesting Engine for Portfolio Agent.

This engine simulates market data over a 5-year period, strictly avoiding look-ahead bias,
and allowing the Agent's Brain to learn over time.
"""

import copy
import logging
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

from portfolio_agent.config.schema import StrategyConfig
from portfolio_agent.features.pipeline import build_features
from portfolio_agent.strategies.base import BaseStrategy
from portfolio_agent.strategies.registry import load_strategy
from portfolio_agent.strategies.types import RiskParams, StrategyContext, StrategySignal
from portfolio_agent.strategies.weighting import evaluate_and_learn

# Import from src module
from .data_store import load_ticker_data
from .liquidity import lower_circuit_locked_days
from .models import AgentBrain
from .regime import DEFAULT_TREND_WINDOW, assess_market_regime, build_market_proxy
from .execution_sim import ExecutionSimulator
from .monte_carlo import CrossSectionalDriftPrior, MonteCarloSettings
from .portfolio import (
    ledoit_wolf_covariance, portfolio_volatility, summarize_book_risk,
)
from .risk import MAX_KELLY_FRACTION, calculate_kelly_quantity, estimate_kelly_inputs
from .sectors import (
    load_sector_map, sector_cap_is_enforceable, sector_capacity_inr, sector_of,
)


logger = logging.getLogger(__name__)


def _score_one_ticker(
    strategy: BaseStrategy,
    ticker: str,
    hist_data: pd.DataFrame,
    required_features: List[str],
    risk_params: RiskParams,
    weights: Dict[str, float],
    mc_settings: MonteCarloSettings,
    regime_label: Optional[str] = None,
    normalize: bool = False,
    normalize_window: int = 252,
) -> Optional[StrategySignal]:
    """Score a single ticker: run Monte Carlo + build features + call strategy.score().

    Module-level (not a method) so it is picklable for ProcessPoolExecutor dispatch.

    `regime_label` is classified once per day by the caller and passed down,
    not re-derived per ticker: the benchmark is the same for everyone, and a
    UMA whose members are regime-gated must see one classification per round.
    """
    try:
        daily_returns = hist_data['close'].pct_change().dropna().tolist()
        mc_result = mc_settings.run(
            symbol=ticker, daily_returns=daily_returns, ohlcv=hist_data
        )
        features = build_features(
            hist_data, required_features,
            normalize=normalize, normalize_window=normalize_window,
        )
        context = StrategyContext(
            risk=risk_params, weights=weights, mc_result=mc_result,
            regime_label=regime_label,
        )
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
_WORKER_MC_SETTINGS: MonteCarloSettings = MonteCarloSettings()
#: Run-constant too, and pinned here for the same reason: a worker that
#: normalized differently from the parent would produce a different signal for
#: the same ticker depending only on whether `--parallel` was passed.
_WORKER_NORMALIZE: bool = False
_WORKER_NORMALIZE_WINDOW: int = 252


def _init_scoring_worker(
    strategy: BaseStrategy,
    required_features: List[str],
    risk_params: RiskParams,
    mc_settings: MonteCarloSettings,
    normalize: bool = False,
    normalize_window: int = 252,
) -> None:
    """ProcessPoolExecutor initializer: pin the run-constant scoring inputs."""
    global _WORKER_STRATEGY, _WORKER_FEATURES, _WORKER_RISK, _WORKER_MC_SETTINGS
    global _WORKER_NORMALIZE, _WORKER_NORMALIZE_WINDOW
    _WORKER_STRATEGY = strategy
    _WORKER_FEATURES = required_features
    _WORKER_RISK = risk_params
    _WORKER_MC_SETTINGS = mc_settings
    _WORKER_NORMALIZE = normalize
    _WORKER_NORMALIZE_WINDOW = normalize_window


def _score_one_ticker_in_worker(
    ticker: str,
    hist_data: pd.DataFrame,
    weights: Dict[str, float],
    regime_label: Optional[str] = None,
    drift_prior: Optional[CrossSectionalDriftPrior] = None,
) -> Optional[StrategySignal]:
    """Worker-side entry point: only the per-day varying arguments travel."""
    if _WORKER_STRATEGY is None or _WORKER_RISK is None:
        raise RuntimeError("Scoring worker was not initialized")
    mc_settings = _WORKER_MC_SETTINGS
    if drift_prior is not None:
        mc_settings = mc_settings.with_drift_prior(drift_prior)
    return _score_one_ticker(
        _WORKER_STRATEGY, ticker, hist_data, _WORKER_FEATURES,
        _WORKER_RISK, weights, mc_settings, regime_label,
        _WORKER_NORMALIZE, _WORKER_NORMALIZE_WINDOW,
    )


class BacktestEngine:
    """
    Event-Driven Backtesting Engine.

    Simulates historical trading with strict look-ahead bias prevention.
    At date T the agent sees bars through T's close and trades at T+1's open —
    the live sequence. See `_history_through` for why that is the decision
    point, and for what stops it being look-ahead.
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
        mc_method: str = "gaussian",
        mc_block_size_days: int = 5,
        mc_jump_intensity_per_year: float = 12.0,
        mc_jump_mean: float = -0.02,
        mc_jump_volatility: float = 0.05,
        use_kelly_sizing: bool = False,
        kelly_fraction: float = MAX_KELLY_FRACTION,
        kelly_min_trades: int = 50,
        kelly_shrinkage_strength: float = 20.0,
        max_sector_pct: float = 0.0,
        max_unknown_sector_pct: float = 0.30,
        sector_map_csv: Optional[str] = None,
        max_portfolio_drawdown_pct: float = 0.0,
        drawdown_reentry_pct: float = 0.10,
        drawdown_halt_max_days: int = 60,
        benchmark_symbol: Optional[str] = None,
        exit_on_lower_circuit_lock: bool = True,
        liquidate_on_drawdown_halt: bool = False,
        portfolio_volatility_target: float = 0.0,
        covariance_lookback_days: int = 252,
        covariance_min_observations: int = 60,
        covariance_max_candidates: int = 40,
        show_progress: bool = False,
        # Defaults match what this engine did before it had the setting at all:
        # it called `build_features` with no `normalize`, taking the pipeline's
        # own default of False. The evaluation harness and both trainers read
        # `features.normalize`, whose *schema* default is True — so the day
        # anyone relies on that default, those paths z-score their features and
        # this one does not. Worse, `_normalize_features` applies an additional
        # `.shift(1)`, so the divergence is not a scale difference but a second
        # session of lag: exactly the disagreement T19 removed, re-created by a
        # config value. `agents/backtester.py` maps both from AppConfig.
        feature_normalize: bool = False,
        feature_normalize_window: int = 252,
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
            mc_method: Monte Carlo shock process — "gaussian", "block_bootstrap"
                or "jump_diffusion" (see src/monte_carlo.py).
            mc_block_size_days: Mean block length for the block bootstrap.
            mc_jump_intensity_per_year: Expected jumps/year for jump diffusion.
            mc_jump_mean: Mean log jump size for jump diffusion.
            mc_jump_volatility: Std dev of log jump size for jump diffusion.
            use_kelly_sizing: If True, size positions with fractional-Kelly
                once enough realized trades exist in this run's trade_log,
                falling back to fixed-fractional sizing otherwise.
            kelly_fraction: Fractional-Kelly multiplier kappa, hard-capped at
                MAX_KELLY_FRACTION (quarter-Kelly) inside calculate_kelly_quantity.
            kelly_min_trades: Minimum realized trades required before Kelly sizing is trusted.
            kelly_shrinkage_strength: Beta-prior strength shrinking the realized
                win rate toward 0.5 before it reaches Kelly (see src/risk.py).
            max_unknown_sector_pct: Aggregate cap on tickers missing from the
                sector map, so an incomplete map is not a way around the cap.
            max_sector_pct: Cap on any one sector's share of portfolio value.
                0 (the default here) disables the cap; BacktesterAgent passes
                the configured value through.
            sector_map_csv: Path to the ticker,sector CSV backing that cap.
            max_portfolio_drawdown_pct: Circuit breaker — stop opening new
                positions once drawdown from the equity peak reaches this
                fraction. 0 disables it.
            drawdown_reentry_pct: Drawdown level at which buying resumes.
            drawdown_halt_max_days: Trading days after which a halted breaker
                re-arms regardless of recovery, resetting the equity peak to
                current equity. Without it the breaker deadlocks — see
                _update_circuit_breaker. 0 disables the cooldown and restores
                the recovery-only behaviour.
            benchmark_symbol: Optional market index (e.g. "^NSEI") whose cached
                history drives the momentum crash filter's trend and
                volatility tests. Without it the filter falls back to a
                composite of the traded universe.
            exit_on_lower_circuit_lock: Queue an immediate exit when a holding
                closes pinned at its lower circuit. A locked-down stock cannot
                be sold at the modelled stop, so waiting for that stop is
                waiting for a fill that will not come at that price; the exit
                is queued for the next session, which is the earliest a real
                order could work. On by default — this is a correction to how
                the modelled stop behaves, not a strategy opinion.
            liquidate_on_drawdown_halt: Also sell every open position when the
                drawdown circuit breaker trips, rather than only suppressing
                new BUYs. Off by default and deliberately so: force-liquidating
                a whole book at a drawdown trough converts a bad quarter into a
                permanent loss, and existing positions already carry stops.
                Turn it on for mandates where a hard equity floor outranks
                recovery potential.
            show_progress: Draw tqdm progress bars for the two phases that take
                real time — loading the universe and replaying the trading days.
                Defaults to False so library and test callers stay silent; the
                CLI turns it on.
        """
        self.show_progress = show_progress
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
        self.mc_settings = MonteCarloSettings(
            horizon_days=mc_horizon_days,
            simulations=mc_simulations,
            seed=mc_seed,
            use_garch_volatility=use_garch_volatility,
            method=mc_method,
            block_size_days=mc_block_size_days,
            jump_intensity_per_year=mc_jump_intensity_per_year,
            jump_mean=mc_jump_mean,
            jump_volatility=mc_jump_volatility,
        )
        self.use_kelly_sizing = use_kelly_sizing
        self.kelly_fraction = kelly_fraction
        self.kelly_min_trades = kelly_min_trades
        self.kelly_shrinkage_strength = kelly_shrinkage_strength

        # Sector concentration cap, loaded once per run. Without a map the cap
        # is inactive rather than applied to a single UNKNOWN pool — pooling
        # would cap total invested capital, not sector concentration.
        self.max_sector_pct = max_sector_pct
        self.max_unknown_sector_pct = max_unknown_sector_pct
        self.sector_map = load_sector_map(sector_map_csv) if max_sector_pct > 0 else {}
        if max_sector_pct > 0 and not sector_cap_is_enforceable(self.sector_map):
            logger.warning(
                "risk.max_sector_pct is set to %.0f%% but no sector map was loaded from %s, "
                "so sector concentration is NOT being limited. Provide a ticker,sector CSV "
                "to enable it.",
                max_sector_pct * 100, sector_map_csv or "data/sector_map.csv",
            )

        # Benchmark index for the momentum crash filter. Loaded once from the
        # same parquet cache as everything else; None when it was never
        # cached, in which case the filter uses its composite fallback.
        self.benchmark_symbol = benchmark_symbol
        self.benchmark_close: Optional[pd.Series] = None
        # The full OHLC frame, kept alongside the closes because ADX — and so
        # the SIDEWAYS_CHOP classification — needs the daily range.
        self.benchmark_ohlcv: Optional[pd.DataFrame] = None
        if benchmark_symbol:
            benchmark_df = load_ticker_data(
                benchmark_symbol,
                start_date=None,  # the trend window reaches back before start_date
                end_date=self.end_date.strftime('%Y-%m-%d'),
            )
            if benchmark_df is not None and 'close' in benchmark_df.columns:
                self.benchmark_close = benchmark_df['close'].sort_index()
                self.benchmark_ohlcv = benchmark_df.sort_index()
                logger.info(
                    f"Loaded benchmark {benchmark_symbol} "
                    f"({len(self.benchmark_close)} bars) for the market-regime filter"
                )
            else:
                logger.info(
                    f"Benchmark {benchmark_symbol} is not cached; the market-regime filter "
                    f"will use a composite of the traded universe"
                )

        # Drawdown circuit breaker state.
        self.max_portfolio_drawdown_pct = max_portfolio_drawdown_pct
        self.drawdown_reentry_pct = drawdown_reentry_pct
        self.drawdown_halt_max_days = drawdown_halt_max_days
        self.equity_peak = initial_capital
        self.buying_halted = False
        self.halted_since_day: Optional[int] = None
        self.circuit_breaker_log: List[Dict[str, Any]] = []

        # Forced-exit triggers (see the constructor docstring).
        self.exit_on_lower_circuit_lock = exit_on_lower_circuit_lock
        self.liquidate_on_drawdown_halt = liquidate_on_drawdown_halt
        self.exit_trigger_log: List[Dict[str, Any]] = []

        # Portfolio-level risk. The *measurement* runs unconditionally — the
        # gap between true and independence-assumed volatility is the finding,
        # and a report that omits it is the state this platform was already in.
        # The *constraint* is opt-in, because choosing a volatility target is a
        # risk-policy decision rather than a bug fix.
        self.portfolio_volatility_target = portfolio_volatility_target
        self.covariance_lookback_days = covariance_lookback_days
        self.covariance_min_observations = covariance_min_observations
        self.covariance_max_candidates = covariance_max_candidates
        self._returns_cache: Dict[str, Optional[pd.Series]] = {}
        self.book_risk_log: List[Dict[str, Any]] = []


        # Track untradeable tickers (delisted or NaN issues) - initialize BEFORE loading data
        self.untradeable_tickers: set = set()

        # Filled on first use by `_required_history_rows`; depends only on the
        # strategy's declared features, which do not change across a run.
        self._required_rows: Optional[int] = None
        self.feature_normalize = bool(feature_normalize)
        self.feature_normalize_window = int(feature_normalize_window)

        # Load *before* the backtest window, so the first scored session has
        # warm features instead of NaNs. This used to load from `start_date`,
        # so a run beginning 2023-01-02 had no bar before it — `sma_200` was
        # undefined for the first 200 sessions no matter how much history the
        # cache held, and `mom_9m_skip1m` for the first 211. The old 20-row
        # eligibility bar let those tickers through, which is how the opening
        # months of every backtest came to rank the universe on NaN.
        #
        # 1.6x converts sessions to calendar days with room for weekends
        # (7/5 = 1.4) and Indian exchange holidays; over-reading costs one
        # slice of a parquet file, under-reading costs the warm-up itself.
        self._load_from = self.start_date - pd.Timedelta(
            days=int(self._required_history_rows() * 1.6) + 7
        )
        
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
    
    def _progress(self, iterable, desc: str, unit: str, total: Optional[int] = None):
        """Wrap `iterable` in a tqdm bar, or return it untouched when disabled.

        `disable=None` is tqdm's "auto" mode: the bar draws on a terminal and
        suppresses itself when output is redirected to a file or captured by a
        test, so enabling progress never pollutes a log.
        """
        return tqdm(
            iterable,
            desc=desc,
            unit=unit,
            total=total,
            disable=None if self.show_progress else True,
            dynamic_ncols=True,
            mininterval=0.5,
        )

    def _load_all_data(self) -> None:
        """Load all ticker data into memory using data_store.load_ticker_data.

        Progress is a tqdm bar rather than periodic ``logger.info`` calls: the
        CLI never configures logging, so those lines went to a root logger with
        no handler and the user watched a silent terminal for the several
        minutes it takes to read a 4000-ticker universe off disk.
        """
        total_tickers = len(self.universe_tickers)

        loaded_count = 0
        ticker_iter = self._progress(
            self.universe_tickers,
            desc="Loading ticker data",
            unit="ticker",
            total=total_tickers,
        )
        for ticker in ticker_iter:
            df = load_ticker_data(ticker,
                                  start_date=self._load_from.strftime('%Y-%m-%d'),
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
                logger.debug(f"No data available for {ticker}, marking as untradeable")

        message = f"Loaded {loaded_count}/{total_tickers} tickers with usable history"
        logger.info(message)
        if self.show_progress:
            print(message)
    
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
    
    def _required_history_rows(self) -> int:
        """Rows a ticker needs before this strategy's features are all defined.

        Cached on the instance: it depends only on `required_features()`, which
        does not change across a run, and the underlying probe is itself
        cached. Recomputing per date would be a feature evaluation per session
        for no new information.
        """
        if self._required_rows is None:
            from portfolio_agent.features.pipeline import warmup_rows

            # `strategy` is optional on this constructor, and a run without one
            # scores nothing — so it needs no warm-up rather than a guessed
            # default that would quietly widen every load.
            features = (
                self.strategy.required_features() if self.strategy is not None else []
            )
            self._required_rows = max(warmup_rows(features), 1)
        return self._required_rows

    def _history_through(self, ticker: str, decision_date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Bars up to and **including** `decision_date`.

        Renamed from `_get_historical_data_up_to`, which sliced strictly
        *before* the decision date. That was one session tighter than every
        other path in the package and it was not a look-ahead guard — it was an
        off-by-one against this engine's own loop.

        Signals are generated in step E, after step D has already marked the
        book to market at `decision_date`'s close. The engine is standing at
        the end of that session; the close it just valued the portfolio with is
        knowable. Orders from these signals fill at the *next* session's open
        (`_create_pending_orders` -> `_execute_pending_orders`), so including
        the decision date's bar reproduces the live sequence — decide after the
        close, trade the next morning — rather than deciding on stale data and
        then waiting an extra day to act on it.

        What actually prevents look-ahead is unchanged and lives elsewhere:
        every feature in `features/technical.py` shifts its own inputs by a
        bar, so a value at row `t` is a function of bars before `t`. The one
        deliberate exception is `close`, which is the decision date's reference
        price and is knowable at that session's close by construction.

        `evaluation/harness.py` slices `frame.loc[:date]` — inclusive — and the
        trainers label row `t` with a forward return measured from `t`. This
        now matches both, which is what lets a rank IC measured by `evaluate`
        describe the signal this engine trades.
        """
        if ticker in self.untradeable_tickers:
            return None

        if ticker not in self.ticker_data:
            return None

        df = self.ticker_data[ticker]
        return df[df.index <= decision_date].copy()
    
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
            
            # Get open, high and low for the day. The open matters as much as
            # the range: it is the first price at which anything could have
            # filled, so it bounds what a gapped stop or target could achieve.
            high = None
            low = None
            open_price = None
            for h_col in ['high', 'High']:
                if h_col in row.index:
                    high = row[h_col]
                    break
            for l_col in ['low', 'Low']:
                if l_col in row.index:
                    low = row[l_col]
                    break
            for o_col in ['open', 'Open']:
                if o_col in row.index:
                    open_price = row[o_col]
                    break

            if high is None or low is None:
                continue
            # A missing or unusable open falls back to the level itself, which
            # is the old behaviour — better to keep the optimistic fill than to
            # invent a gap out of a NaN.
            if open_price is None or pd.isna(open_price) or open_price <= 0:
                open_price = None
            
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
            # Signed distance between the realized fill and the level it was
            # set at, as a percentage of that level: negative when a gap
            # carried a stop past its price, positive when one carried a target
            # past its own, and zero whenever the level itself was available.
            # Recorded rather than folded silently into P&L — the P&L alone
            # cannot distinguish a gapped exit from a clean one, and how often
            # the modelled stop was actually reachable is the thing worth
            # measuring.
            gap_fill_delta_pct = 0.0

            # **Gap-aware fills.** A stop is a resting order, not a guaranteed
            # price: it becomes marketable when the price reaches it and fills
            # at whatever is available then. Booking every stop at exactly
            # `stop_price` assumes a fill that existed only if the level was
            # crossed *during* the session.
            #
            # When the market gaps through the level overnight — the open is
            # already past it — the first available price is the open, and the
            # fill is worse than the stop by the whole gap. This engine booked
            # the stop price regardless, so every gapped exit reported a loss
            # smaller than the one actually taken, and the error is not rare
            # here: NSE opens after both the US close and the Asian session,
            # which is why the platform models overnight gap variance
            # separately in the first place (volatility_models.py).
            #
            # The error also feeds back into sizing: Kelly reads its loss
            # magnitude `l` from realized history
            # (src/risk.py::estimate_kelly_inputs), so an understated `l`
            # inflates f* and the book bets larger precisely because it has
            # been mis-measuring its worst outcomes.
            #
            # The same logic runs the other way for a take-profit: a gap
            # *through* the target fills at the open, above the target, so
            # booking the target understates the gain. Both are corrected, and
            # the stop is checked first so a session that touches both levels
            # is charged the adverse one.
            if stop_price is not None and low <= stop_price:
                triggered = True
                trigger_price = min(open_price, stop_price) if open_price else stop_price
                trigger_type = 'STOP_LOSS'
                if stop_price > 0:
                    gap_fill_delta_pct = (trigger_price - stop_price) / stop_price * 100.0

            elif target_price is not None and high >= target_price:
                triggered = True
                trigger_price = max(open_price, target_price) if open_price else target_price
                trigger_type = 'TAKE_PROFIT'
                if target_price > 0:
                    gap_fill_delta_pct = (trigger_price - target_price) / target_price * 100.0

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
                    'exit_reason': exit_reason,
                    'gap_fill_delta_pct': gap_fill_delta_pct,
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
        Run the configured strategy's signal generation on data through T.

        CRITICAL: uses bars through `current_date` and nothing after it. The
        orders these signals produce fill at T+1's open, so T's close is
        knowable here; `_history_through` states the full argument.

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
        # Derived from what this strategy asked for, not a constant. The bar
        # here was 20 rows, against which six of the ten features `momentum`
        # requires are still NaN — `mom_9m_skip1m` among them, which is the
        # value it ranks on. Neither 20 nor the harness's 252 was wrong so much
        # as unrelated to the question; `warmup_rows` answers the question.
        required_rows = self._required_history_rows()
        eligible: Dict[str, pd.DataFrame] = {}
        for ticker in self.universe_tickers:
            if ticker in self.untradeable_tickers:
                continue
            hist_data = self._history_through(ticker, current_date)
            if hist_data is None or len(hist_data) < required_rows:
                continue
            eligible[ticker] = hist_data

        if not eligible:
            return {}

        weights = dict(self.agent_brain.weights)

        # The drift prior is estimated from this round's cross-section, and
        # `eligible` is exactly the point-in-time panel: every frame in it came
        # from _get_historical_data_up_to(ticker, current_date), so nothing
        # here has seen T. Re-estimated per date rather than once per run for
        # the same reason — a prior fitted over the whole backtest would carry
        # the future into every day's posterior.
        mc_settings = self.mc_settings.with_drift_prior_from_panel(
            history['close'].pct_change().dropna().to_numpy()
            for history in eligible.values()
        )

        if self.strategy.supports_gpu_batch or self.strategy.requires_full_batch:
            features_by_symbol = {
                ticker: build_features(
                    hist_data,
                    self.strategy.required_features(),
                    normalize=self.feature_normalize,
                    normalize_window=self.feature_normalize_window,
                )
                for ticker, hist_data in eligible.items()
            }
            benchmark_close = self._benchmark_up_to(current_date)
            benchmark_ohlcv = self._benchmark_ohlcv_up_to(current_date)
            context = StrategyContext(
                risk=self.risk_params,
                weights=weights,
                # Through T, exactly like every other input: the regime state
                # and the signals it gates must be read off the same session.
                benchmark_close=benchmark_close,
                benchmark_ohlcv=benchmark_ohlcv,
                # Classified once per round and shared, so every model in the
                # round is arbitrated against the same market state.
                regime_label=self._classify_regime(
                    benchmark_close, benchmark_ohlcv, eligible
                ),
            )
            return self.strategy.score_batch(features_by_symbol, context)

        # Classified once for the whole round, so the per-ticker path gates
        # models on the same market state the batched path does.
        regime_label = self._classify_regime(
            self._benchmark_up_to(current_date),
            self._benchmark_ohlcv_up_to(current_date),
            eligible,
        )

        if self.parallel and len(eligible) > 1:
            parallel_signals = self._score_tickers_parallel(
                eligible, weights, regime_label, mc_settings.drift_prior
            )
            if parallel_signals is not None:
                return parallel_signals

        required_features = self.strategy.required_features()

        signals: Dict[str, StrategySignal] = {}
        for ticker, hist_data in eligible.items():
            result = _score_one_ticker(
                self.strategy, ticker, hist_data, required_features,
                self.risk_params, weights, mc_settings, regime_label,
            )
            if result is not None:
                signals[ticker] = result
        return signals

    def _benchmark_up_to(self, current_date: pd.Timestamp) -> Optional[pd.Series]:
        """Benchmark closes through `current_date`, inclusive.

        Same rule as `_history_through`, and for the same reason: the regime
        filter decides at the end of the session, so the index close it is
        deciding against is knowable. Slicing this one exclusively while the
        ticker panel was inclusive would put the regime state a session behind
        the signals it gates.
        """
        if self.benchmark_close is None:
            return None
        truncated = self.benchmark_close[self.benchmark_close.index <= current_date]
        return truncated if not truncated.empty else None

    def _benchmark_ohlcv_up_to(self, current_date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Benchmark OHLC bars through `current_date`, inclusive.

        Kept in step with `_benchmark_up_to`: the two are handed to the same
        regime call, and a close series one bar longer than its own high/low
        range would have `assess_market_regime` reading a trend from one
        session and an ADX from the session before it.
        """
        if self.benchmark_ohlcv is None:
            return None
        truncated = self.benchmark_ohlcv[self.benchmark_ohlcv.index <= current_date]
        return truncated if not truncated.empty else None

    def _classify_regime(
        self,
        benchmark_close: Optional[pd.Series],
        benchmark_ohlcv: Optional[pd.DataFrame],
        eligible: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Optional[str]:
        """Name the market state for this scoring round.

        Derived here, once, rather than inside each strategy: a UMA whose
        members are gated by regime and whose cross-sectional members each
        assess their own regime could otherwise be muted against one
        classification while sizing against another.

        **The composite fallback is what keeps the regime map from going
        silently inert.** A cached index is the better gauge and is used
        whenever it exists, but requiring one means that on any installation
        without `^NSEI` in the cache — which is every installation until
        somebody runs `download-data` against an index directory — the
        classification is UNKNOWN, every model is permitted in every state, and
        the meta-orchestrator's whole point quietly evaporates with nothing in
        the logs to say so. src/regime.py is built to derive the market state
        from the traded universe itself for exactly this reason, so the same
        equal-weighted composite the cross-sectional strategies already fall
        back to is used here too.

        Returns None only when neither an index nor a usable composite is
        available, which the meta-orchestrator reads as "permit every model"
        rather than as a reason to stand the book down.
        """
        if benchmark_close is not None:
            return assess_market_regime(
                benchmark_close, market_ohlcv=benchmark_ohlcv
            ).classification

        if not eligible:
            return None

        # Only the trailing trend window can affect either test, and this runs
        # once per trading day over the full universe.
        proxy = build_market_proxy(
            {
                ticker: history['close']
                for ticker, history in eligible.items()
                if 'close' in history.columns and not history.empty
            },
            lookback=DEFAULT_TREND_WINDOW + 1,
        )
        if proxy is None:
            return None
        # No high/low on a composite, so ADX uses its close-only proxy.
        return assess_market_regime(proxy).classification

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
                    self.mc_settings,
                    self.feature_normalize,
                    self.feature_normalize_window,
                ),
            )
        return self._scoring_executor

    def shutdown_workers(self) -> None:
        """Tear down the scoring process pool, if one was started."""
        if self._scoring_executor is not None:
            self._scoring_executor.shutdown(wait=True)
            self._scoring_executor = None

    def _score_tickers_parallel(
        self,
        eligible: Dict[str, pd.DataFrame],
        weights: Dict[str, float],
        regime_label: Optional[str] = None,
        drift_prior: Optional[CrossSectionalDriftPrior] = None,
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

        `drift_prior` rides with each task rather than through the pool
        initializer: the pool is created once per backtest, but the prior is
        re-estimated every scoring round, so installing it at worker startup
        would pin the whole run to day one's cross-section. It is four floats,
        so per-task pickling costs nothing next to the history frame already
        being shipped.
        """
        try:
            executor = self._get_scoring_executor()
            futures = {
                ticker: executor.submit(
                    _score_one_ticker_in_worker, ticker, hist_data, weights,
                    regime_label, drift_prior,
                )
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

                    stop_level, target_level = self._exit_levels(order, adjusted_price)
                    self.stop_loss_levels[ticker] = stop_level
                    self.take_profit_levels[ticker] = target_level
                    
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
    
    @staticmethod
    def _net_return_pct(trade: Dict[str, Any]) -> float:
        """A closed trade's return on cost basis, net of costs and taxes.

        `net_pnl` already has brokerage, STT, exchange and SEBI charges, GST,
        stamp duty and capital gains tax deducted, so this is the return the
        portfolio actually kept — the only version Kelly's payoff ratio should
        be estimated from.
        """
        entry_price = float(trade.get("entry_price") or 0.0)
        quantity = float(trade.get("quantity") or 0.0)
        cost_basis = entry_price * quantity
        if cost_basis <= 0:
            return 0.0
        return float(trade.get("net_pnl", 0.0)) / cost_basis * 100.0

    def _kelly_quantity(self, entry_price: float, stop_price: Optional[float] = None) -> int:
        """Fractional-Kelly position size from this run's realized trade_log so far.

        Returns 0 (triggering the caller's fixed-fractional fallback) when
        there aren't yet enough realized trades to estimate Kelly inputs
        reliably (see risk.py::estimate_kelly_inputs).

        Args:
            entry_price: Entry price per share.
            stop_price: This signal's stop, used to cap the Kelly size at the
                per-trade risk budget. Omitted (None) leaves only the
                concentration cap in force.
        """
        # Only *closed* round trips are realized outcomes. Open BUY legs carry
        # net_pnl = -transaction_costs, so counting them here classified every
        # open position as a loss and dragged the Kelly win probability down.
        # Both Kelly inputs are measured net of friction. The trade log's
        # `return_pct` is a GROSS figure (gross_pnl / cost basis), so feeding
        # it to Kelly would pair a net win/loss classification with a gross
        # payoff ratio b — and b is what f* is most sensitive to after p. On a
        # trade whose gross reward:risk is 2.0, round-trip costs take the
        # realized ratio nearer 1.8, so a gross b systematically overstates
        # the edge and Kelly over-bets on it.
        realized = [
            {
                "outcome": "WIN" if t.get("net_pnl", 0.0) > 0 else "LOSS",
                "return_pct": self._net_return_pct(t),
            }
            for t in self.trade_log
            if t.get("exit_date") is not None
        ]
        kelly_inputs = estimate_kelly_inputs(
            realized,
            min_trades=self.kelly_min_trades,
            shrinkage_strength=self.kelly_shrinkage_strength,
        )
        if kelly_inputs is None:
            return 0
        kelly_quantity = calculate_kelly_quantity(
            entry_price=entry_price,
            portfolio_value_inr=self.portfolio_value,
            max_single_position_pct=self.risk_params.max_single_position_pct,
            win_probability=kelly_inputs.win_probability,
            avg_win_pct=kelly_inputs.avg_win_pct,
            avg_loss_pct=kelly_inputs.avg_loss_pct,
            kelly_fraction=self.kelly_fraction,
        )

        # Kelly is a ceiling, not a licence — the same contract as
        # risk.py::calculate_position_quantity. With the allocation units
        # corrected, f* is scaled up by 1/l (~17x at a 6% stop), so without the
        # per-trade risk budget alongside it the fix would land as an
        # across-the-board increase in position size. The budget is priced off
        # this signal's own stop distance, which is known exactly, rather than
        # Kelly's historical average loss.
        if stop_price is not None and stop_price > 0:
            risk_per_share = entry_price - stop_price
            if risk_per_share <= 0:
                return 0
            risk_budget = self.portfolio_value * self.risk_params.risk_per_trade_pct
            kelly_quantity = min(kelly_quantity, math.floor(risk_budget / risk_per_share))

        return max(0, kelly_quantity)

    def _update_circuit_breaker(self, current_date: pd.Timestamp) -> None:
        """Track the equity peak and trip/re-arm the drawdown circuit breaker.

        Once drawdown from the running peak reaches max_portfolio_drawdown_pct,
        no new positions are opened until it recovers to within
        drawdown_reentry_pct of the peak. Two distinct thresholds, not one:
        with a single level the breaker would flip on and off every time equity
        wobbled across it, churning the book precisely when conditions are
        worst. Open positions are left alone — their stops and targets are
        already the exit plan, and force-liquidating a whole book at a
        drawdown trough is how a bad quarter becomes a permanent loss.

        **The cooldown is not optional.** Recovery-only re-arming deadlocks,
        and does so silently. The breaker halts buying; the open positions then
        exit through their own stops and targets; the book is now entirely
        cash. Cash does not appreciate, so equity is frozen at its trough
        forever — permanently below the re-entry threshold, which is measured
        against a peak it can no longer approach. A 5-year backtest observed
        exactly this: the breaker tripped 15% down in month 7 and the strategy
        sat in cash for the remaining four years, which reads in the report as
        a flat equity curve rather than as a stuck flag.

        After `drawdown_halt_max_days` trading days halted, the breaker
        therefore re-arms regardless and **resets the peak to current equity**.
        Resetting matters as much as re-arming: leaving the old peak in place
        would put the very next bar straight back over the trip threshold. The
        reset is an admission of what actually happened — this is the capital
        the strategy has now, and the drawdown to measure from here is the one
        that starts today.
        """
        if self.max_portfolio_drawdown_pct <= 0 or self.max_portfolio_drawdown_pct >= 1:
            return

        self.equity_peak = max(self.equity_peak, self.portfolio_value)
        if self.equity_peak <= 0:
            return

        drawdown = (self.equity_peak - self.portfolio_value) / self.equity_peak

        def _log(event: str, note: str) -> Dict[str, Any]:
            entry = {
                'date': current_date.strftime('%Y-%m-%d'),
                'event': event,
                'drawdown_pct': round(drawdown * 100, 2),
                'portfolio_value': round(self.portfolio_value, 2),
                'peak_value': round(self.equity_peak, 2),
                'note': note,
            }
            self.circuit_breaker_log.append(entry)
            return entry

        if not self.buying_halted and drawdown >= self.max_portfolio_drawdown_pct:
            self.buying_halted = True
            self.halted_since_day = self.trading_day_count
            entry = _log('HALT', 'drawdown threshold reached')
            logger.warning(
                f"Drawdown circuit breaker tripped on {entry['date']}: "
                f"{entry['drawdown_pct']:.2f}% below peak; halting new BUY orders"
            )
            return

        if not self.buying_halted:
            return

        if drawdown <= self.drawdown_reentry_pct:
            self.buying_halted = False
            self.halted_since_day = None
            entry = _log('RESUME', 'recovered to the re-entry threshold')
            logger.info(
                f"Drawdown circuit breaker re-armed on {entry['date']}: "
                f"recovered to {entry['drawdown_pct']:.2f}% below peak; buying resumed"
            )
            return

        # `is None`, not `or`: trading_day_count starts at 0, and a falsy check
        # would read a breaker that tripped on day zero as never having tripped,
        # leaving the cooldown permanently unsatisfiable.
        started = self.halted_since_day if self.halted_since_day is not None else self.trading_day_count
        halted_days = self.trading_day_count - started
        if self.drawdown_halt_max_days > 0 and halted_days >= self.drawdown_halt_max_days:
            self.buying_halted = False
            self.halted_since_day = None
            entry = _log(
                'RESUME',
                f'cooldown of {self.drawdown_halt_max_days} trading days elapsed; '
                f'peak reset to current equity',
            )
            # Reset last: the log entry above should report the drawdown that
            # justified the cooldown, not the 0% the reset creates.
            self.equity_peak = self.portfolio_value
            logger.info(
                f"Drawdown circuit breaker re-armed on {entry['date']} after "
                f"{halted_days} halted trading days at {entry['drawdown_pct']:.2f}% "
                f"below peak; peak reset and buying resumed"
            )

    def _is_locked_down(self, ticker: str, current_date: pd.Timestamp) -> bool:
        """Whether `ticker` closed pinned at its lower circuit on `current_date`.

        Read from the raw bar rather than a feature column, because this runs
        against the holdings book — which contains positions in names that may
        have dropped out of the scored universe entirely.
        """
        df = self.ticker_data.get(ticker)
        if df is None or current_date not in df.index:
            return False

        position = df.index.get_loc(current_date)
        if not isinstance(position, int) or position < 1:
            return False

        # Two bars are the minimum: the lock is defined against the prior close.
        window = df.iloc[position - 1:position + 1]
        window = window.rename(columns={c: c.lower() for c in window.columns})
        try:
            return bool(lower_circuit_locked_days(window).iloc[-1])
        except Exception:
            return False

    def _exit_trigger_reasons(self, current_date: pd.Timestamp) -> Dict[str, str]:
        """Holdings that must be exited regardless of what the strategy says.

        Two conditions, both of which invalidate the exit plan a position was
        opened with rather than merely arguing against holding it:

        - **Locked at the lower circuit.** The modelled stop assumes a fill is
          available somewhere near it. On a lock there is no bid, so the stop is
          not a stop — it is a hope. Every further locked session realizes a
          loss the sizing never priced, which is precisely the asymmetry that
          biases the measured payoff ratio Kelly reads. Exiting is queued for
          the next session, the earliest a real order could work.
        - **The drawdown breaker has tripped** and this run is configured to
          liquidate on it (off by default; see the constructor docstring).

        Returns:
            ticker -> reason, for holdings with a live position.
        """
        reasons: Dict[str, str] = {}
        if not self.holdings:
            return reasons

        if self.liquidate_on_drawdown_halt and self.buying_halted:
            for ticker, quantity in self.holdings.items():
                if quantity > 0:
                    reasons[ticker] = "portfolio drawdown circuit breaker tripped"
            return reasons

        if self.exit_on_lower_circuit_lock:
            for ticker, quantity in self.holdings.items():
                if quantity > 0 and self._is_locked_down(ticker, current_date):
                    reasons[ticker] = "locked at the lower circuit; modelled stop is unfillable"

        return reasons

    def _ticker_returns(self, ticker: str) -> Optional[pd.Series]:
        """Daily simple returns for one ticker, computed once and cached.

        Cached because the covariance estimate below is rebuilt on every
        rebalance round; recomputing pct_change over a multi-year frame per
        ticker per day is the difference between a covariance-aware backtest
        that finishes and one that does not.
        """
        if ticker in self._returns_cache:
            return self._returns_cache[ticker]

        frame = self.ticker_data.get(ticker)
        if frame is None or len(frame) < 2 or 'close' not in frame.columns:
            self._returns_cache[ticker] = None
            return None

        returns = frame['close'].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        self._returns_cache[ticker] = returns
        return returns

    def _book_covariance(
        self,
        tickers: List[str],
        current_date: pd.Timestamp,
    ) -> Optional[pd.DataFrame]:
        """Annualized covariance for a set of names, as of `current_date`.

        **Strictly backward-looking.** Only returns dated *before* the decision
        date are used — the whole point of a covariance-aware sizing decision
        is lost if the matrix knows how the names co-moved during the period it
        is sizing for.

        Shrunk toward a constant-correlation target (Ledoit-Wolf): over a
        `covariance_lookback_days` window the sample covariance of 20-60 names
        is poorly conditioned, and its extreme eigenvalues — the ones an
        optimizer or a volatility constraint leans on hardest — are mostly
        estimation noise.

        Returns None when too few names have enough overlapping history to say
        anything, in which case the caller leaves sizing alone rather than
        acting on a matrix it does not trust.
        """
        usable = {}
        for ticker in tickers:
            returns = self._ticker_returns(ticker)
            if returns is None:
                continue
            window = returns[returns.index < current_date].tail(self.covariance_lookback_days)
            if len(window) >= self.covariance_min_observations:
                usable[ticker] = window

        if len(usable) < 2:
            return None

        panel = pd.DataFrame(usable).dropna()
        if len(panel) < self.covariance_min_observations or panel.shape[1] < 2:
            return None

        covariance, _ = ledoit_wolf_covariance(panel, annualize=True)
        return covariance

    def _book_risk_summary(
        self,
        position_values: Dict[str, float],
        current_date: pd.Timestamp,
    ) -> Optional[Dict[str, float]]:
        """Portfolio-level risk of the current book, or None if unmeasurable.

        Reported whether or not the volatility cap is enabled, because the
        measurement is the point: the ratio between true portfolio volatility
        and the volatility independent per-trade sizing implies is exactly the
        factor by which the platform used to understate its own risk.
        """
        held = {t: v for t, v in position_values.items() if v > 0}
        if len(held) < 2 or self.portfolio_value <= 0:
            return None

        covariance = self._book_covariance(sorted(held), current_date)
        if covariance is None:
            return None

        names = [str(c) for c in covariance.columns]
        weights = np.array([held.get(n, 0.0) / self.portfolio_value for n in names])
        return summarize_book_risk(weights, covariance)

    def _apply_portfolio_risk_cap(
        self,
        ticker: str,
        quantity: int,
        price: float,
        position_values: Dict[str, float],
        current_date: pd.Timestamp,
        covariance: Optional[pd.DataFrame],
    ) -> int:
        """Trim a BUY so the whole book's forecast volatility stays under target.

        This is the constraint the platform never had. Every other limit here
        is per-position — risk per trade, the concentration cap, the sector cap
        — and none of them can see that twenty 3% positions correlated 0.6 carry
        3.5x the volatility twenty independent ones would.

        Portfolio variance as a function of the candidate's value v is a
        convex quadratic,

            sigma^2(v) = sigma_ii v^2 + 2 v * cov(i, rest) + sigma^2(0)

        so the feasible set {v : sigma(v) <= target} is an interval containing
        v = 0 whenever the existing book is already inside the target. That
        makes bisection on the upper endpoint exact rather than a heuristic —
        and when the book is *already* over target, the honest answer is to add
        nothing, which is what a zero here means.
        """
        if (
            self.portfolio_volatility_target <= 0
            or quantity <= 0
            or price <= 0
            or covariance is None
            or self.portfolio_value <= 0
        ):
            return quantity

        names = [str(c) for c in covariance.columns]
        if ticker not in names:
            # No usable history for this name, so its contribution to book
            # volatility is unknown. Leave the per-position limits to govern
            # rather than inventing a correlation for it.
            return quantity

        index = names.index(ticker)
        base = np.array([position_values.get(n, 0.0) / self.portfolio_value for n in names])
        candidate_weight = quantity * price / self.portfolio_value

        def volatility_with(extra: float) -> float:
            weights = base.copy()
            weights[index] += extra
            return portfolio_volatility(weights, covariance)

        if volatility_with(candidate_weight) <= self.portfolio_volatility_target:
            return quantity
        if volatility_with(0.0) > self.portfolio_volatility_target:
            logger.debug(
                f"Portfolio vol cap: skipping BUY {ticker} — book already at "
                f"{volatility_with(0.0):.1%} against a "
                f"{self.portfolio_volatility_target:.1%} target"
            )
            return 0

        low, high = 0.0, candidate_weight
        for _ in range(60):
            mid = 0.5 * (low + high)
            if volatility_with(mid) <= self.portfolio_volatility_target:
                low = mid
            else:
                high = mid

        return max(0, int(low * self.portfolio_value / price))

    def _current_position_values(self, current_date: pd.Timestamp) -> Dict[str, float]:
        """Mark every open holding to T's close, for sector-exposure accounting."""
        values: Dict[str, float] = {}
        for ticker, quantity in self.holdings.items():
            if quantity <= 0:
                continue
            price = self._get_price_at_date(ticker, current_date, 'close')
            if price is None or pd.isna(price):
                price = self._get_last_valid_price(ticker, current_date)
            if price is not None and not pd.isna(price):
                values[ticker] = float(quantity) * float(price)
        return values

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

        Three risk controls sit between a BUY signal and a queued order:

        1. The drawdown circuit breaker, which suppresses new positions
           entirely while it is tripped.
        2. `position_scale` from the signal (volatility targeting / market
           regime — see src/regime.py), which shrinks the sized quantity.
        3. The sector concentration cap, which trims or drops an order that
           would push its sector past risk.max_sector_pct. Because BUYs are
           processed in conviction order, the highest-scoring name in a sector
           gets the remaining capacity and later ones are trimmed — the same
           ordering guarantee the cash constraint already relies on.

        Args:
            signals: Dictionary of ticker -> StrategySignal.
            current_date: Current date (orders will execute at T+1 open).
        """
        execution_date = current_date + pd.Timedelta(days=1)

        # Skip weekends
        while execution_date.weekday() >= 5:
            execution_date += pd.Timedelta(days=1)

        # Forced exits are evaluated first and win any tie with a strategy
        # signal: they exist because the position's exit plan has stopped being
        # valid, which no amount of conviction in holding can restore. The
        # breaker is updated before this so a trip on today's equity can fire
        # today's liquidation rather than tomorrow's.
        self._update_circuit_breaker(current_date)
        forced_exits = self._exit_trigger_reasons(current_date)

        signal_sells = {
            t for t, s in signals.items() if s.signal == 'SELL' and t in self.holdings
        }
        # An order that could not fill yesterday (a market holiday, a missing
        # bar) is still queued. Re-queueing the same exit on each subsequent
        # day would stack sells against a position only large enough for one.
        already_queued = {
            o['ticker'] for o in self.pending_orders if o['action'] == 'SELL'
        }
        for ticker in sorted(forced_exits.keys() | signal_sells):
            quantity = self.holdings.get(ticker, 0)
            if quantity <= 0 or ticker in already_queued:
                continue
            reason = forced_exits.get(ticker)
            if reason is not None:
                trigger = 'EXIT_TRIGGER'
                self.exit_trigger_log.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'ticker': ticker,
                    'quantity': quantity,
                    'reason': reason,
                })
                logger.info(
                    f"Exit trigger on {current_date.strftime('%Y-%m-%d')} for {ticker}: {reason}"
                )
            else:
                trigger = signals[ticker].trigger or 'SIGNAL'

            self.pending_orders.append({
                'ticker': ticker,
                'action': 'SELL',
                'quantity': quantity,
                'execution_date': execution_date,
                'trigger': trigger,
            })

        if self.buying_halted:
            return

        buys = sorted(
            (t for t, s in signals.items() if s.signal == 'BUY' and t not in self.holdings),
            key=lambda t: (-signals[t].score, t),
        )

        # Sector exposure accumulates across this round's queued orders too, not
        # just already-open positions: five BUYs in one sector each fit under
        # the cap individually and blow straight through it together.
        position_values = self._current_position_values(current_date)

        # One covariance estimate per rebalance round, over the names that
        # could be in the book by the end of it, rather than one per candidate:
        # the matrix does not change between candidates, and refitting it
        # inside the loop is what would make this unaffordable. Candidates are
        # capped because `buys` can run to hundreds on a wide universe while
        # only the highest-scoring few ever clear the cash and sector limits.
        covariance = None
        if self.portfolio_volatility_target > 0:
            covariance = self._book_covariance(
                sorted(set(position_values) | set(buys[: self.covariance_max_candidates])),
                current_date,
            )

        for ticker in buys:
            sig = signals[ticker]
            price = sig.entry_price or 100
            quantity = (
                self._kelly_quantity(price, sig.stop_price) if self.use_kelly_sizing else 0
            )

            if quantity <= 0:
                # Default / Kelly-unavailable fallback: 10% of portfolio per position.
                position_value = self.portfolio_value * 0.10
                quantity = int(position_value / price)

            quantity = self._apply_position_scale(quantity, sig)
            quantity = self._apply_sector_cap(ticker, quantity, price, position_values)
            quantity = self._apply_portfolio_risk_cap(
                ticker, quantity, price, position_values, current_date, covariance
            )

            if quantity > 0:
                position_values[ticker] = position_values.get(ticker, 0.0) + quantity * price
                self.pending_orders.append({
                    'ticker': ticker,
                    'action': 'BUY',
                    'quantity': quantity,
                    'execution_date': execution_date,
                    'trigger': sig.trigger or 'SIGNAL',
                    # The signal's own exit plan travels with the order; see
                    # _exit_levels for why the distances rather than the levels.
                    'signal_entry_price': sig.entry_price,
                    'stop_price': sig.stop_price,
                    'target_price': sig.target_price,
                })

    # Exit plan used when a strategy supplies no usable stop or target.
    # Deliberately wide relative to the old hardcoded pair: these are a
    # fallback for a strategy that declined to specify an exit, not a
    # replacement for one that did.
    DEFAULT_STOP_FRACTION = 0.05
    DEFAULT_TARGET_FRACTION = 0.10

    def _exit_levels(self, order: Dict[str, Any], fill_price: float) -> tuple:
        """Stop and target for a filled BUY, from the signal that produced it.

        This used to be a hardcoded 5% stop and 10% target, which silently
        discarded every strategy's own exit plan. The consequences reached well
        past the exit itself, because the rest of the platform gates on the
        plan it thought was being used:

        - `atr_stop_multiplier` / `atr_target_multiplier` were dead config.
        - `min_reward_risk` screened signals on a net-of-cost reward:risk
          computed from ATR levels the engine then ignored — gating on one exit
          plan and trading another.
        - The quantile model's stop and target, derived from its predicted
          10th/90th percentiles, were thrown away along with them.
        - Kelly's payoff ratio b was estimated from trades exited under the 5/10
          rule while the signals feeding it were screened under the ATR rule.

        On Indian small- and mid-caps a flat 5% stop is roughly one session of
        noise, which is why backtests exited at a 2-day median holding period
        with 82% of exits at the stop.

        **Distances, not levels.** The signal's stop and target were computed
        off T's close; the fill lands at T+1's open plus slippage. Copying the
        absolute levels across would put a gapped-up entry immediately through
        its own target, so the *fractional* distances are preserved and
        re-applied to the price actually paid.

        Args:
            order: The pending order, carrying the signal's entry/stop/target.
            fill_price: Slippage-adjusted execution price.

        Returns:
            (stop_price, target_price).
        """
        signal_entry = float(order.get('signal_entry_price') or 0.0)
        signal_stop = float(order.get('stop_price') or 0.0)
        signal_target = float(order.get('target_price') or 0.0)

        stop_fraction = self.DEFAULT_STOP_FRACTION
        target_fraction = self.DEFAULT_TARGET_FRACTION
        if signal_entry > 0:
            if 0 < signal_stop < signal_entry:
                stop_fraction = (signal_entry - signal_stop) / signal_entry
            if signal_target > signal_entry:
                target_fraction = (signal_target - signal_entry) / signal_entry

        return (
            fill_price * (1.0 - stop_fraction),
            fill_price * (1.0 + target_fraction),
        )

    @staticmethod
    def _apply_position_scale(quantity: int, signal: StrategySignal) -> int:
        """Shrink a sized quantity by the signal's `position_scale`, if any.

        Strategies that measure their own risk environment (cross-sectional
        momentum's volatility targeting and market-regime filter) publish a
        multiplier in [0, 1] on the signal. Sizing honours it here rather than
        inside the strategy so every sizing rule — fixed-fractional, Kelly,
        or a future one — picks it up automatically. A signal that carries no
        scale is left untouched.
        """
        scale = signal.extra.get("position_scale") if signal.extra else None
        if scale is None:
            return quantity
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            return quantity
        if scale >= 1.0:
            return quantity
        return int(quantity * max(0.0, scale))

    def _apply_sector_cap(
        self,
        ticker: str,
        quantity: int,
        price: float,
        position_values: Dict[str, float],
    ) -> int:
        """Trim a BUY so its sector stays under risk.max_sector_pct."""
        if self.max_sector_pct <= 0 or self.max_sector_pct >= 1 or quantity <= 0 or price <= 0:
            return quantity

        capacity = sector_capacity_inr(
            ticker=ticker,
            portfolio_value_inr=self.portfolio_value,
            position_values=position_values,
            sector_map=self.sector_map,
            max_sector_pct=self.max_sector_pct,
            max_unknown_pct=self.max_unknown_sector_pct,
        )
        if math.isinf(capacity):
            # The cap does not apply to this ticker (disabled, or unmapped
            # sector). Return before int(inf / price) raises OverflowError.
            return quantity
        max_quantity = int(capacity / price)
        if max_quantity >= quantity:
            return quantity

        if max_quantity <= 0:
            logger.debug(
                f"Sector cap: skipping BUY {ticker} — sector "
                f"'{sector_of(ticker, self.sector_map)}' already at "
                f"{self.max_sector_pct:.0%} of portfolio value"
            )
        return max(0, max_quantity)


    def _evaluate_and_learn(
        self,
        learning_rate: float = 0.15,
        min_trades_for_learning: int = 3,
        min_trades_per_component: int = 30,
        shrinkage_strength: float = 20.0,
        significance_level: float = 0.05,
    ) -> None:
        """
        Trigger agent learning every N trading days.

        Uses the same pure weight-adaptation function (strategies/weighting.py)
        as the live orchestrator, so backtest and live learning stay identical.
        That identity is the point of the shared function, and it extends to
        the sample-size and significance guards: a backtest that adapted
        weights on evidence the live path would refuse to act on would be
        measuring a strategy the platform cannot run.
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
                min_trades_per_component=min_trades_per_component,
                shrinkage_strength=shrinkage_strength,
                significance_level=significance_level,
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
        5. Score the universe on bars through T — the close step 4 just used.
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

        day_bar = self._progress(
            self.master_date_index,
            desc="Backtesting",
            unit="day",
            total=len(self.master_date_index),
        )

        try:
            for i, current_date in enumerate(day_bar):
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

                # Step E: Signals on bars through T. Step D just marked the
                # book at T's close, so that bar is knowable here; the orders
                # step F queues fill at T+1's open.
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
                    self._record_book_risk(current_date)

                # The date and equity are what tell an operator the run is
                # progressing *and* whether it is going anywhere — a bar that
                # only counts days looks identical for a working strategy and
                # one that stopped trading in year one.
                if self.show_progress:
                    day_bar.set_postfix(
                        date=date_str,
                        equity=f"{self.portfolio_value:,.0f}",
                        open=len(self.holdings),
                        trades=len(self.trade_log),
                        refresh=False,
                    )
        finally:
            # Workers are per-run, so they are released here whether the run
            # finished or raised.
            self.shutdown_workers()
            # A bar left open corrupts every line printed after it, including
            # the traceback when the run is ending because something raised.
            day_bar.close()

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
            'daily_activity_log': self.daily_activity_log,
            'circuit_breaker_log': self.circuit_breaker_log,
            'exit_trigger_log': self.exit_trigger_log,
            'book_risk_log': self.book_risk_log,
            'book_risk_summary': self.book_risk_statistics(),
        }

    def _record_book_risk(self, current_date: pd.Timestamp) -> None:
        """Snapshot the book's portfolio-level risk, whether or not it is capped.

        Sampled on the same 20-day cadence as weight learning rather than
        daily: the covariance is estimated over a 252-day window and does not
        meaningfully change between adjacent sessions, so a daily estimate
        would cost twenty times as much to say the same thing.
        """
        summary = self._book_risk_summary(
            self._current_position_values(current_date), current_date
        )
        if summary is None:
            return
        self.book_risk_log.append({
            'date': current_date,
            'trading_day': self.trading_day_count,
            **summary,
        })

    def book_risk_statistics(self) -> Dict[str, float]:
        """Aggregate the book-risk snapshots into report-ready figures.

        `mean_correlation_risk_multiple` is the headline: it is how many times
        larger the book's actual volatility was than independent per-trade
        sizing implied. A value of 1 would mean correlation cost nothing; on a
        long-only Indian equity book it does not.
        """
        if not self.book_risk_log:
            return {}

        def _mean(key: str) -> float:
            values = [row[key] for row in self.book_risk_log if key in row]
            return float(np.mean(values)) if values else 0.0

        def _max(key: str) -> float:
            values = [row[key] for row in self.book_risk_log if key in row]
            return float(np.max(values)) if values else 0.0

        return {
            'observations': float(len(self.book_risk_log)),
            'mean_portfolio_volatility': _mean('portfolio_volatility'),
            'max_portfolio_volatility': _max('portfolio_volatility'),
            'mean_independent_volatility': _mean('independent_volatility'),
            'mean_correlation_risk_multiple': _mean('correlation_risk_multiple'),
            'max_correlation_risk_multiple': _max('correlation_risk_multiple'),
            'mean_diversification_ratio': _mean('diversification_ratio'),
            'mean_max_risk_contribution': _mean('max_risk_contribution'),
            'mean_positions': _mean('n_positions'),
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
