"""Main orchestrator module for portfolio agent."""

import logging
import math
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from portfolio_agent.config.schema import AppConfig
from portfolio_agent.config.loader import load_config as get_config
from portfolio_agent.features.pipeline import build_features
from portfolio_agent.strategies.registry import load_strategy
from portfolio_agent.strategies.types import RiskParams, StrategyContext, StrategySignal

from .storage import (
    init_db, save_recommendations,
    save_trade_outcome, log_run, get_trade_history,
    load_brain, save_brain
)
from portfolio_agent.src.data_store import load_or_fetch_data, load_ticker_data
from portfolio_agent.src.monte_carlo import MonteCarloResult, MonteCarloSettings
from portfolio_agent.src.risk import calculate_position_quantity, to_net_realized_trades
from portfolio_agent.src.execution_sim import cost_fraction_per_side
from portfolio_agent.src.compliance import run_compliance_checks
from portfolio_agent.src.learning import evaluate_and_learn
from portfolio_agent.src.regime import DEFAULT_TREND_WINDOW, assess_market_regime, build_market_proxy
from .reporting import export_excel_report
from portfolio_agent.src.models import IndicatorSnapshot, Recommendation
from .outcomes import simulate_outcome as simulate_outcome_fn, update_outcomes_from_market
from portfolio_agent.src.sectors import (
    load_sector_map, sector_cap_is_enforceable, sector_capacity_inr, sector_of,
)
from portfolio_agent.src.logging_utils import get_logger, ContextualLogger

# A plain module logger alongside the contextual one. The indicator-snapshot
# helpers moved in from the deleted src/indicators.py log per-ticker skips, and
# they run before any run_id exists to give a ContextualLogger.
logger = logging.getLogger(__name__)


def _setup_logging(log_file: str, run_id: str) -> ContextualLogger:
    """Setup logging configuration with contextual identifiers.

    Args:
        log_file: Path to the log file.
        run_id: Unique run identifier (UUID).

    Returns:
        ContextualLogger instance.
    """
    return get_logger(
        module_name='orchestrator',
        log_file=log_file,
        run_id=run_id,
        worker_id='main',
        level=logging.INFO
    )


def _prepare_one_ticker(
    ticker: str,
    df: pd.DataFrame,
    required_features: List[str],
    mc_settings: MonteCarloSettings,
):
    """Indicators + Monte Carlo + features for one ticker.

    Module-level (not a closure) so it can be dispatched to a
    ProcessPoolExecutor. Returns None if the ticker cannot be prepared.
    """
    try:
        indicator = calculate_indicators(ticker, df)
        daily_returns = df['close'].pct_change().dropna().tolist()
        mc_result = mc_settings.run(symbol=ticker, daily_returns=daily_returns, ohlcv=df)
        features = build_features(df, required_features)
        return ticker, indicator, mc_result, features
    except Exception:
        return None


def _prepare_all_tickers(
    data: Dict[str, pd.DataFrame],
    required_features: List[str],
    config: AppConfig,
    logger,
) -> List[tuple]:
    """Prepare every ticker, in parallel when configured.

    Output is ordered by `data`'s iteration order in both the serial and the
    parallel path, so downstream scoring, ranking and the exported report do
    not depend on which worker finished first.
    """
    # The drift prior is a property of the universe, not of any one ticker, so
    # it is estimated once over the whole panel and shared by every worker.
    # `data` is the live path's as-of-now history — the same series each ticker
    # is about to be simulated from — so the prior sees nothing its tickers do
    # not already see.
    mc_settings = MonteCarloSettings.from_simulation_config(
        config.simulation
    ).with_drift_prior_from_panel(
        df['close'].pct_change().dropna().to_numpy()
        for df in data.values()
        if 'close' in df.columns and not df.empty
    )
    args = (required_features, mc_settings)

    if not (config.data.parallel_ticker_prep and len(data) > 1):
        results = []
        for ticker, df in data.items():
            prepared = _prepare_one_ticker(ticker, df, *args)
            if prepared is None:
                logger.warning(f"Skipping {ticker}: preparation failed")
            else:
                results.append(prepared)
        return results

    by_ticker = {}
    try:
        with ProcessPoolExecutor(max_workers=config.data.ticker_prep_workers) as executor:
            futures = {
                executor.submit(_prepare_one_ticker, ticker, df, *args): ticker
                for ticker, df in data.items()
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    prepared = future.result()
                except Exception as e:
                    logger.warning(f"Skipping {ticker}: preparation failed in worker ({e})")
                    continue
                if prepared is None:
                    logger.warning(f"Skipping {ticker}: preparation failed")
                else:
                    by_ticker[ticker] = prepared
    except Exception as e:
        # Never lose the run because workers could not start.
        logger.warning(f"Parallel ticker preparation unavailable ({e}); falling back to serial")
        results = []
        for ticker, df in data.items():
            prepared = _prepare_one_ticker(ticker, df, *args)
            if prepared is not None:
                results.append(prepared)
        return results

    return [by_ticker[t] for t in data if t in by_ticker]


def _load_benchmark_frame(config: AppConfig, logger) -> Optional[pd.DataFrame]:
    """Cached OHLC frame for the configured market benchmark, if any.

    Read from the ordinary per-ticker parquet cache, so `download-data`
    populating it is the only setup step. The whole frame is returned rather
    than just the closes because ADX — and therefore the SIDEWAYS_CHOP
    classification — needs the daily range. Returns None when the symbol was
    never cached; src/regime.py then falls back to an equal-weighted composite
    of the traded universe.
    """
    symbol = getattr(config.data, "benchmark_symbol", "")
    if not symbol:
        return None

    df = load_ticker_data(symbol)
    if df is None or 'close' not in df.columns or df.empty:
        logger.info(
            f"Benchmark {symbol} is not cached; the market-regime filter will use a "
            f"composite of the traded universe"
        )
        return None

    logger.info(f"Loaded benchmark {symbol} ({len(df)} bars) for the market-regime filter")
    return df.sort_index()


def _classify_market_regime(
    benchmark_frame: Optional[pd.DataFrame],
    data: Dict[str, pd.DataFrame],
) -> Optional[str]:
    """Name today's market state, from the index if cached and the book if not.

    Mirrors BacktestEngine._classify_regime so live and backtested runs gate
    models on the same definition. The composite fallback is what keeps the
    meta-orchestrator's regime map from going silently inert on an install that
    never downloaded an index: without it the label is UNKNOWN, every model is
    permitted in every state, and nothing in the logs says the gating stopped
    working.

    Args:
        benchmark_frame: Cached OHLC for the configured index, or None.
        data: Per-ticker OHLCV for the traded universe.

    Returns:
        A regime classification, or None when neither an index nor a usable
        composite exists — read downstream as "permit every model".
    """
    if benchmark_frame is not None and 'close' in benchmark_frame.columns:
        return assess_market_regime(
            benchmark_frame['close'], market_ohlcv=benchmark_frame
        ).classification

    proxy = build_market_proxy(
        {
            ticker: df['close']
            for ticker, df in data.items()
            if df is not None and 'close' in df.columns and not df.empty
        },
        lookback=DEFAULT_TREND_WINDOW + 1,
    )
    if proxy is None:
        return None
    # A composite has no meaningful high/low, so ADX uses its close-only proxy.
    return assess_market_regime(proxy).classification


def _scaled_quantity(quantity: int, signal: StrategySignal) -> int:
    """Apply a signal's `position_scale` to a sized quantity.

    Mirrors BacktestEngine._apply_position_scale so live and backtested
    sizing cannot drift: strategies that measure their own risk environment
    (cross-sectional momentum's volatility targeting and market-regime filter,
    src/regime.py) publish a multiplier in [0, 1], and every sizing rule
    honours it here rather than each strategy applying it itself.
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


def _sector_capped_quantity(
    ticker: str,
    quantity: int,
    price: float,
    config: AppConfig,
    sector_map: Dict[str, str],
    planned_sector_values: Dict[str, float],
    logger,
) -> int:
    """Trim a recommendation so its sector stays under risk.max_sector_pct.

    `planned_sector_values` accumulates this run's already-recommended
    positions, so a set of recommendations that each fit under the cap alone
    cannot collectively breach it — the failure mode a per-position check
    would miss entirely.

    Scope, stated precisely: this caps *today's recommendation set*, not the
    live book. The orchestrator is a recommender and holds no positions state,
    so unlike BacktestEngine — which seeds the same accounting with its open
    holdings marked to market — it cannot see exposure carried in from
    previous runs. A reader acting on these recommendations on top of an
    existing portfolio has to apply the cap against their real holdings.
    """
    if config.risk.max_sector_pct <= 0 or config.risk.max_sector_pct >= 1 or price <= 0:
        return quantity

    capacity = sector_capacity_inr(
        ticker=ticker,
        portfolio_value_inr=config.risk.portfolio_value_inr,
        position_values=planned_sector_values,
        sector_map=sector_map,
        max_sector_pct=config.risk.max_sector_pct,
        max_unknown_pct=config.risk.max_unknown_sector_pct,
    )
    if math.isinf(capacity):
        # The cap does not apply to this ticker (disabled, or unmapped sector).
        return quantity
    max_quantity = int(capacity / price)
    if max_quantity >= quantity:
        return quantity

    logger.info(
        f"Sector cap trimmed {ticker} from {quantity} to {max(0, max_quantity)} shares "
        f"(sector '{sector_of(ticker, sector_map)}' at "
        f"{config.risk.max_sector_pct:.0%} portfolio limit)"
    )
    return max(0, max_quantity)


def run_orchestrator(
    force_refresh: bool = False,
    simulate_outcome: bool = False,
    update_outcomes: bool = False,
    config: AppConfig | None = None
) -> str:
    """Run the full daily loop for portfolio optimization.

    Args:
        force_refresh: If True, fetch fresh data instead of using cache.
        simulate_outcome: If True, simulate outcome for top recommendation.
        update_outcomes: If True, fetch market data and update open outcomes.
        config: Optional AppConfig to use (for testing). If None, loads from config file.

    Returns:
        Path to generated Excel report.

    Steps:
        1. Load config.
        2. Initialize SQLite.
        3. Load brain.
        4. Learn from trade outcomes.
        5. Load the configured strategy (rule-based or ML).
        6. Fetch market data.
        7. Build features and run Monte Carlo per ticker.
        8. Score each ticker with the strategy (same code path as backtesting).
        9. Calculate quantity and compliance.
        10. Save recommendations to SQLite.
        11. Optionally simulate outcome for top recommendation.
        12. Optionally update outcomes from market data.
        13. Save brain.
        14. Export Excel report.
        15. Log run status.
    """
    # Generate run_id
    run_id = str(uuid.uuid4())

    # Step 1: Load config
    if config is None:
        config = get_config()
    logger = _setup_logging(config.paths.log_file, run_id)
    logger.info(f"Starting orchestrator run with run_id={run_id}")

    try:
        # Step 2: Initialize SQLite
        init_db(config.paths.sqlite_path)
        logger.info("SQLite initialized")

        # Step 3: Load brain from config.paths.brain_file
        brain = load_brain(config.paths.brain_file)
        logger.info(f"Loaded brain from {config.paths.brain_file}")

        # Step 4: Load trade outcomes from SQLite into brain.trade_history
        #
        # Restated net of round-trip friction on the way in. The stored
        # outcomes carry a gross price move, which is the right thing for a
        # report but the wrong thing for every consumer of brain.trade_history:
        # Kelly sizes off the payoff ratio b (risk.py::estimate_kelly_inputs)
        # and the weight learner off the per-trigger win rate, and a ~0.8%
        # round-trip cost both shrinks b and turns marginally-positive trades
        # into losses. Netting once here keeps a single definition of "did this
        # trade make money" across both.
        trade_outcomes = get_trade_history(config.paths.sqlite_path)
        gross_history = [
            {
                "trade_id": outcome.trade_id,
                "symbol": outcome.symbol,
                "signal_trigger": outcome.signal_trigger,
                "entry_date": outcome.entry_date,
                "entry_price": outcome.entry_price,
                "exit_date": outcome.exit_date,
                "exit_price": outcome.exit_price,
                "outcome": outcome.outcome,
                "return_pct": outcome.return_pct,
                "outcome_source": outcome.outcome_source,
            }
            for outcome in trade_outcomes
        ]
        brain.trade_history.extend(
            to_net_realized_trades(
                gross_history,
                buy_cost_pct=cost_fraction_per_side("BUY", config.risk.slippage_pct_per_side),
                sell_cost_pct=cost_fraction_per_side("SELL", config.risk.slippage_pct_per_side),
            )
        )
        logger.info(f"Loaded {len(trade_outcomes)} trade outcomes from SQLite")

        # Step 5: Run evaluate_and_learn()
        brain = evaluate_and_learn(brain, config)
        logger.info("Learning evaluation complete")

        # Step 6: Load the configured strategy (same code path used by the backtest engine)
        strategy = load_strategy(config.strategy)
        if hasattr(strategy, "load"):
            if not strategy.load():
                logger.warning(f"Strategy '{strategy.name}' failed to load; recommendations may be unavailable")
        risk_params = RiskParams.from_app_config(config)
        logger.info(f"Loaded strategy: {strategy.name}")

        # Step 7: Fetch data using load_or_fetch_data()
        data = load_or_fetch_data(config, force_refresh=force_refresh)
        logger.info(f"Fetched data for {len(data)} tickers")

        # Prepare containers for results
        recommendations: List[Recommendation] = []
        mc_results: List[MonteCarloResult] = []
        indicator_snapshots = []

        # Step 8: Calculate indicators + Monte Carlo + features for every ticker
        # up front. This is the run's CPU-bound hot spot (Monte Carlo dominates),
        # so it is dispatched across a process pool when
        # config.data.parallel_ticker_prep is set — results are collected back
        # in universe order, so recommendations are identical either way.
        prepared = _prepare_all_tickers(data, strategy.required_features(), config, logger)

        mc_by_ticker: Dict[str, MonteCarloResult] = {}
        features_by_ticker: Dict[str, pd.DataFrame] = {}
        for ticker, indicator, mc_result, features in prepared:
            indicator_snapshots.append(indicator)
            mc_by_ticker[ticker] = mc_result
            mc_results.append(mc_result)
            features_by_ticker[ticker] = features

        # Step 9: Score via the strategy — the same code path the backtest engine
        # uses, so live and backtest decisions can never drift apart. Strategies
        # that need the full universe at once (cross-sectional momentum/
        # low-volatility ranking) are scored in a single score_batch() call;
        # everything else is scored per-ticker with its own Monte Carlo result.
        # The market benchmark (Nifty 50 by default) drives the momentum crash
        # filter. It comes from the same parquet cache as everything else; when
        # it was never cached the filter falls back to a composite of the
        # traded universe, so a missing index is a downgrade, not a failure.
        benchmark_frame = _load_benchmark_frame(config, logger)
        benchmark_close = benchmark_frame['close'] if benchmark_frame is not None else None
        regime_label = _classify_market_regime(benchmark_frame, data)
        logger.info(f"Market regime classified as {regime_label or 'UNKNOWN (no usable market series)'}")

        signals: Dict[str, StrategySignal] = {}
        if strategy.requires_full_batch or strategy.supports_gpu_batch:
            batch_context = StrategyContext(
                risk=risk_params,
                weights=dict(brain.weights),
                run_id=run_id,
                benchmark_close=benchmark_close,
                benchmark_ohlcv=benchmark_frame,
                regime_label=regime_label,
            )
            signals = strategy.score_batch(features_by_ticker, batch_context)
        else:
            for ticker, features in features_by_ticker.items():
                context = StrategyContext(
                    risk=risk_params,
                    weights=dict(brain.weights),
                    mc_result=mc_by_ticker.get(ticker),
                    run_id=run_id,
                    benchmark_close=benchmark_close,
                    benchmark_ohlcv=benchmark_frame,
                    regime_label=regime_label,
                )
                try:
                    signals[ticker] = strategy.score(ticker, features, context)
                except Exception as e:
                    logger.exception(f"Error scoring ticker {ticker}: {e}")

        # Step 10-11: Turn each signal into a Recommendation.
        #
        # BUY candidates are processed highest-score-first so the sector cap
        # below allocates its remaining capacity to the strongest names, then
        # everything else follows. Sizing is the live mirror of the backtest
        # engine's order path: base quantity, then the signal's own
        # position_scale (volatility targeting / market regime), then the
        # sector concentration cap.
        sector_map = load_sector_map(config.paths.sector_map_csv) if config.risk.max_sector_pct > 0 else {}
        if config.risk.max_sector_pct > 0 and not sector_cap_is_enforceable(sector_map):
            logger.warning(
                f"risk.max_sector_pct is set to {config.risk.max_sector_pct:.0%} but no sector "
                f"map was loaded from {config.paths.sector_map_csv}, so sector concentration "
                f"is NOT being limited. Provide a ticker,sector CSV to enable it."
            )
        planned_sector_values: Dict[str, float] = {}

        ordered_tickers = sorted(
            signals,
            key=lambda t: (signals[t].signal != "BUY", -signals[t].score, t),
        )
        for ticker in ordered_tickers:
            sig = signals[ticker]
            try:
                mc_result = mc_by_ticker.get(ticker)

                # Calculate quantity (fixed-fractional, or fractional-Kelly
                # once config.risk.use_kelly_sizing is set and enough
                # realized trade history exists — see risk.py)
                quantity = calculate_position_quantity(
                    entry_price=sig.entry_price,
                    stop_price=sig.stop_price,
                    config=config,
                    trade_history=brain.trade_history,
                )
                quantity = _scaled_quantity(quantity, sig)

                if sig.signal == "BUY" and quantity > 0:
                    quantity = _sector_capped_quantity(
                        ticker=ticker,
                        quantity=quantity,
                        price=sig.entry_price,
                        config=config,
                        sector_map=sector_map,
                        planned_sector_values=planned_sector_values,
                        logger=logger,
                    )
                    if quantity > 0:
                        planned_sector_values[ticker] = quantity * sig.entry_price

                # Calculate investment and max loss
                investment_inr = quantity * sig.entry_price
                max_loss_inr = quantity * (sig.entry_price - sig.stop_price)

                # Run compliance
                compliance_status, failed_reasons = run_compliance_checks(
                    symbol=ticker,
                    close=sig.entry_price,
                    quantity=quantity,
                    investment_inr=investment_inr,
                    config=config
                )

                # Create Recommendation object
                rec = Recommendation(
                    recommendation_id=str(uuid.uuid4()),
                    created_at=datetime.now(timezone.utc).isoformat(),
                    symbol=ticker,
                    signal=sig.signal,
                    score=sig.score,
                    trigger=sig.trigger,
                    entry_price=sig.entry_price,
                    stop_price=sig.stop_price,
                    target_price=sig.target_price,
                    reward_risk=sig.reward_risk,
                    quantity=quantity,
                    investment_inr=investment_inr,
                    max_loss_inr=max_loss_inr,
                    mc_probability_profit=sig.probability_profit,
                    mc_var_95_pct=mc_result.var_95 if mc_result else 0.0,
                    mc_cvar_95_pct=mc_result.cvar_95 if mc_result else 0.0,
                    compliance_status=compliance_status,
                    rationale=sig.rationale
                )
                recommendations.append(rec)
                logger.info(f"Processed {ticker}: signal={rec.signal}, score={rec.score:.2f}")
            except Exception as e:
                logger.exception(f"Error processing ticker {ticker}: {e}")
                continue

        # Step 8: Sort recommendations by score descending
        recommendations.sort(key=lambda r: r.score, reverse=True)

        # Step 9: Save recommendations to SQLite
        save_recommendations(config.paths.sqlite_path, recommendations)
        logger.info(f"Saved {len(recommendations)} recommendations to SQLite")

        # Step 10: Optionally simulate outcome for top recommendation
        if simulate_outcome and recommendations:
            top_rec = recommendations[0]
            simulated = simulate_outcome_fn(top_rec)
            save_trade_outcome(config.paths.sqlite_path, simulated)

            # Also add to brain's trade_history — netted the same way the
            # stored history was on load, so one appended trade cannot be the
            # single gross record in an otherwise net series.
            brain.trade_history.extend(to_net_realized_trades(
                [{
                    "trade_id": simulated.trade_id,
                    "symbol": simulated.symbol,
                    "signal_trigger": simulated.signal_trigger,
                    "entry_date": simulated.entry_date,
                    "entry_price": simulated.entry_price,
                    "exit_date": simulated.exit_date,
                    "exit_price": simulated.exit_price,
                    "outcome": simulated.outcome,
                    "return_pct": simulated.return_pct,
                    "outcome_source": simulated.outcome_source,
                }],
                buy_cost_pct=cost_fraction_per_side("BUY", config.risk.slippage_pct_per_side),
                sell_cost_pct=cost_fraction_per_side("SELL", config.risk.slippage_pct_per_side),
            ))
            logger.info(f"Added simulated outcome for {top_rec.symbol}: {simulated.outcome}")

        # Step 11: Optionally update outcomes from market data
        if update_outcomes:
            updated = update_outcomes_from_market(config.paths.sqlite_path, data)
            logger.info(f"Updated {len(updated)} trade outcomes from market data")

        # Step 12: Save updated brain
        brain.updated_at = datetime.now(timezone.utc).isoformat()
        save_brain(config.paths.brain_file, brain)
        logger.info(f"Saved brain to {config.paths.brain_file}")

        # Step 13: Export Excel
        excel_path = export_excel_report(
            config=config,
            brain=brain,
            recommendations=recommendations,
            indicators=indicator_snapshots,
            mc_results=mc_results,
            run_id=run_id
        )
        logger.info(f"Exported Excel report to {excel_path}")

        # Step 14: Log run result
        log_run(
            sqlite_path=config.paths.sqlite_path,
            run_id=run_id,
            status="SUCCESS",
            message=f"Generated {len(recommendations)} recommendations",
            recommendations_count=len(recommendations)
        )
        logger.info(f"Logged run status: SUCCESS")

        # Step 15: Return Excel file path
        return excel_path
    except Exception as e:
        logger.exception(f"Orchestrator run failed: {e}")
        # Log run failure
        log_run(
            sqlite_path=config.paths.sqlite_path,
            run_id=run_id,
            status="FAILED",
            message=str(e),
            recommendations_count=0
        )
        raise


# ---------------------------------------------------------------------------
# Indicator snapshots
# ---------------------------------------------------------------------------
#
# Moved here from the deleted src/indicators.py. This orchestrator is the only
# caller, and under the freeze (T11) the live path keeps its behaviour exactly
# — so the code travels with it rather than being rewritten against
# features/technical.py.
#
# Note these are deliberately *unshifted*, unlike everything in
# features/technical.py. That is correct here and only here: a live snapshot
# describes the state as of the latest bar, and the orchestrator is deciding
# what to do given today's close. The registered features shift because they
# are trained on, where reading today's bar to predict today's move is the
# whole failure mode. Two different jobs, two different conventions — which is
# exactly why having both under one importable name was a hazard.


def calculate_indicators(symbol: str, df: pd.DataFrame) -> IndicatorSnapshot:
    """Calculate technical indicators for a ticker DataFrame.

    Args:
        symbol: Ticker symbol.
        df: DataFrame with columns: open, high, low, close, volume.

    Returns:
        IndicatorSnapshot with latest indicator values.
    """
    # Work on a copy to avoid mutating input
    df = df.copy()
    
    # Calculate SMA20, SMA50, SMA200
    sma20_series = df['close'].rolling(window=20).mean()
    sma50_series = df['close'].rolling(window=50).mean()
    sma200_series = df['close'].rolling(window=200).mean()
    
    # Calculate Donchian Upper 20 (20-day rolling max of high)
    donchian_upper_20_series = df['high'].rolling(window=20).max()
    prev_donchian_upper_20_series = donchian_upper_20_series.shift(1)
    
    # Calculate Avg Volume 20
    avg_volume_20_series = df['volume'].rolling(window=20).mean()
    
    # Calculate Volume Ratio = latest volume / avg volume 20
    volume_ratio_series = df['volume'] / avg_volume_20_series
    
    # Calculate ATR14
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr14_series = true_range.rolling(window=14).mean()
    
    # Calculate daily log returns = ln(close / previous_close)
    log_returns = np.log(df['close'] / df['close'].shift(1))
    
    # Get latest row values
    latest_idx = len(df) - 1
    
    # Extract values, handling None cases
    sma20 = float(sma20_series.iloc[latest_idx]) if not pd.isna(sma20_series.iloc[latest_idx]) else None
    sma50 = float(sma50_series.iloc[latest_idx]) if not pd.isna(sma50_series.iloc[latest_idx]) else None
    sma200_val = sma200_series.iloc[latest_idx]
    sma200 = float(sma200_val) if not pd.isna(sma200_val) else None
    
    donchian_upper_20_val = donchian_upper_20_series.iloc[latest_idx]
    donchian_upper_20 = float(donchian_upper_20_val) if not pd.isna(donchian_upper_20_val) else None
    
    prev_donchian_upper_20_val = prev_donchian_upper_20_series.iloc[latest_idx]
    prev_donchian_upper_20 = float(prev_donchian_upper_20_val) if not pd.isna(prev_donchian_upper_20_val) else None
    
    avg_volume_20_val = avg_volume_20_series.iloc[latest_idx]
    avg_volume_20 = float(avg_volume_20_val) if not pd.isna(avg_volume_20_val) else None
    
    # Volume ratio: None if volume missing or zero
    latest_volume = df['volume'].iloc[latest_idx]
    if pd.isna(latest_volume) or latest_volume == 0:
        volume_ratio = None
    else:
        volume_ratio_val = volume_ratio_series.iloc[latest_idx]
        volume_ratio = float(volume_ratio_val) if not pd.isna(volume_ratio_val) else None
    
    # ATR14: None if cannot be computed
    atr14_val = atr14_series.iloc[latest_idx]
    atr14 = float(atr14_val) if not pd.isna(atr14_val) else None
    
    # Daily log return
    log_return_val = log_returns.iloc[latest_idx]
    daily_log_return = float(log_return_val) if not pd.isna(log_return_val) else None
    
    return IndicatorSnapshot(
        symbol=symbol,
        sma20=sma20,
        sma50=sma50,
        sma200=sma200,
        donchian_upper_20=donchian_upper_20,
        prev_donchian_upper_20=prev_donchian_upper_20,
        avg_volume_20=avg_volume_20,
        volume_ratio=volume_ratio,
        atr14=atr14,
        daily_log_return=daily_log_return
    )


def calculate_all_indicators(data: Dict[str, pd.DataFrame]) -> List[IndicatorSnapshot]:
    """Calculate indicators for all tickers in the data dictionary.

    Args:
        data: Dictionary mapping ticker symbols to DataFrames.

    Returns:
        List of IndicatorSnapshot objects.
    """
    results = []
    
    for symbol, df in data.items():
        try:
            # Validate required columns
            required_columns = {'open', 'high', 'low', 'close', 'volume'}
            if not required_columns.issubset(set(df.columns)):
                logger.warning(f"Skipping ticker {symbol}: missing required columns")
                continue
            
            # Skip if DataFrame is empty
            if df.empty:
                logger.warning(f"Skipping ticker {symbol}: empty DataFrame")
                continue
            
            snapshot = calculate_indicators(symbol, df)
            results.append(snapshot)
            
        except Exception as e:
            logger.warning(f"Skipping ticker {symbol}: error computing indicators - {e}")
            continue
    
    return results
