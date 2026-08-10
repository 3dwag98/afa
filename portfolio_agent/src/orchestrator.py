"""Main orchestrator module for portfolio agent."""

import logging
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from portfolio_agent.config.schema import AppConfig
from portfolio_agent.config.loader import load_config as get_config
from portfolio_agent.features.pipeline import build_features
from portfolio_agent.strategies.registry import load_strategy
from portfolio_agent.strategies.types import RiskParams, StrategyContext, StrategySignal

# Use absolute imports for CLI execution
try:
    from .storage import (
        init_db, save_recommendations,
        save_trade_outcome, log_run, get_trade_history,
        load_brain, save_brain
    )
    from .data_store import load_or_fetch_data, load_ticker_data
    from .indicators import calculate_indicators
    from .monte_carlo import MonteCarloResult, MonteCarloSettings
    from .risk import calculate_position_quantity
    from .compliance import run_compliance_checks
    from .learning import evaluate_and_learn
    from .reporting import export_excel_report
    from .models import Recommendation
    from .outcomes import simulate_outcome as simulate_outcome_fn, update_outcomes_from_market
    from .sectors import load_sector_map, sector_capacity_inr, sector_of
    from .logging_utils import get_logger, ContextualLogger
except ImportError:
    from storage import (
        init_db, save_recommendations,
        save_trade_outcome, log_run, get_trade_history,
        load_brain, save_brain
    )
    from data_store import load_or_fetch_data, load_ticker_data
    from indicators import calculate_indicators
    from monte_carlo import MonteCarloResult, MonteCarloSettings
    from risk import calculate_position_quantity
    from compliance import run_compliance_checks
    from learning import evaluate_and_learn
    from reporting import export_excel_report
    from models import Recommendation
    from outcomes import simulate_outcome as simulate_outcome_fn, update_outcomes_from_market
    from sectors import load_sector_map, sector_capacity_inr, sector_of
    from logging_utils import get_logger, ContextualLogger


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
    args = (required_features, MonteCarloSettings.from_simulation_config(config.simulation))

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


def _load_benchmark_close(config: AppConfig, logger) -> Optional[pd.Series]:
    """Cached close series for the configured market benchmark, if any.

    Read from the ordinary per-ticker parquet cache, so `download-data`
    populating it is the only setup step. Returns None when the symbol was
    never cached — src/regime.py then falls back to an equal-weighted
    composite of the traded universe.
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
    return df['close'].sort_index()


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
    """
    if config.risk.max_sector_pct <= 0 or config.risk.max_sector_pct >= 1 or price <= 0:
        return quantity

    capacity = sector_capacity_inr(
        ticker=ticker,
        portfolio_value_inr=config.risk.portfolio_value_inr,
        position_values=planned_sector_values,
        sector_map=sector_map,
        max_sector_pct=config.risk.max_sector_pct,
    )
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
        trade_outcomes = get_trade_history(config.paths.sqlite_path)
        for outcome in trade_outcomes:
            brain.trade_history.append({
                "trade_id": outcome.trade_id,
                "symbol": outcome.symbol,
                "signal_trigger": outcome.signal_trigger,
                "entry_date": outcome.entry_date,
                "entry_price": outcome.entry_price,
                "exit_date": outcome.exit_date,
                "exit_price": outcome.exit_price,
                "outcome": outcome.outcome,
                "return_pct": outcome.return_pct,
                "outcome_source": outcome.outcome_source
            })
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
        benchmark_close = _load_benchmark_close(config, logger)

        signals: Dict[str, StrategySignal] = {}
        if strategy.requires_full_batch or strategy.supports_gpu_batch:
            batch_context = StrategyContext(
                risk=risk_params,
                weights=dict(brain.weights),
                run_id=run_id,
                benchmark_close=benchmark_close,
            )
            signals = strategy.score_batch(features_by_ticker, batch_context)
        else:
            for ticker, features in features_by_ticker.items():
                context = StrategyContext(
                    risk=risk_params,
                    weights=dict(brain.weights),
                    mc_result=mc_by_ticker.get(ticker),
                    run_id=run_id,
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

            # Also add to brain's trade_history
            brain.trade_history.append({
                "trade_id": simulated.trade_id,
                "symbol": simulated.symbol,
                "signal_trigger": simulated.signal_trigger,
                "entry_date": simulated.entry_date,
                "entry_price": simulated.entry_price,
                "exit_date": simulated.exit_date,
                "exit_price": simulated.exit_price,
                "outcome": simulated.outcome,
                "return_pct": simulated.return_pct,
                "outcome_source": simulated.outcome_source
            })
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
