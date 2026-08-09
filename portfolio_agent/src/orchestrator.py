"""Main orchestrator module for portfolio agent."""

import logging
import uuid
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
    from .data_store import load_or_fetch_data
    from .indicators import calculate_indicators
    from .monte_carlo import run_monte_carlo, run_monte_carlo_garch, MonteCarloResult
    from .risk import calculate_position_quantity
    from .compliance import run_compliance_checks
    from .learning import evaluate_and_learn
    from .reporting import export_excel_report
    from .models import Recommendation
    from .outcomes import simulate_outcome as simulate_outcome_fn, update_outcomes_from_market
    from .logging_utils import get_logger, ContextualLogger
except ImportError:
    from storage import (
        init_db, save_recommendations,
        save_trade_outcome, log_run, get_trade_history,
        load_brain, save_brain
    )
    from data_store import load_or_fetch_data
    from indicators import calculate_indicators
    from monte_carlo import run_monte_carlo, run_monte_carlo_garch, MonteCarloResult
    from risk import calculate_position_quantity
    from compliance import run_compliance_checks
    from learning import evaluate_and_learn
    from reporting import export_excel_report
    from models import Recommendation
    from outcomes import simulate_outcome as simulate_outcome_fn, update_outcomes_from_market
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

        # Step 8: Calculate indicators + Monte Carlo + features for every ticker up front.
        mc_by_ticker: Dict[str, MonteCarloResult] = {}
        features_by_ticker: Dict[str, pd.DataFrame] = {}
        mc_fn = run_monte_carlo_garch if config.simulation.use_garch_volatility else run_monte_carlo

        for ticker, df in data.items():
            try:
                indicator = calculate_indicators(ticker, df)
                indicator_snapshots.append(indicator)

                daily_returns = df['close'].pct_change().dropna().tolist()
                mc_result = mc_fn(
                    symbol=ticker,
                    daily_returns=daily_returns,
                    horizon_days=config.simulation.mc_horizon_days,
                    simulations=config.simulation.mc_simulations,
                    seed=config.simulation.random_seed
                )
                mc_by_ticker[ticker] = mc_result
                mc_results.append(mc_result)

                features_by_ticker[ticker] = build_features(df, strategy.required_features())
            except Exception as e:
                logger.exception(f"Error preparing ticker {ticker}: {e}")
                continue

        # Step 9: Score via the strategy — the same code path the backtest engine
        # uses, so live and backtest decisions can never drift apart. Strategies
        # that need the full universe at once (cross-sectional momentum/
        # low-volatility ranking) are scored in a single score_batch() call;
        # everything else is scored per-ticker with its own Monte Carlo result.
        signals: Dict[str, StrategySignal] = {}
        if strategy.requires_full_batch or strategy.supports_gpu_batch:
            batch_context = StrategyContext(risk=risk_params, weights=dict(brain.weights), run_id=run_id)
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
        for ticker, sig in signals.items():
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
