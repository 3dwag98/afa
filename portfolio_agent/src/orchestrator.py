"""Main orchestrator module for portfolio agent."""

import logging
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

# Use absolute imports for CLI execution
try:
    from .config import AppConfig, get_config
    from .storage import (
        init_db, save_recommendations,
        save_trade_outcome, log_run, get_trade_history,
        load_brain, save_brain
    )
    from .data_ingestion import load_or_fetch_data
    from .indicators import calculate_indicators
    from .monte_carlo import run_monte_carlo, MonteCarloResult
    from .scoring import score_candidate
    from .risk import calculate_stop_target, calculate_quantity
    from .compliance import run_compliance_checks
    from .learning import evaluate_and_learn
    from .reporting import export_excel_report
    from .models import Recommendation, TradeOutcome, AgentBrain
    from .outcomes import simulate_outcome as simulate_outcome_fn, update_outcomes_from_market
except ImportError:
    from config import AppConfig, get_config
    from storage import (
        init_db, save_recommendations,
        save_trade_outcome, log_run, get_trade_history,
        load_brain, save_brain
    )
    from data_ingestion import load_or_fetch_data
    from indicators import calculate_indicators
    from monte_carlo import run_monte_carlo, MonteCarloResult
    from scoring import score_candidate
    from risk import calculate_stop_target, calculate_quantity
    from compliance import run_compliance_checks
    from learning import evaluate_and_learn
    from reporting import export_excel_report
    from models import Recommendation, TradeOutcome, AgentBrain
    from outcomes import simulate_outcome as simulate_outcome_fn, update_outcomes_from_market


def _setup_logging(log_file: str) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger('portfolio_agent')
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()

    # File handler
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


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
        5. Fetch market data.
        6. Calculate indicators.
        7. Run Monte Carlo.
        8. Calculate stop/target.
        9. Score candidates.
        10. Calculate quantity and compliance.
        11. Save recommendations to SQLite.
        12. Optionally simulate outcome for top recommendation.
        13. Optionally update outcomes from market data.
        14. Save brain.
        15. Export Excel report.
        16. Log run status.
    """
    # Generate run_id
    run_id = str(uuid.uuid4())

    # Step 1: Load config
    if config is None:
        config = get_config()
    logger = _setup_logging(config.log_file)
    logger.info(f"Starting orchestrator run with run_id={run_id}")

    # Step 2: Initialize SQLite
    init_db(config.sqlite_path)
    logger.info("SQLite initialized")

    # Step 3: Load brain from config.brain_file
    brain = load_brain(config.brain_file)
    logger.info(f"Loaded brain from {config.brain_file}")

    # Step 4: Load trade outcomes from SQLite into brain.trade_history
    trade_outcomes = get_trade_history(config.sqlite_path)
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

    # Step 6: Fetch data using load_or_fetch_data()
    data = load_or_fetch_data(config, force_refresh=force_refresh)
    logger.info(f"Fetched data for {len(data)} tickers")

    # Prepare containers for results
    recommendations: List[Recommendation] = []
    mc_results: List[MonteCarloResult] = []
    indicator_snapshots = []

    # Step 7-11: Process each ticker
    for ticker, df in data.items():
        # Calculate indicators
        indicator = calculate_indicators(ticker, df)
        indicator_snapshots.append(indicator)

        # Run Monte Carlo using daily returns
        daily_returns = df['close'].pct_change().dropna().tolist()
        mc_result = run_monte_carlo(
            symbol=ticker,
            daily_returns=daily_returns,
            horizon_days=config.mc_horizon_days,
            simulations=config.mc_simulations,
            seed=config.random_seed
        )
        mc_results.append(mc_result)

        # Get current price
        current_price = float(df['close'].iloc[-1])

        # Calculate entry, stop, target using ATR
        atr = indicator.atr14
        stop_price, target_price = calculate_stop_target(current_price, atr, config)

        # Score candidate
        scored = score_candidate(
            indicator=indicator,
            mc_result=mc_result,
            brain=brain,
            config=config,
            entry_price=current_price,
            stop_price=stop_price,
            target_price=target_price
        )

        # Calculate quantity
        quantity = calculate_quantity(
            entry_price=current_price,
            stop_price=stop_price,
            config=config
        )

        # Calculate investment and max loss
        investment_inr = quantity * current_price
        max_loss_inr = quantity * (current_price - stop_price)

        # Run compliance
        compliance_status, failed_reasons = run_compliance_checks(
            symbol=ticker,
            close=current_price,
            quantity=quantity,
            investment_inr=investment_inr,
            config=config
        )

        # Create Recommendation object
        rec = Recommendation(
            recommendation_id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc).isoformat(),
            symbol=ticker,
            signal=scored["signal"],
            score=scored["score"],
            trigger=scored["trigger"],
            entry_price=scored["entry_price"],
            stop_price=scored["stop_price"],
            target_price=scored["target_price"],
            reward_risk=scored["reward_risk"],
            quantity=quantity,
            investment_inr=investment_inr,
            max_loss_inr=max_loss_inr,
            mc_probability_profit=scored["mc_probability_profit"],
            mc_var_95_pct=mc_result.var_95,
            mc_cvar_95_pct=mc_result.cvar_95,
            compliance_status=compliance_status,
            rationale=scored["rationale"]
        )
        recommendations.append(rec)
        logger.info(f"Processed {ticker}: signal={rec.signal}, score={rec.score:.2f}")

    # Step 8: Sort recommendations by score descending
    recommendations.sort(key=lambda r: r.score, reverse=True)

    # Step 9: Save recommendations to SQLite
    save_recommendations(config.sqlite_path, recommendations)
    logger.info(f"Saved {len(recommendations)} recommendations to SQLite")

    # Step 10: Optionally simulate outcome for top recommendation
    if simulate_outcome and recommendations:
        top_rec = recommendations[0]
        simulated = simulate_outcome_fn(top_rec)
        save_trade_outcome(config.sqlite_path, simulated)

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
        updated = update_outcomes_from_market(config.sqlite_path, data)
        logger.info(f"Updated {len(updated)} trade outcomes from market data")

    # Step 12: Save updated brain
    brain.updated_at = datetime.now(timezone.utc).isoformat()
    save_brain(config.brain_file, brain)
    logger.info(f"Saved brain to {config.brain_file}")

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
        sqlite_path=config.sqlite_path,
        run_id=run_id,
        status="SUCCESS",
        message=f"Generated {len(recommendations)} recommendations",
        recommendations_count=len(recommendations)
    )
    logger.info(f"Logged run status: SUCCESS")

    # Step 15: Return Excel file path
    return excel_path


class PortfolioOrchestrator:
    """Main orchestrator for portfolio optimization agent."""

    def __init__(self, config: AppConfig):
        """Initialize orchestrator.

        Args:
            config: Application configuration.
        """
        self.config = config
        self.logger = self._setup_logging()

        # Initialize storage
        self.db = SQLiteStorage(config.sqlite_path)
        self.brain_storage = JSONBrain(config.brain_file)

        # Initialize learning agent
        self.agent = LearningAgent(
            self.brain_storage.data,
            config.learning_rate
        )

        self.logger.info("Portfolio Orchestrator initialized")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('portfolio_agent')
        logger.setLevel(logging.INFO)

        # File handler
        from pathlib import Path
        Path(self.config.log_file).parent.mkdir(parents=True, exist_ok=True)

        fh = logging.FileHandler(self.config.log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def run(self) -> Dict[str, Any]:
        """Run the full portfolio optimization cycle.

        Returns:
            Dictionary with results summary.
        """
        self.logger.info("Starting portfolio optimization cycle")

        # Step 1: Fetch data
        self.logger.info(f"Fetching data for {len(self.config.tickers)} tickers")
        raw_data = fetch_multiple_tickers(
            self.config.tickers,
            self.config.min_history_days
        )

        if not raw_data:
            self.logger.warning("No data fetched")
            return {'error': 'No data available'}

        # Step 2: Process data and add indicators
        processed_data = {}
        for ticker, df in raw_data.items():
            if validate_data(df, self.config.min_history_days):
                df_with_indicators = add_all_indicators(df)
                processed_data[ticker] = df_with_indicators
                self.logger.info(f"Processed {ticker}: {len(df)} rows")

        # Step 3: Run Monte Carlo simulations
        simulations = {}
        for ticker, df in processed_data.items():
            returns = df['Close'].pct_change().dropna().tolist()
            sim_result = run_monte_carlo(
                symbol=ticker,
                daily_returns=returns,
                horizon_days=self.config.mc_horizon_days,
                simulations=self.config.mc_simulations,
                seed=self.config.random_seed
            )
            simulations[ticker] = {
                'probability_profit': sim_result.probability_profit,
                'expected_return_pct': sim_result.expected_return_pct,
                'var_95': sim_result.var_95,
                'cvar_95': sim_result.cvar_95,
                'simulations_count': sim_result.simulations_count,
                'horizon_days': sim_result.horizon_days
            }
            self.logger.info(f"Monte Carlo complete for {ticker}")

        # Step 4: Generate recommendations
        recommendations = []
        compliance_results = []

        for ticker, df in processed_data.items():
            rec = self._generate_recommendation(ticker, df, simulations.get(ticker, {}))
            if rec:
                # Compliance check
                comp_result = compliance_check(
                    rec,
                    self.config,
                    self.db.get_positions('OPEN')
                )
                compliance_results.append({
                    'ticker': ticker,
                    **comp_result
                })

                if comp_result['passed']:
                    recommendations.append(rec)
                    self.logger.info(f"Recommendation generated for {ticker}: {rec['action']}")

        # Step 5: Create portfolio summary
        portfolio_summary = self._create_portfolio_summary(
            processed_data, simulations
        )

        # Step 6: Get learning stats
        learning_stats = self.agent.get_learning_stats()

        # Step 7: Generate Excel report
        output_path = self.create_report(
            recommendations,
            simulations,
            portfolio_summary,
            compliance_results,
            learning_stats
        )

        self.logger.info(f"Report generated: {output_path}")

        return {
            'status': 'success',
            'recommendations_count': len(recommendations),
            'tickers_analyzed': len(processed_data),
            'output_path': output_path,
            'timestamp': datetime.now().isoformat()
        }

    def _generate_recommendation(self, ticker: str, df: pd.DataFrame,
                                  sim_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate recommendation for a single ticker.

        Args:
            ticker: Stock ticker.
            df: DataFrame with indicators.
            sim_result: Monte Carlo simulation result.

        Returns:
            Recommendation dictionary or None.
        """
        import pandas as pd

        current_price = df['Close'].iloc[-1]
        atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else current_price * 0.02

        # Calculate base score
        base_score = calculate_combined_score(df)

        # Prepare features for learning adjustment
        features = {
            'rsi_level': 'high' if df['RSI'].iloc[-1] > 70 else 
                        ('low' if df['RSI'].iloc[-1] < 30 else 'neutral'),
            'trend': 'up' if df['Close'].iloc[-1] > df['SMA_20'].iloc[-1] else 'down'
        }

        # Apply learning adjustments
        adjusted_score = self.agent.get_adjusted_score(base_score, ticker, features)

        # Check against target probability from Monte Carlo
        prob_profit = sim_result.get('probability_profit', 0.5)

        # Determine action
        if adjusted_score >= 0.6 and prob_profit >= self.config.target_prob_profit:
            action = 'BUY'
        elif adjusted_score <= 0.3:
            action = 'SELL'
        else:
            action = 'HOLD'

        # Calculate position size and levels
        stop_loss_pct = (atr / current_price) * 2 if atr > 0 else 0.05
        quantity = calculate_position_size(
            self.config.portfolio_value_inr,
            current_price,
            self.config.risk_per_trade_pct,
            self.config.max_single_position_pct,
            stop_loss_pct
        )

        stop_loss = calculate_stop_loss(current_price, atr=atr)
        target = calculate_target_price(
            current_price,
            self.config.min_reward_risk,
            stop_loss
        )

        return {
            'ticker': ticker,
            'action': action,
            'quantity': quantity,
            'entry_price': current_price,
            'target_price': target if action == 'BUY' else None,
            'stop_loss': stop_loss if action == 'BUY' else None,
            'confidence': adjusted_score,
            'expected_return': sim_result.get('mean_return', 0),
            'risk_score': sim_result.get('std_return', 0),
            'probability_profit': prob_profit,
            'rationale': f"Score: {adjusted_score:.2f}, Prob: {prob_profit:.2%}"
        }

    def _create_portfolio_summary(self, processed_data: Dict[str, pd.DataFrame],
                                   simulations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create portfolio summary metrics.

        Args:
            processed_data: Processed ticker data.
            simulations: Simulation results.

        Returns:
            Summary dictionary.
        """
        total_expected_return = sum(
            s.get('mean_return', 0) for s in simulations.values()
        ) / len(simulations) if simulations else 0

        avg_probability = sum(
            s.get('probability_profit', 0) for s in simulations.values()
        ) / len(simulations) if simulations else 0

        return {
            'Portfolio Value (INR)': self.config.portfolio_value_inr,
            'Risk Per Trade (%)': self.config.risk_per_trade_pct * 100,
            'Max Position (%)': self.config.max_single_position_pct * 100,
            'Tickers Analyzed': len(processed_data),
            'Avg Expected Return': total_expected_return,
            'Avg Probability of Profit': avg_probability,
            'Paper Trading Mode': self.config.paper_trading_mode,
            'MC Simulations': self.config.mc_simulations,
            'MC Horizon (Days)': self.config.mc_horizon_days
        }

    def create_report(self, recommendations: List[Dict],
                      simulations: Dict[str, Dict],
                      portfolio_summary: Dict[str, Any],
                      compliance_results: List[Dict],
                      learning_stats: Dict[str, Any]) -> str:
        """Generate Excel report.

        Args:
            recommendations: List of recommendations.
            simulations: Simulation results.
            portfolio_summary: Portfolio summary.
            compliance_results: Compliance results.
            learning_stats: Learning statistics.

        Returns:
            Path to generated report.
        """
        return create_excel_report(
            recommendations,
            simulations,
            portfolio_summary,
            compliance_results,
            learning_stats,
            self.config.excel_output
        )

    def save_state(self) -> None:
        """Save current agent state to storage."""
        brain_data = self.agent.get_brain()
        self.brain_storage.data = brain_data
        # Trigger save
        self.brain_storage._save_brain()
        self.logger.info("Agent state saved")


# Import pandas at module level for type hints
import pandas as pd
from .learning import LearningAgent
from .data_ingestion import fetch_ohlcv, validate_ohlcv
from .indicators import add_all_indicators
from .scoring import calculate_combined_score
from .risk import calculate_position_size, calculate_stop_loss, calculate_target_price
from .compliance import compliance_check
from .reporting import create_excel_report

# Alias for backward compatibility
fetch_multiple_tickers = fetch_ohlcv
validate_data = validate_ohlcv
