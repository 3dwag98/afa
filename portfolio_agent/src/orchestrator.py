"""Main orchestrator module for portfolio agent."""

import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional

# Use absolute imports for CLI execution
try:
    from .config import AppConfig
    from .storage import SQLiteStorage, JSONBrain
    from .data_ingestion import fetch_multiple_tickers, validate_data
    from .indicators import add_all_indicators
    from .monte_carlo import run_monte_carlo
    from .scoring import calculate_combined_score
    from .risk import calculate_position_size, calculate_stop_loss, calculate_target_price
    from .compliance import compliance_check
    from .learning import LearningAgent
    from .reporting import create_excel_report
except ImportError:
    from config import AppConfig
    from storage import SQLiteStorage, JSONBrain
    from data_ingestion import fetch_multiple_tickers, validate_data
    from indicators import add_all_indicators
    from monte_carlo import run_monte_carlo
    from scoring import calculate_combined_score
    from risk import calculate_position_size, calculate_stop_loss, calculate_target_price
    from compliance import compliance_check
    from learning import LearningAgent
    from reporting import create_excel_report


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
            returns = df['Close'].pct_change().dropna()
            sim_result = run_monte_carlo(
                returns,
                horizon_days=self.config.mc_horizon_days,
                simulations=self.config.mc_simulations,
                seed=self.config.random_seed
            )
            simulations[ticker] = sim_result
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
