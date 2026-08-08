#!/usr/bin/env python3
"""
Point-In-Time (PIT) Backtest Script for Portfolio Agent

This script runs a complete backtest using the run_orchestrator methodology
with strict Point-In-Time (PIT) data access to prevent look-ahead bias.

The agent learns from trade outcomes every N trading days and adjusts its
brain weights accordingly, simulating how a real trading agent would evolve.

Features:
- Strict PIT data access (at date T, only sees data up to T-1)
- T+1 execution delay (signals generated at T, executed at T+1 open)
- Brain weight updates every 20 trading days based on trade outcomes
- Support for both rule-based and ML model-enhanced signals
- Comprehensive Excel reporting with analytics

Usage:
    python pit_backtest.py --start-date 2021-01-01 --end-date 2026-01-01 --universe-size 50
    python pit_backtest.py --use-trained-model --model-path models/lstm_best.pt
"""

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Add portfolio_agent to path
sys.path.insert(0, str(Path(__file__).parent))

from portfolio_agent.config.loader import load_config
from portfolio_agent.config.schema import AppConfig
from portfolio_agent.src.backtest_engine import BacktestEngine
from portfolio_agent.src.backtest_reporting import export_backtest_excel
from portfolio_agent.src.risk_analytics import RiskAnalyzer
from portfolio_agent.src.universe import resolve_backtest_universe
from portfolio_agent.agents.backtester import BacktesterAgent, ModelLoader


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('pit_backtest.log', mode='w')
        ]
    )
    return logging.getLogger('pit_backtest')


def run_pit_backtest(
    config: AppConfig,
    start_date: str,
    end_date: str,
    initial_capital: float,
    universe_tickers: List[str],
    use_trained_model: bool = False,
    model_path: Optional[str] = None,
    output_file: str = "output/PIT_Backtest_Report.xlsx",
    learning_interval: int = 20
) -> Dict[str, Any]:
    """
    Run Point-In-Time backtest with learning agent.
    
    This function implements the complete PIT backtest loop:
    1. Load historical data for all tickers
    2. Iterate through each trading day
    3. At each day T:
       - Generate signals using only data up to T-1 (PIT)
       - Create pending orders for T+1 execution
       - Execute orders from previous day's signals
       - Check stop-loss/take-profit levels
       - Mark-to-market portfolio
    4. Every N trading days, update brain weights based on outcomes
    5. Generate comprehensive Excel report
    
    Args:
        config: Application configuration
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_capital: Initial capital in INR
        universe_tickers: List of ticker symbols
        use_trained_model: Whether to use trained ML model
        model_path: Path to trained model checkpoint
        output_file: Path for Excel report
        learning_interval: Number of trading days between learning updates
        
    Returns:
        Dictionary with backtest results and metrics
    """
    logger = logging.getLogger('pit_backtest')
    logger.info(f"Starting PIT Backtest from {start_date} to {end_date}")
    logger.info(f"Universe size: {len(universe_tickers)} tickers")
    logger.info(f"Initial capital: ₹{initial_capital:,.2f}")
    logger.info(f"Use trained model: {use_trained_model}")
    logger.info(f"Learning interval: {learning_interval} trading days")
    
    # Initialize backtest engine
    engine = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        universe_tickers=universe_tickers
    )
    
    logger.info(f"Loaded data for {len(engine.ticker_data)} tickers")
    logger.info(f"Trading days in period: {len(engine.master_date_index)}")
    
    # Override signal generation if using trained model
    if use_trained_model:
        logger.info("Initializing model for signal generation...")
        
        # Try to load model
        model_loader = ModelLoader(
            models_dir="models",
            device="cpu"
        )
        
        model_name = model_path.split('/')[-1].replace('_best.pt', '') if model_path else 'lstm'
        if model_loader.load_model(model_name):
            logger.info(f"Successfully loaded model: {model_name}")
            
            # Inject model-aware signal generation
            # We need to monkey-patch the engine's _generate_signals method
            original_generate_signals = engine._generate_signals
            
            def generate_signals_with_model(current_date: pd.Timestamp) -> Dict[str, Dict[str, Any]]:
                """Generate signals using trained model predictions."""
                signals = {}
                
                for ticker in engine.universe_tickers:
                    if ticker in engine.untradeable_tickers:
                        continue
                    
                    # Get historical data up to T-1 (PIT)
                    hist_data = engine._get_historical_data_up_to(ticker, current_date)
                    
                    if hist_data is None or len(hist_data) < 60:
                        continue
                    
                    # Prepare features for model
                    try:
                        from portfolio_agent.features.pipeline import build_features
                        
                        feature_names = model_loader.feature_names
                        sequence_length = model_loader.metadata.get('sequence_length', 60)
                        
                        if len(feature_names) > 0:
                            # Build features
                            feature_df = build_features(hist_data, feature_names, normalize=False)
                            feature_df = feature_df.dropna()
                            
                            if len(feature_df) >= sequence_length:
                                # Take last sequence_length samples
                                recent_features = feature_df.iloc[-sequence_length:].values
                                
                                # Convert to tensor
                                import torch
                                features_tensor = torch.FloatTensor(recent_features).unsqueeze(0)
                                
                                # Get prediction
                                with torch.no_grad():
                                    prediction = model_loader.predict(features_tensor)
                                    model_prob = float(prediction.squeeze().item())
                                
                                # Convert to probability-like value
                                if model_prob < 0:
                                    model_prob = 0.5 + (model_prob / 2)
                                elif model_prob > 1:
                                    model_prob = 1.0 - (1.0 / (1.0 + model_prob))
                                model_prob = min(max(model_prob, 0.0), 1.0)
                                
                                # Generate signal
                                if model_prob > 0.6:
                                    signal_type = 'BUY'
                                elif model_prob < 0.4:
                                    signal_type = 'SELL'
                                else:
                                    signal_type = 'HOLD'
                                
                                current_price = float(hist_data['close'].iloc[-1])
                                
                                signals[ticker] = {
                                    'signal': signal_type,
                                    'score': model_prob,
                                    'current_price': current_price,
                                    'trigger': 'MODEL',
                                    'model_probability': model_prob
                                }
                                continue
                    except Exception as e:
                        logger.debug(f"Model prediction failed for {ticker}: {e}")
                    
                    # Fallback to rule-based signals
                    close_prices = hist_data['close'] if 'close' in hist_data.columns else hist_data.iloc[:, 0]
                    
                    if len(close_prices) >= 20:
                        sma_20 = close_prices.rolling(window=20).mean().iloc[-1]
                        current_price = close_prices.iloc[-1]
                        
                        trend_signal = 1 if current_price > sma_20 else -1
                        
                        volume_signal = 0
                        if 'volume' in hist_data.columns:
                            avg_vol = hist_data['volume'].rolling(window=20).mean().iloc[-1]
                            current_vol = hist_data['volume'].iloc[-1]
                            if current_vol > avg_vol * 1.5:
                                volume_signal = 1
                            elif current_vol < avg_vol * 0.5:
                                volume_signal = -1
                        
                        weights = engine.agent_brain.weights
                        combined_score = (
                            weights.get('Trend', 25.0) * trend_signal +
                            weights.get('Volume', 20.0) * volume_signal
                        ) / 100.0
                        
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
                            'trigger': 'Trend' if abs(trend_signal) > 0 else 'Volume',
                            'model_probability': 0.5
                        }
                
                return signals
            
            engine._generate_signals = generate_signals_with_model
            logger.info("Model-enhanced signal generation enabled")
        else:
            logger.warning("Failed to load trained model, falling back to rule-based signals")
            use_trained_model = False
    
    # Run backtest
    logger.info("Running PIT backtest simulation...")
    result = engine.run_backtest()
    
    # Calculate risk analytics
    logger.info("Calculating risk analytics...")
    analyzer = RiskAnalyzer(
        daily_equity_curve=engine.daily_equity_curve,
        trade_log=engine.trade_log
    )
    analytics_report = analyzer.generate_analytics_report()
    
    # Export Excel report
    logger.info(f"Exporting report to {output_file}...")
    
    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    analytics_for_export = {
        'cagr': analytics_report.get('cagr', 0),
        'sharpe': analytics_report.get('sharpe_ratio', 0),
        'sortino': analytics_report.get('sortino_ratio', 0),
        'max_drawdown': analytics_report.get('max_drawdown_pct', 0),
        'profit_factor': analytics_report.get('profit_factor', 0),
        'probability_of_ruin': analytics_report.get('mc_probability_of_ruin_pct', 0),
        'total_return': analytics_report.get('total_return_pct', 0),
        'volatility': analytics_report.get('annualized_volatility_pct', 0),
        'win_rate': analytics_report.get('win_rate_pct', 0),
        'total_trades': analytics_report.get('total_trades', 0),
        'final_portfolio_value': analytics_report.get('final_capital', 0),
        'initial_capital': initial_capital,
        'monte_carlo_results': {
            'percentile_5': analytics_report.get('mc_percentile_5', 0),
            'percentile_50': analytics_report.get('mc_median_terminal_wealth', 0),
            'percentile_95': analytics_report.get('mc_percentile_95', 0)
        },
        'model_used': use_trained_model,
        'model_name': model_name if use_trained_model else 'rule_based',
        'learning_enabled': True,
        'learning_interval': learning_interval
    }
    
    export_backtest_excel(
        analytics=analytics_for_export,
        equity_curve=engine.daily_equity_curve,
        trade_log=engine.trade_log,
        brain_evolution=engine.brain_evolution,
        daily_activity_log=engine.daily_activity_log,
        filepath=output_file
    )
    
    logger.info(f"Backtest complete. Report saved to: {output_file}")
    
    # Log summary
    logger.info("=" * 60)
    logger.info("BACKTEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Trading Days: {len(engine.master_date_index)}")
    logger.info(f"Total Trades: {len(engine.trade_log)}")
    logger.info(f"Initial Capital: ₹{initial_capital:,.2f}")
    logger.info(f"Final Portfolio Value: ₹{analytics_report.get('final_capital', 0):,.2f}")
    logger.info(f"Total Return: {analytics_report.get('total_return_pct', 0):.2f}%")
    logger.info(f"CAGR: {analytics_report.get('cagr', 0):.2f}%")
    logger.info(f"Sharpe Ratio: {analytics_report.get('sharpe_ratio', 0):.2f}")
    logger.info(f"Max Drawdown: {analytics_report.get('max_drawdown_pct', 0):.2f}%")
    logger.info(f"Win Rate: {analytics_report.get('win_rate_pct', 0):.2f}%")
    logger.info("=" * 60)
    
    return {
        'status': 'success',
        'output_file': output_file,
        'metrics': analytics_report,
        'trade_count': len(engine.trade_log),
        'trading_days': len(engine.master_date_index),
        'model_used': use_trained_model,
        'brain_evolution': engine.brain_evolution
    }


def main():
    """Main entry point for PIT backtest script."""
    parser = argparse.ArgumentParser(
        description="Run Point-In-Time backtest with learning agent"
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date (YYYY-MM-DD). Default: 5 years ago'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD). Default: today'
    )
    
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=None,
        help='Initial capital in INR. Default: from config'
    )
    
    parser.add_argument(
        '--universe-size',
        type=int,
        default=None,
        help='Number of tickers in universe. Default: from config'
    )
    
    parser.add_argument(
        '--use-trained-model',
        action='store_true',
        help='Use trained PyTorch model for signals'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to trained model checkpoint'
    )
    
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='output/PIT_Backtest_Report.xlsx',
        help='Output Excel file path'
    )
    
    parser.add_argument(
        '--learning-interval',
        type=int,
        default=20,
        help='Trading days between learning updates'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Load configuration
    config = load_config()
    
    # Override config with command line arguments
    if args.initial_capital:
        config.backtest.initial_capital = args.initial_capital
    
    if args.universe_size:
        config.data.universe_size = args.universe_size
    
    # Calculate date range
    if args.end_date:
        end_date = args.end_date
    else:
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    if args.start_date:
        start_date = args.start_date
    else:
        start_date = (pd.Timestamp.now() - timedelta(days=config.backtest.start_years_ago * 365)).strftime('%Y-%m-%d')
    
    # Resolve universe
    logger.info(f"Resolving universe of {config.data.universe_size} tickers...")
    tickers = resolve_backtest_universe(
        force_full_download=False,
        max_tickers=config.data.universe_size
    )
    
    if not tickers:
        logger.error("No tickers available for backtest")
        sys.exit(1)
    
    logger.info(f"Resolved {len(tickers)} tickers")
    
    # Run backtest
    try:
        result = run_pit_backtest(
            config=config,
            start_date=start_date,
            end_date=end_date,
            initial_capital=config.backtest.initial_capital,
            universe_tickers=tickers,
            use_trained_model=args.use_trained_model,
            model_path=args.model_path,
            output_file=args.output,
            learning_interval=args.learning_interval
        )
        
        if result.get('status') == 'success':
            logger.info(f"\n✓ Backtest completed successfully!")
            logger.info(f"Report: {result['output_file']}")
            sys.exit(0)
        else:
            logger.error("Backtest failed")
            sys.exit(1)
            
    except Exception as e:
        logger.exception(f"Error during backtest: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
