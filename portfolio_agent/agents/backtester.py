"""Backtester agent for portfolio forecasting with trained model integration."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from portfolio_agent.config.schema import AppConfig, BacktestConfig
from portfolio_agent.features.pipeline import build_features
from portfolio_agent.models.registry import get_model
from portfolio_agent.utils.device import get_device
from portfolio_agent.src.backtest_engine import BacktestEngine
from portfolio_agent.src.backtest_reporting import export_backtest_excel
from portfolio_agent.src.risk_analytics import RiskAnalyzer


logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and manage trained PyTorch models for inference.
    
    This class handles:
    - Loading model weights from checkpoint files
    - Loading metadata for feature alignment
    - Moving model to inference device (CPU or GPU)
    - Making predictions with proper preprocessing
    """
    
    def __init__(self, models_dir: str = "models", device: str = "cpu"):
        """Initialize the model loader.
        
        Args:
            models_dir: Directory containing model checkpoints and metadata.
            device: Device for inference ("cpu", "cuda", or "mps").
        """
        self.models_dir = Path(models_dir)
        self.device = torch.device(device)
        self.model: Optional[nn.Module] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self._model_loaded = False
        
    def load_model(self, model_name: str = "lstm") -> bool:
        """Load a trained model from checkpoint.
        
        Args:
            model_name: Name of the model architecture (e.g., "lstm").
            
        Returns:
            True if model loaded successfully, False otherwise.
        """
        # Find the best model checkpoint
        checkpoint_path = self.models_dir / f"{model_name}_best.pt"
        metadata_path = self.models_dir / "metadata.json"
        
        if not checkpoint_path.exists():
            logger.error(f"Model checkpoint not found: {checkpoint_path}")
            return False
            
        if not metadata_path.exists():
            logger.error(f"Metadata file not found: {metadata_path}")
            return False
        
        # Load metadata first
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Extract model configuration from metadata
        n_features = len(self.metadata.get('feature_names', []))
        sequence_length = self.metadata.get('sequence_length', 60)
        
        if n_features == 0:
            logger.error("No feature names found in metadata")
            return False
        
        # Get model class from registry
        try:
            model_class = get_model(model_name)
        except KeyError as e:
            logger.error(f"Model class not found: {e}")
            return False
        
        # Instantiate model
        self.model = model_class(
            n_features=n_features,
            hidden_size=64,
            n_layers=2,
            sequence_length=sequence_length,
            dropout=0.2,
            n_outputs=1,
        )
        
        # Load weights
        checkpoint = torch.load(
            checkpoint_path, 
            map_location=self.device, 
            weights_only=True
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self._model_loaded = True
        logger.info(f"Loaded model '{model_name}' from {checkpoint_path}")
        logger.info(f"  Features ({n_features}): {self.metadata.get('feature_names', [])}")
        logger.info(f"  Sequence length: {sequence_length}")
        logger.info(f"  Device: {self.device}")
        
        return True
    
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Make predictions on input features.
        
        Args:
            features: Input tensor of shape (batch_size, sequence_length, n_features).
            
        Returns:
            Predictions tensor of shape (batch_size, 1).
            
        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        with torch.no_grad():
            features = features.to(self.device, non_blocking=True)
            predictions = self.model(features)
        
        return predictions
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names used during training."""
        if self.metadata is None:
            return []
        return self.metadata.get('feature_names', [])
    
    @property
    def target_name(self) -> str:
        """Get target variable name used during training."""
        if self.metadata is None:
            return ""
        return self.metadata.get('target', '')


class BacktesterAgent:
    """Backtesting agent with optional trained model integration.
    
    This agent extends the basic backtest engine to support:
    - Loading and using trained PyTorch models for signal generation
    - Feature engineering aligned with trained model
    - Passing model probabilities to strategy engine
    - Generating detailed Excel reports
    """
    
    def __init__(
        self,
        config: AppConfig,
        use_trained_model: bool = False,
        inference_device: str = "cpu"
    ):
        """Initialize the backtester agent.
        
        Args:
            config: Application configuration.
            use_trained_model: Whether to use trained model for predictions.
            inference_device: Device for model inference ("cpu", "cuda", or "mps").
        """
        self.config = config
        self.use_trained_model = use_trained_model
        self.inference_device = inference_device
        self.model_loader: Optional[ModelLoader] = None
        
        # Initialize model loader if needed
        if use_trained_model:
            self.model_loader = ModelLoader(
                models_dir="models",
                device=inference_device
            )
        
        logger.info(f"BacktesterAgent initialized")
        logger.info(f"  Use trained model: {use_trained_model}")
        logger.info(f"  Inference device: {inference_device}")
    
    def load_trained_model(self, model_name: str = "lstm") -> bool:
        """Load the trained model for inference.
        
        Args:
            model_name: Name of the model architecture.
            
        Returns:
            True if model loaded successfully.
        """
        if self.model_loader is None:
            logger.error("Model loader not initialized (use_trained_model=False)")
            return False
        
        return self.model_loader.load_model(model_name)
    
    def _prepare_features_for_model(
        self, 
        df: pd.DataFrame, 
        ticker: str
    ) -> Optional[torch.Tensor]:
        """Prepare features for model prediction.
        
        Args:
            df: DataFrame with OHLCV data up to current date.
            ticker: Ticker symbol.
            
        Returns:
            Tensor of shape (1, sequence_length, n_features) or None.
        """
        if self.model_loader is None or self.model_loader.metadata is None:
            return None
        
        feature_names = self.model_loader.feature_names
        sequence_length = self.model_loader.metadata.get('sequence_length', 60)
        
        if len(feature_names) == 0:
            return None
        
        try:
            # Build features using the pipeline
            feature_df = build_features(
                df,
                feature_names,
                normalize=False  # Model expects pre-normalized or raw features
            )
            
            # Drop NaN values
            feature_df = feature_df.dropna()
            
            if len(feature_df) < sequence_length:
                logger.debug(f"Not enough data for {ticker}: {len(feature_df)} < {sequence_length}")
                return None
            
            # Take the last sequence_length samples
            recent_features = feature_df.iloc[-sequence_length:].values
            
            # Convert to tensor and add batch dimension
            features_tensor = torch.FloatTensor(recent_features).unsqueeze(0)
            
            return features_tensor
            
        except Exception as e:
            logger.warning(f"Error preparing features for {ticker}: {e}")
            return None
    
    def _prepare_features_for_model_inline(
        self, 
        df: pd.DataFrame, 
        ticker: str,
        model_loader
    ) -> Optional[torch.Tensor]:
        """Prepare features for model prediction (inline version for BacktestEngine).
        
        Args:
            df: DataFrame with OHLCV data up to current date.
            ticker: Ticker symbol.
            model_loader: ModelLoader instance with feature_names and metadata.
            
        Returns:
            Tensor of shape (1, sequence_length, n_features) or None.
        """
        if model_loader is None or model_loader.metadata is None:
            return None
        
        feature_names = model_loader.feature_names
        sequence_length = model_loader.metadata.get('sequence_length', 60)
        
        if len(feature_names) == 0:
            return None
        
        try:
            # Build features using the pipeline
            from portfolio_agent.agents.trainer import build_features
            feature_df = build_features(
                df,
                feature_names,
                normalize=False  # Model expects pre-normalized or raw features
            )
            
            # Drop NaN values
            feature_df = feature_df.dropna()
            
            if len(feature_df) < sequence_length:
                logger.debug(f"Not enough data for {ticker}: {len(feature_df)} < {sequence_length}")
                return None
            
            # Take the last sequence_length samples
            recent_features = feature_df.iloc[-sequence_length:].values
            
            # Convert to tensor and add batch dimension
            features_tensor = torch.FloatTensor(recent_features).unsqueeze(0)
            
            return features_tensor
            
        except Exception as e:
            logger.warning(f"Error preparing features for {ticker}: {e}")
            return None

    def _get_model_probability(
        self, 
        features: torch.Tensor
    ) -> Optional[float]:
        """Get probability prediction from model.
        
        Args:
            features: Input tensor of shape (1, sequence_length, n_features).
            
        Returns:
            Probability value between 0 and 1, or None.
        """
        if self.model_loader is None:
            return None
        
        try:
            prediction = self.model_loader.predict(features)
            prob = float(prediction.squeeze().item())
            
            # Convert to probability-like value (sigmoid if needed)
            # For regression targets, interpret as directional confidence
            if prob < 0:
                prob = 0.5 + (prob / 2)  # Map negative values to 0-0.5
            elif prob > 1:
                prob = 1.0 - (1.0 / (1.0 + prob))  # Soft cap for large values
            
            return min(max(prob, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Error getting model prediction: {e}")
            return None
    
    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float,
        universe_tickers: List[str],
        output_file: str
    ) -> Dict[str, Any]:
        """Run complete backtest with optional model integration.
        
        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            initial_capital: Initial capital in INR.
            universe_tickers: List of ticker symbols to trade.
            output_file: Path for Excel report.
            
        Returns:
            Dictionary with backtest results and metrics.
        """
        # Load trained model if requested
        if self.use_trained_model and self.model_loader is not None:
            if not self.load_trained_model():
                logger.warning("Failed to load trained model, falling back to rule-based signals")
                self.use_trained_model = False
        
        # Initialize backtest engine
        engine = BacktestEngine(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            universe_tickers=universe_tickers
        )
        
        # Override signal generation if using trained model
        if self.use_trained_model:
            logger.info("Using trained model for signal generation")
            # The engine will call our custom signal generation
            # We need to inject model predictions into the process
            engine._generate_signals = self._generate_signals_with_model.__get__(engine, BacktestEngine)
            engine.model_loader = self.model_loader  # Inject model loader
            # Inject the helper methods as well
            engine._prepare_features_for_model_inline = self._prepare_features_for_model_inline.__get__(engine, BacktestEngine)
            engine._get_model_probability = self._get_model_probability.__get__(engine, BacktestEngine)
        
        # Run simulation
        logger.info(f"Running backtest from {start_date} to {end_date}")
        result = engine.run_backtest()
        
        # Calculate risk analytics
        analyzer = RiskAnalyzer(
            daily_equity_curve=engine.daily_equity_curve,
            trade_log=engine.trade_log
        )
        analytics_report = analyzer.generate_analytics_report()
        
        # Export Excel report
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
            'model_used': self.use_trained_model,
            'model_name': 'lstm' if self.use_trained_model else 'rule_based'
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
        
        return {
            'status': 'success',
            'output_file': output_file,
            'metrics': analytics_report,
            'trade_count': len(engine.trade_log),
            'model_used': self.use_trained_model
        }
    
    def _generate_signals_with_model(
        self,
        current_date: pd.Timestamp
    ) -> Dict[str, Dict[str, Any]]:
        """Generate signals using trained model predictions.
        
        This method replaces the default _generate_signals in BacktestEngine
        when use_trained_model is True.
        
        Args:
            current_date: Current date timestamp.
            
        Returns:
            Dictionary of ticker -> signal info including model_probability.
        """
        signals = {}
        prev_date = current_date - pd.Timedelta(days=1)
        
        # Get historical data up to T-1 for all tickers
        for ticker in self.universe_tickers:
            if ticker in self.untradeable_tickers:
                continue
            
            # Get historical data up to T-1
            hist_data = self._get_historical_data_up_to(ticker, current_date)
            
            if hist_data is None or len(hist_data) < 60:
                continue
            
            # Prepare features for model
            model_loader = getattr(self, 'model_loader', None)
            if model_loader is not None and model_loader._model_loaded:
                # Call _prepare_features_for_model on the BacktestAgent instance
                # stored in model_loader's parent or use inline logic
                features_tensor = self._prepare_features_for_model_inline(hist_data, ticker, model_loader)
                
                if features_tensor is not None:
                    model_prob = self._get_model_probability(features_tensor)
                    
                    if model_prob is not None:
                        # Generate signal based on model probability
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
                            'model_probability': model_prob  # Pass to Strategy engine
                        }
                        continue
            
            # Fallback to rule-based signals if model unavailable
            close_prices = hist_data['close'] if 'close' in hist_data.columns else hist_data.get('Close', hist_data.iloc[:, 0])
            
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
                
                weights = self.agent_brain.weights
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
                    'model_probability': 0.5  # Default for rule-based
                }
        
        return signals


def run_backtest_cli(
    config: AppConfig,
    use_trained_model: bool = False,
    device: str = "cpu",
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """Run backtest from CLI with configuration.
    
    Args:
        config: Application configuration.
        use_trained_model: Whether to use trained model.
        device: Device for model inference.
        output_file: Optional output file path override.
        
    Returns:
        Dictionary with backtest results.
    """
    from datetime import timedelta
    
    # Calculate date range from config
    end_date = pd.Timestamp.now()
    start_date = end_date - timedelta(days=config.backtest.start_years_ago * 365)
    
    # Set output file
    if output_file is None:
        output_file = "output/Backtest_Report.xlsx"
    
    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Get universe tickers
    from portfolio_agent.src.universe import resolve_backtest_universe
    tickers = resolve_backtest_universe(
        force_full_download=False,
        max_tickers=config.data.universe_size
    )
    
    if not tickers:
        raise ValueError("No tickers available for backtest")
    
    # Initialize and run backtester
    agent = BacktesterAgent(
        config=config,
        use_trained_model=use_trained_model,
        inference_device=device
    )
    
    return agent.run_backtest(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        initial_capital=config.backtest.initial_capital,
        universe_tickers=tickers,
        output_file=output_file
    )
