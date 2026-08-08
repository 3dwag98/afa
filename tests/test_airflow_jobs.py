"""Tests for Airflow job wrapper functions."""

import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAirflowJobs(unittest.TestCase):
    """Test cases for src.airflow_jobs module."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock config object
        self.mock_config = MagicMock()
        self.mock_config.paper_trading_mode = True
        self.mock_config.sqlite_path = '/tmp/test.db'
        self.mock_config.brain_file = '/tmp/brain.json'
        self.mock_config.initial_capital_inr = 1000000

    @patch('src.airflow_jobs.get_config')
    @patch('src.airflow_jobs.run_orchestrator')
    def test_run_daily_agent_job_calls_orchestrator(self, mock_run_orchestrator, mock_get_config):
        """Test that run_daily_agent_job calls the agent orchestrator."""
        from src.airflow_jobs import run_daily_agent_job
        
        mock_get_config.return_value = self.mock_config
        mock_run_orchestrator.return_value = '/app/output/report.xlsx'
        
        result = run_daily_agent_job()
        
        mock_get_config.assert_called_once()
        mock_run_orchestrator.assert_called_once_with(force_refresh=False, config=self.mock_config)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['job'], 'DAILY_AGENT')
        self.assertEqual(result['excel_path'], '/app/output/report.xlsx')

    @patch('src.airflow_jobs.get_config')
    def test_run_daily_agent_job_raises_if_not_paper_trading(self, mock_get_config):
        """Test that run_daily_agent_job raises error if paper_trading_mode is false."""
        from src.airflow_jobs import run_daily_agent_job
        
        mock_config = MagicMock()
        mock_config.paper_trading_mode = False
        mock_get_config.return_value = mock_config
        
        with self.assertRaises(RuntimeError) as context:
            run_daily_agent_job()
        
        self.assertIn("Live trading is disabled", str(context.exception))
        self.assertIn("paper_trading_mode=true", str(context.exception))

    @patch('src.airflow_jobs.resolve_backtest_universe')
    @patch('src.airflow_jobs.batch_download_and_cache')
    @patch('src.airflow_jobs.BacktestEngine')
    @patch('src.airflow_jobs.RiskAnalyzer')
    @patch('src.airflow_jobs.export_backtest_excel')
    @patch('src.airflow_jobs.get_config')
    def test_run_backtest_job_calls_backtest_with_safe_defaults(
        self, mock_get_config, mock_export_excel, mock_risk_analyzer, 
        mock_backtest_engine, mock_batch_download, mock_resolve_universe
    ):
        """Test that run_backtest_job calls backtest orchestrator with safe defaults."""
        from src.airflow_jobs import run_backtest_job
        
        # Setup mocks
        mock_get_config.return_value = self.mock_config
        mock_resolve_universe.return_value = ['TICKER1', 'TICKER2']
        mock_batch_download.return_value = True
        
        # Mock engine
        mock_engine = MagicMock()
        mock_engine.daily_equity_curve = MagicMock()
        mock_engine.trade_log = []
        mock_engine.brain_evolution = []
        mock_engine.daily_activity_log = []
        mock_engine.run_backtest.return_value = None
        mock_backtest_engine.return_value = mock_engine
        
        # Mock analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.generate_analytics_report.return_value = {
            'cagr': 0.15,
            'cagr_pct': 15.0,
            'sharpe_ratio': 1.2,
            'sortino_ratio': 1.5,
            'max_drawdown_pct': 10.0,
            'profit_factor': 1.8,
            'mc_probability_of_ruin_pct': 2.0,
            'total_return_pct': 75.0,
            'annualized_volatility_pct': 12.0,
            'win_rate_pct': 60.0,
            'total_trades': 50,
            'final_capital': 1750000,
            'mc_percentile_5': 900000,
            'mc_median_terminal_wealth': 1700000,
            'mc_percentile_95': 2500000
        }
        mock_risk_analyzer.return_value = mock_analyzer
        
        result = run_backtest_job()
        
        # Verify resolve_backtest_universe called with safe defaults
        mock_resolve_universe.assert_called_once_with(
            force_full_download=False,
            max_tickers=20
        )
        
        # Verify BacktestEngine initialized correctly
        mock_backtest_engine.assert_called_once()
        call_args = mock_backtest_engine.call_args
        self.assertEqual(call_args[1]['universe_tickers'], ['TICKER1', 'TICKER2'])
        
        # Verify result
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['job'], 'BACKTEST_SMALL')
        self.assertEqual(result['cagr'], 15.0)
        self.assertEqual(result['sharpe'], 1.2)

    @patch('src.airflow_jobs.resolve_backtest_universe')
    @patch('src.airflow_jobs.batch_download_and_cache')
    @patch('src.airflow_jobs.BacktestEngine')
    @patch('src.airflow_jobs.RiskAnalyzer')
    @patch('src.airflow_jobs.export_backtest_excel')
    @patch('src.airflow_jobs.get_config')
    def test_run_full_backtest_job_uses_five_years_all_tickers(
        self, mock_get_config, mock_export_excel, mock_risk_analyzer,
        mock_backtest_engine, mock_batch_download, mock_resolve_universe
    ):
        """Test that run_full_backtest_job uses 5 years and all tickers."""
        from src.airflow_jobs import run_full_backtest_job
        
        # Setup mocks
        mock_get_config.return_value = self.mock_config
        mock_resolve_universe.return_value = ['TICKER1', 'TICKER2', 'TICKER3']
        mock_batch_download.return_value = True
        
        # Mock engine
        mock_engine = MagicMock()
        mock_engine.daily_equity_curve = MagicMock()
        mock_engine.trade_log = []
        mock_engine.brain_evolution = []
        mock_engine.daily_activity_log = []
        mock_engine.run_backtest.return_value = None
        mock_backtest_engine.return_value = mock_engine
        
        # Mock analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.generate_analytics_report.return_value = {
            'cagr': 0.18,
            'cagr_pct': 18.0,
            'sharpe_ratio': 1.5,
            'total_trades': 200
        }
        mock_risk_analyzer.return_value = mock_analyzer
        
        result = run_full_backtest_job()
        
        # Verify resolve_backtest_universe called with universe_size=None (all tickers)
        mock_resolve_universe.assert_called_once_with(
            force_full_download=False,
            max_tickers=None
        )
        
        # Verify result
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['job'], 'BACKTEST_FULL')

    @patch('src.airflow_jobs.get_config')
    def test_run_backtest_job_raises_if_no_tickers(self, mock_get_config):
        """Test that run_backtest_job raises error if no tickers available."""
        from src.airflow_jobs import run_backtest_job
        
        mock_get_config.return_value = self.mock_config
        
        with patch('src.airflow_jobs.resolve_backtest_universe') as mock_resolve:
            mock_resolve.return_value = []
            
            with self.assertRaises(RuntimeError) as context:
                run_backtest_job()
            
            self.assertIn("No tickers with available data", str(context.exception))

    @patch('src.airflow_jobs.get_config')
    def test_run_full_backtest_job_raises_if_no_tickers(self, mock_get_config):
        """Test that run_full_backtest_job raises error if no tickers available."""
        from src.airflow_jobs import run_full_backtest_job
        
        mock_get_config.return_value = self.mock_config
        
        with patch('src.airflow_jobs.resolve_backtest_universe') as mock_resolve:
            mock_resolve.return_value = []
            
            with self.assertRaises(RuntimeError) as context:
                run_full_backtest_job()
            
            self.assertIn("No tickers with available data", str(context.exception))


if __name__ == '__main__':
    unittest.main()
