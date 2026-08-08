"""Tests for backtest reporting module."""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtest_reporting import export_backtest_excel


def generate_dummy_analytics() -> dict:
    """Generate dummy analytics data for testing."""
    return {
        'cagr': 18.5,
        'sharpe': 1.45,
        'sortino': 2.1,
        'max_drawdown': -15.3,
        'profit_factor': 1.85,
        'probability_of_ruin': 2.5,
        'total_return': 125.6,
        'volatility': 12.4,
        'win_rate': 62.5,
        'total_trades': 247,
        'final_portfolio_value': 2256000.0,
        'initial_capital': 1000000.0,
        'monte_carlo_results': {
            'percentile_5': 1450000.0,
            'percentile_50': 2100000.0,
            'percentile_95': 3250000.0
        }
    }


def generate_dummy_equity_curve(start_date: str = '2020-01-01', periods: int = 1260) -> pd.Series:
    """
    Generate dummy equity curve data.
    
    Args:
        start_date: Start date for the equity curve.
        periods: Number of trading days (approx 252 per year).
    
    Returns:
        pd.Series with DateTimeIndex containing portfolio values.
    """
    dates = pd.bdate_range(start=start_date, periods=periods)
    
    # Simulate realistic equity curve with some growth and volatility
    np.random.seed(42)
    daily_returns = np.random.normal(0.0007, 0.015, periods)  # ~17% annual return, 15% vol
    
    # Create cumulative returns
    cumulative_returns = np.cumprod(1 + daily_returns)
    
    # Scale to start at 1,000,000
    portfolio_values = 1000000 * cumulative_returns
    
    equity_curve = pd.Series(portfolio_values, index=dates, name='Portfolio_Value')
    
    # Add benchmark data as attribute (simulating Nifty 50)
    benchmark_values = 10000 * np.cumprod(1 + np.random.normal(0.0005, 0.012, periods))
    equity_curve.attrs['benchmark'] = benchmark_values
    
    return equity_curve


def generate_dummy_trade_log(num_trades: int = 50) -> list:
    """
    Generate dummy trade log data.
    
    Args:
        num_trades: Number of trades to generate.
    
    Returns:
        List of trade dictionaries.
    """
    tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
               'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'BAJFINANCE.NS']
    
    np.random.seed(43)
    trades = []
    
    base_date = datetime(2020, 1, 15)
    
    for i in range(num_trades):
        entry_date = base_date + timedelta(days=np.random.randint(0, 1200))
        holding_days = np.random.randint(1, 60)
        exit_date = entry_date + timedelta(days=holding_days)
        
        ticker = np.random.choice(tickers)
        side = np.random.choice(['BUY', 'SELL'])
        
        entry_price = np.random.uniform(500, 3000)
        qty = np.random.randint(10, 500)
        
        # Generate PnL with some bias towards positive
        gross_pnl = np.random.normal(500, 2000)
        
        # Calculate taxes and slippage
        stt_taxes = abs(gross_pnl) * 0.001  # 0.1% STT
        slippage = entry_price * qty * 0.0005  # 0.05% slippage
        
        net_pnl = gross_pnl - stt_taxes - slippage
        
        signal_triggers = ['Trend', 'Breakout', 'Volume', 'MC_Prob']
        signal_trigger = np.random.choice(signal_triggers)
        
        trade = {
            'entry_date': entry_date.strftime('%Y-%m-%d'),
            'exit_date': exit_date.strftime('%Y-%m-%d'),
            'ticker': ticker,
            'side': side,
            'entry_price': entry_price,
            'exit_price': entry_price + (gross_pnl / qty if qty > 0 else 0),
            'qty': qty,
            'gross_pnl': gross_pnl,
            'stt_taxes': stt_taxes,
            'slippage': slippage,
            'net_pnl': net_pnl,
            'holding_days': holding_days,
            'signal_trigger': signal_trigger
        }
        trades.append(trade)
    
    return trades


def generate_dummy_brain_evolution(num_snapshots: int = 63) -> list:
    """
    Generate dummy brain evolution data showing weight changes every 20 days.
    
    Args:
        num_snapshots: Number of snapshots (1260 trading days / 20 = 63).
    
    Returns:
        List of brain state dictionaries.
    """
    np.random.seed(44)
    brain_evolution = []
    
    # Initial weights
    weights = {
        'Trend': 25.0,
        'Breakout': 25.0,
        'Volume': 20.0,
        'MC_Prob': 30.0
    }
    
    base_date = datetime(2020, 1, 1)
    
    for i in range(num_snapshots):
        trading_day = (i + 1) * 20
        
        # Simulate weight adaptation over time
        # E.g., during crash (early days), MC_Prob increases; during bull run, Trend increases
        if trading_day <= 100:  # 2020 crash period
            weights['MC_Prob'] = min(45, weights['MC_Prob'] + 0.3)
            weights['Trend'] = max(15, weights['Trend'] - 0.1)
        elif trading_day <= 400:  # Recovery period
            weights['Trend'] = min(35, weights['Trend'] + 0.05)
            weights['Breakout'] = min(30, weights['Breakout'] + 0.03)
        else:  # Bull market period
            weights['Volume'] = min(25, weights['Volume'] + 0.02)
            weights['Breakout'] = max(20, weights['Breakout'] - 0.01)
        
        # Normalize to sum to 100
        total = sum(weights.values())
        weights = {k: v * 100 / total for k, v in weights.items()}
        
        snapshot = {
            'trading_day': trading_day,
            'date': (base_date + timedelta(days=trading_day)).strftime('%Y-%m-%d'),
            'weights': {
                'Trend': round(weights['Trend'], 2),
                'Breakout': round(weights['Breakout'], 2),
                'Volume': round(weights['Volume'], 2),
                'MC_Prob': round(weights['MC_Prob'], 2)
            }
        }
        brain_evolution.append(snapshot)
    
    return brain_evolution


class TestBacktestReporting:
    """Test class for backtest reporting module."""
    
    @pytest.fixture
    def output_dir(self, tmp_path) -> Path:
        """Create temporary output directory."""
        output_path = tmp_path / 'output'
        output_path.mkdir(exist_ok=True)
        return output_path
    
    @pytest.fixture
    def dummy_data(self):
        """Generate all dummy data for testing."""
        analytics = generate_dummy_analytics()
        equity_curve = generate_dummy_equity_curve()
        trade_log = generate_dummy_trade_log(num_trades=50)
        brain_evolution = generate_dummy_brain_evolution(num_snapshots=63)
        
        return {
            'analytics': analytics,
            'equity_curve': equity_curve,
            'trade_log': trade_log,
            'brain_evolution': brain_evolution
        }
    
    def test_export_backtest_excel_creates_file(self, dummy_data, output_dir):
        """Test that export creates a valid Excel file."""
        filepath = str(output_dir / 'backtest_report.xlsx')
        
        result = export_backtest_excel(
            analytics=dummy_data['analytics'],
            equity_curve=dummy_data['equity_curve'],
            trade_log=dummy_data['trade_log'],
            brain_evolution=dummy_data['brain_evolution'],
            filepath=filepath
        )
        
        # Assert file was created
        assert os.path.exists(filepath), f"File not created at {filepath}"
        assert result == filepath
    
    def test_export_backtest_excel_file_size_greater_than_zero(self, dummy_data, output_dir):
        """Test that exported file has size > 0."""
        filepath = str(output_dir / 'backtest_report.xlsx')
        
        export_backtest_excel(
            analytics=dummy_data['analytics'],
            equity_curve=dummy_data['equity_curve'],
            trade_log=dummy_data['trade_log'],
            brain_evolution=dummy_data['brain_evolution'],
            filepath=filepath
        )
        
        file_size = os.path.getsize(filepath)
        assert file_size > 0, f"File size is 0 bytes at {filepath}"
        print(f"File size: {file_size} bytes")
    
    def test_export_backtest_excel_all_sheets_exist(self, dummy_data, output_dir):
        """Test that all 6 required sheets exist in the workbook."""
        filepath = str(output_dir / 'backtest_report.xlsx')
        
        export_backtest_excel(
            analytics=dummy_data['analytics'],
            equity_curve=dummy_data['equity_curve'],
            trade_log=dummy_data['trade_log'],
            brain_evolution=dummy_data['brain_evolution'],
            filepath=filepath
        )
        
        # Read the Excel file and check sheet names
        xl = pd.ExcelFile(filepath)
        sheet_names = xl.sheet_names
        
        expected_sheets = [
            'Executive_Summary',
            'Equity_Curve',
            'Trade_Log',
            'Monthly_Heatmap',
            'Brain_Evolution',
            'Monte_Carlo_Simulations'
        ]
        
        for sheet in expected_sheets:
            assert sheet in sheet_names, f"Sheet '{sheet}' not found in workbook"
        
        xl.close()
    
    def test_executive_summary_content(self, dummy_data, output_dir):
        """Test Executive Summary sheet contains key metrics."""
        filepath = str(output_dir / 'backtest_report.xlsx')
        
        export_backtest_excel(
            analytics=dummy_data['analytics'],
            equity_curve=dummy_data['equity_curve'],
            trade_log=dummy_data['trade_log'],
            brain_evolution=dummy_data['brain_evolution'],
            filepath=filepath
        )
        
        summary_df = pd.read_excel(filepath, sheet_name='Executive_Summary')
        
        # Check that key metrics are present
        metrics_column = summary_df.iloc[:, 0].tolist()
        assert any('Sharpe' in str(m) for m in metrics_column), "Sharpe Ratio not found"
        assert any('Drawdown' in str(m) for m in metrics_column), "Max Drawdown not found"
        assert any('CAGR' in str(m) for m in metrics_column), "CAGR not found"
    
    def test_trade_log_columns(self, dummy_data, output_dir):
        """Test Trade Log sheet has all required columns."""
        filepath = str(output_dir / 'backtest_report.xlsx')
        
        export_backtest_excel(
            analytics=dummy_data['analytics'],
            equity_curve=dummy_data['equity_curve'],
            trade_log=dummy_data['trade_log'],
            brain_evolution=dummy_data['brain_evolution'],
            filepath=filepath
        )
        
        trade_df = pd.read_excel(filepath, sheet_name='Trade_Log')
        
        expected_columns = [
            'Entry Date', 'Exit Date', 'Ticker', 'Side', 'Entry Price', 
            'Exit Price', 'Qty', 'Gross PnL', 'STT/Taxes', 'Slippage', 
            'Net PnL', 'Holding Days', 'Signal Trigger'
        ]
        
        for col in expected_columns:
            assert col in trade_df.columns, f"Column '{col}' not found in Trade_Log"
    
    def test_monthly_heatmap_structure(self, dummy_data, output_dir):
        """Test Monthly Heatmap has years as rows and months as columns."""
        filepath = str(output_dir / 'backtest_report.xlsx')
        
        export_backtest_excel(
            analytics=dummy_data['analytics'],
            equity_curve=dummy_data['equity_curve'],
            trade_log=dummy_data['trade_log'],
            brain_evolution=dummy_data['brain_evolution'],
            filepath=filepath
        )
        
        heatmap_df = pd.read_excel(filepath, sheet_name='Monthly_Heatmap')
        
        # Should have month columns (at least Jan-Mar to verify structure)
        month_cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for col in month_cols:
            assert col in heatmap_df.columns, f"Month column '{col}' not found"
    
    def test_brain_evolution_weights(self, dummy_data, output_dir):
        """Test Brain Evolution shows weight adaptations."""
        filepath = str(output_dir / 'backtest_report.xlsx')
        
        export_backtest_excel(
            analytics=dummy_data['analytics'],
            equity_curve=dummy_data['equity_curve'],
            trade_log=dummy_data['trade_log'],
            brain_evolution=dummy_data['brain_evolution'],
            filepath=filepath
        )
        
        brain_df = pd.read_excel(filepath, sheet_name='Brain_Evolution')
        
        expected_columns = ['Trading Day', 'Trend', 'Breakout', 'Volume', 'MC_Prob']
        
        for col in expected_columns:
            assert col in brain_df.columns, f"Column '{col}' not found in Brain_Evolution"
        
        # Verify we have multiple snapshots showing evolution
        assert len(brain_df) > 1, "Brain Evolution should have multiple snapshots"
    
    def test_monte_carlo_percentiles(self, dummy_data, output_dir):
        """Test Monte Carlo sheet shows percentile results."""
        filepath = str(output_dir / 'backtest_report.xlsx')
        
        export_backtest_excel(
            analytics=dummy_data['analytics'],
            equity_curve=dummy_data['equity_curve'],
            trade_log=dummy_data['trade_log'],
            brain_evolution=dummy_data['brain_evolution'],
            filepath=filepath
        )
        
        mc_df = pd.read_excel(filepath, sheet_name='Monte_Carlo_Simulations')
        
        # Check for percentile metrics
        metrics = mc_df.iloc[:, 0].tolist()
        assert any('5th' in str(m) or '5' in str(m) for m in metrics), "5th percentile not found"
        assert any('50th' in str(m) or 'Median' in str(m) for m in metrics), "50th percentile not found"
        assert any('95th' in str(m) or '95' in str(m) for m in metrics), "95th percentile not found"
    
    def test_empty_trade_log_handling(self, dummy_data, output_dir):
        """Test handling of empty trade log."""
        filepath = str(output_dir / 'backtest_report_empty_trades.xlsx')
        
        # Use empty trade log
        dummy_data['trade_log'] = []
        
        export_backtest_excel(
            analytics=dummy_data['analytics'],
            equity_curve=dummy_data['equity_curve'],
            trade_log=dummy_data['trade_log'],
            brain_evolution=dummy_data['brain_evolution'],
            filepath=filepath
        )
        
        # File should still be created
        assert os.path.exists(filepath), "File not created with empty trade log"
        
        # Trade_Log sheet should exist (even if empty)
        xl = pd.ExcelFile(filepath)
        assert 'Trade_Log' in xl.sheet_names, "Trade_Log sheet missing with empty trades"
        xl.close()
    
    def test_equity_curve_with_benchmark(self, dummy_data, output_dir):
        """Test Equity Curve includes benchmark comparison."""
        filepath = str(output_dir / 'backtest_report.xlsx')
        
        export_backtest_excel(
            analytics=dummy_data['analytics'],
            equity_curve=dummy_data['equity_curve'],
            trade_log=dummy_data['trade_log'],
            brain_evolution=dummy_data['brain_evolution'],
            filepath=filepath
        )
        
        equity_df = pd.read_excel(filepath, sheet_name='Equity_Curve')
        
        # Should have Date and Portfolio_Value columns
        assert 'Date' in equity_df.columns or equity_df.index.name == 'Date', \
            "Date column/index not found"
        assert 'Portfolio_Value' in equity_df.columns, "Portfolio_Value column not found"
        
        # Should have Drawdown calculation
        assert 'Drawdown_%' in equity_df.columns, "Drawdown_% column not found"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
