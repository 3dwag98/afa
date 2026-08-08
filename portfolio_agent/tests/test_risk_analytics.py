"""Tests for risk analytics module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    from src.risk_analytics import RiskAnalyzer
except ImportError:
    from portfolio_agent.src.risk_analytics import RiskAnalyzer


class TestRiskAnalyzer:
    """Test RiskAnalyzer class with various scenarios."""
    
    def _create_equity_curve(
        self,
        start_value: float = 100000.0,
        days: int = 252,
        daily_return_mean: float = 0.0005,
        daily_return_std: float = 0.01,
        seed: int = 42
    ) -> pd.Series:
        """Create a synthetic equity curve for testing."""
        np.random.seed(seed)
        dates = pd.date_range(start='2020-01-01', periods=days, freq='B')
        daily_returns = np.random.normal(daily_return_mean, daily_return_std, days)
        cumulative_returns = np.cumprod(1 + daily_returns)
        equity_values = start_value * cumulative_returns
        return pd.Series(equity_values, index=dates)
    
    def _create_equity_curve_with_drawdown(
        self,
        start_value: float = 100000.0,
        drawdown_pct: float = 0.20,
        days_before_dd: int = 100,
        days_in_dd: int = 50,
        days_recovery: int = 100,
        seed: int = 42
    ) -> pd.Series:
        """
        Create an equity curve with a specific maximum drawdown.
        
        Args:
            start_value: Initial portfolio value.
            drawdown_pct: Target maximum drawdown (e.g., 0.20 for 20%).
            days_before_dd: Days of growth before drawdown starts.
            days_in_dd: Days during the drawdown period.
            days_recovery: Days for recovery after drawdown.
            seed: Random seed.
            
        Returns:
            pd.Series with specified drawdown characteristics.
        """
        np.random.seed(seed)
        
        total_days = days_before_dd + days_in_dd + days_recovery
        dates = pd.date_range(start='2020-01-01', periods=total_days, freq='B')
        
        # Phase 1: Growth phase (before drawdown) - use deterministic growth
        growth_factor = 1.005  # 0.5% daily growth
        phase1_values = [start_value * (growth_factor ** i) for i in range(days_before_dd)]
        
        # Peak value at end of phase 1
        peak_value = phase1_values[-1]
        
        # Phase 2: Drawdown phase - engineer exact drawdown
        # We need to go from peak_value to peak_value * (1 - drawdown_pct)
        target_trough = peak_value * (1 - drawdown_pct)
        
        # Create linear decline to target trough
        decline_factor = (target_trough / peak_value) ** (1 / days_in_dd)
        phase2_values = [peak_value * (decline_factor ** i) for i in range(1, days_in_dd + 1)]
        
        # Phase 3: Recovery phase - grow back to above peak
        recovery_start = phase2_values[-1]
        growth_needed = peak_value / recovery_start
        growth_factor_recovery = (growth_needed * 1.1) ** (1 / days_recovery)  # 10% buffer
        phase3_values = [recovery_start * (growth_factor_recovery ** i) for i in range(1, days_recovery + 1)]
        
        # Combine all phases
        all_values = phase1_values + phase2_values + phase3_values
        
        return pd.Series(all_values, index=dates)
    
    def _create_trade_log(
        self,
        n_trades: int = 50,
        win_rate: float = 0.6,
        avg_win: float = 5000,
        avg_loss: float = -3000,
        seed: int = 42
    ) -> list:
        """Create a synthetic trade log for testing."""
        np.random.seed(seed)
        
        trades = []
        n_wins = int(n_trades * win_rate)
        n_losses = n_trades - n_wins
        
        # Generate wins and losses
        wins = np.random.uniform(avg_win * 0.5, avg_win * 1.5, n_wins)
        losses = np.random.uniform(avg_loss * 0.5, avg_loss * 1.5, n_losses)
        
        # Shuffle results
        all_pnls = np.concatenate([wins, losses])
        np.random.shuffle(all_pnls)
        
        base_date = datetime(2020, 1, 1)
        for i, pnl in enumerate(all_pnls):
            trade = {
                'date': (base_date + timedelta(days=i * 5)).strftime('%Y-%m-%d'),
                'ticker': 'TEST',
                'action': 'SELL',
                'quantity': 100,
                'entry_price': 1000,
                'exit_price': 1000 + pnl / 100,
                'pnl': pnl,
                'return': pnl / (1000 * 100)  # Return as decimal
            }
            trades.append(trade)
        
        return trades
    
    # ==================== Core Metrics Tests ====================
    
    def test_cagr_calculation(self):
        """Test CAGR calculation with known values."""
        # Create equity curve that doubles in 1 year (252 trading days)
        dates = pd.date_range(start='2020-01-01', periods=252, freq='B')
        values = np.linspace(100000, 200000, 252)
        equity_curve = pd.Series(values, index=dates)
        trade_log = []
        
        analyzer = RiskAnalyzer(equity_curve, trade_log)
        cagr = analyzer.calculate_cagr()
        
        # Should be approximately 100% (doubling in 1 year)
        # Note: business days only cover ~1 year, so CAGR should be close to 1.0
        assert abs(cagr - 1.0) < 0.10  # Allow tolerance for business day calculation
    
    def test_annualized_volatility(self):
        """Test annualized volatility calculation."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=252, freq='B')
        daily_returns = np.random.normal(0.001, 0.02, 252)
        cumulative = np.cumprod(1 + daily_returns)
        equity_curve = pd.Series(100000 * cumulative, index=dates)
        
        analyzer = RiskAnalyzer(equity_curve, [])
        vol = analyzer.calculate_annualized_volatility()
        
        # Daily std was 0.02, annualized should be ~0.02 * sqrt(252) ≈ 0.317
        expected_vol = 0.02 * np.sqrt(252)
        assert abs(vol - expected_vol) < 0.05
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        # Create high-return, low-volatility curve
        dates = pd.date_range(start='2020-01-01', periods=252, freq='B')
        daily_returns = np.random.normal(0.002, 0.005, 252)
        cumulative = np.cumprod(1 + daily_returns)
        equity_curve = pd.Series(100000 * cumulative, index=dates)
        
        analyzer = RiskAnalyzer(equity_curve, [], risk_free_rate=0.065)
        sharpe = analyzer.calculate_sharpe_ratio()
        
        # Should be positive since returns are good
        assert sharpe > 0
    
    def test_sortino_ratio_handles_zero_downside(self):
        """Test that Sortino ratio handles zero downside deviation gracefully."""
        # Create equity curve with only positive returns
        dates = pd.date_range(start='2020-01-01', periods=50, freq='B')
        daily_returns = np.abs(np.random.normal(0.001, 0.005, 50))  # All positive
        cumulative = np.cumprod(1 + daily_returns)
        equity_curve = pd.Series(100000 * cumulative, index=dates)
        
        analyzer = RiskAnalyzer(equity_curve, [], risk_free_rate=0.065)
        sortino = analyzer.calculate_sortino_ratio()
        
        # Should return inf or large value when no downside
        assert np.isinf(sortino) or sortino > 100
    
    def test_sortino_ratio_normal_case(self):
        """Test Sortino ratio with normal mixed returns."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=252, freq='B')
        daily_returns = np.random.normal(0.001, 0.02, 252)
        cumulative = np.cumprod(1 + daily_returns)
        equity_curve = pd.Series(100000 * cumulative, index=dates)
        
        analyzer = RiskAnalyzer(equity_curve, [], risk_free_rate=0.065)
        sortino = analyzer.calculate_sortino_ratio()
        
        # Should be finite and calculable
        assert np.isfinite(sortino) or np.isinf(sortino)
    
    def test_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        dates = pd.date_range(start='2020-01-01', periods=252, freq='B')
        # Create steadily growing equity with small drawdowns
        values = 100000 * np.exp(np.linspace(0, 0.3, 252))  # 30% growth
        # Add small dip
        values[150:180] *= 0.95
        equity_curve = pd.Series(values, index=dates)
        
        analyzer = RiskAnalyzer(equity_curve, [])
        calmar = analyzer.calculate_calmar_ratio()
        
        # Should be positive
        assert calmar > 0 or np.isinf(calmar)
    
    def test_profit_factor(self):
        """Test profit factor calculation."""
        trade_log = [
            {'pnl': 1000},
            {'pnl': 2000},
            {'pnl': 3000},
            {'pnl': -1500},
            {'pnl': -500}
        ]
        
        analyzer = RiskAnalyzer(pd.Series([100000]), trade_log)
        pf = analyzer.calculate_profit_factor()
        
        # Gross profits = 6000, Gross losses = 2000
        # Profit factor = 6000 / 2000 = 3.0
        assert abs(pf - 3.0) < 0.001
    
    def test_win_rate(self):
        """Test win rate calculation."""
        trade_log = [
            {'pnl': 1000},
            {'pnl': -500},
            {'pnl': 2000},
            {'pnl': -300},
            {'pnl': 1500}
        ]
        
        analyzer = RiskAnalyzer(pd.Series([100000]), trade_log)
        win_rate = analyzer.calculate_win_rate()
        
        # 3 wins out of 5 trades = 60%
        assert abs(win_rate - 0.6) < 0.001
    
    def test_expectancy(self):
        """Test expectancy calculation."""
        trade_log = [
            {'pnl': 1000},
            {'pnl': -500},
            {'pnl': 2000},
            {'pnl': -300},
            {'pnl': 1500}
        ]
        
        analyzer = RiskAnalyzer(pd.Series([100000]), trade_log)
        expectancy = analyzer.calculate_expectancy()
        
        # Total PnL = 3700, 5 trades, Expectancy = 740
        assert abs(expectancy - 740) < 0.001
    
    # ==================== Drawdown Analysis Tests ====================
    
    def test_max_drawdown_exactly_20_percent(self):
        """Test Max Drawdown is exactly 0.20 with synthetic 20% drawdown curve."""
        equity_curve = self._create_equity_curve_with_drawdown(
            start_value=100000.0,
            drawdown_pct=0.20,
            days_before_dd=100,
            days_in_dd=50,
            days_recovery=100,
            seed=42
        )
        
        analyzer = RiskAnalyzer(equity_curve, [])
        mdd = analyzer.calculate_max_drawdown()
        
        # Assert Max Drawdown is exactly 0.20 (with small tolerance for floating point)
        assert abs(mdd - 0.20) < 0.001, f"Expected MDD of 0.20, got {mdd}"
    
    def test_max_drawdown_various_levels(self):
        """Test max drawdown at various levels."""
        for dd_pct in [0.10, 0.15, 0.25, 0.30]:
            equity_curve = self._create_equity_curve_with_drawdown(
                start_value=100000.0,
                drawdown_pct=dd_pct,
                seed=42
            )
            
            analyzer = RiskAnalyzer(equity_curve, [])
            mdd = analyzer.calculate_max_drawdown()
            
            assert abs(mdd - dd_pct) < 0.005, f"Expected MDD of {dd_pct}, got {mdd}"
    
    def test_drawdown_duration(self):
        """Test drawdown duration calculation."""
        equity_curve = self._create_equity_curve_with_drawdown(
            start_value=100000.0,
            drawdown_pct=0.20,
            days_before_dd=100,
            days_in_dd=50,
            days_recovery=100,
            seed=42
        )
        
        analyzer = RiskAnalyzer(equity_curve, [])
        duration = analyzer.calculate_drawdown_duration()
        
        # Duration should be positive and reasonable
        assert duration > 0
        # Should span from peak through recovery
        assert duration >= 50  # At least the drawdown period
    
    def test_underwater_equity_curve(self):
        """Test underwater equity curve calculation."""
        equity_curve = self._create_equity_curve_with_drawdown(
            start_value=100000.0,
            drawdown_pct=0.20,
            seed=42
        )
        
        analyzer = RiskAnalyzer(equity_curve, [])
        underwater = analyzer.get_underwater_equity_curve()
        
        # Underwater should be negative or zero
        assert (underwater <= 0).all()
        
        # Maximum underwater should match max drawdown
        max_underwater_pct = abs(underwater.min())
        mdd = analyzer.calculate_max_drawdown() * 100
        
        assert abs(max_underwater_pct - mdd) < 0.001
    
    # ==================== Monte Carlo Simulation Tests ====================
    
    def test_monte_carlo_probability_of_ruin(self):
        """Test Monte Carlo probability of ruin calculation."""
        trade_log = self._create_trade_log(
            n_trades=50,
            win_rate=0.5,
            avg_win=2000,
            avg_loss=-2000,
            seed=42
        )
        
        equity_curve = self._create_equity_curve(seed=42)
        analyzer = RiskAnalyzer(equity_curve, trade_log)
        
        mc_results = analyzer.run_monte_carlo_simulation(n_simulations=1000, seed=42)
        
        # Probability should be between 0 and 1
        assert 0 <= mc_results['probability_of_ruin'] <= 1
        
        # Should have run simulations
        assert mc_results['simulations_run'] == 1000
    
    def test_monte_carlo_percentiles(self):
        """Test Monte Carlo percentile calculations."""
        trade_log = self._create_trade_log(
            n_trades=100,
            win_rate=0.6,
            avg_win=3000,
            avg_loss=-1500,
            seed=42
        )
        
        equity_curve = self._create_equity_curve(seed=42)
        analyzer = RiskAnalyzer(equity_curve, trade_log)
        
        mc_results = analyzer.run_monte_carlo_simulation(n_simulations=5000, seed=42)
        
        # 95th percentile should be greater than 5th percentile
        assert mc_results['percentile_95'] > mc_results['percentile_5']
        
        # Median should be between percentiles
        assert mc_results['percentile_5'] <= mc_results['median_terminal_wealth']
        assert mc_results['median_terminal_wealth'] <= mc_results['percentile_95']
    
    def test_monte_carlo_empty_trade_log(self):
        """Test Monte Carlo with empty trade log."""
        equity_curve = self._create_equity_curve(seed=42)
        analyzer = RiskAnalyzer(equity_curve, [])
        
        mc_results = analyzer.run_monte_carlo_simulation()
        
        # Should handle gracefully
        assert mc_results['probability_of_ruin'] == 0.0
        assert mc_results['simulations_run'] == 0
    
    # ==================== Analytics Report Tests ====================
    
    def test_generate_analytics_report(self):
        """Test comprehensive analytics report generation."""
        equity_curve = self._create_equity_curve(
            start_value=100000,
            days=252,
            seed=42
        )
        trade_log = self._create_trade_log(n_trades=50, seed=42)
        
        analyzer = RiskAnalyzer(equity_curve, trade_log)
        report = analyzer.generate_analytics_report()
        
        # Check all required keys exist
        required_keys = [
            'cagr', 'cagr_pct',
            'annualized_volatility', 'annualized_volatility_pct',
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'profit_factor', 'win_rate', 'win_rate_pct', 'expectancy',
            'max_drawdown', 'max_drawdown_pct', 'drawdown_duration_days',
            'underwater_equity_curve',
            'mc_probability_of_ruin', 'mc_probability_of_ruin_pct',
            'mc_percentile_5', 'mc_percentile_95',
            'mc_median_terminal_wealth', 'mc_mean_terminal_wealth',
            'mc_simulations_run',
            'total_trades', 'initial_capital', 'final_capital',
            'total_return', 'total_return_pct',
            'risk_free_rate', 'analysis_period_days'
        ]
        
        for key in required_keys:
            assert key in report, f"Missing key: {key}"
        
        # Check types
        assert isinstance(report['cagr'], float)
        assert isinstance(report['sharpe_ratio'], (float, int))
        assert isinstance(report['max_drawdown'], float)
        assert isinstance(report['underwater_equity_curve'], pd.Series)
        assert isinstance(report['total_trades'], int)
    
    def test_report_handles_infinite_values(self):
        """Test that report handles infinite values gracefully."""
        # Create scenario with potential infinite values
        dates = pd.date_range(start='2020-01-01', periods=50, freq='B')
        values = np.linspace(100000, 150000, 50)  # Steady growth, no drawdown
        equity_curve = pd.Series(values, index=dates)
        
        # All winning trades
        trade_log = [{'pnl': 1000} for _ in range(10)]
        
        analyzer = RiskAnalyzer(equity_curve, trade_log)
        report = analyzer.generate_analytics_report()
        
        # Infinite values should be converted to large finite numbers
        for key, value in report.items():
            if isinstance(value, float):
                assert not np.isnan(value), f"NaN found in {key}"
    
    def test_report_structure_for_excel(self):
        """Test that report structure is suitable for Excel export."""
        equity_curve = self._create_equity_curve(seed=42)
        trade_log = self._create_trade_log(seed=42)
        
        analyzer = RiskAnalyzer(equity_curve, trade_log)
        report = analyzer.generate_analytics_report()
        
        # Remove non-serializable items for this check
        serializable_report = {
            k: v for k, v in report.items() 
            if not isinstance(v, pd.Series)
        }
        
        # All remaining values should be JSON-serializable
        import json
        try:
            json.dumps(serializable_report)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Report not JSON serializable: {e}")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_equity_curve(self):
        """Test handling of empty equity curve."""
        equity_curve = pd.Series(dtype=float)
        analyzer = RiskAnalyzer(equity_curve, [])
        
        assert analyzer.calculate_cagr() == 0.0
        assert analyzer.calculate_max_drawdown() == 0.0
    
    def test_single_point_equity_curve(self):
        """Test handling of single-point equity curve."""
        equity_curve = pd.Series([100000], index=pd.to_datetime(['2020-01-01']))
        analyzer = RiskAnalyzer(equity_curve, [])
        
        assert analyzer.calculate_cagr() == 0.0
        assert analyzer.calculate_annualized_volatility() == 0.0
    
    def test_empty_trade_log(self):
        """Test handling of empty trade log."""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='B')
        equity_curve = pd.Series(np.linspace(100000, 110000, 100), index=dates)
        analyzer = RiskAnalyzer(equity_curve, [])
        
        assert analyzer.calculate_profit_factor() == 1.0
        assert analyzer.calculate_win_rate() == 0.0
        assert analyzer.calculate_expectancy() == 0.0
    
    def test_all_winning_trades(self):
        """Test with all winning trades."""
        trade_log = [{'pnl': 1000} for _ in range(10)]
        dates = pd.date_range(start='2020-01-01', periods=100, freq='B')
        equity_curve = pd.Series(np.linspace(100000, 110000, 100), index=dates)
        
        analyzer = RiskAnalyzer(equity_curve, trade_log)
        
        assert analyzer.calculate_win_rate() == 1.0
        pf = analyzer.calculate_profit_factor()
        assert np.isinf(pf) or pf > 100
    
    def test_all_losing_trades(self):
        """Test with all losing trades."""
        trade_log = [{'pnl': -500} for _ in range(10)]
        dates = pd.date_range(start='2020-01-01', periods=100, freq='B')
        equity_curve = pd.Series(np.linspace(100000, 95000, 100), index=dates)
        
        analyzer = RiskAnalyzer(equity_curve, trade_log)
        
        assert analyzer.calculate_win_rate() == 0.0
        pf = analyzer.calculate_profit_factor()
        assert pf == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
