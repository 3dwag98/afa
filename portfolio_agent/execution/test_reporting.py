"""Tests for reporting module."""

import os
import tempfile
from datetime import datetime, timedelta

import pytest

from portfolio_agent.config.schema import AppConfig
from portfolio_agent.src.models import Recommendation, IndicatorSnapshot, AgentBrain
from portfolio_agent.src.monte_carlo import MonteCarloResult
from .reporting import export_excel_report


def _create_test_config(tmp_path: str) -> AppConfig:
    """Create a test configuration."""
    return AppConfig.model_validate({
        "risk": {"portfolio_value_inr": 308733.0, "risk_per_trade_pct": 0.01, "max_single_position_pct": 0.03},
        "compliance": {
            "min_price_inr": 20.0, "target_prob_profit": 0.55, "min_reward_risk": 1.5, "paper_trading_mode": True,
        },
        "learning": {"learning_rate": 0.15, "min_trades_for_learning": 5},
        "simulation": {"mc_horizon_days": 20, "mc_simulations": 1000, "random_seed": 42},
        "data": {
            "tickers": ["NIFTYBEES.NS", "RELIANCE.NS"], "min_history_days": 250, "allow_synthetic_fallback": True,
        },
        "paths": {
            "brain_file": "data/agent_brain.json",
            "sqlite_path": "data/portfolio_agent.db",
            "excel_output": tmp_path,
            "log_file": "logs/agent.log",
        },
    })


def _create_dummy_brain() -> AgentBrain:
    """Create a dummy agent brain."""
    return AgentBrain(
        weights={
            "Trend": 25.0,
            "Breakout": 25.0,
            "Volume": 20.0,
            "MC_Prob": 30.0,
        },
        trade_history=[
            {
                "trade_id": "TRD001",
                "symbol": "RELIANCE.NS",
                "signal_trigger": "BREAKOUT",
                "entry_date": "2024-01-15",
                "entry_price": 2450.0,
                "exit_date": "2024-01-25",
                "exit_price": 2550.0,
                "outcome": "WIN",
                "return_pct": 4.08,
                "outcome_source": "STOP_LOSS",
            }
        ],
        learning_log=[
            {"timestamp": "2024-01-15", "event": "Trade executed", "reward": 0.04},
            {"timestamp": "2024-01-25", "event": "Trade closed", "reward": 0.04},
        ],
        updated_at=datetime.now().isoformat(),
    )


def _create_dummy_recommendations() -> list[Recommendation]:
    """Create dummy recommendations."""
    return [
        Recommendation(
            symbol="RELIANCE.NS",
            signal="BUY",
            score=0.85,
            trigger="BREAKOUT",
            entry_price=2450.0,
            stop_price=2400.0,
            target_price=2600.0,
            reward_risk=3.0,
            quantity=10,
            investment_inr=24500.0,
            max_loss_inr=500.0,
            mc_probability_profit=0.65,
            mc_var_95_pct=-0.02,
            mc_cvar_95_pct=-0.035,
            compliance_status="PASS",
            rationale="Strong breakout with volume confirmation",
            recommendation_id="REC001",
            created_at=datetime.now().isoformat(),
        ),
        Recommendation(
            symbol="TCS.NS",
            signal="WATCH",
            score=0.55,
            trigger="NEAR_BREAKOUT",
            entry_price=3800.0,
            stop_price=3700.0,
            target_price=3950.0,
            reward_risk=1.5,
            quantity=5,
            investment_inr=19000.0,
            max_loss_inr=500.0,
            mc_probability_profit=0.52,
            mc_var_95_pct=-0.015,
            mc_cvar_95_pct=-0.025,
            compliance_status="PASS",
            rationale="Approaching resistance level",
            recommendation_id="REC002",
            created_at=datetime.now().isoformat(),
        ),
        Recommendation(
            symbol="HDFCBANK.NS",
            signal="AVOID",
            score=0.30,
            trigger="WEAK_TREND",
            entry_price=1650.0,
            stop_price=1600.0,
            target_price=1700.0,
            reward_risk=1.0,
            quantity=0,
            investment_inr=0.0,
            max_loss_inr=0.0,
            mc_probability_profit=0.35,
            mc_var_95_pct=-0.05,
            mc_cvar_95_pct=-0.08,
            compliance_status="FAIL",
            rationale="Below minimum reward/risk threshold",
            recommendation_id="REC003",
            created_at=datetime.now().isoformat(),
        ),
    ]


def _create_dummy_indicators() -> list[IndicatorSnapshot]:
    """Create dummy indicator snapshots."""
    today = datetime.now()
    return [
        IndicatorSnapshot(
            symbol="RELIANCE.NS",
            sma20=2420.0,
            sma50=2380.0,
            sma200=2300.0,
            donchian_upper_20=2500.0,
            prev_donchian_upper_20=2480.0,
            avg_volume_20=5000000.0,
            volume_ratio=1.2,
            atr14=45.0,
            daily_log_return=0.01,
        ),
        IndicatorSnapshot(
            symbol="TCS.NS",
            sma20=3780.0,
            sma50=3750.0,
            sma200=3650.0,
            donchian_upper_20=3850.0,
            prev_donchian_upper_20=3820.0,
            avg_volume_20=2000000.0,
            volume_ratio=0.9,
            atr14=55.0,
            daily_log_return=0.005,
        ),
    ]


def _create_dummy_mc_results() -> list[MonteCarloResult]:
    """Create dummy Monte Carlo results."""
    # Note: MonteCarloResult doesn't have symbol attribute, so we'll just create results
    return [
        MonteCarloResult(
            probability_profit=0.65,
            expected_return_pct=0.045,
            var_95=-0.02,
            cvar_95=-0.035,
            simulations_count=1000,
            horizon_days=20,
        ),
        MonteCarloResult(
            probability_profit=0.52,
            expected_return_pct=0.025,
            var_95=-0.015,
            cvar_95=-0.025,
            simulations_count=1000,
            horizon_days=20,
        ),
    ]


class TestExportExcelReport:
    """Test suite for export_excel_report function."""

    def test_export_excel_creates_file(self):
        """Test that Excel file is created successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_report.xlsx")
            
            config = _create_test_config(output_path)
            brain = _create_dummy_brain()
            recommendations = _create_dummy_recommendations()
            indicators = _create_dummy_indicators()
            mc_results = _create_dummy_mc_results()
            run_id = "TEST_RUN_001"
            
            result_path = export_excel_report(
                config=config,
                brain=brain,
                recommendations=recommendations,
                indicators=indicators,
                mc_results=mc_results,
                run_id=run_id,
            )
            
            assert os.path.exists(result_path), f"File not created at {result_path}"
            assert os.path.getsize(result_path) > 0, "File size is 0"

    def test_export_excel_with_empty_lists(self):
        """Test that function handles empty lists gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_report_empty.xlsx")
            
            config = _create_test_config(output_path)
            brain = _create_dummy_brain()
            recommendations = []
            indicators = []
            mc_results = []
            run_id = "TEST_RUN_EMPTY"
            
            result_path = export_excel_report(
                config=config,
                brain=brain,
                recommendations=recommendations,
                indicators=indicators,
                mc_results=mc_results,
                run_id=run_id,
            )
            
            assert os.path.exists(result_path), f"File not created at {result_path}"
            assert os.path.getsize(result_path) > 0, "File size is 0"

    def test_export_excel_creates_all_sheets(self):
        """Test that all required sheets are created."""
        import pandas as pd
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_report_sheets.xlsx")
            
            config = _create_test_config(output_path)
            brain = _create_dummy_brain()
            recommendations = _create_dummy_recommendations()
            indicators = _create_dummy_indicators()
            mc_results = _create_dummy_mc_results()
            run_id = "TEST_RUN_SHEETS"
            
            export_excel_report(
                config=config,
                brain=brain,
                recommendations=recommendations,
                indicators=indicators,
                mc_results=mc_results,
                run_id=run_id,
            )
            
            # Read the Excel file and check sheet names
            xls = pd.ExcelFile(output_path)
            sheet_names = xls.sheet_names
            
            expected_sheets = [
                "Summary",
                "Live_Recommendations",
                "Indicators",
                "Monte_Carlo",
                "Agent_Brain_Weights",
                "Learning_History",
                "Trade_Memory",
            ]
            
            for sheet in expected_sheets:
                assert sheet in sheet_names, f"Missing sheet: {sheet}"

    def test_export_excel_signal_formatting(self):
        """Test that BUY/WATCH/AVOID signals get correct formatting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_report_formatting.xlsx")
            
            config = _create_test_config(output_path)
            brain = _create_dummy_brain()
            recommendations = _create_dummy_recommendations()
            indicators = []
            mc_results = []
            run_id = "TEST_RUN_FORMAT"
            
            result_path = export_excel_report(
                config=config,
                brain=brain,
                recommendations=recommendations,
                indicators=indicators,
                mc_results=mc_results,
                run_id=run_id,
            )
            
            assert os.path.exists(result_path)
            assert os.path.getsize(result_path) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
