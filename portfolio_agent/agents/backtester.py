"""Backtester agent — CLI-facing orchestration for running backtests.

Thin glue: resolve the configured strategy, construct the unified
BacktestEngine (serial, CPU-parallel, or GPU-batched depending on the
strategy), run it, compute risk analytics, and export the single canonical
Excel report.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from portfolio_agent.config.schema import AppConfig
from portfolio_agent.strategies.registry import load_strategy
from portfolio_agent.strategies.types import RiskParams
from portfolio_agent.src.backtest_engine import BacktestEngine
from portfolio_agent.src.backtest_reporting import export_backtest_excel
from portfolio_agent.src.risk_analytics import RiskAnalyzer
from portfolio_agent.src.trial_log import Trial, config_hash, record_trial

logger = logging.getLogger(__name__)


class BacktesterAgent:
    """Runs a backtest for a configured strategy and exports the Excel report."""

    def _record_trial(self, analytics_report: Dict[str, Any], strategy_name: str) -> None:
        """Append this configuration and its Sharpe to the trial log.

        Never raises: losing a log line must not lose a backtest.
        """
        try:
            record_trial(
                Trial(
                    run_id=uuid.uuid4().hex[:12],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    config_hash=config_hash(self.config),
                    sharpe=float(analytics_report.get('sharpe_ratio', 0.0) or 0.0),
                    cagr=float(analytics_report.get('cagr', 0.0) or 0.0),
                    max_drawdown=float(analytics_report.get('max_drawdown', 0.0) or 0.0),
                    n_observations=int(analytics_report.get('analysis_period_days', 0) or 0),
                    n_trades=int(analytics_report.get('total_trades', 0) or 0),
                    strategy=strategy_name,
                    parameters={
                        'strategy': strategy_name,
                        'simulation_method': self.config.simulation.method,
                        'use_garch_volatility': self.config.simulation.use_garch_volatility,
                        'atr_stop_multiplier': self.config.risk.atr_stop_multiplier,
                        'atr_target_multiplier': self.config.risk.atr_target_multiplier,
                        'use_kelly_sizing': self.config.risk.use_kelly_sizing,
                        'target_prob_profit': self.config.compliance.target_prob_profit,
                    },
                ),
                path=self.trial_log_path,
            )
        except Exception:  # pragma: no cover - defensive
            logger.debug("Could not record the backtest trial", exc_info=True)

    def __init__(
        self,
        config: AppConfig,
        strategy_type: Optional[str] = None,
        strategy_config_path: Optional[str] = None,
        inference_device: str = "cpu",
        parallel: bool = False,
        max_workers: Optional[int] = None,
        show_progress: bool = True,
        trial_log_path: Optional[str] = None,
    ):
        """Initialize the backtester agent.

        Args:
            config: Application configuration.
            strategy_type: Registered strategy name (e.g. "rule_based", "lstm",
                "ensemble"). Defaults to config.strategy.type.
            strategy_config_path: Optional override for the strategy's YAML
                config file (e.g. a UMA ensemble file under
                config/strategies/). Defaults to config.strategy.config_path.
            inference_device: Device for ML strategy inference ("cpu", "cuda", "mps").
            parallel: Whether to parallelize rule-based signal generation across CPU workers.
            max_workers: Max worker processes when parallel=True.
            show_progress: Draw progress bars for data loading and the day-by-day
                replay. On by default: this is the interactive entry point, and a
                multi-year backtest over a few thousand tickers otherwise prints
                nothing at all between "Running backtest..." and the report.
            trial_log_path: Where to append this run's trial record. Defaults
                to <paths.output_dir>/trials.jsonl. Every backtest is a trial,
                and the Deflated Sharpe Ratio needs the count.
        """
        self.config = config
        self.show_progress = show_progress
        self.strategy_config = config.strategy.model_copy(deep=True)
        if strategy_type:
            self.strategy_config.type = strategy_type
        if strategy_config_path:
            self.strategy_config.config_path = strategy_config_path
        self.strategy_config.params = {**self.strategy_config.params, "device": inference_device}
        self.inference_device = inference_device
        self.parallel = parallel
        self.max_workers = max_workers
        self.trial_log_path = trial_log_path or str(
            Path(config.paths.output_dir) / "trials.jsonl"
        )

        logger.info(f"BacktesterAgent initialized: strategy={self.strategy_config.type}, "
                    f"device={inference_device}, parallel={parallel}")

    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float,
        universe_tickers: List[str],
        output_file: str,
    ) -> Dict[str, Any]:
        """Run a complete backtest and export the Excel report.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            initial_capital: Initial capital in INR.
            universe_tickers: List of ticker symbols to trade.
            output_file: Path for the Excel report.

        Returns:
            Dictionary with backtest results and metrics.
        """
        strategy = load_strategy(self.strategy_config)
        if hasattr(strategy, "load"):
            if not strategy.load():
                raise RuntimeError(
                    f"Strategy '{strategy.name}' failed to load (missing trained model checkpoint?)"
                )

        risk_params = RiskParams.from_app_config(self.config)

        engine = BacktestEngine(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            universe_tickers=universe_tickers,
            strategy=strategy,
            risk_params=risk_params,
            parallel=self.parallel,
            max_workers=self.max_workers,
            mc_horizon_days=self.config.simulation.mc_horizon_days,
            mc_simulations=self.config.simulation.mc_simulations,
            mc_seed=self.config.simulation.random_seed,
            use_garch_volatility=self.config.simulation.use_garch_volatility,
            mc_method=self.config.simulation.method,
            mc_block_size_days=self.config.simulation.block_size_days,
            mc_jump_intensity_per_year=self.config.simulation.jump_intensity_per_year,
            mc_jump_mean=self.config.simulation.jump_mean,
            mc_jump_volatility=self.config.simulation.jump_volatility,
            use_kelly_sizing=self.config.risk.use_kelly_sizing,
            kelly_fraction=self.config.risk.kelly_fraction,
            kelly_min_trades=self.config.risk.kelly_min_trades,
            kelly_shrinkage_strength=self.config.risk.kelly_shrinkage_strength,
            max_sector_pct=self.config.risk.max_sector_pct,
            max_unknown_sector_pct=self.config.risk.max_unknown_sector_pct,
            sector_map_csv=self.config.paths.sector_map_csv,
            max_portfolio_drawdown_pct=self.config.risk.max_portfolio_drawdown_pct,
            drawdown_reentry_pct=self.config.risk.drawdown_reentry_pct,
            drawdown_halt_max_days=self.config.risk.drawdown_halt_max_days,
            benchmark_symbol=self.config.data.benchmark_symbol,
            exit_on_lower_circuit_lock=self.config.risk.exit_on_lower_circuit_lock,
            liquidate_on_drawdown_halt=self.config.risk.liquidate_on_drawdown_halt,
            show_progress=self.show_progress,
        )

        logger.info(f"Running backtest from {start_date} to {end_date} with strategy '{strategy.name}'")
        if self.show_progress:
            print(
                f"Replaying {start_date} to {end_date} with strategy '{strategy.name}' "
                f"over {len(engine.master_date_index)} trading days"
            )
        engine.run_backtest()

        # The two post-loop phases are slow enough (a Monte Carlo ruin
        # simulation and a multi-sheet Excel write) that a terminal which went
        # quiet right after the progress bar filled looks like a hang.
        if self.show_progress:
            print("Computing risk analytics...")

        analyzer = RiskAnalyzer(
            daily_equity_curve=engine.daily_equity_curve,
            trade_log=engine.trade_log,
            random_seed=self.config.simulation.random_seed,
            trial_log_path=self.trial_log_path,
        )
        analytics_report = analyzer.generate_analytics_report()

        # Record the trial BEFORE the report is written, so a run that dies
        # during the Excel export still counts toward N. The Deflated Sharpe
        # Ratio is only meaningful if the denominator includes the trials that
        # were abandoned -- those are precisely the ones that inflate the
        # maximum. See src/trial_log.py.
        self._record_trial(analytics_report, strategy_name=strategy.name)

        # Percentage metrics are handed to the exporter in PERCENT units
        # (18.5 == 18.5%) — the contract documented in
        # src/backtest_reporting.py::SUMMARY_METRICS. CAGR used to be passed as
        # a 0-1 decimal here while everything beside it was already a percent,
        # so the sheet's scaling could only be right for one of them.
        analytics_for_export = {
            'cagr': analytics_report.get('cagr_pct', 0),
            'sharpe': analytics_report.get('sharpe_ratio', 0),
            'sortino': analytics_report.get('sortino_ratio', 0),
            # Negated: RiskAnalyzer reports drawdown as a positive magnitude,
            # but a drawdown reads as a loss in the report (and the sheet's
            # "worse than -20%" conditional formatting can only fire on a
            # negative number).
            'max_drawdown': -abs(analytics_report.get('max_drawdown_pct', 0)),
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
            'model_used': strategy.name != 'rule_based',
            'model_name': strategy.name,
        }

        if self.show_progress:
            print(f"Writing report to {output_file}...")

        export_backtest_excel(
            analytics=analytics_for_export,
            equity_curve=engine.daily_equity_curve,
            trade_log=engine.trade_log,
            brain_evolution=engine.brain_evolution,
            daily_activity_log=engine.daily_activity_log,
            filepath=output_file
        )

        logger.info(f"Backtest complete. Report saved to: {output_file}")
        if engine.circuit_breaker_log:
            halts = sum(1 for e in engine.circuit_breaker_log if e['event'] == 'HALT')
            logger.warning(
                f"Drawdown circuit breaker tripped {halts} time(s) during this run; "
                f"see 'circuit_breaker_log' in the returned results"
            )

        if engine.exit_trigger_log:
            logger.info(
                f"Forced exits fired {len(engine.exit_trigger_log)} time(s) during this run "
                f"(lower-circuit locks and/or drawdown liquidation); see 'exit_trigger_log' "
                f"in the returned results"
            )

        return {
            'status': 'success',
            'output_file': output_file,
            'metrics': analytics_report,
            'trade_count': len(engine.trade_log),
            'strategy': strategy.name,
            'circuit_breaker_log': engine.circuit_breaker_log,
            'exit_trigger_log': engine.exit_trigger_log,
        }


def run_backtest_cli(
    config: AppConfig,
    strategy_type: Optional[str] = None,
    strategy_config_path: Optional[str] = None,
    device: str = "cpu",
    output_file: Optional[str] = None,
    parallel: bool = False,
    max_workers: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Run backtest from CLI with configuration.

    Args:
        config: Application configuration.
        strategy_type: Registered strategy name override (defaults to config.strategy.type).
        strategy_config_path: Optional strategy YAML override (e.g. a UMA ensemble file).
        device: Device for ML strategy inference.
        output_file: Optional output file path override.
        parallel: Whether to parallelize rule-based signal generation across CPU workers.
        max_workers: Max worker processes when parallel=True.
        start_date: Optional explicit start date (YYYY-MM-DD), overrides config.backtest.start_years_ago.
        end_date: Optional explicit end date (YYYY-MM-DD), defaults to today.
        show_progress: Draw progress bars during the run (pass False for
            unattended/scripted runs).

    Returns:
        Dictionary with backtest results.
    """
    from datetime import timedelta

    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
    start = pd.Timestamp(start_date) if start_date else end - timedelta(days=config.backtest.start_years_ago * 365)

    if output_file is None:
        output_file = config.paths.backtest_excel_output
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    from portfolio_agent.src.universe import resolve_backtest_universe
    tickers = resolve_backtest_universe(
        force_full_download=False,
        max_tickers=config.data.universe_size
    )

    if not tickers:
        raise ValueError("No tickers available for backtest")

    agent = BacktesterAgent(
        config=config,
        strategy_type=strategy_type,
        strategy_config_path=strategy_config_path,
        inference_device=device,
        parallel=parallel,
        max_workers=max_workers,
        show_progress=show_progress,
    )

    return agent.run_backtest(
        start_date=start.strftime('%Y-%m-%d'),
        end_date=end.strftime('%Y-%m-%d'),
        initial_capital=config.backtest.initial_capital,
        universe_tickers=tickers,
        output_file=output_file
    )
