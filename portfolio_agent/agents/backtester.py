"""Backtester agent — CLI-facing orchestration for running backtests.

Thin glue: resolve the configured strategy, construct the unified
BacktestEngine (serial, CPU-parallel, or GPU-batched depending on the
strategy), run it, compute risk analytics, and export the single canonical
Excel report.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from portfolio_agent.config.schema import AppConfig
from portfolio_agent.strategies.registry import load_strategy
from portfolio_agent.strategies.types import RiskParams
from portfolio_agent.src.backtest_engine import BacktestEngine
from portfolio_agent.src.backtest_reporting import export_backtest_excel
from portfolio_agent.src.performance_stats import (
    Trial, config_hash, distinct_trials, log_trial, read_trials,
    trial_sharpe_variance,
)
from portfolio_agent.src.risk_analytics import RiskAnalyzer, resolve_risk_free_rate

logger = logging.getLogger(__name__)


def _load_failure_message(strategy, config) -> str:
    """Say which member failed and what to run, not just that something did.

    "Strategy 'ensemble' failed to load (missing trained model checkpoint?)" is
    an accurate sentence that leaves the reader no better off: a UMA has
    several sleeves, usually only one of them needs a checkpoint, and the
    message named none of them. This names the members that failed, the model
    path that was looked for, and the two ways forward.
    """
    failed = list(getattr(strategy, "unloadable_members", []) or [])
    model_path = getattr(config.backtest, "model_path", "the configured model path")

    if not failed:
        return (
            f"Strategy '{strategy.name}' failed to load. It expects a trained checkpoint at "
            f"{model_path}; run `portfolio-agent train` first, or pick a strategy that needs "
            f"no model (`--strategy rule_based`, `momentum`, `low_volatility`)."
        )

    return (
        f"UMA '{strategy.name}' could not load {len(failed)} member(s): {', '.join(failed)}. "
        f"Those members expect a trained checkpoint at {model_path}.\n"
        f"  - Train it:            portfolio-agent train\n"
        f"  - Or run without it:   add `drop_unavailable_members: true` to the UMA's YAML to "
        f"run the remaining members (note that this is no longer the strategy you configured)."
    )


class BacktesterAgent:
    """Runs a backtest for a configured strategy and exports the Excel report."""

    def __init__(
        self,
        config: AppConfig,
        strategy_type: Optional[str] = None,
        strategy_config_path: Optional[str] = None,
        inference_device: str = "cpu",
        parallel: bool = False,
        max_workers: Optional[int] = None,
        show_progress: bool = True,
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
        if hasattr(strategy, "load") and not strategy.load():
            raise RuntimeError(_load_failure_message(strategy, self.config))

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
            portfolio_volatility_target=self.config.risk.portfolio_volatility_target,
            covariance_lookback_days=self.config.risk.covariance_lookback_days,
            show_progress=self.show_progress,
            # The harness and both trainers read these; the engine took the
            # pipeline's own default and ignored the config. They agreed only
            # because config.yaml sets normalize false against a schema default
            # of true — and normalization adds its own `.shift(1)`, so a
            # divergence here is a second session of lag, which is what T19
            # removed.
            feature_normalize=self.config.features.normalize,
            feature_normalize_window=self.config.features.normalize_window,
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
            # Explicit rather than inherited from a constructor default: this
            # call site used to omit the argument entirely and silently report
            # every Sharpe against a hardcoded 6.5% hurdle.
            risk_free_rate=resolve_risk_free_rate(
                self.config.paths.risk_free_rate_csv,
                self.config.risk.risk_free_rate,
            ),
            random_seed=self.config.simulation.random_seed,
        )
        # The trial count comes from the log, not from memory: DSR is undefined
        # without N, and N is exactly the quantity a research process forgets.
        # Absent a log this is a single pre-registered run (N = 1, no
        # deflation), which is the honest default rather than a flattering one.
        # Deduplicated first: five recordings of two configurations is a
        # two-trial search, and deflating it as a five-trial one penalizes the
        # Sharpe against a search that never happened.
        trials = distinct_trials(read_trials(self.config.paths.trial_log))
        n_trials, sharpe_variance = trial_sharpe_variance(trials)
        analytics_report = analyzer.generate_analytics_report(
            n_trials=max(1, n_trials),
            sharpe_variance=sharpe_variance if n_trials > 1 else None,
        )
        book_risk = engine.book_risk_statistics()

        # Percentage metrics are handed to the exporter in PERCENT units
        # (18.5 == 18.5%) — the contract documented in
        # src/backtest_reporting.py::SUMMARY_METRICS. CAGR used to be passed as
        # a 0-1 decimal here while everything beside it was already a percent,
        # so the sheet's scaling could only be right for one of them.
        analytics_for_export = {
            'cagr': analytics_report.get('cagr_pct', 0),
            'sharpe': analytics_report.get('sharpe_ratio', 0),
            'probabilistic_sharpe': analytics_report.get('probabilistic_sharpe_ratio', 0),
            'deflated_sharpe': analytics_report.get('deflated_sharpe_ratio', 0),
            'n_trials': analytics_report.get('n_trials', 1),
            'sortino': analytics_report.get('sortino_ratio', 0),
            'book_volatility': book_risk.get('mean_portfolio_volatility', 0) * 100,
            'correlation_risk_multiple': book_risk.get('mean_correlation_risk_multiple', 0),
            'diversification_ratio': book_risk.get('mean_diversification_ratio', 0),
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

        # Record this run *after* reading the log, so a result is never
        # deflated against itself. The next run sees it, which is the point:
        # the count accumulates across a research session instead of being
        # reconstructed from memory at the end of one.
        self._record_trial(strategy, analytics_report, start_date, end_date)

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
            'book_risk': book_risk,
        }

    def _record_trial(
        self,
        strategy,
        analytics_report: Dict[str, Any],
        start_date: str,
        end_date: str,
    ) -> None:
        """Append this configuration and its Sharpe to the trial log.

        The parameters recorded are the ones a research session actually varies
        — they are what makes two runs different trials rather than the same
        trial twice. They are also hand-enumerated, and therefore incomplete by
        construction: every knob added to the config after this list was written
        is invisible to it. The config hash alongside them covers the whole
        resolved configuration, so trial identity does not depend on anyone
        remembering to extend this dict. Failures are swallowed deliberately: a
        research log that cannot be written is a reason to lose the count, not
        the backtest.
        """
        risk = self.config.risk
        try:
            log_trial(
                self.config.paths.trial_log,
                Trial(
                    label=f"{self.strategy_config.type}:{strategy.name}",
                    sharpe=float(analytics_report.get('sharpe_ratio', 0.0)),
                    config_hash=self._config_fingerprint(start_date, end_date),
                    parameters={
                        'strategy': self.strategy_config.type,
                        'strategy_config': self.strategy_config.config_path,
                        'start_date': start_date,
                        'end_date': end_date,
                        'atr_stop_multiplier': risk.atr_stop_multiplier,
                        'atr_target_multiplier': risk.atr_target_multiplier,
                        'use_kelly_sizing': risk.use_kelly_sizing,
                        'portfolio_volatility_target': risk.portfolio_volatility_target,
                        'simulation_method': self.config.simulation.method,
                        'use_garch_volatility': self.config.simulation.use_garch_volatility,
                        'target_transform': self.config.training.target_transform,
                    },
                    metrics={
                        'cagr': float(analytics_report.get('cagr', 0.0)),
                        'max_drawdown': float(analytics_report.get('max_drawdown', 0.0)),
                        'total_trades': float(analytics_report.get('total_trades', 0)),
                    },
                ),
            )
        except OSError as error:
            logger.warning(f"Could not append to the trial log: {error}")

    def _config_fingerprint(self, start_date: str, end_date: str) -> str:
        """Hash the whole resolved config, plus what is not in it.

        The backtest window and the strategy selection come from CLI arguments
        rather than the config file, so two runs of one config over different
        date ranges are genuinely different trials and must hash differently.

        Paths are excluded on purpose: writing the report to a different
        filename is not a new trial, and including them would make every run
        with a timestamped output path unique, which would defeat the
        deduplication entirely.
        """
        payload = self.config.model_dump(mode="json", exclude={"paths"})
        payload["_window"] = {"start": start_date, "end": end_date}
        payload["_strategy"] = {
            "type": self.strategy_config.type,
            "config_path": self.strategy_config.config_path,
            "params": self.strategy_config.params,
        }
        return config_hash(payload)


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
    universe_snapshot: Optional[str] = None,
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
        universe_snapshot: Path to a saved universe snapshot. Without one the
            names come from a seeded draw, and a draw is only comparable to
            another draw taken with the same purpose — so pinning is what lets
            a backtest and an `evaluate` run describe the same market.

    Returns:
        Dictionary with backtest results.
    """
    from datetime import timedelta

    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
    start = pd.Timestamp(start_date) if start_date else end - timedelta(days=config.backtest.start_years_ago * 365)

    if output_file is None:
        output_file = config.paths.backtest_excel_output
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # A snapshot beats a draw whenever one is supplied — it is the only way to
    # put a backtest and an evaluation on identical names. `evaluate` and
    # `train` have accepted one since T09; the backtest did not, so the two
    # sides of "does the measured IC show up in the equity curve" could not be
    # pinned together even in principle.
    from portfolio_agent.src.universe import MEASUREMENT_PURPOSE
    from portfolio_agent.training.universe import resolve_universe

    snapshot = resolve_universe(
        config,
        snapshot=universe_snapshot,
        size=config.data.universe_size,
        name="backtest",
        purpose=MEASUREMENT_PURPOSE,
    )
    tickers = list(snapshot.tickers)

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
