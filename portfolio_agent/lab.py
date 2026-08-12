"""Notebook-facing facade: train, backtest and compare on one pinned universe.

Everything here is available through the CLI and the `training` package; this
exists so a notebook cell is one line instead of fifteen, and so the *same
tickers* are used everywhere without the notebook having to remember to pass
them.

That last point is the reason this class holds a universe rather than taking
one per call. Comparing two strategies is only meaningful when both saw the
same names, and the default path does not give that: `resolve_backtest_universe`
draws from whatever is in the cache, and deliberately offsets its seed by
purpose so training and backtesting sample *differently*. A notebook that
trains in one cell and backtests in another therefore compares two models on
two different samples without anything saying so. A `Lab` resolves the universe
once, in its constructor, and every method uses it.

    from portfolio_agent.lab import Lab

    lab = Lab(universe_size=40)
    lab.save_universe("universe/experiment.json")   # reproduce this later

    run = lab.train("india_sac", epochs=50)
    print(run.summary())

    results = lab.backtest("india_sac")             # same 40 names
    report  = lab.compare(["india_sac", "lstm"])    # still the same 40
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)


class Lab:
    """A pinned universe plus the operations you want to run against it."""

    def __init__(
        self,
        *,
        tickers: Optional[Sequence[str]] = None,
        snapshot: Optional[Path | str] = None,
        universe_size: Optional[int] = None,
        name: str = "lab",
        config: Any = None,
        models_dir: Path | str = "models",
    ) -> None:
        """Resolve the universe once.

        Args:
            tickers: Explicit names — the most reproducible option, and the
                one to use when a notebook should behave identically on someone
                else's machine.
            snapshot: Path to a saved snapshot, for reproducing an earlier run.
            universe_size: Cap on a fresh draw from the cache.
            name: Label carried into reports.
            config: An AppConfig. Loaded from config.yaml when omitted.
            models_dir: Where checkpoints are read from and written to.
        """
        from portfolio_agent.config.loader import load_config
        from portfolio_agent.training.universe import resolve_universe

        self.config = config if config is not None else load_config()
        self.models_dir = Path(models_dir)
        self.universe = resolve_universe(
            self.config,
            tickers=tickers,
            snapshot=snapshot,
            size=universe_size,
            name=name,
        )
        logger.info(
            "Lab pinned to %d tickers (fingerprint %s)",
            len(self.universe), self.universe.fingerprint,
        )

    # -- universe ----------------------------------------------------------

    @property
    def tickers(self) -> List[str]:
        """The pinned names. Every method on this class uses exactly these."""
        return list(self.universe.tickers)

    @property
    def fingerprint(self) -> str:
        """Short hash of the universe — quote it when reporting results."""
        return self.universe.fingerprint

    def save_universe(self, path: Path | str) -> Path:
        """Write the universe so a later session can reproduce this comparison."""
        return self.universe.save(path)

    def __repr__(self) -> str:
        return (
            f"Lab(tickers={len(self.universe)}, "
            f"fingerprint={self.universe.fingerprint!r}, "
            f"name={self.universe.name!r})"
        )

    # -- discovery ---------------------------------------------------------

    @staticmethod
    def trainers() -> List[str]:
        """Registered training procedures."""
        from portfolio_agent.training import list_trainers

        return list_trainers()

    @staticmethod
    def strategies() -> List[str]:
        """Registered strategies."""
        from portfolio_agent.strategies.registry import get_available_strategies

        return sorted(get_available_strategies())

    @staticmethod
    def settings(trainer: str) -> Dict[str, Any]:
        """A trainer's hyperparameters and their defaults."""
        from portfolio_agent.training import get_trainer

        schema = get_trainer(trainer).config_model()
        return {name: field.default for name, field in schema.model_fields.items()}

    # -- training ----------------------------------------------------------

    def train(
        self,
        strategy: Optional[str] = None,
        *,
        trainer: Optional[str] = None,
        save: bool = True,
        model_name: Optional[str] = None,
        **overrides: Any,
    ) -> Any:
        """Train one strategy on the pinned universe.

        Args:
            strategy: Strategy to train. None runs the supervised default.
            trainer: Override the strategy's declared trainer.
            save: Write a checkpoint. Pass False to experiment without
                overwriting a model the backtests are using.
            model_name: Checkpoint basename, so a variant does not overwrite
                the production one.
            **overrides: Hyperparameters, e.g. `epochs=50, gamma=0.9`. Validated
                against the trainer's schema — an unknown name raises rather
                than being ignored.

        Returns:
            A `TrainingRun`; `run.summary()` is a dict suitable for a DataFrame.
        """
        from portfolio_agent.training import run_training_job

        return run_training_job(
            self.config,
            strategy,
            trainer=trainer,
            overrides=overrides,
            universe=self.tickers,
            models_dir=self.models_dir,
            model_name=model_name,
            save=save,
        )

    def compare(
        self,
        strategies: Sequence[str],
        *,
        save: bool = False,
        **overrides: Any,
    ) -> Any:
        """Train several strategies on the same names and tabulate them.

        Args:
            strategies: Strategy names.
            save: Whether to write checkpoints. Off by default so a comparison
                does not replace the models in `models/`.
            **overrides: Hyperparameters applied to every entry.

        Returns:
            A `BulkReport`; `report.to_frame()` is the comparison table.
        """
        from portfolio_agent.training import run_bulk
        from portfolio_agent.training.bulk import BulkJob

        jobs = [
            BulkJob(
                strategy=name,
                overrides=dict(overrides),
                model_name=name if save else None,
                save=save,
            )
            for name in strategies
        ]
        return run_bulk(
            self.config, jobs, universe=self.tickers, models_dir=self.models_dir
        )

    def sweep(
        self,
        strategy: Optional[str],
        grid: Mapping[str, Iterable[Any]],
        *,
        save: bool = False,
        **base_overrides: Any,
    ) -> Any:
        """Train the cross product of a hyperparameter grid on the same names.

        Args:
            strategy: Strategy to sweep.
            grid: Field name -> values, e.g. `{"gamma": [0.0, 0.9]}`.
            save: Whether each grid point writes a checkpoint.
            **base_overrides: Settings held fixed across the sweep.

        Returns:
            A `BulkReport`.
        """
        from portfolio_agent.training import run_bulk
        from portfolio_agent.training.bulk import sweep as build_sweep

        jobs = build_sweep(
            strategy, grid, base_overrides=base_overrides, save=save
        )
        return run_bulk(
            self.config, jobs, universe=self.tickers, models_dir=self.models_dir
        )

    # -- evaluation --------------------------------------------------------

    def backtest(
        self,
        strategy: Optional[str] = None,
        *,
        strategy_config: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        device: str = "cpu",
        output_file: Optional[str] = None,
        parallel: bool = False,
        show_progress: bool = False,
    ) -> Dict[str, Any]:
        """Backtest one strategy on the pinned universe.

        The tickers are this Lab's, not a fresh draw — which is the difference
        between "how did these two models do" and "how did these two models do
        on two different samples".

        Args:
            strategy: Strategy to run. None uses `config.strategy.type`.
            strategy_config: Optional strategy YAML override.
            start_date: ISO start; defaults to `config.backtest.start_years_ago`.
            end_date: ISO end; defaults to today.
            device: Inference device.
            output_file: Excel report path; defaults to the configured one.
            parallel: Parallelize rule-based signal generation.
            show_progress: Draw progress bars. Off by default — notebook output
                is a poor place for a redrawing bar.

        Returns:
            The backtest result dict.
        """
        from datetime import timedelta

        import pandas as pd

        from portfolio_agent.agents.backtester import BacktesterAgent

        end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
        start = (
            pd.Timestamp(start_date)
            if start_date
            else end - timedelta(days=self.config.backtest.start_years_ago * 365)
        )

        if output_file is None:
            output_file = self.config.paths.backtest_excel_output
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        agent = BacktesterAgent(
            config=self.config,
            strategy_type=strategy,
            strategy_config_path=strategy_config,
            inference_device=device,
            parallel=parallel,
            show_progress=show_progress,
        )
        return agent.run_backtest(
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            initial_capital=self.config.backtest.initial_capital,
            universe_tickers=self.tickers,
            output_file=output_file,
        )

    def train_and_backtest(
        self,
        strategy: str,
        *,
        trainer: Optional[str] = None,
        backtest_kwargs: Optional[Dict[str, Any]] = None,
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Train, then immediately backtest, on one universe.

        Returns:
            `{"run": TrainingRun, "backtest": dict | None}`. The backtest is
            skipped and `None` returned when training failed — running one
            against a stale checkpoint would report a number belonging to a
            different model.
        """
        run = self.train(strategy, trainer=trainer, **overrides)
        if not run.ok:
            logger.error("Training failed, skipping backtest: %s", run.error)
            return {"run": run, "backtest": None}

        results = self.backtest(strategy, **(backtest_kwargs or {}))
        return {"run": run, "backtest": results}
