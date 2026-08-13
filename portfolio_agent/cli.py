#!/usr/bin/env python3
"""CLI for Portfolio Agent - Autonomous Financial Advisor.

Commands:
    download-data: Download market data for the configured universe
    train: Train a strategy through its registered trainer
    train-bulk: Train several strategies, or sweep settings, on one universe
    backtest: Run backtesting simulation
    run-agent: Run the daily portfolio agent
    list-strategies: List registered strategies (rule-based, ML, UMA ensembles)
    list-trainers: List registered training procedures and their settings
    gpu-check: Report which compute devices this install can actually use
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


# Set by main() from --config, so every command loads the same file without
# each one having to thread the path through its own signature.
_ACTIVE_CONFIG_PATH = {"path": "config.yaml"}


def get_config() -> "AppConfig":
    """Load application configuration from the path --config selected."""
    from portfolio_agent.config.loader import load_config
    return load_config(_ACTIVE_CONFIG_PATH["path"])


def cmd_download_data(args) -> int:
    """Download market data command."""
    from datetime import timedelta

    import pandas as pd

    from portfolio_agent.src.data_store import fetch_and_cache
    from portfolio_agent.src.universe import resolve_backtest_universe

    config = get_config()
    source = args.source or config.data.source

    end_date = pd.Timestamp.now()
    years = args.years or config.data.default_history_years
    start_date = end_date - timedelta(days=years * 365)

    if source == "huggingface":
        # One columnar download covers the whole universe, so there is no
        # ticker list to resolve first — the dataset defines the universe, and
        # an empty local cache is the normal starting state.
        from portfolio_agent.src.hf_dataset import sync_hf_to_cache

        dataset_id = args.hf_dataset or config.data.hf_dataset_id
        revision = args.hf_revision or config.data.hf_revision
        print(
            f"Loading {dataset_id}/{config.data.hf_asset_dir} "
            f"(revision={revision or 'main'}) from HuggingFace, keeping {years} years "
            f"of history ({start_date:%Y-%m-%d} to {end_date:%Y-%m-%d})"
        )
        try:
            written = sync_hf_to_cache(
                dataset_id=dataset_id,
                revision=revision,
                asset_dir=config.data.hf_asset_dir,
                adjust_prices=config.data.hf_adjust_prices,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                max_symbols=args.universe_size,
                progress=True,
                skip_existing=not args.force,
                workers=args.workers or 8,
            )
        except Exception as e:
            print(f"Error: HuggingFace ingest failed: {e}")
            print("Re-run with --source yfinance to use the per-ticker download path instead.")
            return 1

        if not written:
            print(f"Error: {dataset_id} yielded no tickers for the requested window")
            return 1

        print(f"Cached {len(written)} tickers from {dataset_id}/{config.data.hf_asset_dir}")

        # The benchmark index drives the momentum crash filter, so it is worth
        # one extra small download rather than leaving the filter on its
        # composite fallback.
        benchmark = config.data.benchmark_symbol
        if benchmark:
            cached = sync_hf_to_cache(
                dataset_id=dataset_id,
                revision=revision,
                asset_dir="indices",
                adjust_prices=config.data.hf_adjust_prices,
                tickers=[benchmark],
                skip_existing=not args.force,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
            )
            if cached:
                print(f"Cached benchmark index {benchmark}")
            else:
                print(
                    f"Note: benchmark {benchmark} not found in {dataset_id}/indices; "
                    f"the market-regime filter will use a composite of the traded universe"
                )
        return 0

    tickers = resolve_backtest_universe(
        force_full_download=args.force,
        max_tickers=args.universe_size or config.data.universe_size,
        selection=config.data.universe_selection,
        seed=config.data.universe_seed,
        purpose="backtest",
    )

    if not tickers:
        print("Error: No tickers found with available data")
        return 1

    print(f"Resolved {len(tickers)} tickers")

    workers = args.workers or config.data.download_workers
    print(f"Downloading with {workers} concurrent chunk request(s)")
    config = config.model_copy(deep=True)
    config.data.source = "yfinance"
    config.data.download_workers = workers
    success = fetch_and_cache(
        config,
        tickers=tickers,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        skip_existing=not args.force,
    )

    if success:
        print(f"Successfully downloaded data for {len(tickers)} tickers")
        return 0
    else:
        print("Warning: Some tickers failed to download")
        return 0


def cmd_train(args) -> int:
    """Train a strategy through its registered trainer.

    With no --strategy this resolves to the supervised pipeline and behaves
    exactly as it always has. With one, the strategy's declared trainer runs
    instead — see portfolio_agent/training/ for the registry.
    """
    try:
        from portfolio_agent.training import run_training_job
        from portfolio_agent.training.config import parse_overrides
    except ImportError as e:
        print(f"Error: could not load the training package ({e}).")
        return 1

    config = get_config()

    try:
        overrides = parse_overrides(getattr(args, "set", None))
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Resolve the device once, here. get_device() downgrades an unavailable
    # accelerator to CPU itself, so passing the resolved type down guarantees
    # every later consumer (dataloaders, mixed precision, the checkpoint
    # metadata) agrees with what was printed.
    #
    # Imported inside the branch rather than above, because resolving a device
    # needs PyTorch and the gbm trainer does not. Hoisting it would make
    # `train --trainer gbm` refuse to start on an install that can run it
    # perfectly well.
    if args.device:
        try:
            from portfolio_agent.utils.device import get_device
        except ImportError:
            print(
                f"Warning: PyTorch is not installed, so --device {args.device} is "
                "ignored. Trainers that need it will say so."
            )
        else:
            overrides.setdefault("device", get_device(args.device).type)

    tickers = (
        [t.strip() for t in args.tickers.split(",") if t.strip()]
        if getattr(args, "tickers", None)
        else None
    )

    try:
        run = run_training_job(
            config,
            strategy=args.strategy,
            trainer=args.trainer,
            overrides=overrides,
            universe=tickers,
            snapshot=args.universe_snapshot,
            universe_size=args.universe_size,
            models_dir=args.models_dir,
            model_name=args.model_name,
            strategy_config_file=args.strategy_config,
        )
    except (KeyError, ValueError) as e:
        # A bad trainer name or an unknown hyperparameter. The message already
        # names the offending key and the valid alternatives.
        print(f"Error: {e}")
        return 1

    if not run.ok:
        print(f"\nTraining failed: {run.error}")
        return 1

    if args.save_snapshot:
        run.universe.save(args.save_snapshot)
        print(f"Universe snapshot written to {args.save_snapshot}")

    # "(supervised default)" is only true when the supervised trainer is what
    # actually ran; with an explicit --trainer and no --strategy it is a lie.
    strategy_label = run.strategy or (
        "(supervised default)" if run.trainer == "supervised" else "-"
    )
    print("\nTraining complete!")
    print(f"  Strategy:   {strategy_label}")
    print(f"  Trainer:    {run.trainer}")
    print(f"  Universe:   {len(run.universe)} tickers (fingerprint {run.universe.fingerprint})")
    print(f"  Duration:   {run.duration_seconds:.1f}s")
    for key, value in sorted(run.artifact.metrics.items()):
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
    if run.checkpoint_path:
        print(f"  Checkpoint: {run.checkpoint_path}")
    return 0


def cmd_train_bulk(args) -> int:
    """Train several strategies or hyperparameter settings on one universe."""
    try:
        from portfolio_agent.training import run_bulk
        from portfolio_agent.training.bulk import BulkJob, sweep
        from portfolio_agent.training.config import parse_overrides
    except ImportError as e:
        print(f"Error: could not load the training package ({e}).")
        return 1

    config = get_config()

    try:
        base_overrides = parse_overrides(getattr(args, "set", None))
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    strategies = [s.strip() for s in (args.strategies or "").split(",") if s.strip()]

    if args.sweep:
        if len(strategies) != 1:
            print("Error: --sweep trains one strategy; pass exactly one --strategies value.")
            return 1
        try:
            grid = _parse_sweep_grid(args.sweep)
        except ValueError as e:
            print(f"Error: {e}")
            return 1
        jobs = sweep(
            strategies[0], grid, base_overrides=base_overrides, save=args.save_checkpoints
        )
    else:
        if not strategies:
            print("Error: pass --strategies a,b,c (or --sweep with one strategy).")
            return 1
        jobs = [
            BulkJob(strategy=name, overrides=dict(base_overrides), model_name=name)
            for name in strategies
        ]

    tickers = (
        [t.strip() for t in args.tickers.split(",") if t.strip()]
        if getattr(args, "tickers", None)
        else None
    )

    report = run_bulk(
        config,
        jobs,
        universe=tickers,
        snapshot=args.universe_snapshot,
        universe_size=args.universe_size,
        models_dir=args.models_dir,
        save_snapshot_to=args.save_snapshot,
    )

    print(f"\nBulk run: {len(jobs)} job(s) on {len(report.universe)} tickers "
          f"(universe {report.universe.fingerprint})")
    print(report.to_frame().to_string(index=False))

    best = report.best()
    if best is not None:
        print(f"\nBest: {best.strategy or 'supervised'} ({best.trainer}) "
              f"metric={best.artifact.primary_metric():.6f}")

    if report.failures:
        print(f"\n{len(report.failures)} job(s) failed: {report.failures}")
        return 1
    return 0


def _parse_sweep_grid(specs: list[str]) -> dict:
    """Turn `["epochs=10,50", "gamma=0.0,0.9"]` into a grid mapping.

    Values stay strings; the trainer's schema coerces and validates them, so a
    grid point that is not a legal value fails by name rather than silently.
    """
    grid: dict[str, list[str]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--sweep expects KEY=v1,v2, got {spec!r}")
        key, _, values = spec.partition("=")
        key = key.strip()
        parsed = [v.strip() for v in values.split(",") if v.strip()]
        if not key or not parsed:
            raise ValueError(f"--sweep expects KEY=v1,v2, got {spec!r}")
        grid[key] = parsed
    return grid


def cmd_list_trainers(args) -> int:
    """List registered training procedures and their hyperparameters."""
    from portfolio_agent.training import get_trainer, list_trainers
    from portfolio_agent.training.registry import unavailable_trainers

    names = list_trainers()
    missing = unavailable_trainers()
    if not names:
        print("No trainers are registered, which means the training package "
              "failed to import.")
        return 1

    if args.name:
        if args.name not in names:
            if args.name in missing:
                print(f"Trainer {args.name!r} is installed but unavailable: "
                      f"{missing[args.name]}")
                return 1
            print(f"Unknown trainer: {args.name!r}. Available: {names}")
            return 1
        trainer_class = get_trainer(args.name)
        schema = trainer_class.config_model()
        print(f"{args.name}  ({trainer_class.__name__})")
        if trainer_class.strategy_name:
            print(f"  trains: {trainer_class.strategy_name}")
        reason = trainer_class.availability()
        if reason:
            print(f"  UNAVAILABLE: {reason}")
        print("  settings:")
        for field_name, field in schema.model_fields.items():
            default = field.default
            description = field.description or ""
            print(f"    {field_name} = {default!r}")
            if description:
                print(f"        {description}")
        return 0

    print("Registered trainers:")
    for name in names:
        trainer_class = get_trainer(name)
        target = f" -> {trainer_class.strategy_name}" if trainer_class.strategy_name else ""
        reason = trainer_class.availability()
        suffix = f"   [unavailable: {reason}]" if reason else ""
        print(f"  - {name}{target}{suffix}")
    # Absence and unavailability read very differently to someone scanning this
    # list; a trainer whose module would not import is not a typo.
    for name, reason in sorted(missing.items()):
        print(f"  - {name}   [not loaded: {reason}]")
    print("\nUse --name <trainer> to see its settings, and set them with")
    print("  portfolio-agent train --strategy <s> --set key=value")
    return 0


def _resolve_inference_device(requested: str) -> str:
    """Resolve a --device request for inference, downgrading if unavailable.

    Kept string-typed (and torch-optional) because rule-based backtests must
    run without PyTorch installed at all.
    """
    try:
        from portfolio_agent.utils.device import get_device
    except ImportError:
        if requested != "cpu":
            print(
                f"Warning: PyTorch is not installed, so '{requested}' is unavailable — "
                "using CPU. Install it with: uv sync --extra gpu"
            )
        return "cpu"
    return get_device(requested).type


def cmd_backtest(args) -> int:
    """Run backtest command."""
    from portfolio_agent.agents.backtester import run_backtest_cli

    config = get_config()

    # Resolve strategy: --use-trained-model is a shorthand for --strategy lstm
    strategy_type = args.strategy
    if args.use_trained_model and strategy_type is None:
        strategy_type = "lstm"

    # Determine inference device
    if args.device:
        inference_device = _resolve_inference_device(args.device)
    elif strategy_type == "lstm":
        # Use CPU for inference by default to save VRAM; pass --device cuda to override
        inference_device = "cpu"
        print(f"Using {inference_device} for model inference (pass --device cuda to override)")
    else:
        inference_device = "cpu"

    # Resolve date range
    start_date = args.start_date
    end_date = args.end_date
    if args.years and start_date is None:
        import pandas as pd
        end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
        start_date = (end - pd.Timedelta(days=args.years * 365)).strftime('%Y-%m-%d')

    output_file = args.output or config.paths.backtest_excel_output

    print(f"Running backtest...")
    print(f"  Strategy: {strategy_type or config.strategy.type}")
    if args.strategy_config:
        print(f"  Strategy config: {args.strategy_config}")
    print(f"  Inference device: {inference_device}")
    print(f"  Parallel: {args.parallel} (workers={args.workers or 'auto'})")
    print(f"  Output file: {output_file}")

    try:
        result = run_backtest_cli(
            config=config,
            strategy_type=strategy_type,
            strategy_config_path=args.strategy_config,
            device=inference_device,
            output_file=output_file,
            parallel=args.parallel,
            max_workers=args.workers,
            start_date=start_date,
            end_date=end_date,
            show_progress=not args.no_progress,
        )

        if result.get('status') == 'success':
            print(f"\nBacktest complete!")
            print(f"  Trades executed: {result.get('trade_count', 0)}")
            print(f"  Report saved to: {result.get('output_file', output_file)}")
            return 0
        else:
            print("Backtest failed")
            return 1

    except Exception as e:
        print(f"Error during backtest: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_run_agent(args) -> int:
    """Run the daily portfolio agent."""
    from portfolio_agent.src.orchestrator import run_orchestrator

    config = get_config()

    print("Running orchestrator...")
    try:
        excel_path = run_orchestrator(
            force_refresh=args.force_refresh,
            simulate_outcome=args.simulate_outcome,
            update_outcomes=args.update_outcomes,
            config=config,
        )
        print(f"Done. Report saved to: {excel_path}")
        return 0
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_list_strategies(args) -> int:
    """List registered strategies, and optionally describe one in detail."""
    from portfolio_agent.strategies.registry import get_available_strategies

    strategies = get_available_strategies()

    if args.name:
        if args.name not in strategies:
            print(f"Unknown strategy: {args.name!r}. Available: {sorted(strategies)}")
            return 1

        config = get_config()
        strategy_config = config.strategy.model_copy(deep=True)
        strategy_config.type = args.name
        if args.strategy_config:
            strategy_config.config_path = args.strategy_config

        try:
            strategy = strategies[args.name](strategy_config)
        except Exception as e:
            print(f"Could not load strategy config for {args.name!r}: {e}")
            return 1

        print(f"{strategy.name}  (type={args.name})")
        print(f"  supports_gpu_batch: {strategy.supports_gpu_batch}")
        entry = strategy.entry_rules()
        if entry:
            print(f"  entry_rules: {entry}")
        exit_ = strategy.exit_rules()
        if exit_:
            print(f"  exit_rules: {exit_}")
        try:
            print(f"  required_features: {strategy.required_features()}")
        except Exception as e:
            print(f"  required_features: unavailable ({e})")
        return 0

    print("Registered strategies:")
    for strategy_name in sorted(strategies):
        print(f"  - {strategy_name}")
    print("\nUse --name <strategy> [--strategy-config PATH] to see details for one strategy.")
    print("Combine multiple strategies into a UMA (Unified Multi-strategy Agent) via a YAML file")
    print("(see config/strategies/example_uma.yaml) and run it with --strategy ensemble.")
    return 0


def cmd_gpu_check(args) -> int:
    """Report what compute devices this install can actually use."""
    try:
        from portfolio_agent.utils.device import cuda_is_available, describe_devices, get_device
    except ImportError as e:
        print(f"PyTorch is not installed ({e}).")
        print("GPU acceleration requires the optional 'gpu' extra: uv sync --extra gpu")
        print("Everything except `train` and the 'lstm' strategy works without it.")
        return 1

    print("=" * 60)
    print("Device diagnostics")
    print("=" * 60)
    for line in describe_devices():
        print(line)

    config = get_config()
    print("-" * 60)
    print(f"config.training.device: {config.training.device!r}")
    resolved = get_device(config.training.device, verbose=False)
    print(f"Resolves to:            {resolved}")
    print("=" * 60)

    return 0 if cuda_is_available() or config.training.device in ("cpu", "auto") else 1


def _add_universe_arguments(parser: argparse.ArgumentParser) -> None:
    """Universe-pinning flags shared by `train` and `train-bulk`.

    Comparing two models is only meaningful when they saw the same names, and
    the default cache draw does not guarantee that across two invocations — it
    depends on what happens to be cached at the time. These flags make the
    universe explicit and reproducible.
    """
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated tickers to train on, pinning the universe exactly"
    )
    parser.add_argument(
        "--universe-snapshot",
        type=str,
        default=None,
        help="Path to a saved universe snapshot to train on (see --save-snapshot)"
    )
    parser.add_argument(
        "--save-snapshot",
        type=str,
        default=None,
        help="Write the universe actually used to this JSON file, so a later run "
             "can reproduce the comparison with --universe-snapshot"
    )
    parser.add_argument(
        "--universe-size",
        type=int,
        default=None,
        help="Cap the drawn universe (default: config.data.universe_size)"
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="portfolio-agent",
        description="Portfolio Agent CLI - Autonomous Financial Advisor"
    )

    # Global options, applying to every subcommand.
    #
    # --config is the one that was genuinely missing. Every command loaded
    # config.yaml implicitly from the working directory, so running two
    # experiments meant editing a file in place — on a platform whose purpose
    # is running experiments.
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        metavar="PATH",
        help="Configuration file to use (default: config.yaml, falling back to "
             "the project root and then the packaged default)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable output where a command produces results",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", action="store_true", help="Show INFO-level logging"
    )
    verbosity.add_argument(
        "-q", "--quiet", action="store_true", help="Show warnings and errors only"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # download-data command
    download_parser = subparsers.add_parser(
        "download-data",
        help="Download market data for the configured universe"
    )
    download_parser.add_argument(
        "--source",
        choices=("huggingface", "yfinance"),
        default=None,
        help="Where to pull history from (default: config.data.source). 'huggingface' reads a "
             "versioned Hub dataset in one download; 'yfinance' fetches per ticker."
    )
    download_parser.add_argument(
        "--years",
        type=int,
        default=None,
        help="Years of history to keep (default: config.data.default_history_years)"
    )
    download_parser.add_argument(
        "--hf-dataset",
        default=None,
        help="HuggingFace dataset repo id (default: config.data.hf_dataset_id)"
    )
    download_parser.add_argument(
        "--hf-revision",
        default=None,
        help="Pin the Hub dataset to a git revision (branch, tag or commit) for a "
             "reproducible backtest"
    )
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download data that is already cached. Off by default: a plain "
             "re-run now skips symbols already on disk instead of fetching all "
             "~2,400 of them again."
    )
    download_parser.add_argument(
        "--universe-size",
        type=int,
        default=None,
        help="Cap the number of tickers ingested (default: from config for yfinance, "
             "the dataset's full symbol list for huggingface)"
    )
    download_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Concurrent downloads (huggingface: threads, default 8; yfinance: "
             "chunk workers, default config.data.download_workers). "
             "Use 1 if the data provider rate-limits you."
    )
    download_parser.set_defaults(func=cmd_download_data)
    
    # train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train a strategy through its registered trainer "
             "(no --strategy runs the supervised pipeline, as before)"
    )
    train_parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Strategy to train, e.g. 'india_sac'. Its trainer comes from "
             "config/strategies/<strategy>.yaml. Omit for the supervised default."
    )
    train_parser.add_argument(
        "--trainer",
        type=str,
        default=None,
        help="Override which registered trainer runs (see `list-trainers`)."
    )
    train_parser.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        default=None,
        help="Override one training hyperparameter; repeatable. Validated against "
             "the trainer's own schema, so an unknown key stops the run."
    )
    train_parser.add_argument(
        "--strategy-config",
        type=str,
        default=None,
        help="Path to the strategy's YAML, overriding config/strategies/<strategy>.yaml"
    )
    train_parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "mps", "cpu"],
        default=None,
        help="Device for training (default: auto)"
    )
    _add_universe_arguments(train_parser)
    train_parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory for the checkpoint (default: models)"
    )
    train_parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Checkpoint basename, written as <name>_best<ext> where the extension "
             "is the trainer's (.pt for the neural trainers, .joblib for gbm). "
             "Default: the strategy name."
    )
    train_parser.set_defaults(func=cmd_train)

    # train-bulk command
    bulk_parser = subparsers.add_parser(
        "train-bulk",
        help="Train several strategies, or sweep hyperparameters, on one pinned universe"
    )
    bulk_parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Comma-separated strategies to train, e.g. 'india_sac,lstm'"
    )
    bulk_parser.add_argument(
        "--sweep",
        action="append",
        metavar="KEY=V1,V2",
        default=None,
        help="Sweep one hyperparameter over values; repeatable for a cross product. "
             "Requires exactly one --strategies value."
    )
    bulk_parser.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        default=None,
        help="Hyperparameter held fixed across every job; repeatable."
    )
    bulk_parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Write a checkpoint per sweep point. Off by default: a sweep usually "
             "asks which settings to use rather than producing the model."
    )
    _add_universe_arguments(bulk_parser)
    bulk_parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory for checkpoints (default: models)"
    )
    bulk_parser.set_defaults(func=cmd_train_bulk)

    # list-trainers command
    trainers_parser = subparsers.add_parser(
        "list-trainers",
        help="List registered training procedures and their settings"
    )
    trainers_parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Show one trainer's settings in detail"
    )
    trainers_parser.set_defaults(func=cmd_list_trainers)

    # backtest command
    backtest_parser = subparsers.add_parser(
        "backtest",
        help="Run backtesting simulation"
    )
    backtest_parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Registered strategy to backtest (e.g. 'rule_based', 'lstm', 'ensemble'). Defaults to config.strategy.type"
    )
    backtest_parser.add_argument(
        "--strategy-config",
        type=str,
        default=None,
        help="Path to the strategy's YAML config (e.g. a UMA ensemble file under config/strategies/). "
             "Defaults to config.strategy.config_path"
    )
    backtest_parser.add_argument(
        "--use-trained-model",
        action="store_true",
        help="Shorthand for --strategy lstm"
    )
    backtest_parser.add_argument(
        "--parallel",
        action="store_true",
        help="Parallelize rule-based signal generation across CPU workers"
    )
    backtest_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Max worker processes when --parallel is set (default: CPU count)"
    )
    backtest_parser.add_argument(
        "--years",
        type=int,
        default=None,
        help="Number of years of history to backtest (default: config.backtest.start_years_ago)"
    )
    backtest_parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Explicit backtest start date (YYYY-MM-DD), overrides --years"
    )
    backtest_parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Explicit backtest end date (YYYY-MM-DD), defaults to today"
    )
    backtest_parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "mps", "cpu"],
        default=None,
        help="Device for model inference (default: cpu for backtest)"
    )
    backtest_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output Excel file path"
    )
    backtest_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress the progress bars (useful when logging output to a file)"
    )
    backtest_parser.set_defaults(func=cmd_backtest)
    
    # run-agent command
    agent_parser = subparsers.add_parser(
        "run-agent",
        help="Run the daily portfolio agent"
    )
    agent_parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh of market data"
    )
    agent_parser.add_argument(
        "--simulate-outcome",
        action="store_true",
        help="Simulate outcome for top recommendation"
    )
    agent_parser.add_argument(
        "--update-outcomes",
        action="store_true",
        help="Update outcomes from market data"
    )
    agent_parser.set_defaults(func=cmd_run_agent)

    # list-strategies command
    list_strategies_parser = subparsers.add_parser(
        "list-strategies",
        help="List registered strategies (rule-based, ML, UMA ensembles)"
    )
    list_strategies_parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Show details (entry/exit rules, required features) for one strategy"
    )
    list_strategies_parser.add_argument(
        "--strategy-config",
        type=str,
        default=None,
        help="Strategy YAML to load when using --name (e.g. a UMA ensemble file)"
    )
    list_strategies_parser.set_defaults(func=cmd_list_strategies)

    # gpu-check command
    gpu_check_parser = subparsers.add_parser(
        "gpu-check",
        help="Show which compute devices (CUDA/MPS/CPU) this install can actually use"
    )
    gpu_check_parser.set_defaults(func=cmd_gpu_check)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point."""
    import logging

    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    level = logging.WARNING if args.quiet else (logging.INFO if args.verbose else logging.WARNING)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    # Which configuration a run used is worth an INFO line regardless of
    # verbosity elsewhere — a run that silently used schema defaults is the
    # failure this exists to prevent.
    logging.getLogger("portfolio_agent.config.loader").setLevel(logging.INFO)

    # Every command resolves its config through get_config(), which reads this.
    _ACTIVE_CONFIG_PATH["path"] = args.config

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
