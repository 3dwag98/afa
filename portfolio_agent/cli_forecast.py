"""The forecasting commands: `evaluate`, `compare`, `list-features`, `data build`.

Kept in their own module rather than appended to `cli.py`, which was 873 lines
with zero tests before this task — the one surface everybody touches and the
one place nothing was checked. Splitting is not tidiness: it is what lets these
commands be imported and called directly from a test without dragging the
argparse tree along, which is why they have tests and the rest did not.

Two design choices worth defending
----------------------------------
**`--cv` defaults to purged.** The correct method is what you get by not
thinking about it; the leaky one has to be asked for by name. A default that
requires knowledge to be safe is a default that will be wrong most of the time
it is used.

**`--baseline` is a flag, not a separate run.** The claim that matters is
"better than gradient boosting on identical splits", and any workflow where
that takes a second command is a workflow where it gets skipped. One flag, same
panel, same dates, printed in the same table.

And one thing deliberately absent: there is no `--quick` preset that loosens
cross-validation or shortens the sample. A preset that trades correctness for
speed becomes the default within a week, and the numbers it produces are
indistinguishable from real ones in every report they land in. `--limit`
narrows the universe instead — honestly, visibly, and recorded in the manifest.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: Cross-validation schemes `--cv` accepts. `purged` first because it is the
#: default and the only one that is correct with overlapping labels.
CV_SCHEMES = ("purged", "walkforward", "none")

#: Exposures `--neutralize` accepts.
NEUTRALIZE_KINDS = ("beta", "size", "sector")


def parse_int_list(text: Optional[str], name: str) -> Optional[List[int]]:
    """Turn `"1,5,21"` into `[1, 5, 21]`, or raise naming the offender."""
    if not text:
        return None
    values: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError:
            raise ValueError(
                f"--{name} expects comma-separated integers, got {part!r}"
            ) from None
    if not values:
        raise ValueError(f"--{name} was given no values")
    return values


def parse_str_list(text: Optional[str]) -> Optional[List[str]]:
    if not text:
        return None
    values = [part.strip() for part in text.split(",") if part.strip()]
    return values or None


def build_splitter(scheme: str, horizon: int, embargo: int, n_splits: int):
    """Construct the cross-validation splitter `--cv` asked for.

    `none` returns None, which the harness reads as a single window. It exists
    because a rule-based screen has nothing fitted to leak, so folds add cost
    and no information — but it has to be requested, not inferred.
    """
    if scheme == "none":
        return None
    if scheme not in CV_SCHEMES:
        raise ValueError(f"--cv expects one of {list(CV_SCHEMES)}, got {scheme!r}")

    from portfolio_agent.validation.purged import PurgedWalkForward

    if scheme == "walkforward":
        # Same splitter with the purge switched off. Deliberately reachable and
        # deliberately not the default: it is the scheme whose folds overlap at
        # the boundary, and it exists so the size of that bias can be measured
        # rather than argued about.
        return PurgedWalkForward(n_splits=n_splits, horizon=0, embargo=embargo)
    return PurgedWalkForward(n_splits=n_splits, horizon=horizon, embargo=embargo)


def resolve_universe_for_cli(config, args) -> tuple:
    """Pin the names an evaluation runs on, and say how they were chosen.

    Returns `(tickers, snapshot)`. Explicit `--tickers` wins, then a saved
    snapshot, then a fresh seeded draw. The snapshot is what makes two runs
    comparable, so it is preferred over a draw whenever one is supplied.
    """
    tickers = parse_str_list(getattr(args, "tickers", None))
    if tickers:
        return tickers, None

    from portfolio_agent.training.universe import resolve_universe

    snapshot = resolve_universe(
        config,
        snapshot=getattr(args, "universe_snapshot", None),
        size=getattr(args, "limit", None) or getattr(args, "universe_size", None),
        name="evaluate",
    )
    return list(snapshot.tickers), snapshot


def describe_plan(config, args, tickers: Sequence[str]) -> Dict[str, Any]:
    """Everything a run resolved to, before doing any of it.

    What `--dry-run` prints. The point is that the expensive decisions —
    which names, which split, which horizons — are visible *before* the run
    rather than inferred from its output.
    """
    horizons = parse_int_list(getattr(args, "horizons", None), "horizons")
    return {
        "strategy": getattr(args, "strategy", None),
        "horizon": args.horizon,
        "horizons": horizons,
        "cv": args.cv,
        "embargo": args.embargo,
        "n_splits": args.splits,
        "neutralize": parse_str_list(getattr(args, "neutralize", None)),
        "baseline": getattr(args, "baseline", None),
        "stride": args.stride,
        "n_buckets": args.buckets,
        "seed": getattr(args, "seed", None),
        "universe_size": len(tickers),
        "universe_head": list(tickers[:8]),
        "universe_snapshot": getattr(args, "universe_snapshot", None),
        "start_date": getattr(args, "start_date", None),
        "end_date": getattr(args, "end_date", None),
        "max_dates": getattr(args, "max_dates", None),
        "output": getattr(args, "output", None),
    }


def cmd_evaluate(args) -> int:
    """Measure one strategy's forecast skill. The primary command."""
    from portfolio_agent.config.loader import load_config

    config = load_config(getattr(args, "_config_path", None) or "config.yaml")

    try:
        horizons = parse_int_list(getattr(args, "horizons", None), "horizons")
        neutralize = parse_str_list(getattr(args, "neutralize", None))
        if neutralize:
            unknown = sorted(set(neutralize) - set(NEUTRALIZE_KINDS))
            if unknown:
                raise ValueError(
                    f"--neutralize does not know {unknown}; "
                    f"valid exposures are {list(NEUTRALIZE_KINDS)}"
                )
        splitter = build_splitter(args.cv, args.horizon, args.embargo, args.splits)
    except ValueError as error:
        print(f"Error: {error}")
        return 1

    try:
        tickers, snapshot = resolve_universe_for_cli(config, args)
    except (ValueError, FileNotFoundError) as error:
        print(f"Error: {error}")
        return 1

    if args.dry_run:
        plan = describe_plan(config, args, tickers)
        if args.json:
            print(json.dumps(plan, indent=2, default=str))
        else:
            print("Resolved plan (nothing was run):")
            for key, value in plan.items():
                print(f"  {key:<20} {value}")
        return 0

    if args.seed is not None:
        # Only the numpy global stream: everything in the evaluation path that
        # draws does so through an explicitly-seeded generator, so this is for
        # a strategy that reaches for the global one rather than for us.
        import numpy as np

        np.random.seed(args.seed)

    from portfolio_agent.evaluation import evaluate_forecast

    shared = dict(
        universe=tickers,
        horizon=args.horizon,
        stride=args.stride,
        n_buckets=args.buckets,
        start_date=args.start_date,
        end_date=args.end_date,
        min_history=args.min_history,
        max_dates=args.max_dates,
        use_benchmark=not args.no_benchmark,
        runs_dir=args.output,
        charge_costs=not getattr(args, "gross", False),
        slippage_per_side=(
            None if getattr(args, "slippage_bps", None) is None
            else args.slippage_bps / 1e4
        ),
    )

    try:
        result = evaluate_forecast(config, args.strategy, splitter=splitter, **shared)
    except (ValueError, KeyError, RuntimeError) as error:
        print(f"Error: {error}")
        return 1

    baseline = None
    if args.baseline:
        baseline = _run_baseline(config, args, tickers, shared, splitter)

    neutralized = None
    if neutralize:
        neutralized = _run_neutralized(config, args, tickers, neutralize)

    decay = None
    if horizons:
        decay = _run_decay(config, args, tickers, horizons)

    if args.json:
        document = result.to_dict()
        if baseline is not None:
            document["baseline"] = baseline.to_dict()
        if neutralized is not None:
            document["neutralized"] = neutralized.to_dict()
        if decay is not None:
            document["decay"] = decay.to_dict()
        print(json.dumps(document, indent=2, default=str))
        return 0

    print(result.render())
    if baseline is not None:
        print()
        print(_render_baseline_comparison(result, baseline, args.baseline))
    if neutralized is not None:
        print()
        print(neutralized.render())
    if decay is not None:
        print()
        print(decay.render())
    if result.run_id:
        print(f"\nRun {result.run_id} recorded. Render it with:")
        print(f"  portfolio-agent report --run {result.run_id}")
    return 0


def _run_baseline(config, args, tickers, shared, splitter):
    """Score the baseline trainer's model on the same panel, or explain why not.

    Returns None and prints the reason rather than failing the whole
    evaluation: the primary result is already computed and is worth having even
    when the comparison is unavailable.
    """
    from portfolio_agent.evaluation import evaluate_forecast

    try:
        result = evaluate_forecast(
            config, args.baseline, splitter=splitter,
            **{**shared, "manifest": False},
        )
    except (ValueError, KeyError, RuntimeError) as error:
        print(
            f"\nNote: --baseline {args.baseline} could not be scored ({error}). "
            f"The result above stands on its own; the comparison does not."
        )
        return None
    return result


def _render_baseline_comparison(result, baseline, name: str) -> str:
    """The one table the `--baseline` flag exists to produce."""
    lines = [
        f"Against the {name} baseline, same panel and same dates",
        "=" * 62,
        f"  {'':<18}{'mean IC':>12}{'t':>8}{'spread':>12}{'monotonicity':>14}",
        f"  {result.strategy:<18}{result.ic.mean:>+12.4f}{result.ic.t_stat:>8.2f}"
        f"{result.buckets.spread:>+12.4%}{result.buckets.monotonicity:>+14.3f}",
        f"  {name:<18}{baseline.ic.mean:>+12.4f}{baseline.ic.t_stat:>8.2f}"
        f"{baseline.buckets.spread:>+12.4%}{baseline.buckets.monotonicity:>+14.3f}",
        "",
    ]
    delta = result.ic.mean - baseline.ic.mean
    if delta > 0:
        lines.append(
            f"  {result.strategy} beats {name} by {delta:+.4f} IC. That is the "
            f"claim worth making; note it is one sample, not a significance test "
            f"of the difference."
        )
    else:
        lines.append(
            f"  {result.strategy} does NOT beat {name} ({delta:+.4f} IC). "
            f"On identical features and splits, the simpler model wins."
        )
    return "\n".join(lines)


def _run_neutralized(config, args, tickers, kinds: List[str]):
    """Raw versus residual IC against the requested exposures."""
    from portfolio_agent.evaluation import evaluate_neutralized

    try:
        return evaluate_neutralized(
            config, args.strategy, universe=tickers,
            horizon=args.horizon, stride=args.stride,
            start_date=args.start_date, end_date=args.end_date,
            min_history=args.min_history, max_dates=args.max_dates,
            use_benchmark=not args.no_benchmark,
            # No sector map ships with the repository; the result says so in
            # its own notes rather than quietly reporting "neutralized".
            sector_map=None,
        )
    except (ValueError, KeyError, RuntimeError) as error:
        print(f"\nNote: --neutralize could not be computed ({error}).")
        return None


def _run_decay(config, args, tickers, horizons: List[int]):
    """IC against horizon, scored in one pass."""
    from portfolio_agent.evaluation import decay_curve

    try:
        return decay_curve(
            config, args.strategy, horizons=horizons, universe=tickers,
            stride=args.stride, start_date=args.start_date, end_date=args.end_date,
            min_history=args.min_history, max_dates=args.max_dates,
            use_benchmark=not args.no_benchmark,
        )
    except (ValueError, KeyError, RuntimeError) as error:
        print(f"\nNote: --horizons decay curve could not be computed ({error}).")
        return None


def cmd_compare(args) -> int:
    """Score several strategies on one universe and print one table.

    One universe, resolved once and passed to every strategy. Comparing runs
    that each drew their own sample is the mistake this command exists to make
    impossible — two strategies on two different draws of 200 names differ by
    the draw at least as much as by the strategy.
    """
    from portfolio_agent.config.loader import load_config

    config = load_config(getattr(args, "_config_path", None) or "config.yaml")

    names = parse_str_list(args.strategies)
    if not names:
        print("Error: --strategies expects a comma-separated list")
        return 1

    try:
        splitter = build_splitter(args.cv, args.horizon, args.embargo, args.splits)
        tickers, snapshot = resolve_universe_for_cli(config, args)
    except (ValueError, FileNotFoundError) as error:
        print(f"Error: {error}")
        return 1

    if args.dry_run:
        plan = describe_plan(config, args, tickers)
        plan["strategies"] = names
        if args.json:
            print(json.dumps(plan, indent=2, default=str))
        else:
            print("Resolved plan (nothing was run):")
            for key, value in plan.items():
                print(f"  {key:<20} {value}")
        return 0

    from portfolio_agent.evaluation import compare_forecasts, evaluate_forecast

    results = []
    failures: Dict[str, str] = {}
    for name in names:
        try:
            results.append(evaluate_forecast(
                config, name, universe=tickers, horizon=args.horizon,
                stride=args.stride, n_buckets=args.buckets,
                start_date=args.start_date, end_date=args.end_date,
                min_history=args.min_history, max_dates=args.max_dates,
                use_benchmark=not args.no_benchmark, splitter=splitter,
                runs_dir=args.output,
            ))
        except (ValueError, KeyError, RuntimeError) as error:
            # One bad strategy must not discard the others' results.
            failures[name] = str(error)

    if not results:
        print("Error: no strategy could be evaluated.")
        for name, message in failures.items():
            print(f"  {name}: {message}")
        return 1

    table = compare_forecasts(results)
    columns = [
        c for c in ("strategy", "mean_ic", "icir", "t_stat", "p_value",
                    "spread", "monotonicity", "hit_rate", "score_dispersion",
                    "n_dates", "run_id")
        if c in table.columns
    ]

    if args.json:
        print(json.dumps(
            {"results": table[columns].to_dict(orient="records"), "failed": failures},
            indent=2, default=str,
        ))
        return 0

    print(f"Forecast comparison — {len(results)} strategies, "
          f"{len(tickers)} names, {args.horizon}d horizon")
    if snapshot is not None:
        print(f"Universe {snapshot.fingerprint} — identical for every row")
    print()
    print(table[columns].to_string(index=False))
    if failures:
        print()
        for name, message in failures.items():
            print(f"  {name} failed: {message}")
    return 0


def cmd_list_features(args) -> int:
    """List the registered features. The one registry with no inspection command."""
    from portfolio_agent.features.registry import _FEATURE_REGISTRY, get_feature

    names = sorted(_FEATURE_REGISTRY)
    if not names:
        print("No features are registered, which means the package failed to import.")
        return 1

    if args.json:
        print(json.dumps({
            name: (get_feature(name).__doc__ or "").strip().splitlines()[:1]
            for name in names
        }, indent=2))
        return 0

    print(f"Registered features ({len(names)}):")
    for name in names:
        doc = (get_feature(name).__doc__ or "").strip().splitlines()
        summary = doc[0] if doc else ""
        print(f"  {name:<28}{summary}")
    print("\nUse any of these in a strategy's required_features(), or in")
    print("  portfolio-agent train --set features=<a,b,c>")
    return 0


def cmd_data_build(args) -> int:
    """Download market data, then check what actually arrived.

    `download-data` did the first half. Running the invariants immediately
    afterwards is the half that was missing: a download that half-succeeds
    looks exactly like one that worked, and the gap between them is only
    visible if something looks.
    """
    from portfolio_agent.cli import cmd_download_data

    code = cmd_download_data(args)
    if code != 0:
        return code

    if args.no_validate:
        print("\nSkipping validation (--no-validate). Run "
              "`portfolio-agent data validate` before trusting these bars.")
        return 0

    print("\nChecking what arrived...")
    from portfolio_agent.cli import cmd_data_status, cmd_data_validate

    class _Args:
        pass

    inspect = _Args()
    inspect.cache_dir = None
    inspect.symbols = None
    inspect.limit = args.validate_limit
    inspect.min_sessions = 252
    inspect.json = False
    inspect.per_symbol = False
    cmd_data_status(inspect)

    inspect.extreme_return = 0.25
    inspect.no_calendar = False
    inspect.strict = args.fail_on_warning
    print()
    return cmd_data_validate(inspect)


def add_forecast_commands(subparsers) -> None:
    """Register `evaluate`, `compare`, `list-features` and `data build`."""

    def shared_evaluation_args(parser):
        parser.add_argument(
            "--horizon", type=int, default=5,
            help="Forward-return horizon in sessions (default: 5)",
        )
        parser.add_argument(
            "--cv", type=str, default="purged", choices=CV_SCHEMES,
            help="Cross-validation scheme. 'purged' is the default because it is "
                 "the correct one with overlapping labels; 'walkforward' switches "
                 "the purge off so its bias can be measured; 'none' evaluates a "
                 "single window, which is right for a strategy with nothing fitted.",
        )
        parser.add_argument(
            "--embargo", type=int, default=0,
            help="Sessions excluded after each test fold, on top of the purge "
                 "(default: 0)",
        )
        parser.add_argument(
            "--splits", type=int, default=5,
            help="Number of walk-forward folds (default: 5)",
        )
        parser.add_argument(
            "--stride", type=int, default=1,
            help="Evaluate every Nth date. Trades statistical power for wall "
                 "clock, honestly — the reduced date count travels into the result.",
        )
        parser.add_argument(
            "--buckets", type=int, default=10,
            help="Buckets for the spread and monotonicity checks (default: 10)",
        )
        parser.add_argument("--tickers", type=str, default=None,
                            help="Comma-separated tickers, overriding any draw")
        parser.add_argument("--universe-snapshot", type=str, default=None,
                            help="Saved universe snapshot, for a comparable run")
        parser.add_argument("--universe-size", type=int, default=None,
                            help="Names to draw when no snapshot is given")
        parser.add_argument(
            "--limit", type=int, default=None,
            help="Narrow the universe to N names. The honest way to make a run "
                 "cheap: it changes the sample, visibly, and the count is "
                 "recorded — unlike a preset that loosens the method.",
        )
        parser.add_argument("--start-date", type=str, default=None)
        parser.add_argument("--end-date", type=str, default=None)
        parser.add_argument("--min-history", type=int, default=252,
                            help="Sessions a ticker needs before it is scored")
        parser.add_argument("--max-dates", type=int, default=None,
                            help="Cap on evaluation dates, most recent kept")
        parser.add_argument("--no-benchmark", action="store_true",
                            help="Do not pass the cached index into the strategy context")
        parser.add_argument("--gross", action="store_true",
                            help="Report the spread gross of costs (default: net, "
                                 "charging the NSE delivery schedule at the signal's "
                                 "own measured turnover)")
        parser.add_argument("--slippage-bps", type=float, default=None,
                            help="Slippage per side in basis points (default: 25, "
                                 "conservative for mid-caps)")
        parser.add_argument("--seed", type=int, default=None,
                            help="Seed numpy's global stream before scoring")
        parser.add_argument("--output", type=str, default=None,
                            help="Directory for run manifests (default: runs/)")
        parser.add_argument("--dry-run", action="store_true",
                            help="Resolve everything, print the plan, run nothing")
        parser.add_argument("--json", action="store_true", help="Emit JSON")

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Measure a strategy's forecast skill (rank IC, decile spread, decay)",
    )
    evaluate.add_argument("--strategy", type=str, required=True,
                          help="Registered strategy to evaluate")
    evaluate.add_argument(
        "--horizons", type=str, default=None,
        help="Comma-separated horizons for a decay curve, e.g. 1,2,3,5,10,21. "
             "Scored in one pass, so the curve costs about what one horizon does.",
    )
    evaluate.add_argument(
        "--neutralize", type=str, default=None,
        help=f"Comma-separated exposures to neutralize against "
             f"{list(NEUTRALIZE_KINDS)}. Reports raw and residual IC together; "
             f"sector needs a map, which does not ship with the repository.",
    )
    evaluate.add_argument(
        "--baseline", type=str, default=None,
        help="Also score this strategy on the identical panel and print both. "
             "A flag rather than a second command, because 'better than the "
             "baseline on identical splits' is the claim that matters and any "
             "workflow where it takes two runs is one where it gets skipped.",
    )
    shared_evaluation_args(evaluate)
    evaluate.set_defaults(func=cmd_evaluate)

    compare = subparsers.add_parser(
        "compare",
        help="Score several strategies on one universe and print one table",
    )
    compare.add_argument("--strategies", type=str, required=True,
                         help="Comma-separated registered strategies")
    shared_evaluation_args(compare)
    compare.set_defaults(func=cmd_compare)

    features = subparsers.add_parser(
        "list-features", help="List registered features and what they compute",
    )
    features.add_argument("--json", action="store_true", help="Emit JSON")
    features.set_defaults(func=cmd_list_features)
