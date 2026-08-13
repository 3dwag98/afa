# T07 — Make the package installable, and let the CLI take a config path

**Status:** not started · **Effort:** ~1.5 days · **Depends on:** none
**Plan reference:** `docs/forecasting_plan.html` Architecture register A1, A2; Part 1 (platform)

## Goal

A `pip install` of this repository should be able to load a strategy and run on
a known configuration.

## Why

Both verified by building a wheel and importing it from a clean virtualenv
outside the repository.

**`A1` — no strategy loads.** `strategies/rule_based.py` tries
`from src.risk`, then `from risk`, and never the canonical
`from portfolio_agent.src.risk`, which works. Because `strategies/__init__.py`
imports the registry eagerly, that one chain takes down the registry, the
backtester and the engine together. 17 runtime modules import through the flat
path; 6 have no canonical import at all. The root `src` symlink hides this
during development, because everything resolves as long as the process starts
at the repository root.

**`A2` — config silently substituted.** `config.yaml` is not packaged, so an
installed copy loads nothing and every value falls back to its schema default:
`universe_size: 10` against the repository's 4000. The run still produces
results, charts and a report, all of which look normal.

**No global options.** There is no `--config`, no `--json`, no verbosity
control. Every command loads `config.yaml` implicitly from the working
directory, so a platform whose purpose is running experiments cannot be pointed
at a different configuration.

## Approach

1. Put `portfolio_agent.` first in every fallback chain. `india_sac.py` already
   does this correctly — copy that shape.
2. Once imports resolve canonically, remove the flat fallbacks and the `src`
   symlink. Keeping them preserves the ambiguity and hides the dependency graph
   from static analysis.
3. Package a default `config.yaml`, resolve it as the last fallback, and log at
   INFO which file was loaded. A run should always be able to say which config
   it is on.
4. Add `--config PATH`, `--json` and `--quiet/-v` as global options.

## Acceptance criteria

- [ ] From a clean virtualenv outside the repository:
      `from portfolio_agent.strategies.registry import get_available_strategies`
      returns all six strategies.
- [ ] `load_config()` on an installed copy finds the packaged default and logs
      its path.
- [ ] `--config` selects a configuration file for every command.
- [ ] A CI check builds the wheel and runs the import probe, so this cannot
      regress silently.

## Note

`tools/probe_nse_source.py` is unrelated, but the same pattern applies: a test
that runs the real import path from outside the repository is the only thing
that catches this class of bug.
