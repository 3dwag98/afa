# T10 — Remove dead, duplicated and misleading code

**Status:** not started · **Effort:** ~1 day · **Depends on:** T07 (import fixes) for the symlink removal
**Plan reference:** `docs/forecasting_plan.html` Part 5 (delete)

## Goal

Delete what is actively misleading. Distinct from the freeze list, which
preserves accumulated correctness that is expensive to rebuild.

## Items

| Item | Evidence | Why worse than absent |
| --- | --- | --- |
| Tracked `.pyc` | 6 files, including `__pycache__/setup_afa.cpython-312.pyc` | That one is compiled bytecode for `setup_afa.py`, **a file that no longer exists in the repository**. Tracked bytecode also produces phantom diffs on every machine that imports the package |
| `src/indicators.py` | SMA, RSI, MACD, Bollinger and ATR exist in it *and* in `features/technical.py` | Two RSI implementations is a correctness hazard, not just duplication: they can silently disagree, and which one produced a published number depends on the call path |
| Synthetic fallback on by default | `data.allow_synthetic_fallback = True`, used at `data_store.py:349` | A research platform that silently substitutes generated data when real data is missing will eventually publish a number describing a random-walk generator |
| Market data in git | 2,397 parquet, 112 MB tracked, 80 MB pack | Cannot be updated without bloating history, goes stale silently, every clone pays for it |
| Tracked checkpoint | 1 `.pt` | Implies a canonical model with no record of the config or universe that produced it |
| Five unread settings | `features.lookbacks`, `features.feature_sets`, `paths.log_dir`, `paths.output_dir`, `data.market_data_dir` | Someone will set them and expect an effect. Only 5 of 108 are unread, so delete rather than document |
| `src` symlink and flat fallbacks | 17 runtime modules | Preserves exactly the ambiguity that made the package uninstallable |

## Migration for `src/indicators.py`

`calculate_adx` is the only unique function; move it into
`features/technical.py` and register it. `calculate_indicators` and
`calculate_all_indicators` are used only by the live orchestrator, which is
frozen under T11, so they go with it. Then delete the module.

## Explicitly not deleted

The RL exposure module, the Markov regime model, the portfolio optimizer and
GARCH. All unwired today, which makes them tempting — but under the forecasting
premise they become feature generators and conditioning variables. Deleting
them means rebuilding them in three weeks.

## Acceptance criteria

- [ ] No `.pyc` or `__pycache__` tracked.
- [ ] `src/indicators.py` gone, ADX available from the feature registry, tests
      migrated.
- [ ] Synthetic data requires an explicit opt-in; a missing-data run fails
      loudly.
- [ ] Data and checkpoints untracked, with a documented way to obtain them.
- [ ] The five unread settings removed from schema and YAML.
- [ ] Full test suite still passes.

## Risk

Removing tracked data stops the bloat but does not reclaim the 80 MB already in
history; that needs a rewrite, which is a separate decision with its own
disruption. Note it, do not bundle it.
