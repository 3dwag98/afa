# T06 — Gradient-boosting baseline trainer

**Status:** done · **Effort:** ~2 days · **Depends on:** none (training registry already on main)
**Plan reference:** `docs/forecasting_plan.html` Part 4 (additions)

## Goal

Register a gradient-boosted-tree trainer and make it the baseline every new
idea has to beat.

## Why

On tabular panel data at this sample size — roughly 2,400 names over five
years, and thin per name — gradient boosting typically outperforms sequence
networks, trains in seconds rather than minutes, and yields honest feature
importances. The LSTM is the more interesting model and the less appropriate
default.

Having a fast, strong baseline changes the research loop: an idea that cannot
beat boosting on identical features and splits is not worth the training time,
and finding that out should take a minute rather than an afternoon.

## Approach

Register as a trainer alongside `supervised` and `sac`. The training layer
already provides everything needed — registry dispatch, a per-trainer pydantic
schema with `extra="forbid"`, universe pinning, artifact writing — so this is
almost entirely the model itself.

Predict the same cross-sectional rank target the supervised path uses, so the
two are directly comparable.

## Files

| File | Change |
| --- | --- |
| `portfolio_agent/training/trainers/gbm.py` | New — `HistGradientBoostingRegressor` over the stacked cross-section |
| `portfolio_agent/features/labels.py` | New — the label definition, moved out from under `import torch` and re-exported so nothing that used it had to change |
| `portfolio_agent/training/trainers/__init__.py` | Register it; guard each built-in's import separately |
| `portfolio_agent/training/artifacts.py` | `save_sklearn_artifact` / `load_sklearn_artifact` — joblib plus a JSON sidecar |
| `portfolio_agent/training/base.py` | `checkpoint_suffix` and `write_checkpoint` hooks; `availability()`; rank IC leads `primary_metric` |
| `portfolio_agent/training/runner.py` | Route the write through the trainer's own writer |
| `portfolio_agent/training/registry.py` | `unavailable_trainers()`, and a hint that says which extra is missing |
| `portfolio_agent/cli.py` | Report unavailable trainers; stop requiring torch to reach a torch-free trainer |
| `pyproject.toml` | `gbm` extra (`scikit-learn>=1.3`) |
| `portfolio_agent/tests/test_gbm_trainer.py` | New — 44 tests |

## Acceptance criteria

- [x] `portfolio-agent list-trainers` shows it with its full settings.
- [x] Trains on the same panel as the supervised trainer and writes a
      checkpoint in the standard artifact shape.
- [x] Reproducible for a fixed seed.
- [x] Feature importances are recorded in the artifact metadata.
- [x] Absent library degrades to a clear message, matching how torch is handled.

## What shipped beyond the spec, and why

Three things the spec did not call for turned out to be load-bearing:

**A date split, not a per-ticker one.** `agents/trainer.py` splits every ticker
at a fraction of *its own* history. Against a label ranked across names on a
date, that is a leak: a short-history name's validation rows sit in calendar
time inside a long-history name's training rows, so the training set contains
the answer. This trainer cuts on one global date instead, and purges the
`horizon` dates before it, since a label dated `t` is only realized at
`t + horizon`.

**A second artifact writer.** A fitted estimator cannot go through
`save_artifact`: the strategy loaders read with `torch.load(weights_only=True)`,
which refuses pickled objects by design. Rather than weaken that guarantee, the
estimator gets `save_sklearn_artifact` — joblib, plus a JSON sidecar that
carries the metadata and metrics so comparison tooling never has to unpickle a
checkpoint to find out how it scored.

**Per-module import guards.** `sac` imports PyTorch at module scope, and a
single `try` around all three built-ins meant a torch-less install lost `gbm`
and `supervised` with it. Guarded separately, `pip install
portfolio-agent[gbm]` alone now trains, scores and persists a real forecasting
model with no torch anywhere in the path — asserted in a subprocess test.

## Measured

On 120 names drawn from the parquet cache, 300 iterations, 19 seconds:
mean validation rank IC 0.084, ICIR 0.84, positive on 79% of 236 validation
dates. That is one split, not an evaluation — T04 and T05 are what would turn
it into a claim.
