# T06 — Gradient-boosting baseline trainer

**Status:** not started · **Effort:** ~2 days · **Depends on:** none (training registry already on main)
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
| `portfolio_agent/training/trainers/gbm.py` | New |
| `portfolio_agent/training/trainers/__init__.py` | Register it |
| `pyproject.toml` | Optional extra for the boosting library |
| `portfolio_agent/tests/test_gbm_trainer.py` | New |

## Acceptance criteria

- [ ] `portfolio-agent list-trainers` shows it with its full settings.
- [ ] Trains on the same panel as the supervised trainer and writes a
      checkpoint in the standard artifact shape.
- [ ] Reproducible for a fixed seed.
- [ ] Feature importances are recorded in the artifact metadata.
- [ ] Absent library degrades to a clear message, matching how torch is handled.
