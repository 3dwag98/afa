# Custom models & strategies: pluggable training

How to train a strategy that does not learn the way the built-in supervised
pipeline learns — and how to compare several of them honestly.

## Contents

- [Why this exists](#why-this-exists)
- [The four seams](#the-four-seams)
- [Quick start](#quick-start)
- [Where training settings come from](#where-training-settings-come-from)
- [Pinning a universe](#pinning-a-universe)
- [Bulk training and sweeps](#bulk-training-and-sweeps)
- [Adding a trainer](#adding-a-trainer)
- [Making a strategy trainable](#making-a-strategy-trainable)
- [The checkpoint contract](#the-checkpoint-contract)
- [Notebooks](#notebooks)
- [The SAC trainer](#the-sac-trainer)

---

## Why this exists

The platform was already pluggable in three places: features
(`features/registry.py`), model architectures (`models/registry.py`) and
strategies (`strategies/registry.py`). What was *not* pluggable was **how a
strategy is trained**.

`agents/trainer.py::run_training` is one supervised pipeline — panel, sequence
windows, forward-return label, walk-forward, calibration — and it is a good
one. But a strategy that learns some other way (an RL policy, a ranker, a
meta-labeller) had nowhere to attach, and `TrainingConfig` was a single global
block, so a knob meaningful to only one procedure had nowhere to live.

Two consequences worth naming, because both are failures that do not announce
themselves:

- A hyperparameter aimed at the wrong procedure is **silently dropped**. Nothing
  in a run's output distinguishes "honoured your setting" from "ignored it".
- Two models trained on two different draws from the cache are **not
  comparable**, and a results table built from them says nothing about which
  is better.

This package closes both.

## The four seams

| What | Registry | Registered thing |
| --- | --- | --- |
| Features | `features/registry.py` | a function on an OHLCV frame |
| Architectures | `models/registry.py` | an `nn.Module` |
| Strategies | `strategies/registry.py` | a `BaseStrategy` |
| **Training procedures** | **`training/registry.py`** | **a `BaseTrainer`** |

## Quick start

```bash
# What can train, and with what settings
portfolio-agent list-trainers
portfolio-agent list-trainers --name sac

# Train a strategy through its declared trainer
portfolio-agent train --strategy india_sac

# Override hyperparameters (validated against that trainer's schema)
portfolio-agent train --strategy india_sac --set epochs=300 --set gamma=0.9

# Unchanged: no --strategy runs the supervised pipeline exactly as before
portfolio-agent train
```

## Where training settings come from

Strongest first:

1. `--set key=value` on the command line (or a keyword argument in a notebook)
2. `config/strategies/<strategy>.yaml`, under `training:`
3. the strategy class's `training_defaults()`
4. the global `training:` block in `config.yaml`
5. the trainer's schema defaults

Layer 4 is what keeps this backward compatible: an install with nothing but a
global `training:` block resolves exactly as it did before.

Each trainer declares a **pydantic schema** for its own hyperparameters, with
`extra="forbid"`. So a typo stops the run and names itself:

```
$ portfolio-agent train --strategy india_sac --set buffer_sizee=100
Error: Invalid training config for trainer 'sac':
  - buffer_sizee: not a setting this trainer accepts. Known settings:
    ['auto_entropy', 'batch_size', 'buffer_size', ...]
```

Only keys a trainer actually declares are taken from the *global* block —
otherwise `config.yaml`'s supervised-only settings (`sequence_length`,
`target_transform`) would trip `extra="forbid"` on every other trainer and make
the file un-loadable.

One asymmetry, on purpose: when you override the trainer for a strategy
(`--strategy india_sac --trainer supervised`), that strategy's YAML describes a
*different* procedure's knobs, so the ones the target trainer does not
understand are dropped with a log line rather than raising. When the trainer is
the one the strategy asked for, an unrecognized key is a real mistake and stops
the run.

## Pinning a universe

Comparing models is only meaningful when they saw the same names.
`resolve_backtest_universe` draws from whatever is cached, and offsets its seed
by purpose so training and backtesting deliberately sample *differently*. Both
behaviours are right for one production run and wrong for a comparison.

A **snapshot** freezes the universe and carries a content hash:

```bash
portfolio-agent train --strategy india_sac --save-snapshot universe/exp.json
portfolio-agent train --strategy lstm      --universe-snapshot universe/exp.json
```

```python
from portfolio_agent.training import UniverseSnapshot

snap = UniverseSnapshot.create(config, size=50, name="exp")
snap.save("universe/exp.json")
snap.fingerprint          # 'a3f9c1e0b7d2' — equal iff the names are equal
```

The fingerprint is order-insensitive and travels into every checkpoint's
metadata, so a model can be traced back to the sample it was fitted on.

Pinning a universe for training means giving up the train/backtest separation
that `purpose="train"` provides. That is the right trade when the point of the
run is a comparison, and the wrong one when the point is an honest
generalization estimate. Choose deliberately.

## Bulk training and sweeps

```bash
# Several strategies, one universe
portfolio-agent train-bulk --strategies india_sac,lstm --save-snapshot universe/cmp.json

# One strategy, a hyperparameter grid (cross product)
portfolio-agent train-bulk --strategies india_sac \
    --sweep gamma=0.0,0.9 --sweep entropy_coef=0.05,0.2
```

```python
from portfolio_agent.training import BulkJob, run_bulk, sweep

report = run_bulk(config, [
    BulkJob(strategy="india_sac", overrides={"epochs": 100}),
    BulkJob(strategy="india_sac", overrides={"epochs": 100, "gamma": 0.9}),
], universe_size=40)

report.to_frame()   # comparison table, successes first
report.best()       # highest primary metric
report.failures     # labels of jobs that raised
```

A failing job is recorded and the run continues — losing nine good results
because the tenth had a bad setting is the worse outcome. Sweeps default to
`save=False`, since a sweep usually asks *which settings to use* rather than
producing the model; when they do save, each point gets its own checkpoint name
so they cannot overwrite one another.

## Adding a trainer

```python
# portfolio_agent/training/trainers/my_trainer.py
from pydantic import Field

from ..base import BaseTrainer, TrainerConfig, TrainingArtifact, TrainingData
from ..data import prepare_panel
from ..registry import register_trainer


class MyTrainerConfig(TrainerConfig):
    """Inherits epochs, batch_size, learning_rate, device, seed, train_fraction."""
    horizon: int = Field(default=5, gt=0, description="Label horizon in days.")


@register_trainer("my_trainer")
class MyTrainer(BaseTrainer):
    name = "my_trainer"
    strategy_name = "my_strategy"      # or None if strategy-agnostic

    @classmethod
    def config_model(cls):
        return MyTrainerConfig

    def prepare(self, app_config, universe, cfg):
        # prepare_panel handles the parts that are easy to get wrong: it fits
        # the scaler on *training rows only*, keeps your column order, and
        # re-aligns prices to the surviving feature rows.
        return prepare_panel(
            app_config, universe, ["rsi_14", "macd", "atr_14"],
            train_fraction=cfg.train_fraction,
        )

    def fit(self, data: TrainingData, cfg) -> TrainingArtifact:
        from ..artifacts import build_metadata

        model = ...  # your loop here
        return TrainingArtifact(
            state_dict=model.state_dict(),
            metadata=build_metadata(
                feature_names=data.feature_names,
                scaler=data.scaler,
                trainer=self.name,
                extra={"horizon": cfg.horizon},
            ),
            metrics={"val_sharpe": 1.2},
        )
```

Then add it to `training/trainers/__init__.py` so importing the package
registers it.

Do **not** write the checkpoint yourself — `save_artifact` is the only writer,
which is what keeps the on-disk shape identical across trainers.

## Making a strategy trainable

`TrainableStrategy` is a *declaration*, not a training interface:

```python
from portfolio_agent.strategies.base import TrainableStrategy


class MyStrategy(TrainableStrategy):
    trainer_name = "my_trainer"

    @classmethod
    def training_defaults(cls):
        return {"epochs": 300}

    def load(self) -> bool:
        ...   # the contract the backtest engine already calls
```

The strategy names *which* trainer produces its weights; the trainer owns the
loop. That decoupling is the point: two strategies can share one procedure
without inheriting from each other, and retraining a strategy a different way
is a one-line change that never touches the scoring path.

Loading stays on `load()` — the method `agents/backtester.py` already calls. A
second loading method taking a path would be a third convention alongside
`load()` and `MLStrategy.load_model(name)`, and nothing would call it.

## The checkpoint contract

One writer, one shape:

```python
{
    "model_state_dict": {...},
    "metadata": {
        "feature_names":  [...],                              # column order
        "feature_scaler": {"mean": [...], "std": [...], "clip": 10.0},
        "trainer": "sac",
        "universe_fingerprint": "a3f9c1e0b7d2",
        ...
    },
    "metrics": {...},
}
```

Two failure modes `artifacts.py` exists to prevent, both silent:

- **A checkpoint that ships no scaler.** `FeatureScaler` exposes `.mean`/`.std`
  and serializes through `.to_dict()`. Code written against scikit-learn's
  `.mean_`/`.scale_` spelling finds neither, and behind a `hasattr` guard
  writes `None` without complaint. The model then trains on standardized inputs
  and scores raw ones. `build_metadata` takes the `FeatureScaler` object and
  calls `to_dict()` for you; `save_artifact` refuses an artifact with no
  `feature_names`.
- **A checkpoint that cannot be read back.** The strategy loaders use
  `torch.load(..., weights_only=True)`, which rejects arbitrary pickles.
  Metadata is coerced to plain Python primitives, so an `np.float32` that
  leaked out of a pandas computation does not make the file unloadable.

`test_training_artifacts.py::test_saved_checkpoint_loads_into_the_strategy`
asserts the round trip end to end — what a trainer writes is what
`IndiaSACStrategy.load()` reads, scaler included.

## Notebooks

`notebooks/` holds two, both built on `portfolio_agent.lab.Lab`, which pins a
universe once and uses it for every operation:

```python
from portfolio_agent.lab import Lab

lab = Lab(universe_size=40)
lab.save_universe("universe/exp.json")

run    = lab.train("india_sac", epochs=50)
report = lab.compare(["india_sac", "lstm"])   # same 40 names
result = lab.backtest("india_sac")            # still the same 40
```

See `notebooks/README.md`.

## The SAC trainer

`training/trainers/sac.py` trains `strategies/india_sac.py`'s actor. Points
worth knowing before reading the code:

- **The training actor is a superset of the inference one.** SAC optimizes a
  stochastic policy — a squashed Gaussian whose log-std head supplies the
  entropy term. Inference keeps only the mean head, because sampling at scoring
  time would make two backtest runs disagree. `inference_state_dict()` drops the
  extra head, so the saved weights load into `SACActorNetwork` under
  `strict=True`.
- **`gamma` defaults to 0.** Discounting assumes the action influences the next
  state; a price-taking book does not move the market, so the state sequence is
  exogenous and bootstrapping over it adds variance without signal. With
  `gamma=0` the critic learns `Q(s,a) = E[r|s,a]` and the actor maximizes
  `Q - alpha*log pi` — soft actor-critic on a contextual bandit, which is what
  this decision is. Configurable, because the turnover term does couple steps.
- **The reward is a differential Sortino net of turnover**, matching what the
  strategy's own docstring specifies: `R_t = a_t*ret_{t+1} - friction*|a_t -
  a_{t-1}|` feeds an online Sortino whose per-step increment is the reward. The
  friction term is a function of the *action* — a constant cost cannot penalize
  turnover, since it shifts every action's reward identically.
- **Experience is re-collected each epoch.** Collecting once and training
  against a frozen buffer means the actor only ever sees a randomly-initialized
  policy's decisions.
- **Validation is chronological and the deterministic policy is what is
  scored**, since that is what inference runs. The saved checkpoint is the best
  validation epoch, not the last.

A caveat stated rather than hidden: the turnover term depends on the previous
allocation, but the state cannot carry it — inference builds its state from
features alone, and a twelfth input would change `state_dim` and invalidate
every existing checkpoint. The process is therefore mildly partially observed.
That is a deliberate trade against the fixed inference contract.
