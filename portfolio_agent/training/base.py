"""The trainer contract: config schema, data container, artifact, base class.

A trainer owns *how* a strategy learns. It declares its own hyperparameter
schema, prepares its own data, runs its own loop, and hands back an artifact in
one canonical shape. Everything else in this package — the registry, config
resolution, bulk runs, the notebook facade — is written against these four
objects and knows nothing about any particular training procedure.

The single most important design choice here is that **each trainer declares a
pydantic schema for its own hyperparameters** (`config_model`). Training
knobs used to be one global `training:` block, so a knob that only meant
something to one procedure had nowhere to live, and a misspelled knob was
silently ignored. Both failure modes are real: the SAC work that prompted this
package shipped `--buffer-size` and `--entropy-coef` flags that were parsed,
accepted, printed in the help text, and then never passed to the training
function at all. A per-trainer schema with `extra="forbid"` turns that class of
mistake into a startup error naming the offending key.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Type

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from portfolio_agent.features.scaling import FeatureScaler


class TrainerConfig(BaseModel):
    """Hyperparameters shared by every trainer.

    Subclass this to add procedure-specific knobs. `extra="forbid"` is the
    point of the class: an unrecognized key is a typo or a knob aimed at a
    different trainer, and either way silently dropping it produces a run that
    looks like it honoured a setting it never saw.
    """

    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(default=100, gt=0, description="Passes over the training data.")
    batch_size: int = Field(default=128, gt=0, description="Mini-batch size.")
    learning_rate: float = Field(default=1e-3, gt=0, description="Optimizer step size.")
    device: str = Field(
        default="auto",
        description="'auto', 'cuda', 'mps' or 'cpu'. Resolved through "
        "utils/device.py::get_device, which downgrades an unavailable "
        "accelerator rather than failing.",
    )
    seed: int = Field(
        default=42,
        description="Seeds torch, numpy and every sampler the trainer owns. The "
        "platform requires two runs of one configuration to agree; an "
        "unseeded replay buffer or shuffle is enough to break that.",
    )
    train_fraction: float = Field(
        default=0.8,
        gt=0.0,
        lt=1.0,
        description="Chronological share of each ticker's history used for fitting. "
        "The remainder is held out for validation — never a random split, "
        "which would leak tomorrow into today.",
    )


@dataclass
class TrainingData:
    """Everything a trainer needs, assembled once and shared across trainers.

    Holding prices alongside features is deliberate: a supervised trainer needs
    a forward-return label and an RL trainer needs a reward, and both are
    derived from prices rather than from the standardized feature block.

    Attributes:
        features_by_ticker: Standardized feature frames, one per ticker, in
            chronological order with the columns in `feature_names` order.
        prices_by_ticker: Raw OHLCV frames aligned to the same index.
        tickers: The tickers that survived data preparation, sorted.
        feature_names: Column order the model was fitted against. Travels into
            the artifact, because a state vector assembled in a different order
            than training used is undetectable from the weights.
        scaler: The fitted standardizer, or None when a trainer consumes raw
            features. Travels into the artifact for the same reason.
        split_index_by_ticker: Row index at which each ticker's validation
            segment begins.
    """

    features_by_ticker: Dict[str, pd.DataFrame]
    prices_by_ticker: Dict[str, pd.DataFrame]
    tickers: List[str]
    feature_names: List[str]
    scaler: Optional[FeatureScaler] = None
    split_index_by_ticker: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tickers:
            raise ValueError("TrainingData needs at least one ticker")
        if not self.feature_names:
            raise ValueError("TrainingData needs at least one feature name")

    def split(self, ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (train, validation) frames for one ticker, chronologically."""
        frame = self.features_by_ticker[ticker]
        cut = self.split_index_by_ticker.get(ticker, len(frame))
        return frame.iloc[:cut], frame.iloc[cut:]


@dataclass
class TrainingArtifact:
    """What a trainer produces: weights, provenance, and headline metrics.

    `metadata` is written verbatim into the checkpoint's `metadata` block, so
    whatever a trainer records here is what inference will be able to read
    back. `artifacts.py` fills in the fields every strategy loader depends on
    (`feature_names`, `feature_scaler`) so no individual trainer can forget one.
    """

    state_dict: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def primary_metric(self) -> Optional[float]:
        """The one number worth printing in a summary table, if there is one.

        Bulk runs rank heterogeneous trainers against each other, so this
        prefers validation numbers over training ones and returns None rather
        than inventing a comparison when a trainer reports neither.
        """
        for key in (
            "val_sortino",
            "val_sharpe",
            "best_val_loss",
            "val_loss",
            "final_reward",
            "final_loss",
        ):
            value = self.metrics.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None


class BaseTrainer(ABC):
    """Base class for a registered training procedure.

    Subclasses declare a config schema, prepare data and run a loop. They do
    *not* write checkpoints: `artifacts.py::save_artifact` is the only writer,
    which is what keeps the on-disk contract identical across trainers.
    """

    #: Registry name, for error messages and artifact provenance.
    name: ClassVar[str] = "base"

    #: Which strategy this trainer produces weights for, when that is fixed.
    #: None means the trainer is strategy-agnostic (the supervised one is).
    strategy_name: ClassVar[Optional[str]] = None

    @property
    def writes_own_checkpoint(self) -> bool:
        """Whether `fit` already persisted the weights itself.

        False for everything written against this package — the runner calls
        `save_artifact` so there is one writer and one on-disk shape. True only
        for the supervised adapter, which delegates to a pre-existing pipeline
        that writes its own checkpoint and its own `models/metadata.json`
        sidecar; re-saving over those would replace a populated checkpoint with
        an empty one.
        """
        return False

    @classmethod
    def config_model(cls) -> Type[TrainerConfig]:
        """The pydantic schema for this trainer's hyperparameters."""
        return TrainerConfig

    @classmethod
    def default_config(cls) -> TrainerConfig:
        """This trainer's hyperparameters with every default applied."""
        return cls.config_model()()

    @abstractmethod
    def prepare(self, app_config: Any, universe: List[str], cfg: TrainerConfig) -> TrainingData:
        """Load and featurize `universe` into the block this trainer consumes.

        Args:
            app_config: The loaded AppConfig.
            universe: Exact tickers to train on. Callers pin this (see
                `universe.py`) so two trainers can be compared on identical
                names rather than on two different random draws.
            cfg: Validated hyperparameters.
        """

    @abstractmethod
    def fit(self, data: TrainingData, cfg: TrainerConfig) -> TrainingArtifact:
        """Run the training loop and return weights plus provenance."""
