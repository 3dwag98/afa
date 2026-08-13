"""Gradient-boosted trees over the cross-sectional panel — the baseline.

Why this exists
---------------
This is the model every new idea has to beat before it earns training time.
On tabular panel data at this sample size — a few thousand Indian names, thin
per name, eight engineered features — boosting is not a weak comparator that
makes the neural network look good. It is usually the stronger model, it fits
in seconds rather than minutes, and it reports feature importances that can be
argued with. The LSTM is the more interesting model and the less appropriate
default; the honest way to hold that opinion is to run both on identical data
and print both numbers.

Comparability is the whole point, so it is enforced structurally rather than by
convention. The label comes from `features/labels.py` — the same
`build_forward_return` and `apply_cross_sectional_target` the supervised
pipeline uses — and the inputs get the same per-date cross-sectional
standardization from `features/scaling.py`. Neither is reimplemented here. Two
trainers whose labels differ by a horizon or a rank normalization produce
metrics that cannot be compared, and nothing in a results table would show it.

Where this deliberately differs from the supervised path
--------------------------------------------------------
**The split is by date, not by each ticker's own row count.**
`agents/trainer.py` splits every ticker at 70%/85% of *its own* history. For a
model that scores one name at a time that is merely arbitrary; for a
cross-sectional model it is a leak. A ticker with a short history has its
validation rows sitting in calendar time inside another ticker's training rows,
and since the label is a rank *against those other names on that date*, the
training set contains the answer. One global date cut removes the whole class
of problem: every row before the cut trains, every row after it validates.

**The boundary is purged.** A label dated `t` is only realized at `t + horizon`,
so training rows within one horizon of the cut peek across it. Those rows are
dropped. It costs `horizon` days of a multi-year sample and removes an
optimistic bias that is otherwise unmeasurable after the fact.

**The headline metric is rank IC, not loss.** Mean-squared error against a rank
label is a number that goes down; whether the model *orders* the cross-section
correctly is the question a long-only ranking book actually asks. Both are
reported, and `val_rank_ic` is what `primary_metric()` picks up.

Inherited knobs
---------------
`TrainerConfig` is shared, so a few of its fields need an explicit reading here
rather than being quietly ignored:

* `epochs` is the number of boosting iterations. Each one fits a tree to the
  current residuals over the full training block — one pass over the training
  data, which is what the field says it is.
* `learning_rate` is the shrinkage applied to each tree. Its base default of
  1e-3 is right for a network and useless for boosting, so it is redeclared.
* `batch_size` and `device` genuinely do not apply. They are redeclared with
  descriptions saying so, because `list-trainers` prints those descriptions and
  a knob that silently does nothing is exactly what this package exists to stop.

The estimator is a pickle, not a tensor dict
--------------------------------------------
A fitted scikit-learn model cannot go through `save_artifact`: the strategy
loaders read checkpoints with `torch.load(weights_only=True)`, which refuses
pickled objects on purpose. So this trainer writes a `.joblib` alongside a JSON
sidecar carrying the metadata and metrics — the sidecar being the part that
comparison tooling can read *without* unpickling anything. See
`artifacts.py::save_sklearn_artifact`.

One consequence worth stating: nothing in this module imports PyTorch, and the
joblib path does not either, so a `pip install portfolio-agent[gbm]` with no
torch at all can train, score and persist a real forecasting model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from pydantic import Field

from ...evaluation.metrics import MIN_CROSS_SECTION_NAMES, rank_ic_from_arrays
from ..base import BaseTrainer, TrainerConfig, TrainingArtifact, TrainingData
from ..data import prepare_panel
from ..registry import register_trainer

logger = logging.getLogger(__name__)

#: Features the supervised pipeline is hardcoded against. Duplicated as a
#: fallback rather than imported at module scope, because
#: `agents.trainer.TRAINING_FEATURE_NAMES` lives behind `import torch` and the
#: point of this trainer is that it does not need one. The two are asserted
#: equal in the tests, so a change to either is caught rather than diverging.
DEFAULT_GBM_FEATURES = [
    "sma_20", "sma_50", "rsi_14", "macd",
    "bollinger_pct_b", "atr_14", "return_1d", "return_5d",
]

_MISSING_SKLEARN = (
    "The 'gbm' trainer needs scikit-learn, which is not installed.\n"
    "  uv sync --extra gbm      (or: pip install 'portfolio-agent[gbm]')\n"
    "Every other trainer is unaffected — `portfolio-agent list-trainers` still "
    "shows what this install can run."
)


def _require_sklearn():
    """Import scikit-learn, or fail with a message that says what to do.

    Imported here rather than at module scope so that an install without the
    extra still *registers* this trainer: `list-trainers` should say the gbm
    baseline exists and how to enable it, not omit it and leave the user
    wondering. Mirrors how the registry treats a missing PyTorch.
    """
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise ImportError(_MISSING_SKLEARN) from exc
    return sklearn


class GBMTrainerConfig(TrainerConfig):
    """Hyperparameters for the boosting baseline.

    Tree-shaped knobs are named as scikit-learn names them, so a setting can be
    looked up in its documentation without a translation table.
    """

    # -- inherited fields, restated because their base defaults or their
    #    meanings do not carry over to a tree ensemble --------------------
    learning_rate: float = Field(
        default=0.05, gt=0.0, le=1.0,
        description="Shrinkage applied to each tree. Redeclared because the base "
        "default (1e-3) is a network's step size: at that shrinkage a few hundred "
        "trees have barely moved off the mean.",
    )
    batch_size: int = Field(
        default=128, gt=0,
        description="Unused. Boosting fits each tree against the full training "
        "block; there is no mini-batch. Present only because it is inherited.",
    )
    device: str = Field(
        default="cpu",
        description="Unused. The histogram implementation is CPU-only; a value "
        "other than 'cpu' or 'auto' is reported and ignored rather than "
        "pretending to move anything to a GPU.",
    )
    seed: int = Field(
        default=42,
        description="Seeds the histogram binning subsample and the permutation "
        "importances. Two runs of one configuration must agree; both of those "
        "draw at random and neither would otherwise.",
    )
    train_fraction: float = Field(
        default=0.8, gt=0.0, lt=1.0,
        description="Share of *distinct dates* used for fitting — not of each "
        "ticker's own rows, which is what the supervised pipeline splits on. A "
        "per-ticker split puts a short-history name's validation rows inside "
        "another name's training rows, and against a label ranked across names "
        "on a date that hands the training set the answer.",
    )

    # -- what to predict --------------------------------------------------
    target: str = Field(
        default="return_5d",
        description="Forward-return label, in the same spelling the supervised "
        "trainer uses. The horizon is read out of the digits.",
    )
    target_transform: str = Field(
        default="cross_sectional_rank",
        description="'cross_sectional_rank', 'cross_sectional_demean' or "
        "'absolute'. The rank target is the comparable one and the default.",
    )
    feature_normalization: str = Field(
        default="cross_sectional",
        description="'cross_sectional' z-scores each feature across the universe "
        "per date; 'pooled' fits one standardizer on the training rows; 'none' "
        "leaves features raw. Trees are invariant to a monotone per-feature "
        "transform, so 'pooled' and 'none' fit identically — but 'cross_sectional' "
        "is a different transform on every date and genuinely changes the model, "
        "which is why it is the default against a cross-sectional label.",
    )
    features: Optional[List[str]] = Field(
        default=None,
        description="Feature registry names. None uses the same eight the "
        "supervised pipeline trains on.",
    )

    # -- the split --------------------------------------------------------
    purge_days: Optional[int] = Field(
        default=None, ge=0,
        description="Training rows within this many days of the split date are "
        "dropped, because their labels are only realized after it. None uses the "
        "label horizon, which is the correct value; 0 disables purging and is "
        "there to make the size of the bias measurable rather than theoretical.",
    )
    min_history: int = Field(
        default=252, gt=0,
        description="Rows a ticker needs after cleaning. The long-lookback "
        "features are NaN until their window fills.",
    )

    # -- the ensemble -----------------------------------------------------
    max_leaf_nodes: int = Field(
        default=31, gt=1,
        description="Leaves per tree. The main capacity knob; 31 is the standard "
        "starting point for histogram boosting.",
    )
    max_depth: Optional[int] = Field(
        default=None, gt=0,
        description="Depth cap. None lets max_leaf_nodes govern alone, which is "
        "the usual choice for leaf-wise growth.",
    )
    min_samples_leaf: int = Field(
        default=100, gt=0,
        description="Rows required in a leaf. Higher than scikit-learn's default "
        "of 20: with a rank label whose signal-to-noise is low, small leaves fit "
        "noise that looks like alpha in-sample.",
    )
    l2_regularization: float = Field(
        default=1.0, ge=0.0,
        description="L2 penalty on leaf values. Non-zero by default for the same "
        "reason min_samples_leaf is raised.",
    )
    max_bins: int = Field(
        default=255, gt=1, le=255,
        description="Histogram bins per feature.",
    )
    early_stopping: bool = Field(
        default=True,
        description="Stop when the held-out score stops improving. Scored on the "
        "purged validation block built here, never on a random slice of the "
        "training rows — scikit-learn's own `validation_fraction` would take one, "
        "and a random slice of a time series leaks.",
    )
    n_iter_no_change: int = Field(
        default=20, gt=0,
        description="Iterations without improvement before early stopping fires.",
    )

    # -- reporting --------------------------------------------------------
    importance_repeats: int = Field(
        default=5, ge=0,
        description="Permutation repeats per feature, computed on the validation "
        "block. 0 skips importances. Permutation rather than the impurity "
        "importances trees report by default: those are computed on the training "
        "rows and inflate whichever feature had the most distinct values.",
    )


@dataclass
class GBMPanel:
    """The flat matrices a tree ensemble consumes, plus what produced them.

    Assembled once in `prepare()` and consumed in `fit()`. Kept as its own type
    rather than smuggled into `TrainingData` because the shapes are genuinely
    different: `TrainingData` is per-ticker frames for a model that walks one
    name at a time, and this is a stacked cross-section with a date column.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    val_dates: np.ndarray
    feature_names: List[str]
    split_date: pd.Timestamp
    horizon_days: int
    n_tickers: int
    purged_rows: int = 0
    scaler: Any = None
    train_dates: np.ndarray = field(default_factory=lambda: np.array([]))
    #: Every distinct date in the panel, sorted — *including* the ones the
    #: purge removed. Positions in this index are the only ones the purge
    #: condition can be stated against: an index rebuilt from the surviving
    #: train and validation dates has the purged gap closed up, and measuring
    #: the gap in it silently reports 1 no matter how wide the purge was.
    all_dates: np.ndarray = field(default_factory=lambda: np.array([]))


def horizon_from_target(target: str) -> int:
    """Trading days between the decision and the realized label.

    Reads the digits out of the target name exactly as
    `features/labels.py::build_forward_return` does, so the purge gap and the
    label can never disagree about the horizon.
    """
    digits = "".join(ch for ch in target if ch.isdigit())
    if "return" in target and digits:
        return max(1, int(digits))
    return 1


def build_gbm_panel(
    app_config: Any, universe: List[str], cfg: GBMTrainerConfig
) -> Tuple[GBMPanel, TrainingData]:
    """Assemble the stacked cross-section this trainer fits on.

    Returns both the flat matrices and a populated `TrainingData`, because the
    runner and the notebook facade are written against the latter and a trainer
    that returned an empty placeholder would break the universe bookkeeping.

    Raises:
        ValueError: If either side of the date split ends up empty. That is
            always a configuration problem (a horizon longer than the sample, a
            train_fraction at an extreme) and never something to train through.
    """
    from portfolio_agent.features.labels import (
        apply_cross_sectional_target,
        build_forward_return,
        target_column_name,
    )
    from portfolio_agent.features.scaling import apply_cross_sectional_scaling

    feature_names = list(cfg.features or DEFAULT_GBM_FEATURES)
    horizon = horizon_from_target(cfg.target)
    target_column = target_column_name(cfg.target)

    # Raw features: any standardization happens below, after the label exists,
    # so the two transforms see the same set of surviving rows.
    data = prepare_panel(
        app_config,
        universe,
        feature_names,
        train_fraction=cfg.train_fraction,
        min_history=cfg.min_history,
        fit_scaler=False,
    )

    panel: Dict[str, pd.DataFrame] = {}
    for ticker in data.tickers:
        frame = data.features_by_ticker[ticker]
        close = data.prices_by_ticker[ticker]["close"].astype(float)
        labelled = frame.copy()
        labelled[target_column] = build_forward_return(close, cfg.target)
        # The last `horizon` rows have no realized label by construction.
        labelled = labelled.dropna(subset=[target_column])
        if not labelled.empty:
            panel[ticker] = labelled

    if not panel:
        raise ValueError(
            f"No ticker retained a {cfg.target} label. The horizon "
            f"({horizon} days) is longer than the usable history."
        )

    if cfg.target_transform != "absolute" and len(panel) >= 2:
        panel = apply_cross_sectional_target(panel, target_column, cfg.target_transform)

    scaler = None
    if cfg.feature_normalization == "cross_sectional" and len(panel) >= 2:
        panel = apply_cross_sectional_scaling(panel, feature_names)

    all_dates = pd.DatetimeIndex(
        sorted({d for frame in panel.values() for d in frame.index})
    )
    if len(all_dates) < 3:
        raise ValueError(
            f"Only {len(all_dates)} distinct dates survived; a date split needs more."
        )

    cut = int(len(all_dates) * cfg.train_fraction)
    cut = min(max(cut, 1), len(all_dates) - 1)
    split_date = all_dates[cut]

    # Purge: a row dated t carries a label realized at t + horizon, so training
    # rows inside one horizon of the cut have already seen across it.
    purge = horizon if cfg.purge_days is None else int(cfg.purge_days)
    purge_cutoff_pos = max(cut - purge, 0)
    purge_cutoff = all_dates[purge_cutoff_pos]

    train_blocks: List[pd.DataFrame] = []
    val_blocks: List[pd.DataFrame] = []
    purged_rows = 0
    for frame in panel.values():
        is_train = frame.index < split_date
        train_part = frame[is_train]
        kept = train_part[train_part.index < purge_cutoff] if purge else train_part
        purged_rows += len(train_part) - len(kept)
        if not kept.empty:
            train_blocks.append(kept)
        val_part = frame[~is_train]
        if not val_part.empty:
            val_blocks.append(val_part)

    if not train_blocks or not val_blocks:
        raise ValueError(
            "The date split left one side empty "
            f"(train blocks: {len(train_blocks)}, validation blocks: "
            f"{len(val_blocks)}). Check train_fraction and purge_days against a "
            f"sample of {len(all_dates)} dates."
        )

    train = pd.concat(train_blocks).sort_index()
    validation = pd.concat(val_blocks).sort_index()

    if cfg.feature_normalization == "pooled":
        from portfolio_agent.features.scaling import FeatureScaler

        scaler = FeatureScaler.fit(train[feature_names].to_numpy(dtype=np.float64))
        for block in (train, validation):
            block[feature_names] = scaler.transform(
                block[feature_names].to_numpy(dtype=np.float64)
            )

    gbm_panel = GBMPanel(
        x_train=train[feature_names].to_numpy(dtype=np.float64),
        y_train=train[target_column].to_numpy(dtype=np.float64),
        x_val=validation[feature_names].to_numpy(dtype=np.float64),
        y_val=validation[target_column].to_numpy(dtype=np.float64),
        val_dates=validation.index.to_numpy(),
        train_dates=train.index.to_numpy(),
        all_dates=all_dates.to_numpy(),
        feature_names=feature_names,
        split_date=split_date,
        horizon_days=horizon,
        n_tickers=len(panel),
        purged_rows=purged_rows,
        scaler=scaler,
    )

    # Hand back a TrainingData describing what was actually used, so the
    # universe fingerprint the runner records refers to the surviving names.
    training_data = TrainingData(
        features_by_ticker={t: f[feature_names] for t, f in panel.items()},
        prices_by_ticker={
            t: data.prices_by_ticker[t].reindex(f.index) for t, f in panel.items()
        },
        tickers=sorted(panel),
        feature_names=feature_names,
        scaler=scaler,
        split_index_by_ticker={
            t: int((f.index < split_date).sum()) for t, f in panel.items()
        },
    )
    logger.info(
        "GBM panel: %d train rows, %d validation rows, %d purged, split at %s "
        "(%d names, %d features)",
        len(gbm_panel.y_train), len(gbm_panel.y_val), purged_rows,
        split_date.date(), gbm_panel.n_tickers, len(feature_names),
    )
    return gbm_panel, training_data


def rank_ic_by_date(
    predictions: np.ndarray,
    labels: np.ndarray,
    dates: np.ndarray,
    min_names: int = MIN_CROSS_SECTION_NAMES,
) -> pd.Series:
    """Spearman correlation of prediction against label, one value per date.

    This is the information coefficient in the form that matters to a
    cross-sectional book: on each date, did the model *order* the names
    correctly? Pooling all rows into one correlation instead would mostly
    measure whether the model tracks the market's day-to-day level, which a
    long-only ranking strategy cannot trade.

    Kept as a name because the trainer and its tests read well with it, but the
    arithmetic lives in `evaluation/metrics.py` — one definition of IC, so a
    change to how thin dates or constant columns are handled reaches the
    trainer and the evaluation harness together instead of one of them.
    """
    return rank_ic_from_arrays(predictions, labels, dates, min_names=min_names)


@register_trainer("gbm")
class GBMTrainer(BaseTrainer):
    """Histogram gradient boosting on the cross-sectional panel."""

    name = "gbm"
    strategy_name = None  # a general forecaster, like the supervised trainer

    #: A fitted estimator is a pickle, not a tensor dict; see the module
    #: docstring and `artifacts.py::save_sklearn_artifact`.
    checkpoint_suffix = ".joblib"

    def __init__(self) -> None:
        self._panel: Optional[GBMPanel] = None

    @classmethod
    def availability(cls) -> Optional[str]:
        """Report a missing scikit-learn without importing it."""
        from importlib.util import find_spec

        if find_spec("sklearn") is None:
            return "needs scikit-learn (uv sync --extra gbm)"
        return None

    @classmethod
    def config_model(cls) -> Type[TrainerConfig]:
        return GBMTrainerConfig

    @staticmethod
    def write_checkpoint(artifact: TrainingArtifact, path) -> Any:
        from ..artifacts import save_sklearn_artifact

        return save_sklearn_artifact(artifact, path)

    def prepare(
        self, app_config: Any, universe: List[str], cfg: TrainerConfig
    ) -> TrainingData:
        assert isinstance(cfg, GBMTrainerConfig)
        _require_sklearn()
        if cfg.device not in ("cpu", "auto"):
            logger.info(
                "device=%r is ignored: histogram gradient boosting is CPU-only.",
                cfg.device,
            )
        self._panel, training_data = build_gbm_panel(app_config, universe, cfg)
        return training_data

    def fit(self, data: TrainingData, cfg: TrainerConfig) -> TrainingArtifact:
        assert isinstance(cfg, GBMTrainerConfig)
        from sklearn.ensemble import HistGradientBoostingRegressor

        panel = self._panel
        if panel is None:
            raise RuntimeError("prepare() must run before fit()")

        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=cfg.learning_rate,
            max_iter=cfg.epochs,
            max_leaf_nodes=cfg.max_leaf_nodes,
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            l2_regularization=cfg.l2_regularization,
            max_bins=cfg.max_bins,
            # scikit-learn's own early stopping would carve a validation slice
            # out of the training rows, and on a time series it takes that slice
            # at random. The purged block built above is passed explicitly
            # instead, so the stopping decision is made on data the model has
            # not seen and could not have seen.
            early_stopping=False,
            random_state=cfg.seed,
        )

        if cfg.early_stopping:
            n_iterations = self._fit_with_early_stopping(model, panel, cfg)
        else:
            model.fit(panel.x_train, panel.y_train)
            n_iterations = int(model.n_iter_)

        metrics = self._score(model, panel)
        metrics["n_iterations"] = n_iterations
        metrics["n_train_rows"] = int(len(panel.y_train))
        metrics["n_val_rows"] = int(len(panel.y_val))
        metrics["n_purged_rows"] = int(panel.purged_rows)

        importances = self._importances(model, panel, cfg)

        from ..artifacts import build_metadata

        metadata = build_metadata(
            feature_names=panel.feature_names,
            scaler=panel.scaler,
            trainer=self.name,
            extra={
                "model_architecture": "hist_gradient_boosting",
                "library": "scikit-learn",
                "library_version": _sklearn_version(),
                "target": cfg.target,
                "target_transform": cfg.target_transform,
                "feature_normalization": cfg.feature_normalization,
                "horizon_days": panel.horizon_days,
                "split_date": str(pd.Timestamp(panel.split_date).date()),
                "purged_rows": panel.purged_rows,
                "feature_importances": importances,
                "importance_method": (
                    "permutation_on_validation" if importances else "not_computed"
                ),
                "training_config": cfg.model_dump(),
                "n_tickers": panel.n_tickers,
            },
        )
        # The estimator rides in `state_dict` under a single key. It is not a
        # tensor dict and never claims to be — `save_sklearn_artifact` is the
        # only writer that accepts it, and it writes joblib rather than torch.
        return TrainingArtifact(
            state_dict={"estimator": model}, metadata=metadata, metrics=metrics
        )

    # -- fitting -----------------------------------------------------------

    def _fit_with_early_stopping(
        self, model: Any, panel: GBMPanel, cfg: GBMTrainerConfig
    ) -> int:
        """Grow the ensemble, stopping when held-out loss stops improving.

        `warm_start` refits incrementally: raising `max_iter` and calling `fit`
        again continues from the trees already grown rather than starting over,
        so scoring between checkpoints costs one prediction pass and not a
        retrain. The step is `n_iter_no_change`, which makes the worst-case
        overshoot exactly one patience window.
        """
        model.set_params(warm_start=True)
        step = cfg.n_iter_no_change
        best_loss = np.inf
        best_iter = 0
        grown = 0

        while grown < cfg.epochs:
            grown = min(grown + step, cfg.epochs)
            model.set_params(max_iter=grown)
            model.fit(panel.x_train, panel.y_train)
            loss = float(
                np.mean((model.predict(panel.x_val) - panel.y_val) ** 2)
            )
            logger.debug("iterations=%d val_mse=%.6f", grown, loss)
            if loss < best_loss:
                best_loss = loss
                best_iter = grown
                continue
            logger.info(
                "Early stop at %d iterations; best validation MSE %.6f at %d.",
                grown, best_loss, best_iter,
            )
            break

        # Refit at the best size. Truncating a warm-started ensemble in place is
        # not part of the public API, and shipping an ensemble that is one
        # patience window past its best is exactly the overfit this guards.
        if best_iter and best_iter != grown:
            model.set_params(warm_start=False, max_iter=best_iter)
            model.fit(panel.x_train, panel.y_train)
        model.set_params(warm_start=False)
        return int(model.n_iter_)

    # -- scoring -----------------------------------------------------------

    def _score(self, model: Any, panel: GBMPanel) -> Dict[str, Any]:
        """Loss on both blocks, plus the rank IC statistics that matter."""
        train_pred = model.predict(panel.x_train)
        val_pred = model.predict(panel.x_val)

        metrics: Dict[str, Any] = {
            "train_loss": float(np.mean((train_pred - panel.y_train) ** 2)),
            "val_loss": float(np.mean((val_pred - panel.y_val) ** 2)),
        }

        ic = rank_ic_by_date(val_pred, panel.y_val, panel.val_dates)
        if ic.empty:
            metrics["val_rank_ic"] = 0.0
            metrics["val_ic_dates"] = 0
            return metrics

        mean_ic = float(ic.mean())
        std_ic = float(ic.std(ddof=1)) if len(ic) > 1 else 0.0
        metrics["val_rank_ic"] = mean_ic
        metrics["val_rank_ic_std"] = std_ic
        # ICIR — the IC's own t-statistic in disguise. A mean IC of 0.03 with a
        # std of 0.02 is a strategy; the same mean with a std of 0.30 is noise,
        # and only the ratio distinguishes them.
        metrics["val_icir"] = mean_ic / std_ic if std_ic > 1e-12 else 0.0
        metrics["val_ic_hit_rate"] = float((ic > 0).mean())
        metrics["val_ic_dates"] = int(len(ic))
        return metrics

    def _importances(
        self, model: Any, panel: GBMPanel, cfg: GBMTrainerConfig
    ) -> Dict[str, float]:
        """Permutation importance on the validation block, largest first.

        `HistGradientBoostingRegressor` exposes no `feature_importances_` at
        all, so there is no impurity-based shortcut to fall back on — which is
        just as well. Impurity importances are computed on the training rows
        and systematically favour whichever feature had the most distinct
        values to split on. Permuting a column on held-out rows and measuring
        what the score loses answers the question people think they are asking.
        """
        if cfg.importance_repeats <= 0 or panel.x_val.size == 0:
            return {}
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            model,
            panel.x_val,
            panel.y_val,
            n_repeats=cfg.importance_repeats,
            random_state=cfg.seed,
            scoring="neg_mean_squared_error",
        )
        pairs = zip(panel.feature_names, result.importances_mean)
        return {
            name: float(value)
            for name, value in sorted(pairs, key=lambda kv: -kv[1])
        }


def _sklearn_version() -> str:
    """Installed scikit-learn version, for checkpoint provenance."""
    try:
        import sklearn

        return str(sklearn.__version__)
    except ImportError:  # pragma: no cover - guarded by _require_sklearn
        return "unknown"
