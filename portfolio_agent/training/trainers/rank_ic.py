"""Boosting that optimizes the metric the model is judged on.

Every trainer here fits a pointwise loss — squared error, pinball — and is then
ranked by rank IC. Those are different objectives, and the gap is not academic:
squared error on a cross-sectional rank label is minimized by predicting each
name's *conditional mean rank*, which is an excellent way to be nearly constant
and score an IC of approximately zero. The `gbm` baseline goes further and
chooses **which iteration ships** by validation MSE, so even the model-selection
step never looks at the number in the summary table.

LambdaRankIC (arXiv 2605.00501, May 2026) makes the case for closing that gap
directly, in the LambdaRank tradition: rather than differentiating a rank
metric — you cannot, ranks are piecewise constant — derive per-item gradients
from the metric and hand those to the booster.

What this implements, precisely
-------------------------------
The differentiable surrogate, not the paper's lambda construction. Per date,
the loss is the negative Pearson correlation between the raw scores and the
cross-sectionally ranked label:

    L = -(1/T) Σ_t  corr( s_t , y_t )

This is a surrogate for Spearman rather than Spearman itself, and it becomes
Spearman exactly when the scores are replaced by their own ranks. Two reasons
it is the right one to start with: the label is *already* a cross-sectional
rank under the default `target_transform`, so one side of the correlation is
the rank side; and the gradient is closed-form and exact, which makes it
checkable against a numerical derivative rather than trusted.

    s̃ = s - mean(s)          ỹ = y - mean(y)
    a = <s̃, ỹ>    b = ‖s̃‖    c = ‖ỹ‖    corr = a / (b·c)

    ∂corr/∂s_i = [ ỹ_i - corr·(c/b)·s̃_i ] / (b·c)

Naming it `rank_ic` rather than `lambdarank_ic` is deliberate: it optimizes an
IC surrogate, which is the paper's premise, but it is not the paper's lambda
formulation and should not borrow the name.

Why a hand-rolled boosting loop
-------------------------------
`HistGradientBoostingRegressor` takes no custom objective — scikit-learn
exposes a fixed set of losses. Gradient boosting with a custom objective is,
however, exactly "fit a tree to the negative gradient and take a shrunk step",
so the loop is short. It uses `DecisionTreeRegressor` as the base learner and
shares `build_gbm_panel` with the baseline, so the two trainers see an
identical panel, split, purge and scaler — which is what makes the comparison
between them a comparison of objectives and nothing else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from pydantic import Field

from ...evaluation.metrics import MIN_CROSS_SECTION_NAMES
from ..base import BaseTrainer, TrainerConfig, TrainingArtifact, TrainingData
from ..registry import register_trainer
from .gbm import (
    DEFAULT_GBM_FEATURES,
    GBMPanel,
    GBMTrainerConfig,
    _require_sklearn,
    _sklearn_version,
    build_gbm_panel,
    rank_ic_by_date,
)

logger = logging.getLogger(__name__)

#: Below this the correlation is numerically undefined and the gradient is
#: taken in its limiting direction instead. See `ic_gradient`.
DISPERSION_FLOOR = 1e-12


# --------------------------------------------------------------------------
# The objective
# --------------------------------------------------------------------------


def date_ic_gradient(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """d(corr)/d(scores) for one date, in closed form.

    Returns the gradient of the correlation itself — the quantity to *ascend*.
    The boosting loop negates it once, at the point where it fits a tree to a
    descent direction, so the sign lives in one place.

    The degenerate case matters more than it looks. At the first iteration
    every score is identical, so ‖s̃‖ = 0 and the correlation is undefined: a
    naive implementation divides by zero on iteration one and produces NaNs for
    the rest of the fit. The limit is well defined — with no score dispersion,
    the direction that most increases correlation is simply the centred label —
    so that is what is returned.
    """
    centred_scores = scores - scores.mean()
    centred_labels = labels - labels.mean()

    score_norm = float(np.linalg.norm(centred_scores))
    label_norm = float(np.linalg.norm(centred_labels))

    if label_norm <= DISPERSION_FLOOR:
        # A constant label makes no ordering claim, so there is nothing to
        # correlate with and no direction to move in.
        return np.zeros_like(scores)

    if score_norm <= DISPERSION_FLOOR:
        return centred_labels / label_norm

    correlation = float(centred_scores @ centred_labels) / (score_norm * label_norm)
    return (
        centred_labels - correlation * (label_norm / score_norm) * centred_scores
    ) / (score_norm * label_norm)


def ic_gradient(
    scores: np.ndarray,
    labels: np.ndarray,
    dates: np.ndarray,
    min_names: int = MIN_CROSS_SECTION_NAMES,
) -> np.ndarray:
    """Per-observation gradient of mean per-date correlation.

    Dates thinner than `min_names` contribute nothing — the same threshold the
    evaluation layer drops them at, so the objective is not being optimized on
    cross-sections the metric will refuse to score.

    Averaged over the *scored* dates rather than summed, so the step size does
    not depend on how long the panel is.
    """
    gradient = np.zeros_like(scores, dtype=float)
    frame = pd.DataFrame({"i": np.arange(len(scores)), "date": dates})

    scored = 0
    for _, group in frame.groupby("date", sort=False):
        positions = group["i"].to_numpy()
        if len(positions) < min_names:
            continue
        gradient[positions] = date_ic_gradient(scores[positions], labels[positions])
        scored += 1

    return gradient / scored if scored else gradient


def mean_date_correlation(
    scores: np.ndarray,
    labels: np.ndarray,
    dates: np.ndarray,
    min_names: int = MIN_CROSS_SECTION_NAMES,
) -> float:
    """The objective's value — mean per-date Pearson correlation.

    Reported alongside rank IC so the surrogate and the metric it stands in for
    can be watched separately. They move together but they are not equal, and a
    run where they diverge is worth knowing about.
    """
    total, scored = 0.0, 0
    frame = pd.DataFrame({"i": np.arange(len(scores)), "date": dates})
    for _, group in frame.groupby("date", sort=False):
        positions = group["i"].to_numpy()
        if len(positions) < min_names:
            continue
        s = scores[positions] - scores[positions].mean()
        y = labels[positions] - labels[positions].mean()
        denominator = float(np.linalg.norm(s) * np.linalg.norm(y))
        if denominator <= DISPERSION_FLOOR:
            continue
        total += float(s @ y) / denominator
        scored += 1
    return total / scored if scored else 0.0


# --------------------------------------------------------------------------
# The trainer
# --------------------------------------------------------------------------


class RankICTrainerConfig(GBMTrainerConfig):
    """The baseline's knobs, minus the ones a custom loop does not have.

    Inherits from `GBMTrainerConfig` deliberately: the panel, the split, the
    purge and the label are meant to be identical to the baseline's so that a
    comparison between them isolates the objective. Only the ensemble
    hyperparameters that scikit-learn's histogram implementation owns are
    replaced.
    """

    max_bins: int = Field(
        default=255, gt=1, le=255,
        description="Unused. Histogram binning belongs to the implementation "
        "this trainer does not use; kept so a config written for `gbm` still "
        "validates against `rank_ic`.",
    )
    subsample: float = Field(
        default=1.0, gt=0.0, le=1.0,
        description="Fraction of *dates* sampled for each tree. Sampling dates "
        "rather than rows because the gradient is defined per cross-section — "
        "half a date's names give the other half the wrong centring, so a row "
        "subsample would corrupt the objective rather than regularize it.",
    )
    max_depth: Optional[int] = Field(
        default=3, gt=0,
        description="Depth cap per tree. Shallower than the baseline's leaf-wise "
        "default because each tree here fits a gradient that is centred within "
        "every date, so its scale is small and its structure is noisy.",
    )


@register_trainer("rank_ic")
class RankICTrainer(BaseTrainer):
    """Gradient boosting on a differentiable IC surrogate.

    Same panel, split and label as `gbm`; different objective. The comparison
    that matters is one command:

        portfolio-agent train --trainer gbm
        portfolio-agent train --trainer rank_ic
    """

    name = "rank_ic"
    strategy_name = None

    #: Trees are pickles, not tensors — same reason `gbm` writes joblib.
    checkpoint_suffix = ".joblib"

    def __init__(self) -> None:
        self._panel: Optional[GBMPanel] = None

    @classmethod
    def config_model(cls) -> Type[TrainerConfig]:
        return RankICTrainerConfig

    @staticmethod
    def write_checkpoint(artifact: TrainingArtifact, path) -> Any:
        from ..artifacts import save_sklearn_artifact

        return save_sklearn_artifact(artifact, path)

    def prepare(
        self, app_config: Any, universe: List[str], cfg: TrainerConfig
    ) -> TrainingData:
        assert isinstance(cfg, RankICTrainerConfig)
        _require_sklearn()
        self._panel, training_data = build_gbm_panel(app_config, universe, cfg)
        return training_data

    def fit(self, data: TrainingData, cfg: TrainerConfig) -> TrainingArtifact:
        assert isinstance(cfg, RankICTrainerConfig)
        panel = self._panel
        if panel is None:
            raise RuntimeError("prepare() must run before fit()")

        ensemble, history = self._boost(panel, cfg)
        metrics = self._score(ensemble, panel, history)

        from ..artifacts import build_metadata

        metadata = build_metadata(
            feature_names=panel.feature_names,
            scaler=panel.scaler,
            trainer=self.name,
            extra={
                "model_architecture": "gradient_boosting_rank_ic",
                "library": "scikit-learn",
                "library_version": _sklearn_version(),
                "objective": "mean_per_date_correlation",
                "objective_reference": "LambdaRankIC (arXiv 2605.00501), surrogate form",
                "target": cfg.target,
                "target_transform": cfg.target_transform,
                "feature_normalization": cfg.feature_normalization,
                "horizon_days": panel.horizon_days,
                "split_date": str(pd.Timestamp(panel.split_date).date()),
                "purged_rows": panel.purged_rows,
                "training_config": cfg.model_dump(),
                "n_tickers": panel.n_tickers,
            },
        )
        return TrainingArtifact(
            state_dict={"estimator": ensemble}, metadata=metadata, metrics=metrics
        )

    # -- the loop ----------------------------------------------------------

    def _boost(
        self, panel: GBMPanel, cfg: RankICTrainerConfig
    ) -> Tuple["AdditiveEnsemble", Dict[str, List[float]]]:
        """Fit trees to the ascent direction of the IC surrogate.

        Early stopping is on **validation rank IC**, maximized — not on a loss.
        That is the whole point: the baseline chooses its iteration count by
        validation MSE and is then reported on IC, so the selection step never
        sees the metric that decides whether the model was any good.
        """
        from sklearn.tree import DecisionTreeRegressor

        ensemble = AdditiveEnsemble(learning_rate=cfg.learning_rate)
        train_scores = np.zeros(len(panel.y_train), dtype=float)

        rng = np.random.default_rng(cfg.seed)
        unique_dates = np.unique(panel.train_dates)

        history: Dict[str, List[float]] = {"train_objective": [], "val_rank_ic": []}
        best_ic, best_size, since_best = -np.inf, 0, 0

        for iteration in range(cfg.epochs):
            gradient = ic_gradient(train_scores, panel.y_train, panel.train_dates)
            if not np.any(gradient):
                logger.info("Gradient vanished at iteration %d; stopping.", iteration)
                break

            rows = self._sample_rows(panel, unique_dates, cfg.subsample, rng)
            tree = DecisionTreeRegressor(
                max_depth=cfg.max_depth,
                min_samples_leaf=cfg.min_samples_leaf,
                random_state=cfg.seed + iteration,
            )
            # Ascending the correlation, so the tree fits +gradient. The sign
            # convention is stated once, here.
            tree.fit(panel.x_train[rows], gradient[rows])
            ensemble.append(tree)
            train_scores += cfg.learning_rate * tree.predict(panel.x_train)

            history["train_objective"].append(
                mean_date_correlation(train_scores, panel.y_train, panel.train_dates)
            )

            if not cfg.early_stopping:
                continue

            val_ic = self._validation_ic(ensemble, panel)
            history["val_rank_ic"].append(val_ic)
            if val_ic > best_ic:
                best_ic, best_size, since_best = val_ic, len(ensemble), 0
            else:
                since_best += 1
                if since_best >= cfg.n_iter_no_change:
                    logger.info(
                        "Early stop at %d trees; best validation rank IC %.4f at %d.",
                        len(ensemble), best_ic, best_size,
                    )
                    break

        if cfg.early_stopping and best_size:
            # Truncation is exact here, unlike a warm-started histogram
            # ensemble: the prediction is a plain sum of shrunk tree outputs, so
            # dropping the tail is the same model that existed at `best_size`.
            ensemble.truncate(best_size)
        return ensemble, history

    @staticmethod
    def _sample_rows(
        panel: GBMPanel,
        unique_dates: np.ndarray,
        subsample: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Row indices for one tree, sampled by date rather than by row.

        A row subsample would hand each retained name the wrong cross-sectional
        centring — the gradient is defined against the date's own mean — so it
        would corrupt the objective instead of regularizing it.
        """
        if subsample >= 1.0 or unique_dates.size == 0:
            return np.arange(len(panel.y_train))
        keep = max(1, int(round(subsample * unique_dates.size)))
        chosen = set(rng.choice(unique_dates, size=keep, replace=False).tolist())
        return np.flatnonzero(pd.Series(panel.train_dates).isin(chosen).to_numpy())

    @staticmethod
    def _validation_ic(ensemble: "AdditiveEnsemble", panel: GBMPanel) -> float:
        ic = rank_ic_by_date(
            ensemble.predict(panel.x_val), panel.y_val, panel.val_dates
        )
        return float(ic.mean()) if not ic.empty else 0.0

    # -- scoring -----------------------------------------------------------

    def _score(
        self, ensemble: "AdditiveEnsemble", panel: GBMPanel, history: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """The same keys `gbm` reports, so the two are directly comparable."""
        train_pred = ensemble.predict(panel.x_train)
        val_pred = ensemble.predict(panel.x_val)

        metrics: Dict[str, Any] = {
            "train_loss": float(np.mean((train_pred - panel.y_train) ** 2)),
            "val_loss": float(np.mean((val_pred - panel.y_val) ** 2)),
            "n_iterations": len(ensemble),
            "n_train_rows": int(len(panel.y_train)),
            "n_val_rows": int(len(panel.y_val)),
            "n_purged_rows": int(panel.purged_rows),
            # The surrogate the fit actually descended, on both blocks. Kept
            # next to rank IC because they are not the same number and a run
            # where they disagree is worth noticing.
            "train_objective": mean_date_correlation(
                train_pred, panel.y_train, panel.train_dates
            ),
            "val_objective": mean_date_correlation(
                val_pred, panel.y_val, panel.val_dates
            ),
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
        metrics["val_icir"] = mean_ic / std_ic if std_ic > 1e-12 else 0.0
        metrics["val_ic_hit_rate"] = float((ic > 0).mean())
        metrics["val_ic_dates"] = int(len(ic))
        if history["val_rank_ic"]:
            metrics["best_val_rank_ic"] = float(max(history["val_rank_ic"]))
        return metrics


@dataclass
class AdditiveEnsemble:
    """Shrunk sum of regression trees, with an exact truncation.

    Not a scikit-learn estimator and not pretending to be one — it needs
    `predict` and it needs to survive joblib, and inheriting from `BaseEstimator`
    to gain a `get_params` nothing calls would be ceremony.
    """

    learning_rate: float
    trees: List[Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.trees is None:
            self.trees = []

    def __len__(self) -> int:
        return len(self.trees)

    def append(self, tree: Any) -> None:
        self.trees.append(tree)

    def truncate(self, size: int) -> None:
        self.trees = self.trees[:size]

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores = np.zeros(len(x), dtype=float)
        for tree in self.trees:
            scores += self.learning_rate * tree.predict(x)
        return scores
