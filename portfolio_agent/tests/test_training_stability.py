"""Tests for the training loop's numerical guards (agents/trainer.py).

The defect these pin down: every epoch of a GPU training run printed `nan`.
Three separate things had to be true for that to happen silently, and each one
is covered here — infinities reaching the panel, a non-finite loss being
stepped on, and mixed precision being enabled on a card whose fp16 path cannot
be trusted.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from portfolio_agent.agents.trainer import (
    GRAD_CLIP_NORM,
    _train_model,
    prepare_features,
)
from portfolio_agent.config.schema import AppConfig
from portfolio_agent.data.dataset import TimeSeriesDataset
from portfolio_agent.models.pytorch_models import PointLoss
from portfolio_agent.utils.device import mixed_precision_support


SEQUENCE_LENGTH = 8


def _ohlcv(n: int = 300, level: float = 100.0, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = level + np.cumsum(rng.standard_normal(n) * level * 0.01)
    close = np.maximum(close, level * 0.5)
    return pd.DataFrame({
        'open': close, 'high': close * 1.01, 'low': close * 0.99, 'close': close,
        'volume': rng.integers(1e5, 1e6, n).astype(float),
    }, index=pd.bdate_range("2021-01-04", periods=n))


def _loader(features: np.ndarray, targets: np.ndarray) -> DataLoader:
    dataset = TimeSeriesDataset(features, targets, SEQUENCE_LENGTH)
    return DataLoader(dataset, batch_size=8, shuffle=False, drop_last=True)


class _ExplodingLoss(torch.nn.Module):
    """Emits inf on the first call, then behaves. Stands in for the fp16
    overflow, which is not reproducible without the GPU that causes it."""

    def __init__(self):
        super().__init__()
        self.calls = 0
        self._inner = PointLoss()

    def forward(self, predictions, targets):
        self.calls += 1
        if self.calls == 1:
            return predictions.sum() * float('inf')
        return self._inner(predictions, targets)


def _tiny_model(n_features: int) -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(SEQUENCE_LENGTH * n_features, 1),
    )


class TestNonFiniteRowsAreDropped:
    """A zero price makes every ratio feature — and the forward-return target —
    infinite. dropna() does not remove an infinity."""

    def test_infinite_rows_never_reach_the_panel(self):
        config = AppConfig()
        config.features.normalize = False
        df = _ohlcv()
        df.iloc[120, df.columns.get_loc('close')] = 0.0

        panel = prepare_features(df, config, verbose=False)

        assert len(panel) > 0
        assert np.isfinite(panel.values).all()

    def test_a_clean_series_is_left_alone(self):
        config = AppConfig()
        config.features.normalize = False

        panel = prepare_features(_ohlcv(), config, verbose=False)

        assert np.isfinite(panel.values).all()
        assert len(panel) > 100


class TestNonFiniteLossIsSkipped:
    """Stepping the optimizer on a NaN/inf loss writes NaN into every weight,
    after which every later batch is NaN too — one bad batch killed the run."""

    def test_weights_survive_a_non_finite_batch(self, capsys):
        rng = np.random.default_rng(0)
        features = rng.standard_normal((200, 4))
        targets = rng.standard_normal(200) * 0.01
        model = _tiny_model(4)

        train_losses, _, best = _train_model(
            model=model,
            train_loader=_loader(features, targets),
            val_loader=_loader(features, targets),
            device=torch.device("cpu"),
            epochs=2,
            learning_rate=0.01,
            use_mixed_precision=False,
            loss_fn=_ExplodingLoss(),
            verbose=False,
        )

        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert np.isfinite(train_losses).all()
        assert np.isfinite(best)

    def test_skipped_batches_are_reported_not_swallowed(self, capsys):
        rng = np.random.default_rng(1)
        features = rng.standard_normal((120, 4))
        targets = rng.standard_normal(120) * 0.01

        _train_model(
            model=_tiny_model(4),
            train_loader=_loader(features, targets),
            val_loader=None,
            device=torch.device("cpu"),
            epochs=1,
            learning_rate=0.01,
            use_mixed_precision=False,
            loss_fn=_ExplodingLoss(),
            verbose=False,
        )

        assert "not finite" in capsys.readouterr().out

    def test_an_entirely_dead_epoch_is_nan_not_zero(self):
        """A zero average would read as a perfect fit and would win every
        early-stopping comparison, checkpointing a model that learned nothing."""

        class AlwaysInf(torch.nn.Module):
            def forward(self, predictions, targets):
                return predictions.sum() * float('inf')

        rng = np.random.default_rng(2)
        features = rng.standard_normal((120, 4))
        targets = rng.standard_normal(120) * 0.01

        train_losses, val_losses, best = _train_model(
            model=_tiny_model(4),
            train_loader=_loader(features, targets),
            val_loader=_loader(features, targets),
            device=torch.device("cpu"),
            epochs=1,
            learning_rate=0.01,
            use_mixed_precision=False,
            loss_fn=AlwaysInf(),
            verbose=False,
        )

        assert np.isnan(train_losses[0])
        assert np.isnan(val_losses[0])
        # Never checkpointed: NaN loses the `<` comparison against inf.
        assert best == float('inf')


class TestGradientClipping:
    def test_gradients_are_clipped_to_the_configured_norm(self, monkeypatch):
        """An unclipped step on a huge gradient moves the weights somewhere
        every later loss evaluates to NaN from."""
        observed = []
        real_clip = torch.nn.utils.clip_grad_norm_

        def spy(parameters, max_norm, *args, **kwargs):
            observed.append(max_norm)
            return real_clip(parameters, max_norm, *args, **kwargs)

        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)

        rng = np.random.default_rng(4)
        _train_model(
            model=_tiny_model(4),
            train_loader=_loader(rng.standard_normal((120, 4)), rng.standard_normal(120)),
            val_loader=None,
            device=torch.device("cpu"),
            epochs=1,
            learning_rate=0.01,
            use_mixed_precision=False,
            verbose=False,
        )

        assert observed and set(observed) == {GRAD_CLIP_NORM}


class TestMixedPrecisionSupport:
    """AMP is refused where fp16 is unsafe, with the reason attached — the
    silent version of this decision is what produced a run of `nan` epochs."""

    @staticmethod
    def _fake_cuda(monkeypatch, name: str, major: int = 7, minor: int = 5):
        class Props:
            pass

        props = Props()
        props.name = name
        props.major = major
        props.minor = minor
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: props)

    def test_cpu_is_never_mixed_precision(self):
        supported, reason = mixed_precision_support(torch.device("cpu"))

        assert supported is False
        assert "CUDA" in reason

    def test_gtx_1660_ti_is_refused(self, monkeypatch):
        """TU116 reports compute capability 7.5 like an RTX 2060 but ships no
        tensor cores, and its fp16 path returns NaN."""
        self._fake_cuda(monkeypatch, "NVIDIA GeForce GTX 1660 Ti")

        supported, reason = mixed_precision_support(torch.device("cuda"))

        assert supported is False
        assert "tensor cores" in reason

    def test_pre_volta_is_refused(self, monkeypatch):
        self._fake_cuda(monkeypatch, "NVIDIA GeForce GTX 1080 Ti", major=6, minor=1)

        supported, reason = mixed_precision_support(torch.device("cuda"))

        assert supported is False
        assert "7.0" in reason

    def test_a_tensor_core_card_is_allowed(self, monkeypatch):
        self._fake_cuda(monkeypatch, "NVIDIA GeForce RTX 4070", major=8, minor=9)

        supported, reason = mixed_precision_support(torch.device("cuda"))

        assert supported is True
        assert "tensor cores" in reason
