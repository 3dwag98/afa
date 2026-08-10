"""Tests for the quantile head and the PatchTST architecture.

The property that matters most here is the one MSE fails: a model must not be
able to score well by predicting a constant.
"""

import numpy as np
import pytest
import torch

from portfolio_agent.config.schema import AppConfig
from portfolio_agent.agents.trainer import build_loss, head_width, median_output_index
from portfolio_agent.models.pytorch_models import (
    DEFAULT_QUANTILES,
    PatchTSTForecaster,
    PointLoss,
    QuantileLoss,
    sorted_quantiles,
)
from portfolio_agent.models.registry import get_model


class TestQuantileLoss:
    def test_the_minimizer_of_each_quantile_is_that_quantile(self):
        """The whole point of pinball loss: minimizing L_0.9 lands on the 90th
        percentile, not the mean."""
        torch.manual_seed(0)
        sample = torch.randn(20_000)
        loss = QuantileLoss((0.1, 0.5, 0.9))

        prediction = torch.zeros(1, 3, requires_grad=True)
        optimizer = torch.optim.Adam([prediction], lr=0.05)
        for _ in range(600):
            optimizer.zero_grad()
            loss(prediction.expand(sample.shape[0], 3), sample).backward()
            optimizer.step()

        fitted = prediction.detach().reshape(-1).numpy()
        expected = np.quantile(sample.numpy(), [0.1, 0.5, 0.9])
        assert np.allclose(fitted, expected, atol=0.08)

    def test_a_constant_prediction_cannot_satisfy_three_quantiles(self):
        """The mean-reversion trap, stated as a test: collapsing to one number
        costs strictly more than spreading across the distribution."""
        torch.manual_seed(1)
        targets = torch.randn(4_000)
        loss = QuantileLoss()

        constant = loss(torch.zeros(4_000, 3), targets)
        spread = loss(
            torch.tensor(np.quantile(targets.numpy(), DEFAULT_QUANTILES), dtype=torch.float32)
            .expand(4_000, 3),
            targets,
        )

        assert float(spread) < float(constant)

    def test_the_penalty_is_asymmetric_in_the_expected_direction(self):
        """Under-predicting the 90th percentile costs 0.9/unit, over 0.1/unit."""
        loss = QuantileLoss((0.9,))
        target = torch.tensor([0.0])

        under = float(loss(torch.tensor([[-1.0]]), target))
        over = float(loss(torch.tensor([[1.0]]), target))

        assert under == pytest.approx(0.9)
        assert over == pytest.approx(0.1)

    def test_a_single_sample_batch_keeps_its_batch_dimension(self):
        """The regression the PointLoss/QuantileLoss wrappers exist to prevent:
        `.squeeze()` at the call site collapsed the batch dim on a final batch
        of one."""
        assert float(PointLoss()(torch.zeros(1, 1), torch.tensor([2.0]))) == pytest.approx(4.0)
        assert float(QuantileLoss((0.5,))(torch.zeros(1, 1), torch.tensor([2.0]))) == pytest.approx(1.0)

    @pytest.mark.parametrize("bad", [(), (0.0, 0.5), (0.5, 1.0), (-0.1,)])
    def test_quantiles_outside_the_open_unit_interval_are_rejected(self, bad):
        with pytest.raises(ValueError):
            QuantileLoss(bad)


class TestSortedQuantiles:
    def test_crossed_quantiles_are_repaired(self):
        crossed = torch.tensor([[0.3, -0.1, 0.05]])

        assert sorted_quantiles(crossed).reshape(-1).tolist() == pytest.approx([-0.1, 0.05, 0.3])

    def test_a_one_dimensional_prediction_is_left_alone(self):
        point = torch.tensor([0.3, -0.1])

        assert torch.equal(sorted_quantiles(point), point)

    def test_sorting_never_increases_the_pinball_loss(self):
        torch.manual_seed(2)
        predictions = torch.randn(500, 3)
        targets = torch.randn(500)
        loss = QuantileLoss()

        assert float(loss(sorted_quantiles(predictions), targets)) <= float(
            loss(predictions, targets)
        ) + 1e-6


class TestPatchTST:
    def test_it_is_registered_and_emits_a_quantile_triple(self):
        model = get_model("patchtst")(n_features=8, sequence_length=60, n_outputs=3)

        assert model(torch.randn(4, 60, 8)).shape == (4, 3)

    def test_the_window_is_cut_into_patches(self):
        model = PatchTSTForecaster(n_features=4, sequence_length=60, patch_length=5)

        assert model.n_patches == 12
        assert model.used_length == 60

    def test_a_ragged_window_keeps_the_most_recent_whole_patches(self):
        """Dropping the oldest remainder rather than the newest — the recent
        end is the informative one."""
        model = PatchTSTForecaster(n_features=3, sequence_length=62, patch_length=5)

        assert model.n_patches == 12
        assert model.used_length == 60
        assert model(torch.randn(2, 62, 3)).shape == (2, 3)

    def test_channels_share_encoder_weights(self):
        """Channel independence: the encoder sees one feature at a time, so
        adding a feature must not widen it."""
        narrow = PatchTSTForecaster(n_features=2, sequence_length=60)
        wide = PatchTSTForecaster(n_features=16, sequence_length=60)

        def encoder_params(m):
            return sum(p.numel() for p in m.encoder.parameters())

        assert encoder_params(narrow) == encoder_params(wide)

    def test_the_output_is_invariant_to_a_constant_shift_in_the_input(self):
        """Instance normalization is what lets one set of weights serve a ₹30
        small-cap and a ₹3,000 large-cap."""
        torch.manual_seed(4)
        model = PatchTSTForecaster(n_features=3, sequence_length=60).eval()
        x = torch.randn(2, 60, 3)

        with torch.no_grad():
            base = model(x)
            shifted = model(x + 100.0)

        assert torch.allclose(base, shifted, atol=1e-4)

    def test_gradients_reach_every_parameter(self):
        model = PatchTSTForecaster(n_features=5, sequence_length=60, n_outputs=3)
        QuantileLoss()(model(torch.randn(8, 60, 5)), torch.randn(8)).backward()

        missing = [name for name, p in model.named_parameters() if p.grad is None]
        assert missing == []

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_features": 0},
            {"patch_length": 0},
            {"patch_length": 999},
            {"hidden_size": 65, "n_heads": 4},
        ],
    )
    def test_invalid_geometry_fails_at_construction(self, kwargs):
        params = {"n_features": 4, "sequence_length": 60}
        params.update(kwargs)
        with pytest.raises(ValueError):
            PatchTSTForecaster(**params)


class TestTrainerHeadConfiguration:
    def test_quantile_training_widens_the_head(self):
        config = AppConfig().training

        assert config.loss == "quantile"
        assert head_width(config) == 3
        assert median_output_index(config) == 1
        assert isinstance(build_loss(config), QuantileLoss)

    def test_mse_training_keeps_a_single_output(self):
        config = AppConfig().training
        config.loss = "mse"

        assert head_width(config) == 1
        assert median_output_index(config) is None
        assert isinstance(build_loss(config), PointLoss)

    def test_an_asymmetric_quantile_set_finds_the_real_median(self):
        """Assuming the middle position would score the model on the wrong
        column."""
        config = AppConfig().training
        config.quantiles = [0.05, 0.25, 0.5, 0.95]

        assert median_output_index(config) == 2
