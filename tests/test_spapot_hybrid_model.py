from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spapot.config import ModelConfig, TrainConfig  # noqa: E402
from spapot.data import FeatureScaler, PreparedData, sample_slice  # noqa: E402
from spapot.fields import SpaPOTPotentialModel, make_mlp  # noqa: E402


def _linear_layers(module: nn.Module) -> list[nn.Linear]:
    return [layer for layer in module.modules() if isinstance(layer, nn.Linear)]


class SpaPOTHybridModelTest(unittest.TestCase):
    def test_reference_branch_shapes_are_fixed(self) -> None:
        model = SpaPOTPotentialModel(ModelConfig(spatial_dim=2, latent_dim=10, hidden_dim=128, n_hidden=6))

        spatial_layers = _linear_layers(model.spatial_net)
        self.assertEqual(len(spatial_layers), 7)
        self.assertEqual(spatial_layers[0].weight.shape, torch.Size([128, 13]))
        self.assertEqual(spatial_layers[-1].weight.shape, torch.Size([2, 128]))

        potential_layers = _linear_layers(model.potential_net)
        self.assertEqual(len(potential_layers), 7)
        self.assertEqual(potential_layers[0].weight.shape, torch.Size([128, 13]))
        self.assertEqual(potential_layers[-1].weight.shape, torch.Size([1, 128]))

        growth_layers = _linear_layers(model.growth_net)
        self.assertEqual(len(growth_layers), 4)
        self.assertEqual(growth_layers[0].weight.shape, torch.Size([128, 13]))
        self.assertEqual(growth_layers[-1].weight.shape, torch.Size([1, 128]))

    def test_growth_branch_does_not_follow_n_hidden(self) -> None:
        model = SpaPOTPotentialModel(ModelConfig(spatial_dim=2, latent_dim=10, hidden_dim=64, n_hidden=2))
        self.assertEqual(len(_linear_layers(model.spatial_net)), 3)
        self.assertEqual(len(_linear_layers(model.potential_net)), 3)
        self.assertEqual(len(_linear_layers(model.growth_net)), 4)

    def test_seeded_branch_initialization_order_is_stable(self) -> None:
        torch.manual_seed(19491001)
        model = SpaPOTPotentialModel(ModelConfig(spatial_dim=2, latent_dim=10, hidden_dim=128, n_hidden=6))

        spatial_sum = float(_linear_layers(model.spatial_net)[0].weight.detach().sum())
        potential_sum = float(_linear_layers(model.potential_net)[0].weight.detach().sum())
        growth_sum = float(_linear_layers(model.growth_net)[0].weight.detach().sum())

        self.assertAlmostEqual(spatial_sum, -1.0490612983703613, places=6)
        self.assertAlmostEqual(potential_sum, -4.505341529846191, places=6)
        self.assertAlmostEqual(growth_sum, 8.79880142211914, places=6)

    def test_defaults_are_spapot_fullgrid(self) -> None:
        config = TrainConfig()
        self.assertEqual(config.loss_mode, "spapot_fullgrid")
        self.assertEqual(config.optimizer, "adam")
        self.assertEqual(config.sample_size, 1024)
        self.assertEqual(config.lambda_match, 400000.0)
        self.assertEqual(config.lambda_action, 0.0)
        self.assertEqual(config.grad_clip, 0.0)
        self.assertTrue(config.increase_sample_size)
        self.assertEqual(config.sample_growth_interval, 100)
        self.assertEqual(config.sample_growth_step, 20)

    def test_leakyrelu_matches_torch_default(self) -> None:
        mlp = make_mlp(3, 1, 8, 1, "leakyrelu")
        activations = [layer for layer in mlp.modules() if isinstance(layer, nn.LeakyReLU)]
        self.assertEqual(len(activations), 1)
        self.assertEqual(activations[0].negative_slope, nn.LeakyReLU().negative_slope)

    def test_sample_slice_uses_python_random_sequence(self) -> None:
        state = torch.arange(50, dtype=torch.float32).reshape(10, 5)
        expression = torch.arange(30, dtype=torch.float32).reshape(10, 3)
        labels = np.asarray([f"c{i}" for i in range(10)])
        raw_indices = np.arange(10)
        data = PreparedData(
            adata=None,  # type: ignore[arg-type]
            annotation_key="Annotation",
            time_values=[0.0],
            raw_time_values=[0.0],
            state_by_time=[state],
            expression_by_time=[expression],
            labels_by_time=[labels],
            raw_indices_by_time=[raw_indices],
            scaler=FeatureScaler(
                spatial_mean=np.zeros(2, dtype=np.float32),
                spatial_std=np.ones(2, dtype=np.float32),
                latent_mean=np.zeros(3, dtype=np.float32),
                latent_std=np.ones(3, dtype=np.float32),
                spatial_weight=1.0,
                scale_features=False,
            ),
            spatial_dim=2,
            latent_dim=3,
            expression_dim=3,
        )

        random.seed(19491001)
        sampled = sample_slice(data, 0, 4)
        random.seed(19491001)
        expected = random.sample(range(0, 10), 4)

        np.testing.assert_array_equal(sampled.raw_indices, np.asarray(expected))
        self.assertTrue(torch.equal(sampled.state, state[expected]))


if __name__ == "__main__":
    unittest.main()
