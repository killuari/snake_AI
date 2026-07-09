"""
feature_extractors.py - Custom SB3 Feature Extractor for the CNN Observation Mode

SnakeCombinedExtractor processes the "grid" obs_mode's Dict observation
({"grid": Box(4,H,W), "apple_dir": MultiDiscrete([3,3])}) for use with SB3's
MultiInputPolicy: a small CNN branch for the spatial FOV grid (convolution
filters are shared across all positions, unlike an MLP over a flattened FOV
vector) and a small linear branch for the apple direction, concatenated into
one feature vector.

SB3's default CnnPolicy/NatureCNN assumes large, image-like inputs (>=36x36,
usually 84x84) and produces invalid dimensions for a small FOV window (e.g.
7x7 or 11x11) -- hence this small, purpose-built extractor instead.
"""

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SnakeCombinedExtractor(BaseFeaturesExtractor):
    """Combines a small CNN over the FOV grid with a linear branch over the
    one-hot-encoded apple direction."""

    def __init__(self, observation_space, cnn_features_dim=128, dir_features_dim=8):
        super().__init__(observation_space, cnn_features_dim + dir_features_dim)

        grid_shape = observation_space["grid"].shape
        channels = grid_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flatten = self.cnn(th.zeros(1, *grid_shape)).shape[1]
        self.cnn_linear = nn.Sequential(nn.Linear(n_flatten, cnn_features_dim), nn.ReLU())

        # SB3 one-hot-encodes MultiDiscrete sub-spaces of a Dict space before
        # they reach the extractor, so the "apple_dir" input width is the sum
        # of each sub-dimension's category count.
        n_dir = int(sum(observation_space["apple_dir"].nvec))
        self.dir_linear = nn.Sequential(nn.Linear(n_dir, dir_features_dim), nn.ReLU())

    def forward(self, observations):
        grid_features = self.cnn_linear(self.cnn(observations["grid"].float()))
        dir_features = self.dir_linear(observations["apple_dir"].float())
        return th.cat([grid_features, dir_features], dim=1)
