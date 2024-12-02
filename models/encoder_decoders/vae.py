from typing import List

import torch.nn as nn
from torch.distributions.uniform import Uniform

from mlp.feedforward import FeedForward


class VAE(FeedForward):
    def __init__(self, in_dim=512, hidden_dims: List = ..., out_dim=128, latent_dim=64) -> None:
        super().__init__(in_dim, hidden_dims, out_dim)
        self.mean = nn.Parameter(out_dim, latent_dim)
        self.std = nn.Parameter(out_dim, latent_dim)
        self.epsilon = Uniform(0, 1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.mean + self.epsilon * self.std
        return x
