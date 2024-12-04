from typing import List

import torch.nn as nn

from models.mlp.feedforward_block import FeedForwardBlock


class FeedForward(nn.Module):
    def __init__(self, in_dim=512, hidden_dims: List = [512, 256], out_dim=128) -> None:
        super(FeedForward, self).__init__()
        self.blocks = nn.ModuleList()
        current_dim = in_dim

        for hidden_dim in hidden_dims:
            self.blocks.append(FeedForwardBlock(current_dim, hidden_dim))
            current_dim = hidden_dim

        self.output_layer = nn.Linear(current_dim, out_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.output_layer(x)
        return x
