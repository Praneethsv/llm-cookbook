from typing import List

import torch
import torch.nn as nn

from models.cnns.cnn import CNN
from models.mlp.classification_block import ClassificationBlock


class CNNClassifier(nn.Module):
    def __init__(
        self,
        name,
        in_channels,
        conv_channel_dims: List = ...,
        conv_kernel_dims: List = ...,
        out_dim=10,
        input_size=[32, 32],
    ) -> None:
        super(CNNClassifier, self).__init__()
        self.name = name
        self.cnn = CNN(in_channels, conv_channel_dims, conv_kernel_dims)
        self.flattened_dim = self._compute_flattened_dim(input_size, in_channels)
        self.out_layer = ClassificationBlock(in_dim=self.flattened_dim, out_dim=out_dim)

    def _compute_flattened_dim(self, input_size, in_channels):
        dummy_ip = torch.randn(1, in_channels, *input_size)
        with torch.no_grad():
            output = self.cnn(dummy_ip)
        return output.numel()

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.out_layer(x)
        return x
