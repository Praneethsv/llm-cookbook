from typing import List

import torch.nn as nn

from models.cnns.cnn_block import CNNBlock


class CNN(nn.Module):
    def __init__(
        self,
        in_channels,
        conv_channel_dims: List = [512, 512],
        conv_kernel_dims: List = [3, 3],
    ) -> None:
        super(CNN, self).__init__()
        self.blocks = nn.ModuleList()
        current_channel_dim = in_channels
        for conv_channel_dim, conv_kernel in zip(conv_channel_dims, conv_kernel_dims):
            self.blocks.append(
                CNNBlock(current_channel_dim, conv_channel_dim, conv_kernel)
            )
            current_channel_dim = conv_channel_dim

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
