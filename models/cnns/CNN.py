import torch
import torch.nn as nn
from typing import List


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pooling='max', batch_norm=True) -> None:
        super(CNNBlock, self).__init__()
        self.pooling_layer = nn.MaxPool2d(2) if pooling == 'max' else nn.AvgPool2d(2)
        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding), 
            nn.ReLU(), 
            self.pooling_layer
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        self.cnn_block = nn.Sequential(*layers)


    def forward(self, x):
        return self.cnn_block(x)
    

class CNN(nn.Module):
    def __init__(self, in_channels, conv_channel_dims: List = [512, 512], conv_kernel_dims: List = [3, 3]) -> None:
        super(CNN, self).__init__()
        self.blocks = nn.ModuleList()
        current_channel_dim = in_channels
        for conv_channel_dim, conv_kernel in zip(conv_channel_dims, conv_kernel_dims):
            self.blocks.append(CNNBlock(current_channel_dim,  conv_channel_dim, conv_kernel))
            current_channel_dim = conv_channel_dim
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

