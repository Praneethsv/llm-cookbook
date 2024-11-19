import torch
import torch.nn as nn
from typing import List


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pooling='max', batch_norm=True) -> None:
        super(CNNBlock, self).__init__()
        self.pooling_layer = nn.MaxPool2d(2) if pooling is 'max' else nn.AvgPool2d(2)
        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding), 
            nn.ReLU(), 
            self.pooling_layer,
            nn.BatchNorm2d()

        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        self.cnn = nn.Sequential(*layers)


    def forward(self, x):
        return self.cnn(x)
    

class CNN(nn.Module):
    def __init__(self, in_channels, hidden_channel_dims: List = [512, 512], hidden_kernel_dims: List = [3, 3]) -> None:
        super(CNN, self).__init__()
        self.blocks = nn.ModuleList()
        current_channel_dim = in_channels
        for hidden_channel_dim, hidden_kernel in zip(hidden_channel_dims, hidden_kernel_dims):
            self.blocks.append(CNNBlock(current_channel_dim,  hidden_channel_dim, hidden_kernel))
            current_channel_dim = hidden_channel_dim
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

