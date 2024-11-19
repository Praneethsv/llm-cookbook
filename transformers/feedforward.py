import torch
import torch.nn as nn
from typing import List



class FeedForwardBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU()) 

    def forward(self, x):
        return self.layer(x)



class FeedForward(nn.Module):
    def __init__(self, in_dim=512, hidden_dims: List = [512, 256], out_dim=128)-> None:
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
