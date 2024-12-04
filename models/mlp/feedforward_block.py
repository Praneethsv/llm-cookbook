import torch.nn as nn


class FeedForwardBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())

    def forward(self, x):
        return self.layer(x)
