import torch.nn as nn


class ClassificationBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation="softmax"):
        super(ClassificationBlock, self).__init__()
        self.layer = (
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.Softmax(dim=1))
            if activation == "softmax"
            else nn.Sequential(nn.Linear(in_dim, out_dim), nn.Sigmoid())
        )

    def forward(self, x):
        return self.layer(x)
