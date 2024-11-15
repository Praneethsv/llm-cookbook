import torch
import torch.nn as nn


class Feedforward(nn.Module):
    def __init__(self, in_dim=512, out_dim=512)-> None:
        super(Feedforward, self).__init__()
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear(x) # Wx + b
        x = self.activation(x) # max(0, x)
        return x
