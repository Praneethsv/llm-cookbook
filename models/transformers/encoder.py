import torch
import torch.nn as nn

from models.mlp.feedforward import FeedForward
from models.transformers.attention import Attention


class Encoder(nn.Module):
    def __init__(self, num_attention, num_feedforward) -> None:
        super(Encoder, self).__init__()
        self.num_attention = num_attention
        self.num_feedforward = num_feedforward
        self.attention_blocks = [Attention() for _ in range(self.num_attention)]
        self.attention_blocks = [FeedForward() for _ in range(self.num_feedforward)]

    def forward(self, x):

        pass
