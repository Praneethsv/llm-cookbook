import torch
import torch.nn.functional as F


class RNNCell:
    def __init__(self, x_t, h_prev, w_h, w_a) -> None:
        self.x = x
        self.h_prev = h_prev

    def step(self):
        """Executes the RNN Cell"""
        F.tanh()