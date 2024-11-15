import torch 
import torch.nn as nn



class PositionEncoder(nn.Module):
    """
    This module should supports multiple methods of positional encoding
    """
    def __init__(self) -> None:
        super(PositionEncoder, self).__init__()
        self.cosine = torch.cos()

    def forward(self, embeddings):
        return self.cosine(embeddings)