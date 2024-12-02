import torch 
import torch.nn as nn



class PositionEncoder(nn.Module):
    """
    This module should support multiple methods of positional encoding
    """
    def __init__(self, embeddings) -> None:
        super(PositionEncoder, self).__init__()
        
        # self.embeddings = torch.tensor(embeddings)
        self.register_buffer("embeddings", torch.tensor(embeddings))
        self.seq_len, self.d_model = self.embeddings.shape
        
        self.i = torch.arange(self.d_model)[None, :]
        self.pos = torch.arange(self.seq_len)[:, None]
        self.register_buffer("pos_enc", torch.zeros((self.seq_len, self.d_model))) 

    def forward(self):
        # for odd indices, cos(pos/ 1000 ** (2i/d_model))
        angle_rates = 1 / (1000 ** (2*self.i/self.d_model))
        angle_rads = self.pos * angle_rates
        self.pos_enc[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        self.pos_enc[:, 1::2] = torch.cos(angle_rads[:, 1::2])
        return self.pos_enc + self.embeddings