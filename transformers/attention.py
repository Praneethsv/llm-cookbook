from tokenizer import Tokenizer
from position_encoder import PositionEncoder
from torch.nn.functional import softmax
from torch import nn
import torch


class Attention(nn.Module):
    """
    Attention layer with 
    """
    def __init__(self, positional_embeddings: torch.tensor, context_length: int) -> None:
        super(Attention, self).__init__()
        self.positional_embeddings = positional_embeddings
        
        self.d = context_length
        self.W_q = nn.Parameter(torch.empty(self.d, self.d))
        self.W_k = nn.Parameter(torch.empty(self.d, self.d))
        # nn.init.xavier_uniform_(self.W_q)
        # nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_normal_(self.W_q)
        nn.init.xavier_normal_(self.W_k)
        

    def forward(self):
        # Step1: Tokenize and get embeddings
        # Step2: Create and intialize Wq, Wk and compute Q and K, and V
        # Step3: Compute Attention Matrix A using Softmax((Q * k / sqrt(d)))
        Q = torch.matmul(self.positional_embeddings, self.W_q)
        K = torch.matmul(self.positional_embeddings, self.W_k)
        A = softmax((Q * K) / self.d ** 0.5)
        return A