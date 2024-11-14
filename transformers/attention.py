from tokenizer import Tokenizer
from torch.nn.functional import softmax
from torch.nn import Module
from torch import nn


class Attention(Module):
    """
    Attention layer with 
    """
    def __init__(self, tokenizer: Tokenizer, context_length: int) -> None:
        super.__init__()
        self.tokenizer = tokenizer()
        self.W_q = nn.init.xavier_uniform_()
        self.W_k = nn.init.xavier_uniform_()
        self.d = context_length

    def forward(self):
        # Step1: Tokenize and get embeddings
        # Step2: Create and intialize Wq, Wk and compute Q and K, and V
        # Step3: Compute Attention Matrix A using Softmax((Q * k / sqrt(d)))
        embeddings = self.tokenizer.embeddings()
        Q = self.W_q * embeddings
        K = self.W_k * embeddings
        A = softmax((Q * K) / self.d ** 0.5)
        return A