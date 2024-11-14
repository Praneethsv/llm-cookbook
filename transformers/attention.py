from tokenizer import Tokenizer
from torch.nn.functional import softmax
from torch.nn import Module
from torch import nn
import torch


class Attention(Module):
    """
    Attention layer with 
    """
    def __init__(self, tokenizer: Tokenizer, context_length: int, embedding_size: int, window: int, min_count: int) -> None:
        super(Attention, self).__init__()
        self.tokenizer = tokenizer
        self.d = context_length
        self.W_q = nn.Parameter(torch.empty(self.d, self.d))
        self.W_k = nn.Parameter(torch.empty(self.d, self.d))
        self.embedding_size = embedding_size
        self.window = window
        self.min_count = min_count
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_k)
        

    def forward(self):
        # Step1: Tokenize and get embeddings
        # Step2: Create and intialize Wq, Wk and compute Q and K, and V
        # Step3: Compute Attention Matrix A using Softmax((Q * k / sqrt(d)))
        embeddings = self.tokenizer.embeddings(self.embedding_size, self.window, self.min_count)
        print(embeddings.shape, self.W_q.shape)
        Q = torch.matmul(embeddings, self.W_q) 
        K = torch.matmul(embeddings, self.W_k)
        A = softmax((Q * K) / self.d ** 0.5)
        return A