from typing import List
from gensim.models import Word2Vec
import numpy as np
import torch

from transformers import AutoTokenizer, BertModel


class Tokenizer:
    def __init__(self, text: str, tokenizer_model: str, embeddings_model: str) -> None:
        self.tokens = []
        self.text = text
        self.tokenizer_model = tokenizer_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model)
        self.embeddings_model = embeddings_model

    def tokenize(self) -> List:
        self.tokens = self.tokenizer.tokenize(self.text)
        return self.tokens

    def get_tokens(self):
        return self.tokens

    def embeddings(self, embedding_size, window, count) -> np.array:
        if not self.tokens:
            self.tokens = self.tokenize()
        words_to_vec = Word2Vec([self.tokens], vector_size=embedding_size, window=window, min_count=count)
        embeddings = words_to_vec.wv.vectors
        embeddings = torch.tensor(embeddings)
        return embeddings
    
