from attention import Attention
from tokenizer import Tokenizer
from position_encoder import PositionEncoder
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = Tokenizer('Lord Rama is Maryada Purushotham', 'bert-base-uncased', '')
embeddings = tokenizer.get_embeddings(embedding_size=100, 
                      window=5, count=1)
positional_encoder = PositionEncoder(embeddings).to(device)
positional_embeddings = positional_encoder()


attention = Attention(positional_embeddings, context_length=100).to(device)
A = attention.forward()
print(A)
