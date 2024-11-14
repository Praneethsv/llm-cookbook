from attention import Attention
from tokenizer import Tokenizer


tokenizer = Tokenizer('Lord Rama is Maryada Purushotham', 'bert-base-uncased', '')
# print(tokenizer.embeddings(embedding_size=100, window=5, count=1))
attention = Attention(tokenizer=tokenizer, context_length=100, embedding_size=100, window=5, min_count=1)
A = attention.forward()

