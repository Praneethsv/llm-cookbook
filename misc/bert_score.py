from transformers import BertTokenizer


class BERTScoreCalculator:
    def __init__(self, reference_txt: str, candidate_txt: str):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.reference_tokens, self.candidate_tokens = self.tokenizer.tokenize(
            reference_txt
        ), self.tokenizer.tokenize(candidate_txt)
        self.reference = reference_txt
        self.candidate = candidate_txt

    def calculate(self):
        ref_embeds, cand_embeds = self.calc_embeddings(
            self.reference_tokens, self.candidate_tokens
        )
        self.cosine_similarity(ref_embeds, cand_embeds)

        pass

    def calc_embeddings(self, ref_tokens, cand_tokens):
        pass

    def cosine_similarity(self, ref_embeddings, cand_embeddings):
        pass

    def precision(self):
        pass

    def recall(self):
        pass

    def f1_score(self):
        pass
