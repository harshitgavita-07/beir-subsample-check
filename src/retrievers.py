"""Two sparse retrievers that are genuinely close on BEIR full-test nDCG@10:
BM25 with default Okapi parameters vs BM25 with a lower k1 (less term-frequency
saturation). Both standard, neither tuned to any benchmark."""
import re
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25Retriever:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.doc_ids = list(corpus.keys())
        self.model = BM25Okapi([tokenize(corpus[d]) for d in self.doc_ids], k1=k1, b=b)

    def rank(self, query, top_k=10):
        scores = self.model.get_scores(tokenize(query))
        best = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [self.doc_ids[i] for i in best]


def build_systems(corpus):
    """Returns dict name -> retriever. Deliberately close configurations."""
    return {
        "bm25_k1.5": BM25Retriever(corpus, k1=1.5, b=0.75),
        "bm25_k1.2": BM25Retriever(corpus, k1=1.2, b=0.75),
    }
