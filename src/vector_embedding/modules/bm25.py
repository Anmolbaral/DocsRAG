from rank_bm25 import BM25Okapi
import re
import json
from config import Config

STOPWORDS = {
    "the",
    "is",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "by",
    "as",
    "at",
    "from",
    "that",
    "this",
    "it",
    "be",
    "are",
    "was",
    "were",
    "will",
    "can",
    "into",
    "over",
}


# BM25 index for keyword-based text retrieval. Complements semantic search
# (embeddings) by providing exact keyword matching capabilities.
class BM25Index:
    def __init__(self, texts):
        self.texts = texts
        self.tokenizedTexts = self._tokenize(texts)
        self.bm25 = BM25Okapi(self.tokenizedTexts)

    # Tokenize text: lowercase, extract words
    def _tokenize(self, texts):
        tokenized = []
        for text in texts:
            text = text.lower()
            tokens = [
                token
                for token in re.findall(r"[a-z0-9]+", text)
                if token not in STOPWORDS and len(token) >= 3
            ]
            tokenized.append(tokens)
        return tokenized

    # Search for top k results using BM25
    def search(self, query: str, config:Config):
        q = [
            x
            for x in re.findall(r"[a-z0-9]+", query.lower())
            if x not in STOPWORDS and len(x) >= 3
        ]
        scores = self.bm25.get_scores(q)
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:config.retrieval.bm25TopK]
        return [(i, float(scores[i]), self.texts[i]) for i in top]
