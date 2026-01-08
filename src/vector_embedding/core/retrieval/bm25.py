from rank_bm25 import BM25Okapi
import re
from ..config.config import Config
import heapq

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
    def __init__(self, texts, config: Config):
        self.texts = texts
        self.config = config  # Store config
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
    def search(self, query: str):
        maxHeap = []
        heapq.heapify(maxHeap)
        tokenizedQuery = self._tokenize([query])[0]
        scores = self.bm25.get_scores(tokenizedQuery)
        for i, score in enumerate(scores):
            if len(maxHeap) < self.config.retrieval.bm25TopK:
                heapq.heappush(maxHeap, (-score, i))
            else:
                heapq.heappushpop(maxHeap, (-score, i))
        return [(i, float(-score), self.texts[i]) for score, i in maxHeap]
