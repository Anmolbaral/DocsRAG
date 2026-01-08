from sentence_transformers import CrossEncoder
from ..config.config import Config


class RerankerService:
    def __init__(self, config: Config):
        self.config = config
        self.reranker = CrossEncoder(config.reranker.model)

    def rerank_candidates(self, query: str, candidates: list[dict]):
        if not candidates:
            return []
        pairs = [(query, candidate["text"]) for candidate in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        topK = self.config.reranker.topK
        reranked = []
        for score, c in ranked[:topK]:
            c = dict(c)
            c["rerank_score"] = float(score)
            reranked.append(c)
        return reranked
