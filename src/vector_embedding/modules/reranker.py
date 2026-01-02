from sentence_transformers import CrossEncoder
from config import Config

# Initialize a cross-encoder reranker model. 
# Used on smaller candidate sets (top 50-120) rather than entire corpus.
def initialize_reranker(config:Config):
    return CrossEncoder(config.reranker.model)


# Rerank search candidates using a cross-encoder model. Returns topK results with
# added "rerank_score" key, sorted by score (highest first).
def rerank_candidates(query: str, candidates: list[dict], reranker, config:Config):
    if not candidates:
        return []
    pairs = [(query, candidate["text"]) for candidate in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    topK = config.reranker.topK
    reranked = []
    for score, c in ranked[:topK]:
        c = dict(c)
        c["rerank_score"] = float(score)
        reranked.append(c)

    return reranked
