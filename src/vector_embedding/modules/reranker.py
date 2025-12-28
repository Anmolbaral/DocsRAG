from sentence_transformers import CrossEncoder


def initialize_reranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Initialize a reranker model"""
    return CrossEncoder(model)


def rerank_candidates(query: str, candidates: list[dict], reranker, top_k=10):
    """Rerank candidates using the reranker model"""
    if not candidates:
        return []
    pairs = [(query, candidate["text"]) for candidate in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

    reranked = []
    for score, c in ranked[:top_k]:
        c = dict(c)
        c["rerank_score"] = float(score)
        reranked.append(c)

    return reranked
