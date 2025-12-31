from sentence_transformers import CrossEncoder


# Initialize a cross-encoder reranker model. Cross-encoders process query-document
# pairs together for more accurate scores but higher computational cost. Used on
# smaller candidate sets (top 50-120) rather than entire corpus.
def initialize_reranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    return CrossEncoder(model)


# Rerank search candidates using a cross-encoder model. Takes initial results from
# FAISS/BM25 and reranks them for better precision. Returns top_k results with
# added "rerank_score" key, sorted by score (highest first).
def rerank_candidates(query: str, candidates: list[dict], reranker, top_k=10):
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
