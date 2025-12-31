from .embeddings import get_embedding_single, get_embedding_batch
from .vectordb import VectorDB
from .reranker import initialize_reranker, rerank_candidates as rerank_candidates_func
import openai
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .bm25 import BM25Index

openai.api_key = os.getenv("OPENAI_API_KEY")


class RAGPipeline:
    def __init__(
        self,
        chunks,
        embedder=None,
        chat_client=None,
        cache_file="cache/embeddings.json",
    ):
        self.db = VectorDB(dim=3072)
        self.texts = [chunk["text"] for chunk in chunks]
        self.chunks = chunks
        self.bm25Index = BM25Index(self.texts)
        self._embedder_single = (
            getattr(embedder, "get_embedding", None)
            if embedder
            else get_embedding_single
        )
        self._embedder_batch = (
            getattr(embedder, "get_embeddings", None)
            if embedder
            else get_embedding_batch
        )
        self._chat_client = chat_client
        self._cache_file = cache_file
        # If no texts are provided, return early
        if not self.texts or not self.chunks:
            self.embeddings = []
            self.chunks = []
        else:
            self.embeddings = self._embedder_batch(self.texts)
            for i, t in enumerate(self.chunks):
                self.db.add(
                    self.embeddings[i],
                    self.chunks[i]["text"],
                    self.chunks[i]["metadata"],
                )
        self.add_to_cache()

        # Adding conversation memory
        self.conversationHistory = []
        self.maxHistory = 10

        # Cache category embeddings once during initialization
        self.categoryEmbeddings = self.create_category_embeddings()

        # Initialize reranker
        self.reranker = initialize_reranker()

    @classmethod
    def from_cache(
        cls, cacheFile="cache/embeddings.json", embedder=None, chat_client=None
    ):
        with open(cacheFile, "r") as f:
            metadata = json.load(f)

        obj = cls.__new__(cls)
        obj.db = VectorDB(dim=3072)
        obj.chunks = metadata["chunks"]
        obj.embeddings = metadata["embeddings"]
        obj.texts = [chunk["text"] for chunk in obj.chunks]
        obj.bm25Index = BM25Index(obj.texts)
        obj._embedder_single = (
            getattr(embedder, "get_embedding", None)
            if embedder
            else get_embedding_single
        )
        obj._embedder_batch = (
            getattr(embedder, "get_embeddings", None)
            if embedder
            else get_embedding_batch
        )
        obj._chat_client = chat_client
        obj._cache_file = cacheFile

        # Intializing conversation history
        obj.conversationHistory = []
        obj.maxHistory = 10

        # Cache category embeddings once during initialization
        obj.categoryEmbeddings = obj.create_category_embeddings()

        # Initialize reranker
        obj.reranker = initialize_reranker()

        for i, chunk in enumerate(obj.chunks):
            obj.db.add(obj.embeddings[i], chunk["text"], chunk["metadata"])
        return obj

    def create_category_embeddings(self):
        category = {
            "resume": "Professional career document containing work experience, technical skills, education background, projects, and quantified achievements. Formal tone with bullet points, dates, company names, job titles, and measurable accomplishments. Focus on qualifications, competencies, and professional history.",
            "cover_letters": "Personal application letters expressing interest in specific job roles and companies. Conversational tone showing enthusiasm, motivation, and personal fit for positions. Contains phrases like 'excited about', 'passionate about', 'would love to', and explanations of why someone wants a particular role or company.",
            "misc_docs": "Academic papers, essays, fellowship applications, and personal writings about broader topics like education, community impact, research, and personal philosophy. Reflective tone discussing values, social impact, academic work, and personal growth experiences.",
        }
        categoryEmbedding = {
            cat: self._embedder_single(desc) for cat, desc in category.items()
        }
        return categoryEmbedding

    def calculate_confidence(self, queryEmb, categoryEmbedding, debug=False):
        categoryScore = {
            cat: cosine_similarity([queryEmb], [categoryEmbedding[cat]])[0][0]
            for cat in categoryEmbedding
        }

        sortedScore = sorted(categoryScore.values(), reverse=True)
        if debug:
            print(f"Sorted Score: {sortedScore}")

        relativeConfidence = (
            sortedScore[0] - sortedScore[1] if len(sortedScore) > 1 else sortedScore[0]
        )
        if debug:
            print(f"Relative Confidence: {relativeConfidence}")

        return relativeConfidence, categoryScore

    def ask(self, query, debug=False):
        if not self.chunks or not self.embeddings:
            raise ValueError("Cannot query: No documents loaded.")

        queryEmb = self._embedder_single(query)

        # Hybrid search parameters (fixed)
        BM25_K = 20
        VECTOR_K = 20
        RERANK_TOP_N = 10

        # 1. BM25 search (k=20)
        bm25Results = self.bm25Index.search(query, k=BM25_K)
        # Convert BM25 results to same format as vector results
        bm25Candidates = []
        for idx, score, text in bm25Results:
            bm25Candidates.append(
                {
                    "text": text,
                    "metadata": self.chunks[idx]["metadata"],
                    "bm25_score": score,
                    "source": "bm25",
                }
            )

        # 2. Vector search (k=20)
        vectorCandidates = self.db.search(queryEmb, VECTOR_K)
        # Add source identifier
        for candidate in vectorCandidates:
            candidate["source"] = "vector"

        # 3. Merge results (deduplicate by text)
        mergedCandidates = []
        seenTexts = set()

        # Add BM25 results first
        for candidate in bm25Candidates:
            textKey = candidate["text"].strip().lower()
            if textKey not in seenTexts:
                seenTexts.add(textKey)
                mergedCandidates.append(candidate)

        # Add vector results (skip duplicates)
        for candidate in vectorCandidates:
            textKey = candidate["text"].strip().lower()
            if textKey not in seenTexts:
                seenTexts.add(textKey)
                mergedCandidates.append(candidate)
            else:
                # If duplicate, mark as hybrid (found by both methods)
                for existing in mergedCandidates:
                    if existing["text"].strip().lower() == textKey:
                        existing["source"] = "hybrid"
                        if "distance" in candidate:
                            existing["vector_distance"] = candidate["distance"]
                        break

        # 4. Rerank top 10
        results = rerank_candidates_func(
            query, mergedCandidates, self.reranker, top_k=RERANK_TOP_N
        )

        contextParts = []
        for result in results:
            contextParts.append(
                f"Page {result['metadata']['page']} - {result['metadata']['filename']}: {result['text']}"
            )

        context = "\n".join(contextParts)
        messages = self.build_converation_context(context)
        messages.append({"role": "user", "content": query})
        if self._chat_client is not None:
            response_text = self._chat_client(messages)
            self.add_to_conversation_history(query, response_text)
            # Add source information for custom chat client
            sources = [
                f"{result['metadata']['category']}/{result['metadata']['filename']}"
                for result in results[:3]
            ]
            queryAnswer = response_text + f"\n\n-----Sources: {', '.join(sources)}"
            return queryAnswer

        response = openai.chat.completions.create(
            model="gpt-4o-mini", messages=messages
        )
        self.add_to_conversation_history(query, response.choices[0].message.content)
        # Add source information for OpenAI response
        sources = [
            f"{result['metadata']['category']}/{result['metadata']['filename']}"
            for result in results[:3]
        ]
        queryAnswer = (
            response.choices[0].message.content
            + f"\n\n-----Sources: {', '.join(sources)}"
        )
        return queryAnswer

    def build_converation_context(self, documentContext):
        conversationContext = []
        conversationContext.append(
            {
                "role": "system",
                "content": "You are a helpful assistant that can only answer questions from the context provided. Use conversation history to understant the references and context",
            }
        )
        conversationContext.append(
            {"role": "user", "content": f"Document Context: {documentContext}"}
        )
        if self.conversationHistory:
            for conversation in self.conversationHistory:
                conversationContext.append(conversation["user"])
                conversationContext.append(conversation["assistant"])
        return conversationContext

    def add_to_conversation_history(self, query, response):
        self.conversationHistory.append(
            {
                "user": {"role": "user", "content": query},
                "assistant": {"role": "assistant", "content": response},
            }
        )
        if len(self.conversationHistory) > self.maxHistory:
            self.conversationHistory.pop(0)

    def rerank_candidates(self, query: str, candidates: list, top_k: int = 10):
        """Rerank candidates using the reranker model"""
        return rerank_candidates_func(query, candidates, self.reranker, top_k=top_k)

    def add_to_cache(self):
        """Ensure embeddings are JSON serializable (convert numpy arrays to lists)"""
        serializable_embeddings = []
        for emb in self.embeddings:
            if isinstance(emb, np.ndarray):
                serializable_embeddings.append(emb.tolist())
            else:
                try:
                    serializable_embeddings.append(np.asarray(emb).tolist())
                except Exception:
                    serializable_embeddings.append(emb)

        cacheData = {
            "chunks": self.chunks,
            "embeddings": serializable_embeddings,
        }
        with open(self._cache_file, "w") as f:
            json.dump(cacheData, f, indent=2)

    def search_bm25(self, query: str, k: int = 20):
        return self.bm25Index.search(query, k)
