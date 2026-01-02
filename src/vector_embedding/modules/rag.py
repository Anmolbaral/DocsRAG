from .embeddings import get_embedding_single, get_embedding_batch
from .vectordb import VectorDB
from .reranker import initialize_reranker, rerank_candidates as rerankCandidatesFunc
import openai
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .bm25 import BM25Index
from config import Config

openai.api_key = os.getenv("OPENAI_API_KEY")


class RAGPipeline:
    def __init__(
        self,
        chunks,
        config: Config,
        embedder=None,
        chatClient=None,
        cacheFile="cache/embeddings.json",
    ):
        self.config = config
        self.db = VectorDB(dim=config.vectorDB.dim)
        self.texts = [chunk["text"] for chunk in chunks]
        self.chunks = chunks
        self.bm25Index = BM25Index(self.texts)
        self._embedderSingle = (
            getattr(embedder, "get_embedding_single", None)
            if embedder
            else get_embedding_single
        )
        self._embedderBatch = (
            getattr(embedder, "get_embedding_batch", None)
            if embedder
            else get_embedding_batch
        )
        self._chatClient = chatClient
        self._cacheFile = cacheFile
        # If no texts are provided, return early
        if not self.texts or not self.chunks:
            self.embeddings = []
            self.chunks = []
        else:
            self.embeddings = self._embedderBatch(self.texts, self.config)
            for i, t in enumerate(self.chunks):
                self.db.add(
                    self.embeddings[i],
                    self.chunks[i]["text"],
                    self.chunks[i]["metadata"],
                )
        self.add_to_cache()

        # Adding conversation memory
        self.conversationHistory = []

        # Cache category embeddings once during initialization
        self.categoryEmbeddings = self.create_category_embeddings()

        # Initialize reranker
        self.reranker = initialize_reranker(self.config)

    @classmethod
    def from_cache(
        cls,
        config: Config,
        cacheFile="cache/embeddings.json",
        embedder=None,
        chatClient=None,
    ):
        with open(cacheFile, "r") as f:
            metadata = json.load(f)

        obj = cls.__new__(cls)
        obj.db = VectorDB(dim=config.vectorDB.dim)
        obj.chunks = metadata["chunks"]
        obj.embeddings = metadata["embeddings"]
        obj.texts = [chunk["text"] for chunk in obj.chunks]
        obj.bm25Index = BM25Index(obj.texts)
        obj._embedderSingle = (
            getattr(embedder, "getEmbedding", None)
            if embedder
            else get_embedding_single
        )
        obj._embedderBatch = (
            getattr(embedder, "getEmbeddings", None)
            if embedder
            else get_embedding_batch
        )
        obj._chatClient = chatClient
        obj._cacheFile = cacheFile
        obj.config = config

        # Intializing conversation history
        obj.conversationHistory = []

        # Cache category embeddings once during initialization
        obj.categoryEmbeddings = obj.create_category_embeddings()

        # Initialize reranker
        obj.reranker = initialize_reranker(config)

        for i, chunk in enumerate(obj.chunks):
            obj.db.add(obj.embeddings[i], chunk["text"], chunk["metadata"])
        return obj

    def create_category_embeddings(self):
        categoryEmbedding = {
            "resume": self._embedderSingle(
                self.config.categoryEmbedding.resume, self.config
            ),
            "cover_letters": self._embedderSingle(
                self.config.categoryEmbedding.cover_letters, self.config
            ),
            "misc_docs": self._embedderSingle(
                self.config.categoryEmbedding.misc_docs, self.config
            ),
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

        queryEmb = self._embedderSingle(query, self.config)

        # 1. BM25 search
        bm25Results = self.bm25Index.search(query, self.config)
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

        # 2. Vector search
        vectorCandidates = self.db.search(queryEmb, k=self.config.retrieval.vectorTopK)
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
        results = rerankCandidatesFunc(
            query, mergedCandidates, self.reranker, self.config
        )

        contextParts = []
        for result in results:
            contextParts.append(
                f"Page {result['metadata']['page']} - {result['metadata']['filename']}: {result['text']}"
            )

        context = "\n".join(contextParts)
        messages = self.build_conversation_context(context)
        messages.append({"role": "user", "content": query})
        if self._chatClient is not None:
            responseText = self._chatClient(messages)
            self.add_to_conversation_history(query, responseText)
            # Add source information for custom chat client
            sources = [
                f"{result['metadata']['category']}/{result['metadata']['filename']}"
                for result in results[:3]
            ]
            queryAnswer = responseText + f"\n\n-----Sources: {', '.join(sources)}"
            return queryAnswer

        response = openai.chat.completions.create(
            model=self.config.generation.model, messages=messages
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

    def build_conversation_context(self, documentContext):
        conversationContext = []
        conversationContext.append(
            {
                "role": "system",
                "content": self.config.conversation.systemPrompt,
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
        if len(self.conversationHistory) > self.config.conversation.maxHistory:
            self.conversationHistory.pop(0)

    def rerank_candidates(self, query: str, candidates: list):
        """Rerank candidates using the reranker model"""
        return rerankCandidatesFunc(query, candidates, self.reranker, self.config)

    def add_to_cache(self):
        """Ensure embeddings are JSON serializable (convert numpy arrays to lists)"""
        serializableEmbeddings = []
        for emb in self.embeddings:
            if isinstance(emb, np.ndarray):
                serializableEmbeddings.append(emb.tolist())
            else:
                try:
                    serializableEmbeddings.append(np.asarray(emb).tolist())
                except Exception:
                    serializableEmbeddings.append(emb)

        cacheData = {
            "chunks": self.chunks,
            "embeddings": serializableEmbeddings,
        }
        with open(self._cacheFile, "w") as f:
            json.dump(cacheData, f, indent=2)

    def search_bm25(self, query: str):
        return self.bm25Index.search(query, self.config)
