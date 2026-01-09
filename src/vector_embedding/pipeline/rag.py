from ..core.retrieval.embeddings import EmbeddingService
from ..core.retrieval.vectordb import VectorDB
import json
import numpy as np
from ..core.config.config import Config
from ..core.retrieval.reranker import RerankerService
from ..core.retrieval.bm25 import BM25Index
from ..core.llm.client import LLMChat


class RAGPipeline:
    def __init__(
        self,
        chunks,
        config: Config,
        embedder=None,
        chatClient=None,
        cachedChunks="cache/cached_chunks.json",
        cachedEmbeddings="cache/cached_embeddings.npy",
    ):
        self.config = config
        self.db = VectorDB(dim=config.vectorDB.dim)
        self.texts = [chunk["text"] for chunk in chunks]
        self.chunks = chunks

        # Initialize services
        self.embeddingService = (
            EmbeddingService(config) if embedder is None else embedder
        )
        self.rerankerService = RerankerService(config)
        self.bm25Index = BM25Index(self.texts, config)

        self._chatClient = LLMChat(config) if chatClient is None else chatClient
        self._cachedChunks = cachedChunks
        self._cachedEmbeddings = cachedEmbeddings

        if not self.texts or not self.chunks:
            self.embeddings = []
            self.chunks = []
        else:
            # Use service methods
            if hasattr(self.embeddingService, "get_embedding_batch"):
                self.embeddings = self.embeddingService.get_embedding_batch(self.texts)
            else:
                # Fallback for custom embedder
                self.embeddings = self.embeddingService(self.texts)

            metadataList = [chunk["metadata"] for chunk in self.chunks]
            self.db.add(self.embeddings, self.texts, metadataList)

        self.add_to_cache()
        self.conversationHistory = []

    @classmethod
    def from_cache(
        cls,
        config: Config,
        cachedChunks="cache/cached_chunks.json",
        cachedEmbeddings="cache/cached_embeddings.npy",
        embedder=None,
        chatClient=None,
    ):
        with open(cachedChunks, "r") as f:
            metadata = json.load(f)
        with open(cachedEmbeddings, "rb") as f:
            embeddings = np.load(f)

        obj = cls.__new__(cls)
        obj.db = VectorDB(dim=config.vectorDB.dim)
        obj.chunks = metadata
        obj.embeddings = embeddings
        obj.texts = [chunk["text"] for chunk in obj.chunks]
        obj.config = config

        obj.embeddingService = (
            EmbeddingService(config) if embedder is None else embedder
        )
        obj.rerankerService = RerankerService(config)
        obj.bm25Index = BM25Index(obj.texts, config)
        obj._chatClient = LLMChat(config) if chatClient is None else chatClient

        # Populate the vector database with cached embeddings
        metadataList = [chunk["metadata"] for chunk in obj.chunks]
        obj.db.add(obj.embeddings, obj.texts, metadataList)

        # Initialize conversation history
        obj.conversationHistory = []
        return obj

    def ask(self, query):
        if not self.chunks or len(self.embeddings) == 0:
            raise ValueError("Cannot query: No documents loaded.")

        queryEmb = self.embeddingService.get_embedding_single(query)

        bm25Results = self.bm25Index.search(query)
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
        textToCandidate = {}

        # Add BM25 results first
        for candidate in bm25Candidates:
            textKey = candidate["text"].strip().lower()
            if textKey not in seenTexts:
                seenTexts.add(textKey)
                textToCandidate[textKey] = candidate
                mergedCandidates.append(candidate)

        # Add vector results (skip duplicates)
        for candidate in vectorCandidates:
            textKey = candidate["text"].strip().lower()
            if textKey not in seenTexts:
                seenTexts.add(textKey)
                mergedCandidates.append(candidate)
            else:
                # O(1) lookup
                if textKey in textToCandidate:
                    textToCandidate[textKey]["source"] = "hybrid"
                    if "distance" in candidate:
                        textToCandidate[textKey]["vector_distance"] = candidate[
                            "distance"
                        ]

        # 4. Rerank using service
        results = self.rerankerService.rerank_candidates(query, mergedCandidates)

        contextParts = []

        for result in results:
            contextParts.append(
                f"Page {result['metadata']['page']} - {result['metadata']['filename']}: {result['text']}"
            )

        context = "\n".join(contextParts)
        messages = self.build_conversation_context(context)
        messages.append({"role": "user", "content": query})

        responseText = self._chatClient.chat(messages)
        self.add_to_conversation_history(query, responseText)
        # Add source information for custom chat client
        sources = [
            f"{result['metadata']['category']}/{result['metadata']['filename']}"
            for result in results[:3]
        ]
        queryAnswer = responseText + f"\n\n-----Sources: {', '.join(sources)}"
        return queryAnswer

        responseText = self._chatClient.chat(messages)
        self.add_to_conversation_history(query, responseText)
        return responseText

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

    def add_to_cache(self):
        """Ensure embeddings are JSON serializable (convert numpy arrays to lists)"""
        if self.embeddings:
            embeddingsArray = np.array(self.embeddings, dtype=np.float32)
            np.save(self._cachedEmbeddings, embeddingsArray)

        with open(self._cachedChunks, "w") as f:
            json.dump(self.chunks, f, indent=2)
