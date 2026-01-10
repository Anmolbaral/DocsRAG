"""
Comprehensive tests for RAGPipeline.

These tests cover the critical bugs that were found:
1. NumPy array boolean evaluation
2. Empty FAISS index when loading from cache
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vector_embedding.pipeline.rag import RAGPipeline
from vector_embedding.core.config import Config


@pytest.fixture
def mock_config():
    """Create a minimal config for testing."""
    config_data = {
        "vectorDB": {"dim": 384},
        "retrieval": {"vectorTopK": 5, "bm25TopK": 5, "contextTopK": 3},
        "reranker": {"model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "topK": 5},
        "conversation": {
            "systemPrompt": "You are a helpful assistant.",
            "maxHistory": 10,
        },
        "llm": {"provider": "openai", "model": "gpt-4o-mini"},
        "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
    }
    return Config.from_dict(config_data)


@pytest.fixture
def sample_chunks():
    """Create sample document chunks."""
    return [
        {
            "text": "Python is a programming language.",
            "metadata": {"filename": "test.pdf", "page": 1, "category": "test"},
        },
        {
            "text": "JavaScript is used for web development.",
            "metadata": {"filename": "test.pdf", "page": 2, "category": "test"},
        },
        {
            "text": "Machine learning requires data.",
            "metadata": {"filename": "test2.pdf", "page": 1, "category": "test"},
        },
    ]


class MockEmbedder:
    """Mock embedder that returns random vectors."""

    def __init__(self, dim=384):
        self.dim = dim

    def get_embedding_single(self, text):
        return np.random.rand(self.dim).astype("float32")

    def get_embedding_batch(self, texts):
        return np.random.rand(len(texts), self.dim).astype("float32")


class MockLLM:
    """Mock LLM that returns a simple response."""

    def chat(self, messages):
        return "This is a test response."


def test_pipeline_initialization_with_empty_chunks(mock_config):
    """Test that pipeline handles empty chunks correctly (Bug #1)."""
    # This should not raise an error
    pipeline = RAGPipeline(
        [], config=mock_config, embedder=MockEmbedder(), chatClient=MockLLM()
    )

    assert pipeline.chunks == []
    assert pipeline.embeddings == []

    # Querying should raise a clear error
    with pytest.raises(ValueError, match="Cannot query: No documents loaded"):
        pipeline.ask("test query")


def test_pipeline_initialization_with_chunks(mock_config, sample_chunks):
    """Test that pipeline initializes correctly with documents."""
    pipeline = RAGPipeline(
        sample_chunks, config=mock_config, embedder=MockEmbedder(), chatClient=MockLLM()
    )

    assert len(pipeline.chunks) == 3
    assert len(pipeline.embeddings) > 0
    assert len(pipeline.texts) == 3


def test_pipeline_from_cache_populates_index(mock_config, sample_chunks, tmp_path):
    """
    Test that from_cache properly populates FAISS index (Bug #2).

    This was the critical bug - from_cache created an empty FAISS index,
    causing "list index out of range" errors during search.
    """
    # Create a pipeline and save to cache
    cache_chunks = tmp_path / "cached_chunks.json"
    cache_embeddings = tmp_path / "cached_embeddings.npy"

    embedder = MockEmbedder()

    # Create and cache
    _ = RAGPipeline(
        sample_chunks,
        config=mock_config,
        embedder=embedder,
        chatClient=MockLLM(),
        cachedChunks=str(cache_chunks),
        cachedEmbeddings=str(cache_embeddings),
    )

    # Load from cache
    pipeline2 = RAGPipeline.from_cache(
        config=mock_config,
        cachedChunks=str(cache_chunks),
        cachedEmbeddings=str(cache_embeddings),
        embedder=embedder,
        chatClient=MockLLM(),
    )

    # Critical assertion: FAISS index should be populated
    assert len(pipeline2.db.texts) == 3, "FAISS index not populated from cache!"
    assert len(pipeline2.db.metadata) == 3

    # Should be able to query without errors
    try:
        result = pipeline2.ask("What is Python?")
        assert result is not None
        assert "test response" in result.lower()
    except IndexError:
        pytest.fail(
            "IndexError raised - FAISS index was not properly populated from cache!"
        )


def test_pipeline_query_execution(mock_config, sample_chunks):
    """Test that queries execute without errors."""
    pipeline = RAGPipeline(
        sample_chunks, config=mock_config, embedder=MockEmbedder(), chatClient=MockLLM()
    )

    result = pipeline.ask("What is Python?")
    assert result is not None
    assert isinstance(result, str)


def test_pipeline_handles_numpy_array_correctly(mock_config, sample_chunks):
    """
    Test that embeddings (NumPy arrays) are handled correctly (Bug #1).

    The bug was using 'not self.embeddings' which causes
    "truth value of array is ambiguous" error.
    """
    pipeline = RAGPipeline(
        sample_chunks, config=mock_config, embedder=MockEmbedder(), chatClient=MockLLM()
    )

    # Embeddings should be a NumPy array
    assert isinstance(pipeline.embeddings, np.ndarray)

    # This should not raise "ambiguous truth value" error
    try:
        result = pipeline.ask("test query")
        assert result is not None
    except ValueError as e:
        if "ambiguous" in str(e).lower():
            pytest.fail(f"NumPy array boolean evaluation bug: {e}")


def test_conversation_history(mock_config, sample_chunks):
    """Test that conversation history is maintained."""
    pipeline = RAGPipeline(
        sample_chunks, config=mock_config, embedder=MockEmbedder(), chatClient=MockLLM()
    )

    # First query
    pipeline.ask("What is Python?")
    assert len(pipeline.conversationHistory) == 1

    # Second query
    pipeline.ask("What is JavaScript?")
    assert len(pipeline.conversationHistory) == 2

    # Check history structure
    assert "user" in pipeline.conversationHistory[0]
    assert "assistant" in pipeline.conversationHistory[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
