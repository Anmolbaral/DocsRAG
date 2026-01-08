"""Simple tests for VectorDB."""
import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from vector_embedding.core.retrieval.vectordb import VectorDB


def test_vectordb_init():
    """VectorDB should initialize correctly."""
    db = VectorDB(dim=128)
    
    assert db.texts == []
    assert db.metadata == []


def test_add_single_vector():
    """Should add a single vector."""
    db = VectorDB(dim=128)
    
    vector = np.random.rand(128).tolist()
    db.add(vector, "Test text", {"id": 1})
    
    assert len(db.texts) == 1
    assert db.texts[0] == "Test text"


def test_add_multiple_vectors():
    """Should add multiple vectors."""
    db = VectorDB(dim=128)
    
    vectors = np.random.rand(5, 128).tolist()
    texts = [f"Text {i}" for i in range(5)]
    metadata = [{"id": i} for i in range(5)]
    
    db.add(vectors, texts, metadata)
    
    assert len(db.texts) == 5


def test_search_returns_results():
    """Search should return results."""
    db = VectorDB(dim=128)
    
    vectors = np.random.rand(10, 128).tolist()
    texts = [f"Text {i}" for i in range(10)]
    metadata = [{"id": i} for i in range(10)]
    db.add(vectors, texts, metadata)
    
    query = np.random.rand(128).tolist()
    results = db.search(query, k=3)
    
    assert len(results) == 3
    assert "text" in results[0]
    assert "metadata" in results[0]
    assert "distance" in results[0]
