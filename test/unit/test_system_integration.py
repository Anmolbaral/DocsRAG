"""
Integration tests for DocumentRAGSystem.

Tests the full system initialization and query flow.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vector_embedding.system import DocumentRAGSystem
from vector_embedding.core.config import Config


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def mock_config():
    """Create a test configuration."""
    config_data = {
        "vectorDB": {"dim": 384},
        "retrieval": {
            "vectorTopK": 5,
            "bm25TopK": 5,
            "contextTopK": 3
        },
        "reranker": {
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "topK": 5
        },
        "conversation": {
            "systemPrompt": "You are a helpful assistant.",
            "maxHistory": 10
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini"
        },
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small"
        },
        "chunking": {
            "chunkSize": 300,
            "overlap": 60,
            "minChunkChars": 150
        }
    }
    return Config.from_dict(config_data)


def test_system_initialization_no_cache(temp_data_dir, temp_cache_dir, mock_config):
    """Test system initialization without existing cache."""
    # Note: This test would need actual PDFs to work fully
    # For now, it tests the structure
    system = DocumentRAGSystem(
        cacheDir=str(temp_cache_dir),
        dataDir=str(temp_data_dir),
        config=mock_config
    )
    
    assert system.cacheManager is not None
    assert system.config is not None
    assert system.ragPipeline is None  # Not initialized yet


def test_system_requires_initialization_before_query(temp_data_dir, temp_cache_dir, mock_config):
    """Test that querying before initialization raises error."""
    system = DocumentRAGSystem(
        cacheDir=str(temp_cache_dir),
        dataDir=str(temp_data_dir),
        config=mock_config
    )
    
    with pytest.raises(ValueError, match="not initialized"):
        system.query("test query")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
