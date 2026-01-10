"""Simple tests for CacheManager."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from vector_embedding.core.cache.manager import CacheManager


def test_creates_cache_dir(temp_dir):
    """CacheManager should create cache directory."""
    cache_dir = temp_dir / "cache"
    data_dir = temp_dir / "data"
    data_dir.mkdir()

    manager = CacheManager(str(cache_dir), str(data_dir))

    assert cache_dir.exists()


def test_save_and_load_metadata(temp_dir):
    """Should save and load metadata correctly."""
    cache_dir = temp_dir / "cache"
    data_dir = temp_dir / "data"
    data_dir.mkdir()

    manager = CacheManager(str(cache_dir), str(data_dir))

    test_data = {"file1.pdf": {"fileSize": 1024, "fileHash": "abc123"}}
    manager.save_file_metadata(test_data)

    loaded = manager.load_file_metadata()
    assert loaded == test_data


def test_load_missing_metadata_returns_empty(temp_dir):
    """Loading non-existent metadata should return empty dict."""
    cache_dir = temp_dir / "cache"
    data_dir = temp_dir / "data"
    data_dir.mkdir()

    manager = CacheManager(str(cache_dir), str(data_dir))
    loaded = manager.load_file_metadata()

    assert loaded == {}
