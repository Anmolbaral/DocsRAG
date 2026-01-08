"""Simple tests for file hashing."""
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from vector_embedding.core.cache.hashing import get_file_hash


def test_hash_same_content_same_hash(temp_dir):
    """Same content = same hash."""
    file1 = temp_dir / "file1.txt"
    file2 = temp_dir / "file2.txt"
    
    file1.write_text("Hello World")
    file2.write_text("Hello World")
    
    assert get_file_hash(str(file1)) == get_file_hash(str(file2))


def test_hash_different_content_different_hash(temp_dir):
    """Different content = different hash."""
    file1 = temp_dir / "file1.txt"
    file2 = temp_dir / "file2.txt"
    
    file1.write_text("Hello")
    file2.write_text("World")
    
    assert get_file_hash(str(file1)) != get_file_hash(str(file2))


def test_hash_is_consistent(temp_dir):
    """Hash should be consistent."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("Test content")
    
    hash1 = get_file_hash(str(test_file))
    hash2 = get_file_hash(str(test_file))
    
    assert hash1 == hash2
