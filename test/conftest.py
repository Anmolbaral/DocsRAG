"""Basic pytest fixtures for unit tests."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for tests."""
    return tmp_path


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "Machine learning is a subset of artificial intelligence."
