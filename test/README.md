# Test Suite

## Running Tests

```bash
# Run all unit tests
./vector_venv/bin/python -m pytest test/unit -v

# Run specific file
./vector_venv/bin/python -m pytest test/unit/test_hashing.py -v

# Quick run
./vector_venv/bin/python -m pytest test/unit -q
```

## Current Coverage

- **test_hashing.py** - File hashing (3 tests)
- **test_cache_manager.py** - Cache management (3 tests)
- **test_vectordb.py** - Vector database (4 tests)

**Total**: 10 tests, all passing

## Adding New Tests

Create `test/unit/test_<module>.py`:

```python
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from vector_embedding.core.<module> import <Class>


def test_something(temp_dir):
    """Test description."""
    assert True
```

Available fixtures: `temp_dir`, `sample_text`
