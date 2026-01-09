"""
Core business logic modules.

This package contains the core components of the VectorEmbedding system,
organized by domain:
- retrieval: Search and retrieval components (embeddings, vector DB, BM25, reranker)
- documents: Document processing and loading
- llm: LLM client and interactions
- cache: Cache management
- config: Configuration management
- analysis: Document analysis and semantic profiling
- utils: Shared utilities
"""

from .utils import get_project_root

__all__ = ["get_project_root"]
