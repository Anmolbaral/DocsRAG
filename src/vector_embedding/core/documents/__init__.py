"""
Document processing module.

This module provides functions for loading and processing documents,
including PDF extraction and text chunking.
"""

from .loader import load_pdf, load_and_chunk_pdf

__all__ = ["load_pdf", "load_and_chunk_pdf"]
