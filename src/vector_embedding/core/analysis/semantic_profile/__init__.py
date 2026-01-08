"""
Semantic Profile Module

This module provides tools for extracting structured metadata from documents
and generating insights from that metadata.
"""

from .extractor import MetadataExtractor, DocumentMetadata, CompanyProfile
from .insights import InsightEngine

__all__ = [
    "MetadataExtractor",
    "DocumentMetadata", 
    "CompanyProfile",
    "InsightEngine"
]

