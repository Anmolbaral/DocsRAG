"""
Analysis module for forensic document profiling.

Provides atomic claims extraction and deterministic insights generation.
"""

from .claims_db import ClaimsDatabase
from .schema import AtomicClaim, EvidencePointer, ClaimExtractionResult
from .canonicalizer import Canonicalizer, get_canonicalizer
from .semantic_profile.extractor import AtomicClaimsExtractor, MetadataExtractor
from .semantic_profile.insights import InsightEngine

__all__ = [
    "ClaimsDatabase",
    "AtomicClaim",
    "EvidencePointer", 
    "ClaimExtractionResult",
    "Canonicalizer",
    "get_canonicalizer",
    "AtomicClaimsExtractor",
    "MetadataExtractor",
    "InsightEngine"
]
