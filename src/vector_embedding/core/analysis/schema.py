"""
Atomic Claims Schema for Forensic Analysis.

Defines the data structures for evidence-based claim storage.
Each claim is atomic, traceable, and auditable.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any
import hashlib


__all__ = ["EvidencePointer", "AtomicClaim", "ClaimExtractionResult"]


@dataclass
class EvidencePointer:
    """
    GPS coordinates for a claim's source evidence with span-level precision.

    Attributes:
        filename: Source document filename
        page: Page number where evidence appears (1-indexed)
        start_char: Start character offset in page text (0-indexed)
        end_char: End character offset in page text (exclusive)
        quote: Exact text snippet that supports the claim
        text_hash: SHA256 hash of the source page text (for change detection)
        context_before: Optional text before the quote (for context)
        context_after: Optional text after the quote (for context)
    """

    filename: str
    page: int
    start_char: int
    end_char: int
    quote: str
    text_hash: str
    context_before: str = ""
    context_after: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidencePointer":
        """
        Create from dictionary with backward compatibility.
        
        Handles old format (page-level) and new format (span-level).
        """
        # Check if this is old format (missing span fields)
        if 'start_char' not in data:
            # Old format: page-level evidence only
            # Use placeholder values for new fields
            return cls(
                filename=data['filename'],
                page=data['page'],
                start_char=0,
                end_char=0,
                quote="[Legacy claim - no span data]",
                text_hash=data.get('text_hash', ''),
                context_before="",
                context_after=""
            )
        
        # New format: has all span fields
        return cls(**data)


@dataclass
class AtomicClaim:
    """
    A single, atomic, auditable claim extracted from documents.

    Each claim represents one fact (e.g., "User knows Python") with
    full traceability to its source evidence.

    Attributes:
        claim_id: Unique ID (SHA256 of claim_type|value|filename|page)
        claim_type: Type of claim (skill, value, experience, achievement, education)
        value: The actual claim content (e.g., "Python", "Collaboration")
        context: Context category (production, academic, internship, hobby)
        confidence: Confidence score (0.4=implicit, 0.7=clear, 1.0=explicit)
        evidence: Pointer to source evidence
        document_date: Date of document (YYYY-MM-DD) or file mtime
        metadata: Additional context (dict)
    """

    claim_id: str
    claim_type: str  # skill, value, experience, achievement, education
    value: str
    context: str  # production, academic, internship, hobby
    confidence: float  # 0.4, 0.7, 1.0
    evidence: EvidencePointer
    document_date: str  # YYYY-MM-DD
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AtomicClaim":
        """Create from dictionary (for loading from JSON)."""
        # Convert evidence dict to EvidencePointer
        if isinstance(data.get("evidence"), dict):
            data["evidence"] = EvidencePointer.from_dict(data["evidence"])
        return cls(**data)

    @staticmethod
    def generate_claim_id(claim_type: str, value: str, filename: str, page: int) -> str:
        """
        Generate deterministic claim ID.

        Args:
            claim_type: Type of claim
            value: Claim value
            filename: Source filename
            page: Page number

        Returns:
            SHA256 hash as hex string
        """
        # Normalize inputs for consistent hashing
        normalized = f"{claim_type.lower()}|{value.lower()}|{filename}|{page}"
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def generate_text_hash(text: str) -> str:
        """
        Generate hash of source text block.

        Args:
            text: Source text content

        Returns:
            SHA256 hash as hex string
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class ClaimExtractionResult:
    """
    Result of extracting claims from a single page.

    Attributes:
        filename: Source document
        page: Page number
        claims: List of extracted claims
        extraction_metadata: Metadata about the extraction process
    """

    filename: str
    page: int
    claims: list[AtomicClaim]
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "filename": self.filename,
            "page": self.page,
            "claims": [claim.to_dict() for claim in self.claims],
            "extraction_metadata": self.extraction_metadata,
        }
