"""
Atomic Claims Extractor - Evidence Miner.

Extracts atomic, traceable claims from documents page-by-page.
Each claim is a single fact with full evidence pointer.
"""
import json
import os
import logging
import datetime
from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field
from openai import OpenAI
from pathlib import Path

from ...documents.loader import load_pdf
from ...config.config import Config
from ...utils import get_project_root
from ..schema import AtomicClaim, EvidencePointer, ClaimExtractionResult
from ..canonicalizer import get_canonicalizer

logger = logging.getLogger(__name__)


# Pydantic models for LLM structured output
class ExtractedClaim(BaseModel):
    """Single claim extracted by LLM."""
    claim_type: Literal["skill", "value", "experience", "achievement", "education"]
    value: str
    context: Literal["production", "academic", "internship", "hobby", "unknown"]
    confidence: float = Field(..., description="0.4=implicit, 0.7=clear, 1.0=explicit")
    notes: str = Field(default="", description="Optional notes about the claim")


class PageClaims(BaseModel):
    """All claims extracted from a single page."""
    document_date: str = Field(..., description="Best guess YYYY-MM-DD or UNKNOWN")
    claims: List[ExtractedClaim]


class AtomicClaimsExtractor:
    """
    Extracts atomic claims from documents using page-level processing.
    
    Features:
    - Page-by-page processing for accuracy
    - Confidence-scored claims (0.4, 0.7, 1.0)
    - Context tagging (production/academic/internship/hobby)
    - Full evidence pointers
    - Date fallback to file mtime
    - Canonicalization of values
    """
    
    def __init__(self, config: Config):
        """
        Initialize extractor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = config.llm.parseModel
        self.loader = load_pdf
        self.canonicalizer = get_canonicalizer()
    
    def extract_claims_from_page(
        self, 
        page_text: str, 
        page_num: int, 
        filename: str
    ) -> ClaimExtractionResult:
        """
        Extract atomic claims from a single page.
        
        Args:
            page_text: Text content of the page
            page_num: Page number (1-indexed)
            filename: Source filename
            
        Returns:
            ClaimExtractionResult with extracted claims
        """
        logger.debug(f"Extracting claims from {filename} page {page_num}")
        
        # Build extraction prompt
        system_prompt = """You are a forensic evidence extractor. Extract atomic, verifiable claims.

Rules:
- Extract ONLY explicit facts (skills, experiences, achievements, values, education)
- Each claim must be atomic (one fact)
- Assign context: production, academic, internship, hobby, unknown
- Assign confidence:
  * 1.0 = Explicitly stated with details
  * 0.7 = Clearly implied from context
  * 0.4 = Weakly suggested
- For dates: Extract best guess as YYYY-MM-DD. If no date found, use "UNKNOWN"
"""
        
        user_prompt = f"""Document: {filename}
Page: {page_num}

Content:
{page_text}

Extract all atomic claims from this page."""
        
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=PageClaims,
                temperature=0.1  # Low temp for consistency
            )
            
            page_claims = completion.choices[0].message.parsed
            
            # Convert to AtomicClaim objects with evidence pointers
            atomic_claims = []
            text_hash = AtomicClaim.generate_text_hash(page_text)
            
            for extracted in page_claims.claims:
                # Canonicalize value
                canonical_value = self.canonicalizer.canonicalize_value(
                    extracted.claim_type, 
                    extracted.value
                )
                
                # Generate claim ID
                claim_id = AtomicClaim.generate_claim_id(
                    extracted.claim_type,
                    canonical_value,
                    filename,
                    page_num
                )
                
                # Create evidence pointer
                evidence = EvidencePointer(
                    filename=filename,
                    page=page_num,
                    text_hash=text_hash
                )
                
                # Create atomic claim
                claim = AtomicClaim(
                    claim_id=claim_id,
                    claim_type=extracted.claim_type,
                    value=canonical_value,
                    context=extracted.context,
                    confidence=extracted.confidence,
                    evidence=evidence,
                    document_date=page_claims.document_date,
                    metadata={"notes": extracted.notes} if extracted.notes else {}
                )
                
                atomic_claims.append(claim)
            
            return ClaimExtractionResult(
                filename=filename,
                page=page_num,
                claims=atomic_claims,
                extraction_metadata={
                    "model": self.model,
                    "extracted_date": page_claims.document_date
                }
            )
        
        except Exception as e:
            logger.error(f"Error extracting claims from {filename} page {page_num}: {e}")
            return ClaimExtractionResult(
                filename=filename,
                page=page_num,
                claims=[],
                extraction_metadata={"error": str(e)}
            )
    
    def extract_claims_from_document(
        self, 
        pdf_path: str
    ) -> List[AtomicClaim]:
        """
        Extract all atomic claims from a PDF document.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of atomic claims from all pages
        """
        filename = Path(pdf_path).name
        logger.info(f"Extracting claims from: {filename}")
        
        # Load PDF pages
        try:
            pages = self.loader(pdf_path)
            if not pages:
                logger.warning(f"No pages extracted from {filename}")
                return []
        
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return []
        
        # Extract claims from each page
        all_claims = []
        document_date = None
        
        for page_data in pages:
            page_num = page_data["metadata"]["page"]
            page_text = page_data["text"]
            
            if not page_text.strip():
                continue
            
            result = self.extract_claims_from_page(page_text, page_num, filename)
            all_claims.extend(result.claims)
            
            # Capture document date from first page that has one
            extracted_date = result.extraction_metadata.get("extracted_date", "UNKNOWN")
            if not document_date and extracted_date != "UNKNOWN":
                document_date = extracted_date
        
        # Date fallback: Use file modification time if no date found
        if not document_date or document_date == "UNKNOWN":
            file_mtime = os.path.getmtime(pdf_path)
            document_date = datetime.datetime.fromtimestamp(file_mtime).strftime('%Y-%m-%d')
            logger.info(f"Using file mtime as document date: {document_date}")
            
            # Update all claims with fallback date
            for claim in all_claims:
                claim.document_date = document_date
        
        logger.info(f"Extracted {len(all_claims)} claims from {filename}")
        return all_claims
    
    def extract_claims_from_directory(
        self, 
        data_dir: str
    ) -> List[AtomicClaim]:
        """
        Extract claims from all PDFs in a directory.
        
        Args:
            data_dir: Path to directory containing PDFs
            
        Returns:
            List of all atomic claims from all documents
        """
        # Resolve path
        data_path = Path(data_dir)
        if not data_path.is_absolute():
            data_path = get_project_root() / data_dir
        
        logger.info(f"Scanning for PDFs in: {data_path}")
        
        # Find all PDFs
        pdf_files = list(data_path.rglob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {data_path}")
            return []
        
        # Extract claims from each PDF
        all_claims = []
        
        for idx, pdf_file in enumerate(pdf_files, 1):
            logger.info(f"[{idx}/{len(pdf_files)}] Processing: {pdf_file.name}")
            
            try:
                claims = self.extract_claims_from_document(str(pdf_file))
                all_claims.extend(claims)
                logger.info(f"  ✓ {len(claims)} claims extracted")
            
            except Exception as e:
                logger.error(f"  ✗ Error processing {pdf_file.name}: {e}")
                continue
        
        logger.info(f"\nTotal: {len(all_claims)} claims from {len(pdf_files)} documents")
        return all_claims


# For backwards compatibility and testing
class MetadataExtractor(AtomicClaimsExtractor):
    """Alias for backwards compatibility."""
    pass


if __name__ == "__main__":
    """Test extraction on sample documents."""
    from ...utils import get_project_root
    from ..claims_db import ClaimsDatabase
    
    # Setup
    project_root = get_project_root()
    config = Config.from_file(project_root / "config.toml")
    cache_dir = project_root / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    # Extract claims
    extractor = AtomicClaimsExtractor(config)
    claims = extractor.extract_claims_from_directory("data")
    
    # Save to database
    db = ClaimsDatabase(str(cache_dir / "claims.json"))
    result = db.add_claims(claims)
    db.save()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Extraction Complete:")
    logger.info(f"  Added: {result['added']} claims")
    logger.info(f"  Duplicates: {result['duplicates']} claims")
    logger.info(f"  Total in DB: {len(db)} claims")
    logger.info(f"{'='*60}\n")
    
    # Show stats
    stats = db.get_stats()
    logger.info(f"Database Statistics:")
    logger.info(f"  By Type: {stats['by_type']}")
    logger.info(f"  By Context: {stats['by_context']}")
    logger.info(f"  By Confidence: {stats['by_confidence']}")
    logger.info(f"  Date Range: {stats['date_range']}")
