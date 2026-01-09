"""
Claims Database Manager.

Manages atomic claims storage with automatic deduplication and querying.
Provides an append-only, idempotent interface for claim storage.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from collections import defaultdict
from datetime import datetime

from .schema import AtomicClaim, EvidencePointer

logger = logging.getLogger(__name__)


class ClaimsDatabase:
    """
    Manager for atomic claims storage and querying.
    
    Features:
    - Automatic deduplication by claim_id
    - Idempotent inserts (same claim_id = no duplicate)
    - Flexible querying interface
    - JSON-based storage
    """
    
    def __init__(self, db_path: str):
        """
        Initialize claims database.
        
        Args:
            db_path: Path to claims.json file
        """
        self.db_path = Path(db_path)
        self.claims: Dict[str, AtomicClaim] = {}  # claim_id -> claim
        self._load()
    
    def _load(self) -> None:
        """Load claims from JSON file."""
        if not self.db_path.exists():
            logger.info(f"No existing database at {self.db_path}, starting fresh")
            return
        
        try:
            with open(self.db_path, 'r') as f:
                data = json.load(f)
            
            # Convert dicts to AtomicClaim objects
            for claim_dict in data.get('claims', []):
                claim = AtomicClaim.from_dict(claim_dict)
                self.claims[claim.claim_id] = claim
            
            logger.info(f"Loaded {len(self.claims)} claims from {self.db_path}")
        
        except Exception as e:
            logger.error(f"Error loading claims database: {e}")
            self.claims = {}
    
    def save(self) -> None:
        """Save claims to JSON file."""
        try:
            # Ensure directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert claims to serializable format
            data = {
                "metadata": {
                    "total_claims": len(self.claims),
                    "last_updated": datetime.now().isoformat(),
                    "version": "1.0"
                },
                "claims": [claim.to_dict() for claim in self.claims.values()]
            }
            
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.claims)} claims to {self.db_path}")
        
        except Exception as e:
            logger.error(f"Error saving claims database: {e}")
            raise
    
    def add_claim(self, claim: AtomicClaim) -> bool:
        """
        Add a single claim (idempotent by claim_id).
        
        Args:
            claim: Claim to add
            
        Returns:
            True if claim was newly added, False if it already existed
        """
        if claim.claim_id in self.claims:
            logger.debug(f"Claim {claim.claim_id[:8]}... already exists, skipping")
            return False
        
        self.claims[claim.claim_id] = claim
        return True
    
    def add_claims(self, claims: List[AtomicClaim]) -> Dict[str, int]:
        """
        Add multiple claims with deduplication.
        
        Args:
            claims: List of claims to add
            
        Returns:
            Dict with counts: {"added": N, "duplicates": M}
        """
        added = 0
        duplicates = 0
        
        for claim in claims:
            if self.add_claim(claim):
                added += 1
            else:
                duplicates += 1
        
        return {"added": added, "duplicates": duplicates}
    
    def query(self, filters: Optional[Dict[str, Any]] = None) -> List[AtomicClaim]:
        """
        Query claims with filters.
        
        Args:
            filters: Dictionary of filters:
                - claim_type: str or list of str
                - context: str or list of str
                - confidence: float or {"$gte": float, "$lte": float}
                - document_date: str or {"$gte": str, "$lte": str}
                - filename: str or list of str
                - value: str (exact match or substring with "$contains")
        
        Returns:
            List of matching claims
        """
        if not filters:
            return list(self.claims.values())
        
        results = []
        
        for claim in self.claims.values():
            if self._matches_filters(claim, filters):
                results.append(claim)
        
        return results
    
    def _matches_filters(self, claim: AtomicClaim, filters: Dict[str, Any]) -> bool:
        """Check if claim matches all filters."""
        for key, value in filters.items():
            if key == "claim_type":
                if isinstance(value, list):
                    if claim.claim_type not in value:
                        return False
                elif claim.claim_type != value:
                    return False
            
            elif key == "context":
                if isinstance(value, list):
                    if claim.context not in value:
                        return False
                elif claim.context != value:
                    return False
            
            elif key == "confidence":
                if isinstance(value, dict):
                    if "$gte" in value and claim.confidence < value["$gte"]:
                        return False
                    if "$lte" in value and claim.confidence > value["$lte"]:
                        return False
                elif claim.confidence != value:
                    return False
            
            elif key == "document_date":
                if isinstance(value, dict):
                    if "$gte" in value and claim.document_date < value["$gte"]:
                        return False
                    if "$lte" in value and claim.document_date > value["$lte"]:
                        return False
                elif claim.document_date != value:
                    return False
            
            elif key == "filename":
                if isinstance(value, list):
                    if claim.evidence.filename not in value:
                        return False
                elif claim.evidence.filename != value:
                    return False
            
            elif key == "value":
                if isinstance(value, dict) and "$contains" in value:
                    if value["$contains"].lower() not in claim.value.lower():
                        return False
                elif claim.value != value:
                    return False
        
        return True
    
    def get_by_type(self, claim_type: str) -> List[AtomicClaim]:
        """Get all claims of a specific type."""
        return self.query({"claim_type": claim_type})
    
    def get_by_context(self, context: str) -> List[AtomicClaim]:
        """Get all claims from a specific context."""
        return self.query({"context": context})
    
    def get_by_confidence(self, min_confidence: float = 0.7) -> List[AtomicClaim]:
        """Get claims with confidence >= threshold."""
        return self.query({"confidence": {"$gte": min_confidence}})
    
    def get_timeline(self) -> Dict[str, List[AtomicClaim]]:
        """
        Get claims grouped by document date.
        
        Returns:
            Dict mapping date -> list of claims
        """
        timeline = defaultdict(list)
        
        for claim in self.claims.values():
            timeline[claim.document_date].append(claim)
        
        # Sort dates
        return dict(sorted(timeline.items()))
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dict with various statistics
        """
        if not self.claims:
            return {
                "total_claims": 0,
                "by_type": {},
                "by_context": {},
                "by_confidence": {},
                "date_range": None
            }
        
        # Count by type
        by_type = defaultdict(int)
        by_context = defaultdict(int)
        by_confidence = defaultdict(int)
        dates = []
        
        for claim in self.claims.values():
            by_type[claim.claim_type] += 1
            by_context[claim.context] += 1
            
            # Bucket confidence
            if claim.confidence == 1.0:
                by_confidence["explicit (1.0)"] += 1
            elif claim.confidence >= 0.7:
                by_confidence["clear (0.7)"] += 1
            else:
                by_confidence["implicit (0.4)"] += 1
            
            dates.append(claim.document_date)
        
        return {
            "total_claims": len(self.claims),
            "by_type": dict(by_type),
            "by_context": dict(by_context),
            "by_confidence": dict(by_confidence),
            "date_range": {
                "earliest": min(dates) if dates else None,
                "latest": max(dates) if dates else None
            }
        }
    
    def clear(self) -> None:
        """Clear all claims from database."""
        self.claims.clear()
        logger.warning("Cleared all claims from database")
    
    def __len__(self) -> int:
        """Return number of claims."""
        return len(self.claims)
    
    def __contains__(self, claim_id: str) -> bool:
        """Check if claim_id exists."""
        return claim_id in self.claims
