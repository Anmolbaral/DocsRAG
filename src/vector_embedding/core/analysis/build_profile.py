"""
Build Profile - Extraction Orchestrator.

Orchestrates atomic claims extraction from documents.
Pure extraction - no critique or analysis.
"""
import logging
import sys
from pathlib import Path

from .semantic_profile.extractor import AtomicClaimsExtractor
from .claims_db import ClaimsDatabase
from ..config.config import Config
from ..utils import get_project_root

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def build_claims_database(force_refresh: bool = False) -> ClaimsDatabase:
    """
    Build or load claims database.
    
    Args:
        force_refresh: If True, re-extract all claims. If False, use existing cache.
        
    Returns:
        ClaimsDatabase instance
    """
    # Setup paths
    project_root = get_project_root()
    config = Config.from_file(project_root / "config.toml")
    cache_dir = project_root / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    db_path = cache_dir / "claims.json"
    data_dir = project_root / "data"
    
    # Check if we need to run extraction
    if db_path.exists() and not force_refresh:
        logger.info(f"Loading existing claims database from {db_path}")
        db = ClaimsDatabase(str(db_path))
        
        if len(db) > 0:
            logger.info(f"✓ Loaded {len(db)} existing claims")
            return db
        else:
            logger.warning("Database is empty, will regenerate")
    
    # Run extraction
    logger.info("="*60)
    logger.info("Building Claims Database (Extracting from PDFs)")
    logger.info("="*60)
    
    extractor = AtomicClaimsExtractor(config)
    claims = extractor.extract_claims_from_directory(str(data_dir))
    
    if not claims:
        logger.error("No claims extracted! Check if PDFs exist in data/ directory")
        sys.exit(1)
    
    # Save to database
    db = ClaimsDatabase(str(db_path))
    result = db.add_claims(claims)
    db.save()
    
    logger.info("\n" + "="*60)
    logger.info("Extraction Complete:")
    logger.info(f"  Added: {result['added']} claims")
    logger.info(f"  Duplicates: {result['duplicates']} claims")
    logger.info(f"  Total in DB: {len(db)} claims")
    logger.info("="*60 + "\n")
    
    # Show basic stats
    stats = db.get_stats()
    logger.info("Database Statistics:")
    logger.info(f"  By Type: {stats['by_type']}")
    logger.info(f"  By Context: {stats['by_context']}")
    logger.info(f"  By Confidence: {stats['by_confidence']}")
    logger.info(f"  Date Range: {stats['date_range']}")
    
    return db


if __name__ == "__main__":
    """
    Main entry point for building profile database.
    
    Usage:
        python -m src.vector_embedding.core.analysis.build_profile         # Use cache if exists
        python -m src.vector_embedding.core.analysis.build_profile --force # Force regeneration
    """
    # Check for --force flag
    force_refresh = "--force" in sys.argv or "-f" in sys.argv
    
    if force_refresh:
        logger.info("Force refresh enabled - will re-extract all claims\n")
    
    # Build database
    try:
        db = build_claims_database(force_refresh=force_refresh)
        
        # Success message
        logger.info(f"\n✓ Profile database ready with {len(db)} claims")
        logger.info(f"  Location: {db.db_path}")
        
        # Suggest next steps
        logger.info("\nNext steps:")
        logger.info("  - Generate insights: python -m src.vector_embedding.core.analysis.semantic_profile.insights")
        logger.info("  - View claims: cat cache/claims.json")
    
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"\nError building profile: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
