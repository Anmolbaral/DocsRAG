# Forensic Analysis System

## Overview

The forensic analysis system extracts **atomic, auditable claims** from your documents (resumes, cover letters, papers) and generates insights using **deterministic logic** instead of LLM interpretation.

## Key Features

✅ **Atomic Claims** - Every fact is a separate, traceable entry  
✅ **Evidence Pointers** - Each claim links to exact source (file + page + hash)  
✅ **Deduplication** - Idempotent inserts (same claim_id = no duplicate)  
✅ **Time-Aware** - Document dates enable timeline queries  
✅ **Confidence Scoring** - 0.4 (implicit) → 0.7 (clear) → 1.0 (explicit)  
✅ **Context Tagging** - production, academic, internship, hobby  
✅ **Deterministic Insights** - Pure logic, no hallucinations  

## Quick Start

### 1. Build Claims Database

```bash
# Extract claims from all PDFs in data/
./vector_venv/bin/python -m src.vector_embedding.core.analysis.build_profile

# Force regeneration
./vector_venv/bin/python -m src.vector_embedding.core.analysis.build_profile --force
```

### 2. Generate Insights

```bash
# Generate insights report from claims
./vector_venv/bin/python -m src.vector_embedding.core.analysis.semantic_profile.insights
```

### 3. View Claims Database

```bash
# View raw claims
cat cache/claims.json | jq '.'

# Count total claims
cat cache/claims.json | jq '.metadata.total_claims'

# View insights report
cat cache/insights_report.json | jq '.'
```

## Data Structure

### Atomic Claim

```json
{
  "claim_id": "abc123...",
  "claim_type": "skill",
  "value": "Python",
  "context": "production",
  "confidence": 1.0,
  "evidence": {
    "filename": "resume.pdf",
    "page": 1,
    "text_hash": "def456..."
  },
  "document_date": "2024-01-15",
  "metadata": {}
}
```

### Claim Types

- **skill** - Technical skills (Python, Docker, React)
- **value** - Core values (Collaboration, Innovation)
- **experience** - Work experiences (internships, jobs)
- **achievement** - Specific accomplishments
- **education** - Educational background

### Context Categories

- **production** - Professional work experience
- **academic** - School projects, coursework
- **internship** - Internship experiences
- **hobby** - Personal projects
- **unknown** - Cannot determine

### Confidence Levels

- **1.0** - Explicitly stated with details
- **0.7** - Clearly implied from context
- **0.4** - Weakly suggested

## Python API

### Extract Claims

```python
from vector_embedding.core.analysis import AtomicClaimsExtractor
from vector_embedding.core.config import Config

config = Config.from_file("config.toml")
extractor = AtomicClaimsExtractor(config)

# Extract from single document
claims = extractor.extract_claims_from_document("data/resume/resume.pdf")

# Extract from directory
all_claims = extractor.extract_claims_from_directory("data")
```

### Query Claims

```python
from vector_embedding.core.analysis import ClaimsDatabase

db = ClaimsDatabase("cache/claims.json")

# Get all skills
skills = db.get_by_type("skill")

# Get production skills with high confidence
prod_skills = db.query({
    "claim_type": "skill",
    "context": "production",
    "confidence": {"$gte": 0.7}
})

# Get timeline
timeline = db.get_timeline()
```

### Generate Insights

```python
from vector_embedding.core.analysis.semantic_profile import InsightEngine

engine = InsightEngine("cache/claims.json")

# Top skills
top_skills = engine.get_top_skills(min_confidence=0.7, limit=10)

# Missing skills (papers vs resume)
gaps = engine.get_missing_skills()

# Growth metrics
growth = engine.get_growth_metrics()

# Full report
report = engine.generate_summary_report()
```

## File Structure

```
src/vector_embedding/core/analysis/
├── schema.py              # Atomic claim data structures
├── claims_db.py           # Database manager with deduplication
├── canonicalizer.py       # Skill normalization ("React.js" → "React")
├── build_profile.py       # Extraction orchestrator
└── semantic_profile/
    ├── extractor.py       # Page-level atomic claims miner
    └── insights.py        # Deterministic insight generators

cache/
├── claims.json            # Atomic claims database
├── insights_report.json   # Generated insights
├── canon_map.json         # Custom normalization rules
└── llm_view_archive.json  # Old format (archived)
```

## Benefits Over Old System

| Feature | Old System | Forensic System |
|---------|-----------|-----------------|
| **Stability** | Different results each run | Deterministic, reproducible |
| **Traceability** | "Trust me, I read it" | Every claim → exact source |
| **Updates** | Re-extract everything | Append new docs only |
| **Growth Tracking** | No timeline | Full temporal analysis |
| **Deduplication** | Manual | Automatic via claim_id |
| **Hallucination** | LLM interprets | Pure logic/math |

## Advanced Usage

### Custom Normalization

Edit `cache/canon_map.json` to add custom aliases:

```json
{
  "skill_aliases": {
    "my_custom_framework": "CustomFramework"
  }
}
```

### Filter by Date Range

```python
recent_claims = db.query({
    "document_date": {"$gte": "2024-01-01"}
})
```

### Context Breakdown

```python
# See production vs academic distribution
breakdown = engine.get_context_breakdown(claim_type="skill")
```

## Testing

Run forensic system tests:

```bash
# Unit tests for claims database
./vector_venv/bin/python -m pytest test/unit -v

# Quick validation
./vector_venv/bin/python -c "
from src.vector_embedding.core.analysis import ClaimsDatabase
db = ClaimsDatabase('cache/claims.json')
print(f'Total claims: {len(db)}')
print(f'Stats: {db.get_stats()}')
"
```

## Next Steps

1. Build claims database from your documents
2. Explore insights report
3. Query specific claims
4. Use timeline for growth analysis
5. Integrate with RAG system (future: separate vector index for claims)
