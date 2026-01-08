from src.vector_embedding.modules.semantic_profile.extractor import MetadataExtractor
from src.vector_embedding.modules.semantic_profile.insights import InsightEngine
from openai import OpenAI
import os
import json
import logging
from config import Config
from pathlib import Path
from src.vector_embedding.core.llm.client import LLMChat

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

projectRoot = Path(__file__).parent.parent.parent.parent
config = Config.from_file(projectRoot / "config.toml")

# Define the output file path (in project root)
llm_view_path = projectRoot / "llm_view.json"

# 1. Run Extraction - Force regeneration if file doesn't exist or is empty
should_regenerate = False

if not llm_view_path.exists():
    logger.info("llm_view.json doesn't exist. Will generate...")
    should_regenerate = True
else:
    # Check if file is empty or invalid
    try:
        with open(llm_view_path, "r") as f:
            existing_data = json.load(f)
            if not existing_data or len(existing_data) == 0:
                logger.warning("llm_view.json is empty. Will regenerate...")
                should_regenerate = True
            else:
                logger.info(f"Found existing llm_view.json with {len(existing_data)} documents")
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"llm_view.json is invalid ({e}). Will regenerate...")
        should_regenerate = True

if should_regenerate:
    logger.info("=" * 60)
    logger.info("Starting LLM View Extraction...")
    logger.info("=" * 60)
    extractor = MetadataExtractor(config)
    data = extractor.extract_metadata(projectRoot / "data")
    
    if not data:
        logger.error("No metadata extracted! Check if PDFs exist in data/ directory")
        exit(1)
    
    with open(llm_view_path, "w") as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"✓ Saved {len(data)} document metadata to llm_view.json")
    logger.info("=" * 60)

# 2. Run Deterministic Logic
logger.info("\nAnalyzing extracted metadata...")
engine = InsightEngine(str(llm_view_path))
missing_skills = engine.find_missing_skills()
tone_analysis = engine.analyze_tone_consistency()

logger.info(f"Skills in papers: {len(missing_skills['presentInPapers'])}")
logger.info(f"Missing from resume: {len(missing_skills['missingFromResume'])}")
logger.info(f"Tone analysis: {tone_analysis}")

# 3. Generate the Final Insight (Safe LLM Call)
# We only feed the LLM the small, verified list of missing skills.
if missing_skills['missingFromResume']:
    logger.info("\nGenerating AI critique...")
    client = LLMChat(config)
    prompt = f"""
Facts derived from analysis:
1. The user has proven experience in these skills (from papers): {missing_skills['presentInPapers']}
2. The user FAILED to list these skills on their resume: {missing_skills['missingFromResume']}
3. Average Tone: {tone_analysis}

Task:
Write a harsh critique of the resume based ONLY on these facts. 
Explain why missing {missing_skills['missingFromResume']} is a critical mistake.
"""

    response = client.chat(messages=[{"role": "user", "content": prompt}])

    print("\n" + "=" * 60)
    print("AI INSIGHT")
    print("=" * 60)
    print(response)
    print("=" * 60)
else:
    logger.info("\n✓ No missing skills found - Resume appears complete!")
    print("\n✓ Analysis Complete: No critical issues found with resume skills coverage.")