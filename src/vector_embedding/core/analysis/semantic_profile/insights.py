import json
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InsightEngine:
    def __init__(self, json_path="llm_view.json"):
        """
        Initialize InsightEngine with metadata from JSON file.
        
        Args:
            json_path: Path to the JSON file containing document metadata
        """
        logger.info(f"Loading metadata from: {json_path}")
        
        with open(json_path, "r") as f:
            self.data = json.load(f)
        
        self.df = pd.DataFrame(self.data)
        
        if self.df.empty:
            logger.warning("DataFrame is empty - no documents to analyze")
        else:
            logger.info(f"Loaded {len(self.df)} documents")
            if 'docType' in self.df.columns:
                doc_types = self.df['docType'].value_counts().to_dict()
                logger.info(f"Document types: {doc_types}")

    def find_missing_skills(self):
        """
        Find skills that appear in research papers but not in resume.
        
        Logic: Skills in Papers - Skills in Resume = The Gap
        
        Returns:
            Dictionary with 'presentInPapers' and 'missingFromResume' lists
        """
        logger.info("Analyzing skill gaps...")
        
        # Check if DataFrame is empty or missing required columns
        if self.df.empty:
            logger.warning("No data to analyze - empty DataFrame")
            return {
                "presentInPapers": [],
                "missingFromResume": []
            }
        
        if 'docType' not in self.df.columns or 'hardSkills' not in self.df.columns:
            logger.error(f"Missing required columns. Available: {list(self.df.columns)}")
            return {
                "presentInPapers": [],
                "missingFromResume": []
            }
        
        # 1. Get all skills mentioned in Papers
        paper_skills = set()
        paper_docs = self.df[self.df['docType'] == "research_paper"]
        
        if not paper_docs.empty:
            logger.info(f"Found {len(paper_docs)} research paper(s)")
            for skills in paper_docs['hardSkills']:
                if isinstance(skills, list):
                    paper_skills.update([s.lower() for s in skills])
            logger.info(f"Extracted {len(paper_skills)} unique skills from papers")
        else:
            logger.info("No research papers found")

        # 2. Get all skills mentioned in Resume
        resume_skills = set()
        resume_docs = self.df[self.df['docType'] == 'resume']
        
        if not resume_docs.empty:
            logger.info(f"Found {len(resume_docs)} resume(s)")
            for skills in resume_docs['hardSkills']:
                if isinstance(skills, list):
                    resume_skills.update([s.lower() for s in skills])
            logger.info(f"Extracted {len(resume_skills)} unique skills from resume")
        else:
            logger.warning("No resume found!")

        # 3. STRICT MATH: What is in A but not B?
        missing = paper_skills - resume_skills
        
        if missing:
            logger.warning(f"Found {len(missing)} skills missing from resume: {sorted(list(missing))[:10]}{'...' if len(missing) > 10 else ''}")
        else:
            logger.info("✓ All paper skills are present in resume")
        
        return {
            "presentInPapers": sorted(list(paper_skills)),
            "missingFromResume": sorted(list(missing))
        }

    def analyze_tone_consistency(self):
        """
        Compare tone scores across different document types.
        
        Returns:
            Dictionary mapping document types to average tone scores
        """
        logger.info("Analyzing tone consistency...")
        
        if self.df.empty or 'docType' not in self.df.columns or 'toneScore' not in self.df.columns:
            logger.warning("Cannot analyze tone - missing required columns")
            return {}
        
        tone_by_type = self.df.groupby('docType')['toneScore'].mean().to_dict()
        
        for doc_type, score in tone_by_type.items():
            logger.info(f"  {doc_type}: {score:.1f}/10")
        
        return tone_by_type