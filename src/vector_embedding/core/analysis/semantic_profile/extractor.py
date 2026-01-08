import json
import os
import logging
from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from ...documents.loader import load_pdf
from ...config.config import Config
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompanyProfile(BaseModel):
    targetCompany: Optional[str] = Field(None, description="Company applied to, if clear")
    companydescription: Optional[str] = Field(None, description="Description of the company, if clear")
    companyLocation: Optional[str] = Field(None, description="Location of the company, if clear")
    companyIndustry: Optional[str] = Field(None, description="Industry of the company, if clear")

class DocumentMetadata(BaseModel):
    filename: str
    docType: Literal["resume", "cover_letter", "research_paper", "misc"]
    companyProfile: Optional[CompanyProfile] = Field(None, description="Company profile, if clear")
    # We force the LLM to extract lists, not paragraphs
    hardSkills: List[str] = Field(..., description="e.g. Python, Docker, CUDA")
    softSkills: List[str] = Field(..., description="e.g. Leadership, Mentoring, Hardworking, Passionate, Problem Solver")
    keyClaims: List[str] = Field(..., description="Specific facts claimed, e.g. 'Reduced latency by 50%'")
    toneScore: int = Field(..., description="1 (Timid) to 10 (Arrogant)")

class CareerVault(BaseModel):
    documents: List[DocumentMetadata]

class MetadataExtractor:
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = config.llm.model
        self.loader = load_pdf
    
    def extract_metadata(self, dataDir: str) -> List[Dict[str, Any]]:
        """
        Extract structured metadata from all PDF files in the given directory.
        
        Args:
            dataDir: Path to directory containing PDFs (can be relative or absolute)
            
        Returns:
            List of dictionaries containing extracted metadata
        """
        metadataList = []
        
        # Resolve path
        dataPath = Path(dataDir)
        if not dataPath.is_absolute():
            dataPath = Path(__file__).parent.parent.parent.parent / dataDir
        
        logger.info(f"Scanning for PDFs in: {dataPath}")
        
        # Find all PDFs
        pdf_files = list(dataPath.rglob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {dataPath}")
            return metadataList
        
        # Process each PDF
        for idx, pdfFile in enumerate(pdf_files, 1):
            logger.info(f"Processing [{idx}/{len(pdf_files)}]: {pdfFile.name}")
            
            try:
                # Load and extract text
                cleanedDocument = self.loader(str(pdfFile))
                
                if not cleanedDocument:
                    logger.warning(f"No text extracted from {pdfFile.name}, skipping...")
                    continue
                
                fullText = " ".join([page["text"] for page in cleanedDocument])
                
                if not fullText.strip():
                    logger.warning(f"Empty text from {pdfFile.name}, skipping...")
                    continue
                
                logger.info(f"Extracted {len(fullText)} characters from {pdfFile.name}")
                
                # Call OpenAI API for structured extraction
                completion = self.client.beta.chat.completions.parse(
                    model=self.config.llm.parseModel,
                    messages=[
                        {"role": "system", "content": "Extract structured metadata. Be precise."},
                        {"role": "user", "content": f"Filename: {pdfFile.name}\n\nContent:\n{fullText[:8000]}"},  # Limit to avoid token limits
                    ],
                    response_format=DocumentMetadata,
                )
                
                metadata = completion.choices[0].message.parsed
                metadata_dict = metadata.model_dump()
                metadataList.append(metadata_dict)
                
                logger.info(f"✓ Successfully extracted metadata from {pdfFile.name} - Type: {metadata_dict['docType']}")

            except Exception as e:
                logger.error(f"✗ Error extracting metadata from {pdfFile.name}: {e}")
                # Continue processing other files instead of crashing
                continue
        
        logger.info(f"Extraction complete: {len(metadataList)}/{len(pdf_files)} files processed successfully")
        return metadataList

if __name__ == "__main__":
    from pathlib import Path
    projectRoot = Path(__file__).parent.parent.parent.parent
    config = Config.from_file(projectRoot / "config.toml")
    metadata = MetadataExtractor(config).extract_metadata(str(projectRoot / "data"))
    with open(projectRoot / "llm_view.json", "w") as f:
        json.dump(metadata, f, indent=4)
