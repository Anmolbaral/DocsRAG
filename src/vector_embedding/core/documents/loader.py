import fitz
import re
import os
import unicodedata
from config import Config
from typing import List, Dict, Any


# Load PDF and extract text per page (no chunking)
# Returns list of dicts with "text" (full page text) and "metadata" (page, filename, category).
def load_pdf(path: str) -> List[Dict[str, Any]]:
    try:
        if os.path.getsize(path) == 0:
            return []

        category = path.split("/")[-2]
        filename = path.split("/")[-1]

        doc = fitz.open(path)
        pages = []

        for pageNum in range(len(doc)):
            page = doc[pageNum]
            text = page.get_text("text") or ""
            text = text.strip()
            
            if not text:
                continue

            # Clean the text
            text = clean_text(text)

            pages.append(
                {
                    "text": text,
                    "metadata": {
                        "page": pageNum + 1,
                        "path": path,
                        "category": category,
                        "filename": filename,
                    },
                }
            )
        
        doc.close()
        return pages

    except Exception as e:
        print(f"Error loading file {path}: {e}")
        return []


# Chunk text into overlapping chunks
# Returns list of dicts with "text" (chunk text) and "metadata" (includes chunkId).
def chunk_text(
    pages: List[Dict[str, Any]], 
    config: Config,
    chunkSize: int = None,
    overlap: int = None,
    minChunkChars: int = None
) -> List[Dict[str, Any]]:
    # Use config defaults if not provided
    if chunkSize is None:
        chunkSize = config.chunking.chunkSize
    if overlap is None:
        overlap = config.chunking.overlap
    if minChunkChars is None:
        minChunkChars = config.chunking.minChunkChars
    
    allChunks = []

    for page in pages:
        pageText = page["text"]
        pageMetadata = page["metadata"]

        chunkTexts = create_overlap_chunks(
            pageText, chunkSize, overlap
        )

        for chunkIndex, chunkText in enumerate(chunkTexts):
            chunkText = chunkText.strip()

            # Drop junk/tiny chunks
            if len(chunkText) < minChunkChars:
                continue

            # Create metadata for this chunk
            chunkMetadata = pageMetadata.copy()
            chunkMetadata["chunkId"] = chunkIndex

            allChunks.append(
                {
                    "text": chunkText,
                    "metadata": chunkMetadata,
                }
            )
    
    return allChunks


# Convenience function: Load PDF and chunk it (backward compatibility)
def load_and_chunk_pdf(
    path: str, 
    config: Config,
    chunkSize: int = None,
    overlap: int = None,
    minChunkChars: int = None
) -> List[Dict[str, Any]]:
    pages = load_pdf(path)
    return chunk_text(pages, config, chunkSize, overlap, minChunkChars)


# Clean text to remove common PDF formatting issues and normalize
# the text coming from the PDF before chunking them
def clean_text(text: str) -> str:
    # Remove zero-width characters (common in PDFs)
    text = re.sub(
        r"[\u2013\u2019\u200B\u200C\u200D\uFEFF\u200b\u200c\u200d\n\t\r]", "", text
    )

    # Normalize unicode (fix weird characters)
    text = unicodedata.normalize("NFKC", text)

    # Replace common PDF bullets with "-"
    text = text.replace("\u25cf", "-").replace("•", "-").replace("●", "-")

    # Fix hyphenated line breaks: "inter-\nnational" -> "international"
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Convert newlines to spaces (after fixing hyphen breaks)
    text = re.sub(r"\s*\n\s*", " ", text)

    # Collapse spaced-out letters: "a n m o l" -> "anmol"
    # This targets sequences of single letters separated by spaces.
    def _collapse_spaced_letters(match):
        return match.group(0).replace(" ", "")

    text = re.sub(
        r"(?:\b[A-Za-z]\b\s+){3,}\b[A-Za-z]\b", _collapse_spaced_letters, text
    )

    # Remove common leading junk like "pp" (PDF artifact)
    text = re.sub(r"^(pp|p)\b\s*", "", text, flags=re.IGNORECASE)

    # Collapse multiple spaces
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text


# Create overlapping chunks from raw text. Chunks by words with overlap.
def create_overlap_chunks(text: str, chunkSize: int, overlap: int) -> List[str]:
    # Split text into words
    words = text.split()

    if len(words) <= chunkSize:
        return [text]

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunkSize
        chunkWords = words[start:end]
        chunks.append(" ".join(chunkWords))
        start += chunkSize - overlap

        if start >= len(words):
            break

    return chunks
