from pypdf import PdfReader
import fitz
import re
import os
import unicodedata
from vector_embedding.modules.bm25 import BM25Index


# Load and chunk a PDF into overlapping text chunks (5 sentences, 2 overlap).
# Returns list of dicts with "text" and "metadata" (page, filename, category).
def load_pdf(path):
    try:
        if os.path.getsize(path) == 0:
            return []

        category = path.split("/")[-2]
        groundTruth = category == "resume"

        doc = fitz.open(path)
        allChunks = []
        allTexts = []

        for pageNum in range(len(doc)):
            page = doc[pageNum]
            text = page.get_text("text") or ""
            text = text.strip()
            if not text:
                continue

            text = clean_text(text)
            allTexts.append(text)
            chunkTexts = create_overlap_chunks(text, chunkSize=500, overlap=100)

            for chunkIndex, chunkText in enumerate(chunkTexts):
                chunkText = chunkText.strip()

                # drop junk/tiny chunks
                if len(chunkText) < 200:
                    continue

                allChunks.append(
                    {
                        "text": chunkText,
                        "metadata": {
                            "page": pageNum + 1,
                            "path": path,
                            "category": category,
                            "filename": path.split("/")[-1],
                            "isGroundTruth": groundTruth,
                            "chunkId": chunkIndex,
                        },
                    }
                )
        return allChunks

    except Exception as e:
        print(f"Error loading file {path}: {e}")
        return []


# Clean text to remove common PDF formatting issues and normalize
# the text coming from the PDF before chunking them
def clean_text(text: str) -> str:
    # Remove zero-width characters (common in PDFs)
    text = re.sub(r"[\u2013\u2019\u200B\u200C\u200D\uFEFF]", "", text)

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
# chunkSize and overlap are in words. Overlap preserves context across
# chunk boundaries (useful for RAG systems).
def create_overlap_chunks(text, chunkSize=500, overlap=100):
    # Split text into words
    words = text.split()

    if len(words) <= chunkSize:
        return [text]  # Returning whole text if it's smaller than chunk size

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunkSize
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += chunkSize - overlap

        if start >= len(words):
            break

    return chunks
