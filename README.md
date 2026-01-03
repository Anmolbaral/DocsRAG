# Personal Document RAG System

Retrieval-Augmented Generation (RAG) system for intelligent question-answering over personal documents using hybrid search (BM25 + Vector embeddings), FAISS vector database, and cross-encoder reranking.

## Features

- **Hybrid Search**: Combines BM25 keyword search with semantic vector search for improved retrieval
- **Multi-Document Processing**: Automatically processes all PDF files in the data directory
- **Text Chunking**: Word-based overlapping chunks (500 words, 100 overlap) for context preservation
- **Vector Search**: FAISS IndexHNSWFlat for approximate nearest neighbor search
- **BM25 Search**: Keyword-based retrieval using rank-bm25 for exact term matching
- **Reranking**: Cross-encoder reranker improves retrieval relevance
- **Metadata Tracking**: Maintains source information (filename, page number, category)
- **Terminal Chat Interface**: Interactive command-line chat for real-time Q&A
- **Text Cleaning**: Handles PDF formatting issues with regex preprocessing
- **Embedding Model**: Uses "text-embedding-3-large" with 3072 dimensions



## Installation

1. Clone the repository
   ```bash
   git clone <your-repo-url>
   cd Vector\ Embedding
   ```

2. Create and activate virtual environment
   ```bash
   python -m venv vector_venv
   source vector_venv/bin/activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables
   Create a `.env` file in the root directory:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. Add your documents
   Place your PDF files in the `data/` directory

## Usage

### Terminal Chat Interface (Recommended)

Run the interactive chat interface:

```bash
python chat.py
```

Commands:
- Type your question and press Enter
- Type `quit` or `exit` to exit
- Type `clear` to clear the screen

### Programmatic Usage

Import and use as a library:

```python
from src.vector_embedding.system import DocumentRAGSystem

system = DocumentRAGSystem()
system.initialize()
answer = system.query("What is your question?")
```

## Configuration

### Chunking Parameters
```python
chunkSize = 500    # Number of words per chunk
overlap = 100      # Number of overlapping words between chunks
```

### Search Parameters
```python
# Hybrid search configuration
BM25_K = 20          # Number of BM25 keyword search results
VECTOR_K = 20        # Number of vector semantic search results
RERANK_TOP_N = 10    # Final number of reranked results sent to LLM

# LLM Model
model = "gpt-4o-mini"
```

**How it works:**
1. BM25 retrieves top 20 candidates based on keyword matching
2. Vector search retrieves top 20 candidates based on semantic similarity
3. Results are merged and deduplicated
4. Cross-encoder reranks the merged candidates
5. Top 10 reranked results are used as context for the LLM

### Embedding Model
```python
model = "text-embedding-3-small"
```

Note: The VectorDB dimension is hardcoded to 3072 to match text-embedding-3-large. If changing the embedding model, you must also update `VectorDB(dim=...)` in `rag.py` to match the model's output dimensions (e.g., 1536 for text-embedding-3-small).

### Reranker Model
```python
model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

## Dependencies

- **faiss-cpu**: Vector similarity search
- **openai**: Embedding generation and LLM inference
- **sentence-transformers**: Cross-encoder reranking
- **rank-bm25**: BM25 keyword search implementation
- **numpy**: Numerical computations
- **scikit-learn**: Similarity calculations
- **python-dotenv**: Environment variable management
- **pypdf**: PDF text extraction
- **PyMuPDF (fitz)**: Advanced PDF text extraction
- **tenacity**: Retry logic
- **ragas**: RAG evaluation
- **langchain**: Framework components
- **langchain-openai**: LangChain OpenAI integration
- **langsmith**: LangChain monitoring

## License

This project is licensed under the MIT License.
