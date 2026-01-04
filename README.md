# Personal Document RAG System

Retrieval-Augmented Generation (RAG) system for intelligent question-answering over personal documents using hybrid search (BM25 + Vector embeddings), FAISS vector database, and cross-encoder reranking.

## Features

- **Hybrid Search**: Combines BM25 keyword search with semantic vector search for improved retrieval
- **Multi-Document Processing**: Automatically processes all PDF files in the data directory
- **Intelligent Caching**: Incremental updates only process changed files, reducing processing time
- **Configuration Management**: Centralized configuration via `config.toml` for easy customization
- **Text Chunking**: Word-based overlapping chunks with configurable size and overlap
- **Vector Search**: FAISS IndexHNSWFlat for approximate nearest neighbor search
- **BM25 Search**: Keyword-based retrieval using rank-bm25 for exact term matching
- **Reranking**: Cross-encoder reranker improves retrieval relevance
- **Metadata Tracking**: Maintains source information (filename, page number, category)
- **Terminal Chat Interface**: Interactive command-line chat for real-time Q&A with query timing
- **Text Cleaning**: Handles PDF formatting issues with regex preprocessing
- **Conversation History**: Maintains context across multiple queries
- **Dynamic Embedding Dimensions**: Automatically adjusts vector database dimensions based on embedding model



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

5. Configure the system (optional)
   Edit `config.toml` to customize embedding models, chunk sizes, retrieval parameters, etc.
   The system will use sensible defaults if the config file is not found.

6. Add your documents
   Place your PDF files in the `data/` directory (organized in subdirectories like `resume/`, `cover_letters/`, `misc_docs/`)

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

# Initialize system (automatically loads config.toml)
system = DocumentRAGSystem()
system.initialize()

# Query the system
answer = system.query("What is your question?")

# Query with timing information
answer, queryTime = system.query("What is your question?", showTiming=True)
print(f"Query processed in {queryTime:.2f} seconds")
```

### Custom Configuration

You can also pass a custom config:

```python
from src.vector_embedding.system import DocumentRAGSystem
from config import Config
from pathlib import Path

# Load custom config
customConfig = Config.from_file(Path("custom_config.toml"))
system = DocumentRAGSystem(config=customConfig)
system.initialize()
```

## Configuration

All configuration is managed through `config.toml` in the project root. The system automatically loads this file on initialization.

### Configuration File Structure

```toml
[embedding]
model = "text-embedding-3-small"  # OpenAI embedding model

[vectorDB]
dim = 1536  # Vector dimension (automatically matches embedding model)

[generation]
model = "gpt-4o-mini"  # LLM model for response generation

[chunking]
chunkSize = 500        # Number of words per chunk
overlap = 100          # Number of overlapping words between chunks
minChunkChars = 200    # Minimum characters required for a chunk

[retrieval]
vectorTopK = 40        # Number of vector search results
bm25TopK = 40          # Number of BM25 keyword search results
rerankTopK = 25        # Number of candidates to rerank
contextTopK = 5        # Final number of results sent to LLM

[reranker]
model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
topK = 10

[conversation]
maxHistory = 10       # Maximum conversation history entries
systemPrompt = "You are a helpful assistant..."
```

### How Retrieval Works

1. **BM25 Search**: Retrieves top `bm25TopK` candidates based on keyword matching
2. **Vector Search**: Retrieves top `vectorTopK` candidates based on semantic similarity
3. **Merge & Deduplicate**: Combines results from both methods, removing duplicates
4. **Reranking**: Cross-encoder reranks the top `rerankTopK` merged candidates
5. **Context Selection**: Top `contextTopK` reranked results are used as context for the LLM

### Embedding Models

Supported models and their dimensions:
- `text-embedding-3-small`: 1536 dimensions
- `text-embedding-3-large`: 3072 dimensions

The vector database dimension is automatically configured based on the embedding model specified in `config.toml`. No manual dimension updates needed!

### Cache Management

The system includes intelligent caching:
- **Incremental Updates**: Only processes files that have changed
- **File Tracking**: Monitors file modifications using MD5 hashes
- **Automatic Detection**: Identifies new, changed, and removed files
- **Cache Validation**: Ensures cache consistency with current files

Cache files are stored in the `cache/` directory:
- `embeddings.json`: Cached embeddings and chunks
- `file_metadata.json`: File modification tracking

## Project Structure

```
Vector Embedding/
├── config.toml              # Configuration file
├── config.py                # Configuration loader
├── chat.py                  # Interactive chat interface
├── data/                    # Document directory
│   ├── resume/              # Resume PDFs
│   ├── cover_letters/       # Cover letter PDFs
│   └── misc_docs/          # Other documents
├── cache/                   # Cache directory
│   ├── embeddings.json     # Cached embeddings
│   └── file_metadata.json  # File tracking metadata
└── src/vector_embedding/
    ├── system.py            # Main system class
    └── modules/
        ├── rag.py           # RAG pipeline
        ├── loader.py        # PDF loading and chunking
        ├── embeddings.py    # Embedding generation
        ├── vectordb.py      # Vector database
        ├── bm25.py          # BM25 search
        ├── reranker.py      # Cross-encoder reranking
        ├── cache_manager.py # Cache management
        └── hashing.py       # File hashing utilities
```

## Dependencies

Core dependencies:
- **faiss-cpu**: Vector similarity search
- **openai**: Embedding generation and LLM inference
- **sentence-transformers**: Cross-encoder reranking
- **rank-bm25**: BM25 keyword search implementation
- **numpy**: Numerical computations
- **scikit-learn**: Similarity calculations
- **python-dotenv**: Environment variable management
- **PyMuPDF (fitz)**: Advanced PDF text extraction

Optional dependencies:
- **tenacity**: Retry logic
- **ragas**: RAG evaluation
- **langchain**: Framework components
- **langchain-openai**: LangChain OpenAI integration
- **langsmith**: LangChain monitoring

## Code Conventions

- **Function names**: `snake_case` (e.g., `get_embedding_single`, `load_pdf`)
- **Variable names**: `camelCase` (e.g., `allChunks`, `fileMetadata`)
- **Class names**: `PascalCase` (e.g., `RAGPipeline`, `DocumentRAGSystem`)

## License

This project is licensed under the MIT License.
