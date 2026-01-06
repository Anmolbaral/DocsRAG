# Personal Document RAG System

Retrieval-Augmented Generation (RAG) system for intelligent question-answering over personal documents using hybrid search (BM25 + Vector embeddings), FAISS vector database, and cross-encoder reranking.

## Features

- **Hybrid Search**: Combines BM25 keyword search with semantic vector search for improved retrieval
- **Multi-Provider Support**: Works with OpenAI and Ollama for both embeddings and LLM generation
- **Service-Based Architecture**: Clean, modular design with dedicated services for embeddings, reranking, and chat
- **Multi-Document Processing**: Automatically processes all PDF files in the data directory
- **Intelligent Caching**: Incremental updates only process changed files, reducing processing time
- **Configuration-Driven**: Everything configured through `config.toml` - no code changes needed to switch providers
- **Text Chunking**: Word-based overlapping chunks with configurable size and overlap
- **Vector Search**: FAISS IndexHNSWFlat for approximate nearest neighbor search
- **BM25 Search**: Keyword-based retrieval using rank-bm25 for exact term matching
- **Reranking**: Cross-encoder reranker improves retrieval relevance
- **Metadata Tracking**: Maintains source information (filename, page number, category)
- **Terminal Chat Interface**: Interactive command-line chat for real-time Q&A with query timing
- **Text Cleaning**: Handles PDF formatting issues with regex preprocessing
- **Conversation History**: Maintains context across multiple queries



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

4. Set up environment variables (if using OpenAI)
   Create a `.env` file in the root directory:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```
   **Note**: If using Ollama, no API key is needed - just ensure Ollama is running locally.

5. Configure the system
   Edit `config.toml` to customize:
   - Embedding provider and model (OpenAI or Ollama)
   - LLM provider and model (OpenAI or Ollama)
   - Chunk sizes, retrieval parameters, etc.
   
   See the [Configuration](#configuration) section below for details.

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

All configuration is managed through `config.toml` in the project root. The system automatically loads this file on initialization. **No code changes needed** - just edit the config file to switch providers, models, or adjust parameters.

### Configuration File Structure

```toml
# LLM Configuration (for generating answers)
[llm]
provider = "openai"      # Options: "openai" or "ollama"
model = "gpt-4o-mini"    # Model name (e.g., "gpt-4o-mini", "llama3.1:8b")

# Embedding Configuration (for vector search)
[embedding]
provider = "openai"                    # Options: "openai" or "ollama"
model = "text-embedding-3-small"       # Model name

# Vector Database Configuration
[vectorDB]
dim = 1536  # Vector dimension - MUST match your embedding model's output size
            # text-embedding-3-small: 1536
            # text-embedding-3-large: 3072
            # nomic-embed-text: 768

# Text Chunking Configuration
[chunking]
chunkSize = 300        # Number of words per chunk
overlap = 60           # Number of overlapping words between chunks
minChunkChars = 150    # Minimum characters required for a chunk

# Retrieval Configuration
[retrieval]
vectorTopK = 20        # Number of vector search results
bm25TopK = 20          # Number of BM25 keyword search results
rerankTopK = 10        # Number of candidates to rerank
contextTopK = 5        # Final number of results sent to LLM

# Reranker Configuration
[reranker]
model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
topK = 10

# Conversation Configuration
[conversation]
maxHistory = 10       # Maximum conversation history entries
systemPrompt = "You are a helpful assistant that can only answer questions from the context provided. Use conversation history to understand the references and context"
```

### Provider Examples

#### Using OpenAI (requires API key)
```toml
[llm]
provider = "openai"
model = "gpt-4o-mini"

[embedding]
provider = "openai"
model = "text-embedding-3-small"

[vectorDB]
dim = 1536  # Matches text-embedding-3-small
```

#### Using Ollama (local, no API key needed)
```toml
[llm]
provider = "ollama"
model = "llama3.1:8b"

[embedding]
provider = "ollama"
model = "nomic-embed-text"

[vectorDB]
dim = 768  # Matches nomic-embed-text
```

#### Mixed Setup (OpenAI embeddings + Ollama LLM)
```toml
[llm]
provider = "ollama"
model = "llama3.1:8b"

[embedding]
provider = "openai"
model = "text-embedding-3-small"

[vectorDB]
dim = 1536
```

### How Retrieval Works

1. **BM25 Search**: Retrieves top `bm25TopK` candidates based on keyword matching
2. **Vector Search**: Retrieves top `vectorTopK` candidates based on semantic similarity
3. **Merge & Deduplicate**: Combines results from both methods, removing duplicates
4. **Reranking**: Cross-encoder reranks the top `rerankTopK` merged candidates
5. **Context Selection**: Top `contextTopK` reranked results are used as context for the LLM

### Embedding Models & Dimensions

**Important**: The `vectorDB.dim` value must match your embedding model's output dimension.

#### OpenAI Models
- `text-embedding-3-small`: **1536 dimensions**
- `text-embedding-3-large`: **3072 dimensions**

#### Ollama Models
- `nomic-embed-text`: **768 dimensions**
- Other Ollama embedding models: Check model documentation for dimensions

**Remember**: Always set `[vectorDB] dim` to match your embedding model's output size, or you'll get errors when initializing the vector database!

### Cache Management

The system includes intelligent caching:
- **Incremental Updates**: Only processes files that have changed
- **File Tracking**: Monitors file modifications using MD5 hashes
- **Automatic Detection**: Identifies new, changed, and removed files
- **Cache Validation**: Ensures cache consistency with current files

Cache files are stored in the `cache/` directory:
- `embeddings.json`: Cached embeddings and chunks
- `file_metadata.json`: File modification tracking

## Architecture

The system uses a **service-based architecture** where each component manages its own configuration:

- **EmbeddingService**: Handles embedding generation (OpenAI or Ollama)
- **RerankerService**: Manages cross-encoder reranking
- **BM25Index**: Provides keyword-based search
- **LLMChat**: Handles LLM interactions (OpenAI or Ollama)
- **VectorDB**: FAISS-based vector storage and search

All services receive the config object during initialization and use it internally - no need to pass config parameters around!

## Project Structure

```
Vector Embedding/
├── config.toml              # Configuration file (edit this to change behavior)
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
    ├── system.py            # Main system class (DocumentRAGSystem)
    └── modules/
        ├── rag.py           # RAG pipeline (orchestrates services)
        ├── loader.py        # PDF loading and chunking
        ├── embeddings.py    # EmbeddingService class
        ├── reranker.py      # RerankerService class
        ├── llm_chat_client.py # LLMChat class
        ├── vectordb.py      # Vector database
        ├── bm25.py          # BM25Index class
        ├── cache_manager.py # Cache management
        └── hashing.py       # File hashing utilities
```

## Dependencies

### Core Dependencies
- **faiss-cpu**: Vector similarity search
- **openai**: OpenAI API client (for OpenAI provider)
- **ollama**: Ollama API client (for Ollama provider, optional)
- **sentence-transformers**: Cross-encoder reranking
- **rank-bm25**: BM25 keyword search implementation
- **numpy**: Numerical computations
- **python-dotenv**: Environment variable management
- **PyMuPDF (fitz)**: Advanced PDF text extraction

### Optional Dependencies
- **tenacity**: Retry logic
- **ragas**: RAG evaluation
- **langchain**: Framework components
- **langchain-openai**: LangChain OpenAI integration
- **langsmith**: LangChain monitoring

**Note**: If you only use Ollama, you don't need the `openai` package. If you only use OpenAI, you don't need `ollama`.

## Recent Improvements

### Service-Based Architecture
- **Cleaner code**: Each service (EmbeddingService, RerankerService, LLMChat) manages its own config
- **No parameter passing**: Services access config internally, reducing function signatures
- **Better organization**: Each module has a single responsibility

### Multi-Provider Support
- **Flexible**: Switch between OpenAI and Ollama via config file
- **No code changes**: Just edit `config.toml` to change providers
- **Mixed setups**: Use OpenAI for embeddings and Ollama for LLM (or vice versa)

### Configuration-Driven
- **Everything in config**: All behavior controlled through `config.toml`
- **Easy experimentation**: Try different models without touching code
- **Clear documentation**: Config file serves as documentation

## Code Conventions

- **Function names**: `snake_case` (e.g., `get_embedding_single`, `load_pdf`)
- **Variable names**: `camelCase` (e.g., `allChunks`, `fileMetadata`)
- **Class names**: `PascalCase` (e.g., `RAGPipeline`, `DocumentRAGSystem`)
- **Service classes**: End with "Service" (e.g., `EmbeddingService`, `RerankerService`)

## License

This project is licensed under the MIT License.
