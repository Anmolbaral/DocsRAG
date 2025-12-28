# Personal Document RAG System

Retrieval-Augmented Generation (RAG) system for intelligent question-answering over personal documents using OpenAI embeddings, FAISS vector search, and cross-encoder reranking.

## Features

- Multi-Document Processing: Automatically processes all PDF files in the data directory
- Text Chunking: Sentence-based overlapping chunks for context preservation
- Vector Search: FAISS IndexHNSWFlat for approximate nearest neighbor search
- Reranking: Cross-encoder reranker improves retrieval relevance
- Metadata Tracking: Maintains source information (filename, page number, category)
- Interactive Q&A: Real-time question answering with source attribution
- Text Cleaning: Handles PDF formatting issues with regex preprocessing
- Embedding Model: Uses "text-embedding-3-large" with 3072 dimensions



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

Import and use as a library:

```python
from vector_embedding import DocumentRAGSystem

system = DocumentRAGSystem()
system.initialize()
answer = system.query("What is your question?")
```

## Configuration

### Chunking Parameters
```python
chunkSize = 5
overlap = 2
```

### Search Parameters
```python
K_retrieve = 50-120
K_context = 3-10
model = "gpt-4o-mini"
```

### Embedding Model
```python
model = "text-embedding-3-large"
```

Note: The VectorDB dimension is hardcoded to 3072 to match text-embedding-3-large. If changing the embedding model, you must also update `VectorDB(dim=...)` in `rag.py` to match the model's output dimensions (e.g., 1536 for text-embedding-3-small).

### Reranker Model
```python
model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

## Dependencies

- faiss-cpu: Vector similarity search
- openai: Embedding generation and LLM inference
- sentence-transformers: Cross-encoder reranking
- numpy: Numerical computations
- scikit-learn: Similarity calculations
- python-dotenv: Environment variable management
- pypdf: PDF text extraction
- tenacity: Retry logic
- ragas: RAG evaluation
- langchain: Framework components
- langchain-openai: LangChain OpenAI integration
- langsmith: LangChain monitoring

## License

This project is licensed under the MIT License.
