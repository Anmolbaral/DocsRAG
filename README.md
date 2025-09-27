# Personal Document RAG System

I built this Retrieval-Augmented Generation (RAG) system built with Python that enables intelligent question-answering over personal documents using OpenAI embeddings and FAISS vector search.

## Features

- Multi-Document Processing: Automatically processes all PDF files in the data directory
- Advanced Text Chunking: Implements sentence-based overlapping chunks for better context preservation
- Vector Search: Uses FAISS (Facebook AI Similarity Search) with L2 distance for efficient similarity search
   - Uses FAISS IndexHNSWFlat for approximate nearest neighbor search with HNSW graph structure
- Metadata Tracking: Maintains source information (filename, page number) for transparency
   - Stored in JSON format with nested structures for readability and easy debugging 
- Interactive Q&A: Real-time question answering with source attribution
- Smart Text Cleaning: Handles PDF formatting issues with regex-based preprocessing
- **Embedding Model**: Uses "text-embedding-3-large" with 3072 dimensions for enhanced semantic understanding
- **Alternative**: Can use "text-embedding-3-small" (1536 dimensions) for lower API costs



## Installation

1. Clone the repository
   ```bash
   git clone <your-repo-url>
   cd Vector\ Embedding
   ```

2. Create and activate virtual environment
   ```bash
   python -m venv vector_venv
   source vector_venv/bin/activate  # On Windows: vector_venv\Scripts\activate
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

1. Start the system
   ```bash
   python -m src.vector_embedding.main
   ```

2. Ask questions
   ```
   Ask a question: What is Anmol's work experience?
   ```

3. View results with source attribution
   ```
   Page 1 - resume.pdf: I worked at Apple as an Engineering Intern...
   Page 2 - context1.pdf: During my internship at Tufts University...
   ```

## Configuration

### Chunking Parameters
```python
# In modules/loader.py
chunkSize = 5    # Number of sentences per chunk
overlap = 2      # Number of overlapping sentences
```

### Search Parameters
```python
# In modules/rag.py
k = 5           # Number of chunks to retrieve
model = "gpt-4o-mini"  # LLM model for response generation
```

### Embedding Model
```python
# In modules/embeddings.py
model = "text-embedding-3-large"  # OpenAI embedding model (3072 dimensions)
```

## Dependencies

- faiss-cpu: Vector similarity search
- openai: Embedding generation and LLM inference
- numpy: Numerical computations
- python-dotenv: Environment variable management
- pypdf: PDF text extraction

## License

This project is licensed under the MIT License.
