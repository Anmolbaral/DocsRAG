import tomllib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class EmbeddingConfig:
    model: str

@dataclass
class VectorDBConfig:
    dim: int

@dataclass
class GenerationConfig:
    model: str

@dataclass
class ChunkingConfig:
    chunkSize: int
    overlap: int
    minChunkChars: int

@dataclass
class RetrievalConfig:
    vectorTopK: int
    bm25TopK: int
    rerankTopK: int
    contextTopK: int

@dataclass
class RerankerConfig:
    model: str
    topK: int

@dataclass
class ConversationConfig:
    maxHistory: int
    systemPrompt: str

@dataclass
class CategoryEmbeddingConfig:
    resume: str
    cover_letters: str
    misc_docs: str

@dataclass
class Config:
    embedding: EmbeddingConfig
    vectorDB: VectorDBConfig
    generation: GenerationConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    reranker: RerankerConfig
    conversation: ConversationConfig
    categoryEmbedding: CategoryEmbeddingConfig

    @classmethod
    def from_file(cls, configPath: Path) -> "Config":
        if not configPath.exists():
            projectRoot = Path(__file__).parent
            configPath = projectRoot / "config.toml"
        
        try:
            with open(configPath, "rb") as f:
                config = tomllib.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load config from {configPath}: {e}")

        return cls(
            embedding=EmbeddingConfig(**config["embedding"]),
            vectorDB=VectorDBConfig(**config["vectorDB"]),
            generation=GenerationConfig(**config["generation"]),
            chunking=ChunkingConfig(**config["chunking"]),
            retrieval=RetrievalConfig(**config["retrieval"]),
            reranker=RerankerConfig(**config["reranker"]),
            conversation=ConversationConfig(**config["conversation"]),
            categoryEmbedding=CategoryEmbeddingConfig(**config["categoryEmbedding"]),
        )