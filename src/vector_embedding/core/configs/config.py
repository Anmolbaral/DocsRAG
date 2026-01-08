import tomllib
from pathlib import Path
from dataclasses import dataclass


@dataclass
class LLMConfig:
    provider: str
    parseModel: str
    model: str


@dataclass
class EmbeddingConfig:
    provider: str
    model: str


@dataclass
class VectorDBConfig:
    dim: int


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
class Config:
    llm: LLMConfig
    embedding: EmbeddingConfig
    vectorDB: VectorDBConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    reranker: RerankerConfig
    conversation: ConversationConfig

    @classmethod
    def from_file(cls, configPath: Path) -> "Config":
        if configPath is None:
            # Auto-find config.toml in project root
            projectRoot = Path(__file__).parent
            configPath = projectRoot / "config.toml"
        elif isinstance(configPath, str):
            configPath = Path(configPath)

        try:
            with open(configPath, "rb") as f:
                config = tomllib.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load config from {configPath}: {e}")

        return cls(
            llm=LLMConfig(**config["llm"]),
            embedding=EmbeddingConfig(**config["embedding"]),
            vectorDB=VectorDBConfig(**config["vectorDB"]),
            chunking=ChunkingConfig(**config["chunking"]),
            retrieval=RetrievalConfig(**config["retrieval"]),
            reranker=RerankerConfig(**config["reranker"]),
            conversation=ConversationConfig(**config["conversation"]),
        )
