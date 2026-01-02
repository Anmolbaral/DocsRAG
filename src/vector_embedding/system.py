from .modules.rag import RAGPipeline
from .modules.loader import load_pdf
from .modules.cache_manager import CacheManager
from pathlib import Path
from config import Config


class DocumentRAGSystem:
    def __init__(
        self,
        embedder=None,
        chatClient=None,
        cacheDir="cache",
        config: Config = None,
        dataDir="data",
    ):
        self.ragPipeline = None
        self.cacheManager = CacheManager(cacheDir, dataDir)
        self._embedder = embedder
        self._chatClient = chatClient

        if config is None:
            projectRoot = Path(__file__).parent.parent.parent
            configPath = projectRoot / "config.toml"
            self.config = Config.from_file(configPath)
        else:
            self.config = config

    def initialize(self):
        """Initialize RAGPipeline once at startup"""
        fileChanges = self.cacheManager.get_file_changes()
        if (
            fileChanges["isValid"]
            and not fileChanges["changedFiles"]
            and not fileChanges["newFiles"]
            and not fileChanges["removedFiles"]
        ):
            print("Using cached embeddings")
            self.ragPipeline = RAGPipeline.from_cache(
                config=self.config,
                cacheFile=f"{self.cacheManager.cacheDir}/embeddings.json",
                embedder=self._embedder,
                chatClient=self._chatClient,
            )
        elif fileChanges["isValid"] and (
            fileChanges["changedFiles"]
            or fileChanges["newFiles"]
            or fileChanges["removedFiles"]
        ):
            print(
                f"Updating embeddings for {len(fileChanges['changedFiles']) + len(fileChanges['newFiles'])} changed file(s)"
            )
            self.ragPipeline = self.incremental_update(fileChanges)
        else:
            print("Using new embeddings")
            self.ragPipeline = self.data_pipeline()

    def data_pipeline(self):
        """Process all files in the data directory"""
        allTexts = []
        fileMetadata = {}

        try:
            for datafile in self.cacheManager.dataDir.rglob("*.pdf"):
                datafile = str(datafile)
                docs = load_pdf(datafile, config=self.config)
                fileMetadata[datafile] = self.cacheManager.get_file_metadata_for_path(
                    datafile
                )
                allTexts.extend(docs)

            self.cacheManager.save_file_metadata(fileMetadata)

            return RAGPipeline(
                allTexts,
                embedder=self._embedder,
                config=self.config,
                chatClient=self._chatClient,
                cacheFile=f"{self.cacheManager.cacheDir}/embeddings.json",
            )
        except Exception as e:
            print(f"Error processing files: {e}")
            raise e

    def query(self, query: str) -> str:
        """Process a query and return the answer"""
        if not self.ragPipeline:
            raise ValueError("System not initialized. Call initialize() first.")
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        return self.ragPipeline.ask(query.strip())

    def incremental_update(self, fileChanges):
        """Update embeddings only for changed files"""
        cacheData = self.cacheManager.load_embeddings_cache()

        existingChunks = cacheData["chunks"]
        existingEmbeddings = cacheData["embeddings"]

        for i, chunk in enumerate(existingChunks):
            chunkPath = chunk["metadata"]["path"]
            if (
                chunkPath in fileChanges["changedFiles"]
                or chunkPath in fileChanges["newFiles"]
            ):
                chunk["metadata"].update(
                    self.cacheManager.get_file_metadata_for_path(chunkPath)
                )

        self.cacheManager.update_file_metadata(fileChanges)

        return RAGPipeline(
            existingChunks,
            config=self.config,
            embedder=self._embedder,
            chatClient=self._chatClient,
            cacheFile=f"{self.cacheManager.cacheDir}/embeddings.json",
        )
