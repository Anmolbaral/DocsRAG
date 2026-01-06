from .modules.rag import RAGPipeline
from .modules.loader import load_pdf
from .modules.cache_manager import CacheManager
from pathlib import Path
from config import Config
from typing import Union, Tuple
import time
import os


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
        cacheFile = f"{self.cacheManager.cacheDir}/embeddings.json"

        cacheExists = os.path.exists(cacheFile) and os.path.getsize(cacheFile) > 0

        if (
            cacheExists
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
        elif (
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

    def query(
        self, query: str, showTiming: bool = True
    ) -> Union[Tuple[str, float], str]:
        """Process a query and return the answer"""

        if not self.ragPipeline:
            raise ValueError("System not initialized. Call initialize() first.")

        startTime = time.time()
        answer = self.ragPipeline.ask(query.strip())
        elapsedTime = time.time() - startTime

        if showTiming:
            return answer, elapsedTime
        return answer

    def incremental_update(self, fileChanges):
        """Update embeddings only for changed files"""
        keptChunks, filesToUpdate = self.cacheManager.get_updated_chunks(fileChanges)

        # Load new/changed files
        newChunks = []
        for datafile in self.cacheManager.dataDir.rglob("*.pdf"):
            datafile = str(datafile)
            if datafile in filesToUpdate:
                docs = load_pdf(datafile, config=self.config)
                newChunks.extend(docs)

        # Combine: kept chunks (unchanged files) + new chunks (changed/new files)
        allChunks = keptChunks + newChunks

        self.cacheManager.update_file_metadata(fileChanges)

        return RAGPipeline(
            allChunks,
            config=self.config,
            embedder=self._embedder,
            chatClient=self._chatClient,
            cacheFile=f"{self.cacheManager.cacheDir}/embeddings.json",
        )
