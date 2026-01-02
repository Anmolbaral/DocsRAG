import os
import json
from pathlib import Path
from .hashing import get_file_hash


class CacheManager:
    """Manages file metadata cache and detects file changes"""
    
    def __init__(self, cacheDir: str, dataDir: str):
        self.cacheDir = cacheDir
        self.dataDir = Path(dataDir)
        os.makedirs(self.cacheDir, exist_ok=True)
    
    def get_file_changes(self):
        """Check which files have changed, are new, or were removed"""
        changedFiles = []
        newFiles = []
        removedFiles = []
        isValid = True

        try:
            metadataPath = f"{self.cacheDir}/file_metadata.json"
            embeddingsPath = f"{self.cacheDir}/embeddings.json"
            
            if not os.path.exists(metadataPath) or not os.path.exists(embeddingsPath):
                return {
                    "isValid": False,
                    "changedFiles": [],
                    "newFiles": [],
                    "removedFiles": [],
                }

            if os.path.getsize(metadataPath) == 0 or os.path.getsize(embeddingsPath) == 0:
                return {
                    "isValid": False,
                    "changedFiles": [],
                    "newFiles": [],
                    "removedFiles": [],
                }

            with open(metadataPath, "r") as f:
                cacheMetadata = json.load(f)

            currentFiles = {str(file) for file in self.dataDir.rglob("*.pdf")}
            cachedFiles = set(cacheMetadata.keys())

            removedFiles = list(cachedFiles - currentFiles)

            for file in currentFiles:
                file = str(file)
                if file not in cacheMetadata.keys():
                    newFiles.append(file)
                elif get_file_hash(file) != cacheMetadata[file]["fileHash"]:
                    changedFiles.append(file)

            return {
                "isValid": True,
                "changedFiles": changedFiles,
                "newFiles": newFiles,
                "removedFiles": removedFiles,
            }
        except Exception as e:
            print(f"Error checking file changes: {e}")
            return {
                "isValid": False,
                "changedFiles": [],
                "newFiles": [],
                "removedFiles": [],
            }
    
    def load_embeddings_cache(self):
        """Load embeddings from cache file"""
        with open(f"{self.cacheDir}/embeddings.json", "r") as f:
            return json.load(f)
    
    def save_file_metadata(self, fileMetadata: dict):
        """Save file metadata to cache"""
        os.makedirs(self.cacheDir, exist_ok=True)
        with open(f"{self.cacheDir}/file_metadata.json", "w") as f:
            json.dump(fileMetadata, f, indent=2)
    
    def load_file_metadata(self):
        """Load file metadata from cache"""
        try:
            with open(f"{self.cacheDir}/file_metadata.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def update_file_metadata(self, fileChanges: dict):
        """Update file metadata for changed files"""
        fileMetadata = self.load_file_metadata()
        
        for file in fileChanges["changedFiles"] + fileChanges["newFiles"]:
            fileMetadata[file] = {
                "fileSize": os.path.getsize(file),
                "fileModifiedTime": os.path.getmtime(file),
                "fileHash": get_file_hash(file),
            }
        
        self.save_file_metadata(fileMetadata)
    
    def get_file_metadata_for_path(self, filePath: str):
        """Get metadata for a single file"""
        return {
            "fileSize": os.path.getsize(filePath),
            "fileModifiedTime": os.path.getmtime(filePath),
            "fileHash": get_file_hash(filePath),
        }

