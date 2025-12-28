from .modules.rag import RAGPipeline
from .modules.loader import load_pdf
from .modules.hashing import get_file_hash
import os, json, pathlib

class DocumentRAGSystem:
	def __init__(self, embedder=None, chat_client=None, cacheDir="cache", dataDir="data"):
		self.ragPipeline = None
		self.cacheDir = cacheDir
		self.allFolders = pathlib.Path(dataDir)
		self._embedder = embedder
		self._chat_client = chat_client

	def initialize(self):
		"""Initiializing RAGPipeline once at startup"""
		# Ensure cache directory exists before reading/writing
		os.makedirs(self.cacheDir, exist_ok=True)

		file_changes = self.get_file_changes()
		if file_changes["is_valid"] and not file_changes["changed_files"] and not file_changes["new_files"] and not file_changes["removed_files"]:
			print("Using cached embeddings")
			self.ragPipeline = RAGPipeline.from_cache(cacheFile=f"{self.cacheDir}/embeddings.json", embedder=self._embedder, chat_client=self._chat_client)
		elif file_changes["is_valid"] and (file_changes["changed_files"] or file_changes["new_files"] or file_changes["removed_files"]):
			print(f"Updating embeddings for {len(file_changes['changed_files']) + len(file_changes['new_files'])} changed file(s)")
			self.ragPipeline = self.incremental_update(file_changes)
		else:
			print("Using new embeddings")
			self.ragPipeline = self.data_pipeline()

	
	def data_pipeline(self):
		"""Processing all files in the data directory"""
		allTexts = []
		fileMetadata = {}

		try:
			for datafile in self.allFolders.rglob("*.pdf"):
				datafile = str(datafile)
				docs = load_pdf(datafile)
				fileMetadata[datafile] = ({
					"fileSize": os.path.getsize(datafile),
					"fileModifiedTime": os.path.getmtime(datafile),
					"fileHash": get_file_hash(datafile),
				})
				allTexts.extend(docs)
			# Ensure cache directory exists before writing metadata
			os.makedirs(self.cacheDir, exist_ok=True)
			with open(f"{self.cacheDir}/file_metadata.json", "w") as f:
				json.dump(fileMetadata, f, indent=2)
			return RAGPipeline(allTexts, embedder=self._embedder, chat_client=self._chat_client, cache_file=f"{self.cacheDir}/embeddings.json")
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

	def get_file_changes(self):
		"""Check which files have changed, are new, or were removed"""
		changed_files = []
		new_files = []
		removed_files = []
		is_valid = True
		
		try:
			if not os.path.exists(f"{self.cacheDir}/file_metadata.json") or not os.path.exists(f"{self.cacheDir}/embeddings.json"):
				return {
					"is_valid": False,
					"changed_files": [],
					"new_files": [],
					"removed_files": []
				}
			
			if os.path.getsize(f"{self.cacheDir}/file_metadata.json") == 0 or os.path.getsize(f"{self.cacheDir}/embeddings.json") == 0:
				return {
					"is_valid": False,
					"changed_files": [],
					"new_files": [],
					"removed_files": []
				}

			with open(f"{self.cacheDir}/file_metadata.json", "r") as f:
				cacheMetadata = json.load(f)
			
			current_files = {str(file) for file in self.allFolders.rglob("*.pdf")}
			cached_files = set(cacheMetadata.keys())
			
			removed_files = list(cached_files - current_files)
			
			for file in current_files:
				file = str(file)
				if file not in cacheMetadata.keys():
					new_files.append(file)
				elif get_file_hash(file) != cacheMetadata[file]["fileHash"]:
					changed_files.append(file)
			
			return {
				"is_valid": True,
				"changed_files": changed_files,
				"new_files": new_files,
				"removed_files": removed_files
			}
		except Exception as e:
			print(f"Error checking file changes: {e}")
			return {
				"is_valid": False,
				"changed_files": [],
				"new_files": [],
				"removed_files": []
			}
	
	def incremental_update(self, file_changes):
		"""Update embeddings only for changed files"""
		try:
			with open(f"{self.cacheDir}/embeddings.json", "r") as f:
				cache_data = json.load(f)
			
			existing_chunks = cache_data["chunks"]
			existing_embeddings = cache_data["embeddings"]
			
			files_to_update = set(file_changes["changed_files"] + file_changes["new_files"])
			files_to_remove = set(file_changes["removed_files"])
			
			updated_chunks = []
			updated_embeddings = []
			
			for i, chunk in enumerate(existing_chunks):
				chunk_path = chunk["metadata"]["path"]
				if chunk_path not in files_to_update and chunk_path not in files_to_remove:
					updated_chunks.append(chunk)
					updated_embeddings.append(existing_embeddings[i])
			
			allTexts = []
			fileMetadata = {}
			
			for datafile in self.allFolders.rglob("*.pdf"):
				datafile = str(datafile)
				if datafile in files_to_update:
					docs = load_pdf(datafile)
					allTexts.extend(docs)
					fileMetadata[datafile] = {
						"fileSize": os.path.getsize(datafile),
						"fileModifiedTime": os.path.getmtime(datafile),
						"fileHash": get_file_hash(datafile),
					}
			
			if allTexts:
				from .modules.embeddings import get_embeddings
				_embedder_batch = getattr(self._embedder, "get_embeddings", None) if self._embedder else get_embeddings
				new_texts = [chunk["text"] for chunk in allTexts]
				new_embeddings = _embedder_batch(new_texts)
				
				updated_chunks.extend(allTexts)
				updated_embeddings.extend(new_embeddings)
			
			with open(f"{self.cacheDir}/file_metadata.json", "r") as f:
				existing_metadata = json.load(f)
			
			for datafile in self.allFolders.rglob("*.pdf"):
				datafile = str(datafile)
				if datafile not in files_to_update:
					fileMetadata[datafile] = existing_metadata.get(datafile, {
						"fileSize": os.path.getsize(datafile),
						"fileModifiedTime": os.path.getmtime(datafile),
						"fileHash": get_file_hash(datafile),
					})
			
			os.makedirs(self.cacheDir, exist_ok=True)
			with open(f"{self.cacheDir}/file_metadata.json", "w") as f:
				json.dump(fileMetadata, f, indent=2)
			
			return RAGPipeline(updated_chunks, embedder=self._embedder, chat_client=self._chat_client, cache_file=f"{self.cacheDir}/embeddings.json")
		except Exception as e:
			print(f"Error in incremental update: {e}")
			print("Falling back to full rebuild")
			return self.data_pipeline()