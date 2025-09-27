from .modules.rag import RAGPipeline
from .modules.loader import load_pdf
from .modules.hashing import get_file_hash
import glob
import os, json, pathlib
import time

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

		if self.cache_is_valid():
			print("Using cached embeddings")
			self.ragPipeline = RAGPipeline.from_cache(cacheFile=f"{self.cacheDir}/embeddings.json", embedder=self._embedder, chat_client=self._chat_client)
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
	
	def user_interaction(self):
		while True:
			try:
				query = input("\nAsk a question or type 'exit' to quit: ")
				start_time = time.time()
				if query == "exit":
					return
				elif query.strip() == "":
					continue
				else:
					answer = self.ragPipeline.ask(query)
					end_time = time.time()
					print("\n---------Answer---------\n", answer)
					print(f'Time taken: {end_time - start_time} seconds')

			except Exception as e:
				print(f"Error answering query: {e}")
		return

	"""Check if the file metadata is valid"""
	"""No need to make a new embedding cache if the file metadata is valid"""
	def cache_is_valid(self):
		try:
			if os.path.getsize(f"{self.cacheDir}/file_metadata.json") == 0 or os.path.getsize(f"{self.cacheDir}/embeddings.json") == 0:
				return False

			with open(f"{self.cacheDir}/file_metadata.json", "r") as f:
				cacheMetadata = json.load(f)			
			
			for file in self.allFolders.rglob("*.pdf"):
				file = str(file)
				
				if file not in cacheMetadata.keys() or get_file_hash(file) != cacheMetadata[file]["fileHash"]:
					return False

			return True
		except Exception as e:
			print(f"Error checking cache: {e}")
			return False