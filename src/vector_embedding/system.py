from .modules.rag import RAGPipeline
from .modules.loader import load_pdf
from .modules.hashing import get_file_hash
import glob
import os, json, pathlib

class DocumentRAGSystem:
	def __init__(self):
		self.ragPipeline = None
		self.cacheDir = "cache"
		self.allFolders = pathlib.Path("data")

	def initialize(self):
		"""Initiializing RAGPipeline once at startup"""

		if self.cache_is_valid():
			print("Using cached embeddings")
			self.ragPipeline = RAGPipeline.from_cache()
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
			with open("cache/file_metadata.json", "w") as f:
				json.dump(fileMetadata, f, indent=2)
			return RAGPipeline(allTexts)
		except Exception as e:
			print(f"Error processing files: {e}")
			raise e
	
	def user_interaction(self):
		while True:
			try:
				query = input("Ask a question or type 'exit' to quit: ")
				if query == "exit":
					return
				answer = self.ragPipeline.ask(query)
				print("Answer: ", answer)

			except Exception as e:
				print(f"Error answering query: {e}")
		return

	"""Check if the file metadata is valid"""
	"""No need to make a new embedding cache if the file metadata is valid"""
	def cache_is_valid(self):
		try:
			if os.path.getsize("cache/file_metadata.json") == 0:
				return False

			with open("cache/file_metadata.json", "r") as f:
				cacheMetadata = json.load(f)			
			
			for file in self.allFolders.rglob("*.pdf"):
				filePath = f"{file}"

				if file not in cacheMetadata.keys() or get_file_hash(filePath) != cacheMetadata[file]["fileHash"]:
					return False

			return True
		except Exception as e:
			print(f"Error checking cache: {e}")
			return False