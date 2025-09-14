from modules.rag import RAGPipeline
from modules.loader import load_pdf
from modules.hashing import get_file_hash
import os, json


class DocumentRAGSystem:
	def __init__(self):
		self.ragPipeline = None
		self.cacheDir = "cache"
		self.allFiles = os.listdir("data")

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

		for file in self.allFiles:
			fileLoc = "data/" + file
			docs = load_pdf(fileLoc)
			fileMetadata[file] = ({
				"fileSize": os.path.getsize(fileLoc),
				"fileModifiedTime": os.path.getmtime(fileLoc),
				"fileHash": get_file_hash(fileLoc),
			})
			allTexts.extend(docs)
		with open("cache/file_metadata.json", "w") as f:
			json.dump(fileMetadata, f, indent=2)
		return RAGPipeline(allTexts)
	
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
			
			for file in self.allFiles:
				filePath = f"data/{file}"

				if file not in cacheMetadata.keys() or get_file_hash(filePath) != cacheMetadata[file]["fileHash"]:
					return False

			return True
		except Exception as e:
			print(f"Error checking cache: {e}")
			return False


if __name__ == "__main__":
	system = DocumentRAGSystem()
	system.initialize()
	system.user_interaction()