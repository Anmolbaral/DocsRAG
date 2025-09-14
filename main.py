from modules.rag import RAGPipeline
from modules.loader import load_pdf
from modules.hashing import get_file_hash

import os, json

allFiles = os.listdir("data")
allTexts = []
fileMetadata = {}

# Check if the file metadata is valid
# No need to make a new embedding cache if the file metadata is valid
def cache_is_valid():
	try:
		if os.path.getsize("cache/file_metadata.json") == 0:
			return False

		with open("cache/file_metadata.json", "r") as f:
			cacheMetadata = json.load(f)			
		
		for file in allFiles:
			filePath = f"data/{file}"

			if file not in cacheMetadata.keys() or get_file_hash(filePath) != cacheMetadata[file]["fileHash"]:
				return False

		return True
	except Exception as e:
		print(f"Error checking cache: {e}")
		return False

while True:
	query = input("Ask a question or type 'exit' to quit: ")

	if query == "exit":
		break
	#if file metadata is invalid, we make a new embedding cache
	if cache_is_valid():
		print("Using cached embeddings")
		rag = RAGPipeline.from_cache()
	else:
		print("Using new embeddings")
		for file in allFiles:
			fileLoc = "data/" + file
			docs = load_pdf(fileLoc)
			fileMetadata[file] = ({
				"fileSize": os.path.getsize(fileLoc),
				"fileModifiedTime": os.path.getmtime(fileLoc),
				"fileHash": get_file_hash(fileLoc),
			})
			allTexts.extend(docs)
		with open("cache/file_metadata.json", "w") as f:
			json.dump(fileMetadata, f, indent=4)
		rag = RAGPipeline(allTexts)
	
	answer = rag.ask(query)
	print("Answer: ", answer)