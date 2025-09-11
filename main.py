from modules.rag import RAGPipeline
from modules.loader import load_pdf

import os, json

allFiles = os.listdir("data")
allTexts = []
fileMetadata = []
for file in allFiles:
	fileName = "data/" + file
	docs = load_pdf(fileName)
	fileMetadata.append({
		"fileName": fileName,
		"fileSize": os.path.getsize(fileName),
		"fileModifiedTime": os.path.getmtime(fileName),
		"fileContent": docs
	})
	allTexts.extend(docs)
	with open("cache/file_metadata.json", "w") as f:
		json.dump(fileMetadata, f, indent=4)

rag = RAGPipeline(allTexts)

while True:
	query = input("Ask a question: ")
	answer = rag.ask(query)
	print("Answer: ", answer)