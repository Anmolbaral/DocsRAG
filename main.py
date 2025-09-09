from modules.rag import RAGPipeline
from modules.loader import load_pdf

import os
allFiles = os.listdir("data")
allTexts = []
for file in allFiles:
	docs = load_pdf("data/" + file)
	allTexts.extend(docs)
rag = RAGPipeline(allTexts)

while True:
	query = input("Ask a question: ")
	answer = rag.ask(query)
	print("Answer: ", answer)