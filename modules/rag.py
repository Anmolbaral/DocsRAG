from modules.embeddings import get_embedding, get_embeddings
from modules.vectordb import VectorDB
import openai, os
openai.api_key = os.getenv("OPENAI_API_KEY")

class RAGPipeline:
	def __init__(self, chunks):
		self.db = VectorDB(dim=1536)
		texts = [chunk["text"] for chunk in chunks]
		embeddings = get_embeddings(texts)
		for i, t in enumerate(chunks):
			self.db.add(embeddings[i], t["text"], t["metadata"])
	
	def ask(self, query):
		queryEmb = get_embedding(query)
		results = self.db.search(queryEmb, k = 5)
		contextParts = []
		for result in results:
			contextParts.append(f"Page {result['metadata']['page']} - {result['metadata']['section']}: {result['text']}")
		context = "\n".join(contextParts)	
		prompt = f"Answer using this context: {context}\n\nQuestion: {query}"
		response = openai.chat.completions.create(
			model = "gpt-4o-mini",
			messages = [{"role": "user", "content": prompt}]
		)
		return response.choices[0].message.content