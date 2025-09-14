from modules.embeddings import get_embedding, get_embeddings
from modules.vectordb import VectorDB
import openai, os, json
openai.api_key = os.getenv("OPENAI_API_KEY")

class RAGPipeline:
	def __init__(self, chunks):
		self.db = VectorDB(dim=1536)
		self.texts = [chunk["text"] for chunk in chunks]
		self.chunks = chunks
		self.embeddings = get_embeddings(self.texts)
		for i, t in enumerate(chunks):
			self.db.add(self.embeddings[i], t["text"], t["metadata"])
		self.add_to_cache()
	

	@classmethod
	def from_cache(cls, cacheFile = "cache/embeddings.json"):
		with open(cacheFile, "r") as f:
			metadata = json.load(f)

		obj = cls.__new__(cls)
		obj.db = VectorDB(dim=1536)
		obj.chunks = metadata["chunks"]
		obj.embeddings = metadata["embeddings"]
		# obj.texts = [chunk["text"] for chunk in obj.chunks]
		for i, chunk in enumerate(obj.chunks):
			obj.db.add(obj.embeddings[i], chunk["text"], chunk["metadata"])
		return obj
	
	
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
	
	def add_to_cache(self):
		cacheData= {
			"chunks": self.chunks,
			"embeddings":self.embeddings,
		}
		with open("cache/embeddings.json", "w") as f:
			json.dump(cacheData, f, indent=2)
			

