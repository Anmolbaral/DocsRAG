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

		#Adding conversation memory
		self.conversationHistory = []
		self.maxHistory = 10
	

	@classmethod
	def from_cache(cls, cacheFile = "cache/embeddings.json"):
		with open(cacheFile, "r") as f:
			metadata = json.load(f)

		obj = cls.__new__(cls)
		obj.db = VectorDB(dim=1536)
		obj.chunks = metadata["chunks"]
		obj.embeddings = metadata["embeddings"]

		# Intializing conversation history
		obj.conversationHistory = []
		obj.maxHistory = 10

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
		messages = self.build_converation_context(context)
		messages.append({"role": "user", "content": query})
		response = openai.chat.completions.create(
			model = "gpt-4o-mini",
			messages = messages
		)
		self.add_to_conversation_history(query, response.choices[0].message.content)
		return response.choices[0].message.content
	
	def build_converation_context(self, documentContext):
		conversationContext = []
		conversationContext.append({'role': 'system', 'content': f'You are a helpful assistant that can answer questions about the context provided. USe conversation history to understant the references and context'})
		conversationContext.append({'role':'user', 'content': f'Document Context: {documentContext}'})
		if self.conversationHistory:
			for conversation in self.conversationHistory:
				conversationContext.append(conversation["user"])
				conversationContext.append(conversation["assistant"])
		return conversationContext

	def add_to_conversation_history(self, query, response):
		self.conversationHistory.append({"user":{"role": "user", "content": query}, "assistant":{'role':'assistant', 'content':response}})
		if len(self.conversationHistory) > self.maxHistory:
			self.conversationHistory.pop(0)

	def add_to_cache(self):
		cacheData= {
			"chunks": self.chunks,
			"embeddings":self.embeddings,
		}
		with open("cache/embeddings.json", "w") as f:
			json.dump(cacheData, f, indent=2)
			

