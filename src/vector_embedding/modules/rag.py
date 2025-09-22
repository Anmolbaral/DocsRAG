from .embeddings import get_embedding, get_embeddings
from .vectordb import VectorDB
import openai, os, json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
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
		
		# Cache category embeddings once during initialization
		self.categoryEmbeddings = self.create_category_embeddings()
	

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
		
		# Cache category embeddings once during initialization
		obj.categoryEmbeddings = obj.create_category_embeddings()

		for i, chunk in enumerate(obj.chunks):
			obj.db.add(obj.embeddings[i], chunk["text"], chunk["metadata"])
		return obj
	
	def create_category_embeddings(self):
		category = {	
        	"resume": "Professional career document containing work experience, technical skills, education background, projects, and quantified achievements. Formal tone with bullet points, dates, company names, job titles, and measurable accomplishments. Focus on qualifications, competencies, and professional history.",
        	"cover_letters": "Personal application letters expressing interest in specific job roles and companies. Conversational tone showing enthusiasm, motivation, and personal fit for positions. Contains phrases like 'excited about', 'passionate about', 'would love to', and explanations of why someone wants a particular role or company.",
        	"misc_docs": "Academic papers, essays, fellowship applications, and personal writings about broader topics like education, community impact, research, and personal philosophy. Reflective tone discussing values, social impact, academic work, and personal growth experiences."
		}
		categoryEmbedding = {cat: get_embedding(desc) for cat, desc in category.items()}
		return categoryEmbedding

	def calculate_confidence(self, queryEmb, categoryEmbedding):
		categoryScore = {cat: cosine_similarity([queryEmb], [categoryEmbedding[cat]])[0][0] for cat in categoryEmbedding}
		
		sortedScore = sorted(categoryScore.values(), reverse = True)
		print(f"Sorted Score: {sortedScore}")

		relativeConfidence = sortedScore[0] - sortedScore[1] if len(sortedScore) > 1 else sortedScore[0]
		print(f"Relative Confidence: {relativeConfidence}")
		
		return relativeConfidence, categoryScore


	def ask(self, query):
		queryEmb = get_embedding(query)
		# Use cached category embeddings instead of recalculating
		categoryEmbedding = self.categoryEmbeddings		

		confidence, categoryScore = self.calculate_confidence(queryEmb, categoryEmbedding)
		closestCategory = max(categoryScore, key=lambda x: categoryScore[x])
		print(f"-----Closest Category-----:\n{closestCategory}")

		# Thresholding the confidence
		if confidence > 0.25:
			k = 3
		elif confidence > 0.15:
			k = 5
		else:
			k = 8
		results = self.db.search(queryEmb, k)

		contextParts = []
		if confidence > 0.15:
			for result in results:
				if result['metadata']['category'] == closestCategory:
					contextParts.append(f"Page {result['metadata']['page']} - {result['metadata']['filename']}: {result['text']}")
		else:
			for result in results:
				contextParts.append(f"Page {result['metadata']['page']} - {result['metadata']['filename']}: {result['text']}")


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
			

