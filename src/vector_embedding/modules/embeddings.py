import openai
import os
from dotenv import load_dotenv
load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(texts, model="text-embedding-3-small"):
	response = client.embeddings.create(input=texts, model=model)
	return response.data[0].embedding

def get_embeddings(texts, model="text-embedding-3-small"):
	cleanTexts = [str(text).strip() for text in texts]
	response = client.embeddings.create(input=cleanTexts, model=model)
	return [item.embedding for item in response.data]

