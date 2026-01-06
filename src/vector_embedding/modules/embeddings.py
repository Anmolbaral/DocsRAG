# In modules/embeddings.py
import openai
import os
import ollama
from config import Config

class EmbeddingService:
    def __init__(self, config: Config):
        self.config = config
        self.provider = config.embedding.provider
        self.model = config.embedding.model
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if config.embedding.provider == "openai" else None

    def get_embedding_single(self, text):
        if self.provider == "openai":
            response = self.client.embeddings.create(input=text, model=self.model)
            return response.data[0].embedding
        elif self.provider == "ollama":
            response = ollama.embeddings(model=self.model, prompt=text)
            return response.get("embedding", response)
        raise ValueError(f"Invalid provider: {self.provider}")

    def get_embedding_batch(self, texts):
        if self.provider == "openai":
            resp = self.client.embeddings.create(input=texts, model=self.model)
            return [item.embedding for item in resp.data]
        elif self.provider == "ollama":
            return [self.get_embedding_single(text) for text in texts]
        raise ValueError(f"Invalid provider: {self.provider}")