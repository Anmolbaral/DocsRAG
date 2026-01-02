import openai
import os
from config import Config

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Get embedding for a single text. Use get_embedding_batch() for multiple texts.
# Returns a 3072-dimensional vector for text-embedding-3-large model.
def get_embedding_single(text, config:Config):
    model = config.embedding.model
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


# Get embeddings for multiple texts in a single API call. More efficient than
def get_embedding_batch(texts, config:Config):
    model = config.embedding.model
    cleanTexts = [str(text).strip() for text in texts]
    response = client.embeddings.create(input=cleanTexts, model=model)
    return [item.embedding for item in response.data]
