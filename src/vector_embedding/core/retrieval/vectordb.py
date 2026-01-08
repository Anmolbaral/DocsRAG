import faiss
import numpy as np


# Vector database using FAISS IndexHNSWFlat for fast approximate nearest neighbor search.
class VectorDB:
    def __init__(self, dim):
        self.index = faiss.IndexHNSWFlat(dim, 32)
        self.texts = []  # list of text strings
        self.metadata = []  # list of metadata dictionaries

    def add(self, vectors, texts, metadata=None):
        vectors = np.array(vectors).astype("float32")
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)
            texts = [texts] if texts else [None]
            metadata = [metadata] if metadata else [None]
        elif len(vectors.shape) == 2:
            texts = texts
            metadata = metadata
        else:
            raise ValueError("Invalid vector shape")

        self.index.add(vectors)
        self.texts.extend(texts)
        self.metadata.extend(metadata)

    # Search for k most similar vectors. Returns list of dicts with "text", "metadata", and "distance" (lower distance = more similar).
    def search(self, queryVector, k=5):
        vec = np.array([queryVector]).astype("float32")
        distances, indices = self.index.search(vec, k)
        results = []
        for idx, match in enumerate(indices[0]):
            results.append(
                {
                    "text": self.texts[match],
                    "metadata": self.metadata[match],
                    "distance": distances[0][idx],
                }
            )
        return results
