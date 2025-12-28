import faiss
import numpy as np

class VectorDB:
    def __init__(self, dim):
        self.index = faiss.IndexHNSWFlat(dim, 32)
        self.texts = []
        self.metadata = []

    def add(self, vector, text, metadata=None):
        vec = np.array([vector]).astype("float32")
        self.index.add(vec)
        self.texts.append(text)
        self.metadata.append(metadata or {})

    def search(self, query_vector, k=5):
        vec = np.array([query_vector]).astype("float32")
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
