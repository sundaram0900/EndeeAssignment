from sentence_transformers import SentenceTransformer
import numpy as np

class RAGEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)

    def generate_embeddings(self, texts):
        """
        Convert a list of texts to embeddings.
        Returns a list of lists of floats.
        """
        embeddings = self.encoder.encode(texts)
        # Convert numpy array to list of lists for JSON serialization
        return embeddings.tolist()

    def chunk_text(self, text, chunk_size=200):
        """
        Simple text chunking by words.
        """
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
