from sentence_transformers import SentenceTransformer
from functools import lru_cache
import numpy as np


class EmbeddingGenerator:
    """
    Gera embeddings de texto usando Sentence Transformers.
    Usado para busca semantica de palavras similares.
    """
    
    def __init__(self):
        self.model_name = "all-MiniLM-L6-v2"
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            print("Carregando modelo de embeddings...")
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def generate(self, text: str) -> list[float]:
        """
        Args:
            text: Texto para gerar embedding
        Returns:
            Vetor de 384 floats
        """
        if not text or not text.strip():
            return [0.0] * 384
        
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def generate_batch(self, texts: list[str]) -> list[list[float]]:
        """Gera embeddings para multiplos textos"""
        if not texts:
            return []
        
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calcula similaridade entre dois textos.
        Returns:
            Float entre 0 e 1 (1 = identicos)
        """
        emb1 = np.array(self.generate(text1))
        emb2 = np.array(self.generate(text2))
        
        # Similaridade do cosseno
        similarity = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2)
        )
        
        return float(similarity)


@lru_cache()
def get_embedding_generator() -> EmbeddingGenerator:
    return EmbeddingGenerator()