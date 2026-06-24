"""Embedding service for RAG using HuggingFace models"""
import os
import hashlib
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# Set HuggingFace mirror for China
os.environ['HF_ENDPOINT'] = os.getenv('HF_ENDPOINT', 'https://hf-mirror.com')


class DeepSeekEmbedding:
    def __init__(self, model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", dimension: int = 384):
        self.model_name = model
        self.dimension = dimension
        self._embeddings = None
        self._use_fallback = False

    @property
    def embeddings(self):
        if self._embeddings is None and not self._use_fallback:
            try:
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings
                except ImportError:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            except Exception as e:
                print(f"Warning: Could not load HuggingFace embeddings: {e}")
                print("Using fallback hash-based embeddings")
                self._use_fallback = True
        return self._embeddings

    def _hash_embed(self, text: str) -> List[float]:
        import math
        embeddings = []
        for i in range(self.dimension):
            h = hashlib.md5(f"{text}_{i}".encode()).hexdigest()
            val = (int(h[:8], 16) / (2**32)) * 2 - 1
            embeddings.append(val)
        norm = math.sqrt(sum(x*x for x in embeddings))
        if norm > 0:
            embeddings = [x / norm for x in embeddings]
        return embeddings

    def embed_text(self, text: str) -> List[float]:
        if self._use_fallback or self.embeddings is None:
            return self._hash_embed(text)
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            print(f"Embedding failed, using fallback: {e}")
            self._use_fallback = True
            return self._hash_embed(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self._use_fallback or self.embeddings is None:
            return [self._hash_embed(text) for text in texts]
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            print(f"Batch embedding failed, using fallback: {e}")
            self._use_fallback = True
            return [self._hash_embed(text) for text in texts]

    def get_embedding_dimension(self) -> int:
        if self._use_fallback:
            return self.dimension
        try:
            sample = self.embed_text("sample")
            return len(sample)
        except:
            return self.dimension


_embedding_service: Optional[DeepSeekEmbedding] = None


def get_embedding_service() -> DeepSeekEmbedding:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = DeepSeekEmbedding()
    return _embedding_service
