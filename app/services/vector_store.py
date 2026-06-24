"""FAISS Vector Store for RAG"""
import pickle
import uuid
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import faiss
from langchain_core.documents import Document as LangchainDocument
from app.services.embedding import DeepSeekEmbedding


class VectorStore:
    def __init__(self, embedding_service: Optional[DeepSeekEmbedding] = None, index_path: Optional[Path] = None, dimension: Optional[int] = None):
        self.embedding_service = embedding_service or DeepSeekEmbedding()
        self.index_path = index_path or Path("data/vector_store")
        if dimension is None:
            dimension = self.embedding_service.get_embedding_dimension()
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents: List[LangchainDocument] = []
        self.doc_ids: List[str] = []

    def add_documents(self, documents: List[LangchainDocument], document_id: Optional[str] = None) -> List[str]:
        if not documents:
            return []
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_service.embed_texts(texts)
        embedding_array = np.array(embeddings, dtype=np.float32)
        self.index.add(embedding_array)
        chunk_ids = []
        for i, doc in enumerate(documents):
            chunk_id = document_id or f"chunk_{uuid.uuid4().hex}"
            if len(documents) > 1:
                chunk_id = f"{chunk_id}_{i}"
            doc.metadata['chunk_id'] = chunk_id
            if document_id:
                doc.metadata['document_id'] = document_id
            self.documents.append(doc)
            self.doc_ids.append(chunk_id)
            chunk_ids.append(chunk_id)
        return chunk_ids

    def similarity_search(self, query: str, k: int = 3, score_threshold: Optional[float] = None) -> List[Tuple[LangchainDocument, float]]:
        if self.index.ntotal == 0:
            return []
        query_embedding = self.embedding_service.embed_text(query)
        query_array = np.array([query_embedding], dtype=np.float32)
        k_results = min(k * 2 if score_threshold else k, self.index.ntotal)
        distances, indices = self.index.search(query_array, k_results)
        results = []
        for distance, doc_idx in zip(distances[0], indices[0]):
            if doc_idx == -1:
                continue
            similarity = 1.0 / (1.0 + float(distance))
            if score_threshold is not None and similarity < score_threshold:
                continue
            doc = self.documents[doc_idx]
            results.append((doc, similarity))
            if len(results) >= k:
                break
        return results

    def save(self, path: Optional[Path] = None) -> None:
        save_path = Path(path or self.index_path)
        save_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(save_path / "index.faiss"))
        with open(save_path / "metadata.pkl", 'wb') as f:
            pickle.dump({'documents': self.documents, 'doc_ids': self.doc_ids, 'dimension': self.dimension}, f)

    def load(self, path: Optional[Path] = None) -> None:
        load_path = Path(path or self.index_path)
        index_file = load_path / "index.faiss"
        metadata_file = load_path / "metadata.pkl"
        if not index_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f"Vector store not found at {load_path}")
        self.index = faiss.read_index(str(index_file))
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        self.documents = metadata['documents']
        self.doc_ids = metadata['doc_ids']
        self.dimension = metadata['dimension']

    def clear(self) -> None:
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.doc_ids = []

    @property
    def document_count(self) -> int:
        return len(self.documents)

    def get_document_ids(self) -> List[str]:
        unique_ids = set()
        for doc in self.documents:
            doc_id = doc.metadata.get('document_id')
            if doc_id:
                unique_ids.add(doc_id)
        return list(unique_ids)


_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        vector_store_path = Path("data/vector_store")
        if vector_store_path.exists():
            _vector_store = VectorStore(index_path=vector_store_path)
            try:
                _vector_store.load()
            except FileNotFoundError:
                pass
        else:
            _vector_store = VectorStore()
    return _vector_store
