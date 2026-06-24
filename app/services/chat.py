"""Chat service for RAG-based conversation"""
import uuid
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
from dotenv import load_dotenv
import os

from app.services.embedding import get_embedding_service
from app.services.vector_store import get_vector_store
from app.utils.prompts import PromptBuilder, get_conversation_history
from app.models import RAGConfig, Document

load_dotenv()


class ChatService:
    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None):
        self.embedding_service = get_embedding_service()
        self.vector_store = get_vector_store()
        self.conversation_history = get_conversation_history()
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_base = api_base or os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY must be set")
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)

    def chat(self, query: str, session_id: Optional[str] = None, config: Optional[RAGConfig] = None) -> Tuple[str, List[Document]]:
        if not session_id:
            session_id = f"session_{uuid.uuid4().hex}"
        if config is None:
            from app.config import get_config
            config = get_config()
        retrieved_docs = self._retrieve_documents(query, config)
        messages = PromptBuilder.build_chat_messages(
            question=query, context_documents=retrieved_docs,
            conversation_history=self.conversation_history.get_history(session_id)
        )
        self.conversation_history.add_message(session_id, "user", query)
        answer = self._call_llm(messages, config)
        self.conversation_history.add_message(session_id, "assistant", answer)
        source_documents = self._convert_documents(retrieved_docs)
        return answer, source_documents

    def _retrieve_documents(self, query: str, config: RAGConfig) -> List:
        results = self.vector_store.similarity_search(query, k=config.top_k_retrieval)
        return [doc for doc, _ in results]

    def _call_llm(self, messages: List[Dict[str, str]], config: RAGConfig) -> str:
        try:
            response = self.client.chat.completions.create(
                model=config.model_name, messages=messages,
                max_tokens=config.max_tokens, temperature=config.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"抱歉，生成回答时出错：{str(e)}"

    def _convert_documents(self, docs: List) -> List[Document]:
        converted = []
        for doc in docs:
            converted.append(Document(
                id=doc.metadata.get('chunk_id', 'unknown'),
                content=doc.page_content,
                source=doc.metadata.get('source', 'unknown'),
                metadata=doc.metadata
            ))
        return converted

    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        return self.conversation_history.get_history(session_id)

    def clear_conversation(self, session_id: str) -> None:
        self.conversation_history.clear_session(session_id)

    def get_session_stats(self) -> Dict[str, int]:
        return {"total_sessions": self.conversation_history.get_session_count(), "vector_store_docs": self.vector_store.document_count}


_chat_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service
