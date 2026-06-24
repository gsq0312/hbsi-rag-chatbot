"""Data models for RAG Service"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class DocumentSource(str, Enum):
    TXT = "txt"
    PDF = "pdf"
    DOCX = "docx"


class RAGConfig(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunk_size": 500,
                "chunk_overlap": 50,
                "top_k_retrieval": 3,
                "model_name": "deepseek-chat",
                "max_tokens": 2000,
                "temperature": 0.7
            }
        }
    )
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)
    top_k_retrieval: int = Field(default=3, ge=1, le=10)
    model_name: str = Field(default="deepseek-chat")
    max_tokens: int = Field(default=2000, ge=100, le=8000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class Document(BaseModel):
    id: str = Field(description="Unique document identifier")
    content: str = Field(description="Document content")
    source: str = Field(description="Document source/file name")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    config: Optional[RAGConfig] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[Document] = Field(default_factory=list)
    session_id: str


class UploadResponse(BaseModel):
    success: bool
    message: str
    document_id: Optional[str] = None
    chunks_created: int = 0


class ErrorResponse(BaseModel):
    error: str
    message: str
    detail: Optional[str] = None
