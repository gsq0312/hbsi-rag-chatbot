"""RAG API Routes"""
import uuid
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from app.models import ChatRequest, ChatResponse, UploadResponse, RAGConfig
from app.services.chat import get_chat_service
from app.services.document import DocumentProcessor
from app.services.vector_store import get_vector_store
from app.config import get_config_manager

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        chat_service = get_chat_service()
        session_id = request.session_id or f"session_{uuid.uuid4().hex}"
        answer, sources = chat_service.chat(query=request.query, session_id=session_id, config=request.config)
        return ChatResponse(answer=answer, sources=sources, session_id=session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_id: Optional[str] = Query(None)
) -> UploadResponse:
    try:
        allowed_extensions = {'.txt', '.pdf', '.docx'}
        file_ext = '.' + file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        if file_ext not in allowed_extensions:
            return UploadResponse(success=False, message=f"Unsupported file type: {file_ext}", chunks_created=0)
        doc_id = document_id or f"doc_{uuid.uuid4().hex}"
        content = await file.read()
        processor = DocumentProcessor()
        chunks = processor.process_bytes(content, file.filename, doc_id)
        vector_store = get_vector_store()
        chunk_ids = vector_store.add_documents(chunks, doc_id)
        vector_store.save()
        return UploadResponse(success=True, message=f"Document '{file.filename}' uploaded successfully", document_id=doc_id, chunks_created=len(chunk_ids))
    except Exception as e:
        return UploadResponse(success=False, message=f"Upload failed: {str(e)}", chunks_created=0)


@router.get("/config", response_model=RAGConfig)
async def get_config() -> RAGConfig:
    return get_config_manager().get_config()


@router.put("/config", response_model=RAGConfig)
async def update_config(config: RAGConfig) -> RAGConfig:
    config_manager = get_config_manager()
    updated_config = config_manager.update_config(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap,
        top_k_retrieval=config.top_k_retrieval, model_name=config.model_name,
        max_tokens=config.max_tokens, temperature=config.temperature
    )
    config_manager.save()
    return updated_config


@router.delete("/chat/{session_id}")
async def clear_conversation(session_id: str):
    chat_service = get_chat_service()
    chat_service.clear_conversation(session_id)
    return {"success": True, "message": f"Conversation history cleared for session {session_id}"}


@router.get("/stats")
async def get_stats():
    chat_service = get_chat_service()
    vector_store = get_vector_store()
    return {
        "vector_store": {"document_count": vector_store.document_count, "document_ids": vector_store.get_document_ids()},
        "sessions": chat_service.get_session_stats()
    }
