"""FastAPI Main Application for RAG Service"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from app.routers import rag
from app.config import get_config_manager
from app.services.vector_store import get_vector_store

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting RAG Service...")
    config_manager = get_config_manager()
    config = config_manager.get_config()
    print(f"Configuration loaded: chunk_size={config.chunk_size}, top_k={config.top_k_retrieval}")
    try:
        vector_store = get_vector_store()
        print(f"Vector store initialized with {vector_store.document_count} documents")
    except Exception as e:
        print(f"Vector store initialization warning: {e}")
    print("RAG Service started successfully")
    yield
    print("Shutting down RAG Service...")


app = FastAPI(
    title="RAG Admission Chatbot Service",
    description="RAG-based intelligent admission consultation service",
    version="1.0.0",
    lifespan=lifespan
)

allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173,*"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"error": "ValueError", "message": str(exc)})


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    return JSONResponse(status_code=404, content={"error": "FileNotFoundError", "message": str(exc)})


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": "InternalServerError", "message": str(exc)})


@app.get("/health")
async def health_check():
    try:
        vector_store = get_vector_store()
        config = get_config_manager().get_config()
        return {
            "status": "healthy",
            "vector_store": {"document_count": vector_store.document_count},
            "config": {"chunk_size": config.chunk_size, "top_k_retrieval": config.top_k_retrieval}
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


app.include_router(rag.router, prefix="/api", tags=["RAG"])


@app.get("/")
async def root():
    return {
        "service": "RAG Admission Chatbot Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {"/health": "Health check", "/api/chat": "Chat", "/api/upload": "Upload", "/docs": "API docs"}
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("RAG_SERVICE_PORT", 8002))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
