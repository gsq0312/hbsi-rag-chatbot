"""Document processing module for RAG Service"""
from pathlib import Path
from typing import List, Optional, Union
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document as LangchainDocument


class DocumentLoader:
    SUPPORTED_FORMATS = {'.txt', '.pdf', '.docx'}

    @staticmethod
    def load_from_file(file_path: Union[str, Path]) -> List[LangchainDocument]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if file_path.suffix.lower() not in DocumentLoader.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        if file_path.suffix.lower() == '.txt':
            loader = TextLoader(str(file_path), encoding='utf-8')
        elif file_path.suffix.lower() == '.pdf':
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix.lower() == '.docx':
            loader = Docx2txtLoader(str(file_path))
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        return loader.load()

    @staticmethod
    def load_from_bytes(file_content: bytes, filename: str, source: Optional[str] = None) -> List[LangchainDocument]:
        file_path = Path(filename)
        if file_path.suffix.lower() not in DocumentLoader.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_path.suffix) as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
        try:
            documents = DocumentLoader.load_from_file(temp_path)
            if source:
                for doc in documents:
                    doc.metadata['source'] = source
            return documents
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @staticmethod
    def load_from_text(text: str, source: str = "text_input") -> List[LangchainDocument]:
        return [LangchainDocument(page_content=text, metadata={'source': source})]


class TextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""], length_function=len
        )

    def split_documents(self, documents: List[LangchainDocument]) -> List[LangchainDocument]:
        return self.splitter.split_documents(documents)

    def split_text(self, text: str) -> List[str]:
        return self.splitter.split_text(text)


class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.loader = DocumentLoader()
        self.splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def process_file(self, file_path: Union[str, Path], document_id: Optional[str] = None) -> List[LangchainDocument]:
        documents = self.loader.load_from_file(file_path)
        if document_id:
            for doc in documents:
                doc.metadata['document_id'] = document_id
        chunks = self.splitter.split_documents(documents)
        for idx, chunk in enumerate(chunks):
            chunk.metadata['chunk_index'] = idx
        return chunks

    def process_bytes(self, file_content: bytes, filename: str, document_id: Optional[str] = None) -> List[LangchainDocument]:
        source = document_id or filename
        documents = self.loader.load_from_bytes(file_content, filename, source)
        if document_id:
            for doc in documents:
                doc.metadata['document_id'] = document_id
        chunks = self.splitter.split_documents(documents)
        for idx, chunk in enumerate(chunks):
            chunk.metadata['chunk_index'] = idx
        return chunks

    def process_text(self, text: str, document_id: Optional[str] = None) -> List[LangchainDocument]:
        source = document_id or "text_input"
        documents = self.loader.load_from_text(text, source)
        if document_id:
            for doc in documents:
                doc.metadata['document_id'] = document_id
        chunks = self.splitter.split_documents(documents)
        for idx, chunk in enumerate(chunks):
            chunk.metadata['chunk_index'] = idx
        return chunks
