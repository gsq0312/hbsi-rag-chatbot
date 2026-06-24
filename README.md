# HBSI RAG Chatbot Template / 招生咨询 RAG 问答模板

This repository is a FastAPI template for a RAG-based admissions chatbot. It keeps the application code only. The public version does not include school knowledge-base text, uploaded documents, vector-store indexes, databases, or API keys.

本仓库是一个基于 FastAPI 的招生咨询 RAG 问答模板。公开版本只保留应用代码，不包含学校知识库正文、上传文档、向量库索引、数据库或 API Key。

## Features / 功能

- Document upload and chunking for TXT, PDF, and DOCX files
- FAISS-based local vector retrieval
- HuggingFace embeddings with a hash-based fallback
- DeepSeek-compatible chat completion calls through environment variables
- Runtime configuration endpoints

## Public Data Policy / 公开数据说明

No private or production data is committed.

本仓库不提交任何私有或生产数据。

Ignored local runtime data includes:

- `data/*.txt`
- `data/*.pdf`
- `data/*.docx`
- `data/*.csv`
- `data/*.xlsx`
- `data/*.jsonl`
- `data/vector_store/`
- `*.faiss`
- `*.pkl`
- `*.db`
- `.env`

If you deploy or reuse this project, prepare your own knowledge base locally and upload it at runtime. Do not commit generated indexes or source documents.

如需部署或复用，请自行准备本地知识库并在运行时上传。不要提交生成的向量索引或原始资料文件。

## API Key Configuration / API Key 配置

DeepSeek is configured through environment variables only. This repository does not include any usable API key.

DeepSeek 只通过环境变量配置。本仓库不包含任何可用 API Key。

```bash
cp .env.example .env
```

Then edit `.env`:

```env
DEEPSEEK_API_KEY=your-own-api-key
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
```

## Local Development / 本地运行

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8002
```

Open API docs:

```text
http://localhost:8002/docs
```

## API / 接口

- `POST /api/chat`
- `POST /api/upload`
- `GET /api/config`
- `PUT /api/config`
- `GET /health`

## Tech Stack / 技术栈

- FastAPI
- LangChain
- FAISS
- DeepSeek-compatible LLM API
- HuggingFace Embeddings
