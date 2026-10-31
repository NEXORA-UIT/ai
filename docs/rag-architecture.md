# Nexora RAG Architecture

This document records the agreed **responsibilities and technologies**; implementation contracts are still open. The service has a FastAPI `/health` endpoint and a read-only PostgreSQL/pgvector connectivity check. Standalone ingestion and Board-scoped dense-retrieval cores are implemented; the end-to-end flows below are not yet wired to a worker or service API.

## Components and responsibilities

| Component | Agreed responsibility |
| --- | --- |
| Frontend | Calls the public Express API; does not call the AI service directly. |
| Express | Authenticates users, authorizes access by Board, owns the public API, uploads files to R2, manages document metadata and Prisma migrations, enqueues ingestion through BullMQ, and connects to FastAPI for RAG queries. |
| Python Worker | Independent BullMQ consumer for RAG jobs; reads documents, runs ingestion, and writes vector data. No intermediate Node.js RAG worker is needed. |
| FastAPI | Retrieves information within a Board's scope and generates answers from documents with source references. |
| Shared infrastructure | PostgreSQL + pgvector stores vectors; Redis + BullMQ carries background work. |

The backend's existing Node.js workers continue to handle other jobs; they are not part of RAG ingestion.

## Ingestion flow

The frontend sends a document to Express. Express checks authorization, manages the file and metadata, then queues background work through BullMQ. The Python Worker reads the document, creates chunks and embeddings using **BAAI/bge-m3**, then writes data to PostgreSQL + pgvector using psycopg 3. Ingestion must handle failures and retries safely. When a document is reprocessed, an older result must not overwrite the new one. Queue messages must not contain secrets or large document contents.

## Retrieval and answer flow

The frontend sends a question through Express. Express handles authentication and Board-level authorization before connecting to FastAPI. FastAPI retrieves only data in the authorized Board scope and generates an answer with references to the source documents. The Worker and FastAPI must use the same BGE-M3 configuration so document and query embeddings are compatible.

## Agreed technology stack

| Area | Technology |
| --- | --- |
| Runtime/API | Python 3.12, FastAPI, Uvicorn |
| Background ingestion | Python BullMQ Worker, Redis + BullMQ |
| RAG framework | LangChain |
| Embeddings | BAAI/bge-m3, dense vectors with **1024 dimensions** |
| Vector database | PostgreSQL + pgvector; Python uses psycopg 3, with no added ORM |
| Generation | LLM API; answers include source references |
| Tracing/evaluation | Langfuse |

The Worker, LangChain, embeddings, generation, and Langfuse are planned for later phases and do not have runtime implementations in Phase 0.

## Decisions for implementation phases

- Ingestion: queue/job names, payload, retry/backoff policy, and coordination between writes and enqueueing.
- Express ↔ FastAPI communication: route, DTO, internal authentication, response format, and whether to use streaming/SSE.
- Source references: citation structure and how document positions are represented.
- Data: processing states, prevention of stale results, chunk/vector schema, index, and concrete migrations.
- Embeddings: BGE-M3 runtime, batching, caching, and operational configuration. The model and 1024-dimensional output are already agreed.
- Generation and evaluation: specific LLM, dataset, metrics, and quality thresholds.

Agree on these details when ingestion and retrieval/generation begin, including any required changes in both repositories.
