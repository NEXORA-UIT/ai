# Nexora RAG Architecture

The AI repository provides an internal FastAPI demo with `/health`,
`POST /v1/documents`, `POST /v1/search`, and `POST /v1/answer`. The document
route stages a local file and ingests it synchronously. Search and answer use
Board-scoped data from PostgreSQL. Run the demo with Uvicorn and connect it
to a dedicated development database and an Apache Tika server.

Express remains responsible for user authorization and the public API.
The Redis/BullMQ worker and Express integration described below are the
target deployment flow. The internal demo routes accept a service key.

## Components and responsibilities

| Component | Agreed responsibility |
| --- | --- |
| Frontend | Calls the public Express API; does not call the AI service directly. |
| Express | Authenticates users, authorizes access by Board, owns the public API, uploads files to R2, manages document metadata and Prisma migrations, enqueues ingestion through BullMQ, and connects to FastAPI for RAG queries. |
| Python Worker | Independent BullMQ consumer for RAG jobs; reads documents, runs ingestion, and writes vector data. No intermediate Node.js RAG worker is needed. |
| FastAPI | Retrieves information within a Board's scope and generates answers from documents with source references. |
| Shared infrastructure | PostgreSQL + pgvector stores vectors; Redis + BullMQ carries background work. |

The backend's existing Node.js workers continue to handle other jobs; they are not part of RAG ingestion.

## Source layout

- `src/api/` owns HTTP contracts. `src/main.py` assembles the FastAPI app and
  controls warmup.
- `src/ingestion/` owns file processing. `src/documents/` defines the document
  models and status used by ingestion, embeddings, and persistence.
- `src/persistence/` owns PostgreSQL errors, scoped reads, and transactional
  writes. Both retrieval and ingestion use it.
- `src/retrieval/` owns the hybrid workflow shared by the API and evaluation.
  `src/answering/` turns retrieved evidence into answers and citations.
  `src/providers/deepseek/` handles the provider protocol and transport.
- `scripts/ingestion/` holds local commands. `scripts/db/bootstrap.py` checks
  the connected database name before applying the dev schema fixture.
  `evaluation/` holds the datasets, metrics, and reports.

See the [README](../../README.md#repository-map) for the full source map.

## Target ingestion flow

The frontend sends a document to Express. Express checks authorization, manages the file and metadata, then queues background work through BullMQ. The Python Worker reads the document, creates chunks and embeddings using **BAAI/bge-m3**, then writes data to PostgreSQL + pgvector using psycopg 3. Ingestion must handle failures and retries safely. When a document is reprocessed, an older result must not overwrite the new one. Queue messages must not contain secrets or large document contents.

## Target retrieval and answer flow

The frontend sends a question through Express. Express handles authentication and Board-level authorization before connecting to FastAPI. FastAPI retrieves only data in the authorized Board scope and generates an answer with references to the source documents. The Worker and FastAPI must use the same BGE-M3 configuration so document and query embeddings are compatible.

## Agreed technology stack

| Area | Technology |
| --- | --- |
| Runtime/API | Python 3.12, FastAPI, Uvicorn |
| Background ingestion (planned) | Python BullMQ worker, Redis + BullMQ |
| Embeddings | BAAI/bge-m3, dense vectors with **1024 dimensions** |
| Vector database | PostgreSQL + pgvector; Python uses psycopg 3, with no added ORM |
| Retrieval | Exact cosine dense search, `rank-bm25`, RRF, BGE CPU reranking |
| Generation | DeepSeek JSON completion and evidence-backed citation IDs |
| Evaluation | A/B/C harness with optional Langfuse traces |

Dense and sparse reads share one PostgreSQL eligibility predicate for Board,
READY status, and embedding fingerprint. Evaluation and the API share the
candidate, fusion, and reranking stages. The default shortlist is five
candidates. Set `AI_RERANK_MAX_LENGTH=192` for the optional CPU profile.
Full chunk text is retained for citations and generation.

## Decisions for implementation phases

- Ingestion: queue/job names, payload, retry/backoff policy, and coordination between writes and enqueueing.
- Express ↔ FastAPI communication: route, DTO, internal authentication, response format, and whether to use streaming/SSE.
- Source references: citation structure and how document positions are represented.
- Data: processing states, prevention of stale results, chunk/vector schema, index, and concrete migrations.
- Embeddings: BGE-M3 runtime, batching, caching, and operational configuration. The model and 1024-dimensional output are already agreed.
- Generation and evaluation: specific LLM, dataset, metrics, and quality thresholds.

Agree on these details when ingestion and retrieval/generation begin, including any required changes in both repositories.
