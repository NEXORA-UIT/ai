# Nexora AI Foundation and RAG Handoff

> Historical foundation design. The current package layout is documented in the [README](../../../README.md): `agents/` is the single Agent placeholder, and empty RAG subpackages were removed.

## Intent

Prepare the existing Python Phase 0 service for the next implementation stage: a minimal end-to-end document ingestion and RAG query slice that Đạt and Bảo can build without guessing service ownership or changing the SRS architecture. The repository is named `ai`; its Python package is `nexora_ai`.

## Goals

- Keep two standalone Python processes: FastAPI for internal retrieval/query requests and a Python BullMQ worker for document ingestion.
- Keep Express as the public API, authentication/RBAC boundary, R2 upload owner, Prisma migration owner, and BullMQ ingestion producer.
- Remove the proposed Node extraction/ingestion worker from the RAG design. The existing Node workers for email and maintenance stay in the BE repo.
- Keep BullMQ and Apache Tika because the SRS calls for both. Python consumes the RAG queue, reads the object with read-only storage credentials, and calls a resource-limited Tika service for extraction.
- Let Python write agreed chunk/vector data and document processing states through psycopg 3; Express continues to own schema migrations.
- Separate `nexora_ai.rag` (ingestion, retrieval, answer generation) from `nexora_ai.agent` (future tool orchestration). No Agent or real ingestion/retrieval behavior is added in this foundation step.
- Define shared Pydantic citation/SSE event models and document revision guards and contract synchronization so implementation does not invent them later.

## Non-goals

- No real upload, Tika parsing, chunking, embedding, vector search, generation, SSE endpoint, Agent loop, proposal execution, or Langfuse integration. Contract models and JSON Schemas are allowed without implementing these behaviors.
- No PostgreSQL queue, Celery/RQ, ORM, Python migration framework, LangGraph, or second vector database.
- No change to existing BE email or maintenance workers.

## Approaches considered

1. **Keep a Node extraction worker.** Rejected for RAG: it duplicates responsibilities and forces an extra text/result handoff between Node and Python. It conflicts with the agreed Python ingestion owner. Existing unrelated Node workers remain.
2. **Replace BullMQ with a PostgreSQL polling queue.** Rejected: the SRS specifies Redis/BullMQ for asynchronous work, retry, and progress. A `SKIP LOCKED` queue would be another queue implementation.
3. **Python BullMQ worker + Tika + shared database (selected).** Express creates the document and enqueues the versioned job; Python reads R2, calls Tika, chunks/embeds and writes rows/status under a processing-revision guard. This keeps one ingestion worker and follows the SRS while retaining Express ownership of auth, public APIs, and migrations.

## Architecture decisions

### Ingestion

1. Express validates access/file limits, writes the file to R2, commits `Document(PENDING, processingRevision)`, then enqueues `document.ingest.v1` with IDs, revision, and correlation ID only.
2. Python Worker is the only consumer for the RAG queue. It downloads the object from R2 using a dedicated read-only key and sends it to the isolated Tika service with the SRS resource/time limits. The queue message contains only identifiers, never signed URLs or storage credentials.
3. The worker chunks extracted text, requests dense BGE-M3 embeddings from one shared embedding runtime, and writes `DocumentChunk` rows using psycopg 3.
4. Express owns `PENDING`; Python owns `PROCESSING`, `READY`, and `FAILED`. Chunk writes and final state updates must check the current `processingRevision`; stale jobs cannot publish chunks or mark a newer revision ready. `READY` is written only after a successful full revision.
5. BullMQ tracks technical job state; PostgreSQL `Document.status` is the business state. Job IDs are deterministic from document ID plus processing revision. Queue retry is bounded and exponential per the SRS.

### Retrieval and citations

1. FE calls Express only. Express authenticates, checks board authorization, and calls FastAPI over the internal service boundary.
2. FastAPI embeds the query with the same BGE-M3 model/configuration as ingestion, filters by `boardId` and ready documents, joins `Document` metadata, and returns citations with `documentName`, chunk IDs, snippets, and locators.
3. `pageNumber` is optional; `sourceLocator` supports TXT/DOCX and non-page-based sources.
4. The public answer contract is streamed as SSE to satisfy the SRS first-token target. Events are JSON in SSE `data:` lines and include the request `correlationId`:
   - `answer.delta`: `{schemaVersion, correlationId, delta}`; emitted for each generated text segment.
   - `citation`: `{schemaVersion, correlationId, citation}`; may repeat, with `documentId`, `documentName`, `chunkId`, `snippet`, optional `pageNumber`, and optional `sourceLocator`.
   - `done`: `{schemaVersion, correlationId, answer, citations}`; emitted once after a completed answer.
   - `error`: `{schemaVersion, correlationId, code, message, retryable}`; `code` is `RAG_QUERY_FAILED`, `MODEL_UNAVAILABLE`, or `REQUEST_TIMEOUT`; sanitized public error only, followed by stream close.
   Pydantic DTOs and JSON Schemas define these payloads now; no stream endpoint is implemented here.

### Agent boundary

`rag` is a reusable capability: it indexes knowledge and answers grounded document questions. `agent` is a separate future module that plans tool calls such as `getCards` and `searchDocuments`. Keep the tool loop separate from ingestion/retrieval. Express remains the authorization and mutation boundary; proposed changes are returned for approval and executed by Express with the SRS idempotency/OCC rules. Do not add LangGraph in this foundation.

### Embedding and model runtime

Use BGE-M3 dense vectors at dimension 1024 as the current target. API and worker must share the same model version and preprocessing/normalization via one embedding runtime, avoiding separate 568M parameter copies. Hosting (local service or compatible embedding endpoint) remains an explicit implementation decision and must be benchmarked against the SRS document-time target before deployment.

### Contracts and schema ownership

- Pydantic is authoritative for internal HTTP and BullMQ payload schemas; Express owns public OpenAPI.
- `CitationV1` requires `documentName`, `documentId`, `chunkId`, and `snippet`; only `pageNumber` and `sourceLocator` are optional. Each SSE event payload has its own versioned Pydantic model/schema.
- The BE stores a committed snapshot of `schemas/v1`. Contract changes update both repositories in one change set; v1 rejects unknown keys, so any shape change requires coordinated rollout.
- Queue payloads never contain document bytes, raw extracted text, secrets, or signed URLs.
- Prisma owns migrations and table shape; Python uses explicit psycopg SQL and must not create or migrate business tables.

## Constraints and open items for the next implementation

- Confirm `Document`/`DocumentChunk` column names, constraints, indexes, cascade/delete policy, and pgvector dimension in BE before writing ingestion SQL.
- Configure the Tika service/container and enforce SRS isolation, time, CPU, memory, file/page/text limits.
- Provision a dedicated read-only R2 credential for Python; rotate/revoke it independently from Express write credentials.
- Select and benchmark one shared BGE-M3 runtime; verify vector dimension and normalization on both ingestion and query.
- Define how an Express enqueue failure after `Document(PENDING)` is retried/reconciled without introducing a second queue. This foundation records the concern but does not implement an outbox.
- Implement a revision-guarded transaction: discard stale work, replace prior chunks for the document revision safely, and set `READY` only after all chunks are stored.
- Implement the documented SSE event flow and internal FastAPI service authentication before opening production query routes.
- Verify citation/document metadata and page-less locators against the FE/public OpenAPI contract.

## Source authority

The SRS PDF supplied by the user is authoritative for system behavior, in particular §2.6.2, §4.5.2–4.5.5, §4.6.1–4.6.3, §5.5.4, §6.1.4, §6.2.3, §6.3.4, and §6.4.1. This design simplifies implementation boundaries without replacing those requirements.
