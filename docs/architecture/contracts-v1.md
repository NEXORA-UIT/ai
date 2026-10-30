# Nexora AI Service Contracts — v1

Python Pydantic models define internal FastAPI/Worker payloads and the BullMQ protocol. Express owns public OpenAPI. Export JSON Schemas from the AI repo:

    uv run python scripts/export_contracts.py

The BE repo keeps a snapshot at backend/contracts/rag/v1. For any internal DTO change, update the Python export and BE snapshot in the same change set, then run contract tests in both repos.

## Boundaries and ownership

- FE calls public Express endpoints only.
- Express authenticates users, verifies board membership, handles R2 writes and Prisma migrations, creates Document(PENDING), and enqueues ingestion jobs.
- Python Worker is the only consumer of nexora-rag-ingestion. It downloads R2 objects using a dedicated read-only key; secrets and object URLs never enter queue payloads.
- Python owns PROCESSING, READY and FAILED transitions and chunk writes. Each write must check documentId plus processingRevision; stale revisions are no-ops.
- FastAPI is internal to Express, filters retrieval by boardId and READY state, and joins Document metadata for citations.
- PostgreSQL holds business state; BullMQ holds technical job state and retry/progress. Do not create a second PostgreSQL queue.
- Prisma owns schema/migrations. Python uses psycopg 3 and does not run Alembic or alter business schema.
- Existing Node email and maintenance workers are unrelated and remain in the BE repo.

## Contracts

| Model | Direction | Required fields | Purpose |
| --- | --- | --- | --- |
| PingJobV1 | Node producer → BullMQ → Python Worker | schemaVersion, correlationId | Phase 0 transport probe |
| PingResultV1 | Python Worker → BullMQ completed result | schemaVersion, correlationId, accepted, expressStatus | Phase 0 probe result |
| IngestionJobV1 | Express → BullMQ → Python Worker | schemaVersion, correlationId, documentId, boardId, processingRevision | Versioned processing signal; identifiers only |
| RagQueryRequestV1 | Express → FastAPI | schemaVersion, boardId, correlationId, question | Internal board-scoped query |
| RagAnswerV1 | FastAPI → Express | schemaVersion, correlationId, boardId, answer, citations | Structured final answer DTO |
| CitationV1 | RagAnswerV1 or event | documentId, documentName, chunkId, snippet | pageNumber and sourceLocator are optional |
| RagAnswerDeltaEventV1 | FastAPI → Express stream | schemaVersion, correlationId, delta | Payload for SSE answer.delta |
| RagCitationEventV1 | FastAPI → Express stream | schemaVersion, correlationId, citation | Payload for SSE citation |
| RagAnswerDoneEventV1 | FastAPI → Express stream | schemaVersion, correlationId, answer, citations | Payload for SSE done |
| RagErrorEventV1 | FastAPI → Express stream | schemaVersion, correlationId, code, message, retryable | Sanitized payload for SSE error; code is RAG_QUERY_FAILED, MODEL_UNAVAILABLE, or REQUEST_TIMEOUT |

Wire keys are camelCase. IDs are UUID strings. Schema version is integer 1; booleans do not count as integers. Unknown fields are rejected. Whitespace-only text is invalid. Query text is capped at 2,000 characters and citation snippets at 1,000. pageNumber and processingRevision are positive integers. documentName is required; pageNumber and sourceLocator may be absent for TXT/DOCX.

## BullMQ semantics

- Queue: nexora-rag-ingestion.
- Phase 0 job: phase0.ping, transport verification only.
- Ingestion job: document.ingest.v1.
- Derive the ingestion jobId from documentId plus processingRevision, never correlationId.
- Retry transient worker failures with bounded exponential backoff per SRS. Mark Document FAILED only after attempts are exhausted and only if processingRevision is current.
- BullMQ retries the same payload. Handler behavior must be idempotent for documentId + processingRevision.
- Express commits Document before enqueueing. Define recovery for enqueue failure; this foundation does not add an outbox or alternate queue.

## SSE event contract

Each SSE data line carries one JSON payload. The SSE event name selects the Pydantic DTO. Payloads include schemaVersion and correlationId but do not repeat the event name.

| Event | Pydantic payload | Data |
| --- | --- | --- |
| answer.delta | RagAnswerDeltaEventV1 | Delta text |
| citation | RagCitationEventV1 | Full citation; may be repeated |
| done | RagAnswerDoneEventV1 | Final answer and citations; exactly once on success |
| error | RagErrorEventV1 | RAG_QUERY_FAILED, MODEL_UNAVAILABLE, or REQUEST_TIMEOUT; safe message and retryable flag; exactly once on failure, then close |

Never expose exception text, SQL, credentials, signed URLs or stack traces in stream events. DTOs are contract only; Phase 0 has no streaming route.

## Change and version policy

- Regenerate Python schemas and update backend/contracts/rag/v1 for every internal DTO or queue name change.
- Update public shape in backend/docs/api/openapi and FE client types separately.
- V1 rejects unknown fields, so additions need coordinated rollout; deploy compatible consumers before producers.
- Breaking field, meaning, status, retry, or event changes require a new version.
- Never put document text, file bytes, storage URLs, access tokens or secrets in job messages or traces.
