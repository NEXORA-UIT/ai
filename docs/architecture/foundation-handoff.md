# Foundation Handoff: What to Build Next

This note is the short checklist to use before starting the full RAG flow. The architecture and event details are in [rag-overview.md](rag-overview.md); shared internal DTO rules are in [contracts-v1.md](contracts-v1.md).

## What Phase 0 gives the team

- FastAPI and Python Worker are separate local processes in the AI repo.
- Both are Python modules in one installable package and share Pydantic settings/contracts.
- Express can enqueue the Phase 0 probe on the shared BullMQ queue; only the Python Worker consumes the RAG queue.
- PostgreSQL access and pgvector probing use psycopg 3; Prisma owns migrations.
- The `rag/` and `agents/` packages are separate placeholders. Add real RAG modules as the flow is implemented; keep a single `agents/` package for now.
- No business ingestion/query logic is included.

## Before Đạt implements ingestion

BE/Bảo should provide and commit:

- Prisma models and migration for Document/DocumentChunk, including pgvector dimension, board/document relationships, indexes and delete/reupload behavior.
- A processingRevision rule and state-transition ownership: Express creates PENDING; Python owns PROCESSING, READY and FAILED.
- A retry policy and behavior when BullMQ enqueue fails after Document is committed.
- A dedicated read-only R2 credential for Python and a local config example that contains no real secrets.
- Tika service endpoint and enforcement for the SRS CPU, memory, time, file, page and text limits.
- Agreement that stale revisions cannot write chunks or mark current document READY.

Then Đạt can deliver a vertical slice with one TXT file:

1. Consume document.ingest.v1 and validate its Pydantic payload.
2. Load Document by ID/revision and transition to PROCESSING only if the revision is current.
3. Download the R2 object using the read-only credential.
4. Call Tika and preserve page/paragraph source locators.
5. Split text and request embeddings from the shared BGE-M3 runtime.
6. In one revision-guarded write flow, replace the current chunks and mark READY.
7. On retryable errors, throw for BullMQ retry; mark FAILED only after final failure and only for the current revision.

## Before Bảo implements retrieval

Agree on the internal FastAPI request and caller authentication, then:

1. Validate the query DTO.
2. Embed with the exact same BGE-M3 runtime and preprocessing as ingestion.
3. Filter SQL by boardId and Document.status = READY.
4. JOIN Document to return documentName with each citation.
5. Allow missing pageNumber; use sourceLocator for formats without pages.
6. Generate a grounded answer from retrieved context.
7. Stream answer.delta, citation, done, or sanitized error events through FastAPI and Express.

## Shared acceptance walkthrough

- Upload a small TXT to an authorized board through Express.
- Confirm Express returns the document as PENDING and enqueues one versioned job.
- Confirm Python Worker consumes it and DB status changes through PROCESSING to READY.
- Inspect DocumentChunk rows and verify stored vector dimension and sourceLocator.
- Ask a question from that board and verify streaming output includes documentName and a source locator.
- Confirm another board cannot retrieve the same chunks.
- Re-enqueue a stale revision and verify it does not replace current chunks or status.
- Retry a transient Tika/embedding failure and verify terminal failure behavior after the configured attempts.

PDF/DOCX, Langfuse experiments, Agent workflows and proposal actions follow after this TXT acceptance path.
