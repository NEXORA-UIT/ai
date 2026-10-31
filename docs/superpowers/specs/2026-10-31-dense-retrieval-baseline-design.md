# Dense Retrieval Baseline — Design

**Status:** Approved with review amendments on 2026-10-31.

## Intent

Implement milestone 1 of Nexora RAG retrieval inside the `ai` repository: embed a question with the same pinned BGE-M3 implementation used by ingestion, run pgvector cosine-distance search, and return the top chunks from one Board's `READY` documents. The caller supplies `board_id` as the search scope. Express will authorize access to that Board when the services are integrated.

## Scope and behavior

- A synchronous, standalone retrieval module in `src/retrieval/` exposes two small operations: a vector-search core accepting a 1024-dimensional query vector, and a question-search wrapper that calls the existing `BgeM3Embedder.embed_texts([question])` before the core. Reuse one lazily loaded default `BgeM3Embedder` instance across queries; allow an embedder to be injected for tests.
- Both operations require `board_id`; `top_k` defaults to 5 and must be positive. The SQL joins `document_chunks` to `documents`, filters `d.board_id` and `d.status = 'READY'` **inside the query**, orders by pgvector cosine distance (`<=>`), and applies `LIMIT top_k`. A stable chunk-ID tie-breaker keeps equal-distance results deterministic.
- Each result includes chunk ID and index, document ID and file name, content, page number when present, character offsets, storage key, and cosine distance (lower means more similar). Return an empty list when no eligible chunks exist. Do not apply a relevance threshold in this milestone.
- Reuse the ingestion `Embedder` protocol, pinned `BgeM3Embedder` configuration, `DocumentStorage.get_connection()` for a registered pgvector psycopg connection, and the existing dev/test schema. Keep the retrieval code in its own module without changing ingestion or shared service files.
- Validate a whitespace-only question, malformed Board UUID, invalid `top_k`, and malformed vector before querying. A valid vector has exactly 1024 finite values and is not the zero vector. Parameterize values in SQL.
- Bind query vectors explicitly as `pgvector.Vector` so psycopg sends the `vector` type, and verify the binding against a live PostgreSQL/pgvector test database.

## Approaches considered

1. **Single end-to-end function:** embeds and queries in one operation. It has fewer functions, but tests and future search methods cannot exercise the SQL with a fixed vector as cleanly.
2. **Vector core plus question wrapper (chosen):** the wrapper reuses BGE-M3; the core can be tested with fixed vectors and reused by later retrieval methods. This adds one small function boundary.

## Constraints and non-goals

- Work directly in the current `ai` checkout and branch; do not create a worktree. Preserve existing uncommitted files.
- `scripts/bootstrap_dev_db.sql` remains a dev/test fixture, not a Backend Prisma migration or production schema contract.
- Milestone 2 handles `model_id`/revision compatibility and additional metadata filters. This milestone assumes the test corpus was embedded with the pinned BGE-M3 revision.
- No hybrid search, RRF, reranking, LLM, citation generation, FastAPI route, Agent API, or authorization logic. Returning source fields now enables later citation work without claiming that citations are implemented.

## Verification

- Fast tests use a fake embedder and fixed 1024-dimensional vectors to check input validation, wrapper-to-core behavior, ordering, `top_k`, result fields, and empty results.
- A PostgreSQL/pgvector test using the existing test schema seeds two Boards and `READY`/`PROCESSING`/`FAILED` documents, then verifies that Board and status filtering occur in SQL and that only eligible chunks are ranked. It runs only against an explicitly configured test database.
- With `top_k = 1`, seed a closer chunk in Board B than any chunk in Board A; querying Board A must still return Board A's best chunk. This guards against applying Board filtering after the limit.
- Run the repository test suite, Ruff lint, and Ruff format checks. A real-model smoke test may run separately when BGE-M3 weights are available; it must not slow the fast suite.

## Builder decisions left open

The implementer may choose internal function names and exception types consistent with the repository. The output remains a typed result object with the fields above; no stable Agent-facing `search_documents()` contract is introduced until milestone 8.
