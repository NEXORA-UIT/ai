# Parallel RAG Work in the AI Repository — Đạt and Bảo

**Status:** Alignment plan for the two developers before implementation; not a BE ↔ AI contract.

**Scope:** Only the `ai` repository. Two independent branches start from the same `staging` commit. Bảo reviews and merges after both branches pass their independent checks.

**First goal:** Ingest a TXT file into PostgreSQL + pgvector, then retrieve the correct passage from the correct Board. No LLM answer generation in this phase.

## 1. Agreed boundaries

| Owner | Responsibility |
| --- | --- |
| Đạt | Read TXT files, chunk text, embed documents, and write chunks, vectors, and source metadata. |
| Bảo | Embed test queries, implement semantic retrieval, filter by Board and document status, and return chunks with source information. |
| Backend (later phase) | Own Prisma schema/migrations, Board authorization, R2 uploads, BullMQ producer, and the public API. |

Both developers work **AI-only** against a test database. Do not change the backend, create business migrations in Python, or treat the test schema as a production schema approved by the backend team.

## 2. Decisions to align on before dependent work

A short working session should be enough to record the choices in the **Decision log** at the end of this document. If an item is still open, each developer may work on independent code, but the shared vertical slice must not be marked complete.

| Decision | Minimum agreement | Why it matters |
| --- | --- | --- |
| Embeddings | `BAAI/bge-m3` and dense 1024-dimensional vectors are agreed; pin the exact model revision, normalization, and distance metric. | Vectors can have the same dimensions and still rank results differently if their configuration differs. |
| Test data | Board ownership, searchable document states, the Document-to-Chunk relationship, and where text/vectors live. | Both branches must write and read the same fixture. The backend will decide the production schema later. |
| Source references | Each chunk has document/chunk identifiers, text, and source position; page numbers may be absent for TXT. | Retrieval must return verifiable sources rather than citations invented by an LLM. |
| Visibility and reprocessing | Only `READY` documents are searchable; deleted or superseded document versions must not appear. The backend will decide the concrete revision/column mechanism. | Prevent stale results and cross-Board leakage. |
| Shared test | One small TXT file with an unambiguous expected answer, plus data for Boards A/B and a document that is not ready. | The branches can run independently while measuring the same behavior. |

**Proposal requiring agreement:** Use normalized dense vectors and cosine distance for the first vertical slice; pin the same BGE-M3 revision. This is a proposal for the AI test slice and does not decide the production index or model deployment.

## 3. Environment setup

- [ ] Each developer has a separate checkout/worktree of the `ai` repository at the **same SHA from `origin/staging`**; record that SHA in the Decision log. Do not commit directly to `staging`.
- [ ] Python 3.12 and `uv sync --locked` work; baseline `uv run pytest -q` and Ruff checks pass.
- [ ] The test PostgreSQL database has the `vector` extension. Use a dedicated test database/schema, never the business database. `scripts.check_db` confirms connectivity.
- [ ] Both machines can download and run the same BGE-M3 revision. Fast tests use fixed vectors; only a smoke test loads the real model to keep the test suite quick.
- [ ] A sample TXT file and expected question are available; test content contains no real user data or secrets.

## 4. Đạt's branch — ingestion core

**Suggested branch:** `feat/ai-ingestion-core`.

**Main code areas:** `src/ingestion/`, `src/chunking/`, `src/embedding/`, `tests/ingestion/`. Đạt owns the shared embedding module after merge.

### Implementation order

1. Write a failing test: read the sample TXT into text chunks with source metadata; empty or unreadable input returns a clear error.
2. Implement TXT reading and minimal chunking. Do not add PDF, DOCX, Tika, or R2 support.
3. Write a failing embedding test: output is a dense 1024-dimensional vector using the agreed configuration. Use a fake in fast tests and one smoke test with real BGE-M3.
4. Write a failing PostgreSQL test: chunk count, text, vectors, and source metadata match the input. Mark data ready only after all writes succeed; a partial failure must not make an incomplete document searchable.
5. Run focused and full-repository tests plus Ruff; record commands and results in the PR.

**Independent acceptance:** A small TXT file is chunked, embedded, and written to pgvector. Read-back returns the correct number of chunks and source metadata, with controlled error behavior. No BullMQ Worker runtime or Express job is needed yet.

## 5. Bảo's branch — retrieval core

**Suggested branch:** `feat/ai-retrieval-core`.

**Main code areas:** `src/retrieval/`, `tests/retrieval/`. Do not edit Đạt's `src/embedding/` module on this branch.

### Implementation order

1. Seed a **test-only** schema with the agreed data: at least two Boards, `READY`/`PROCESSING`/`FAILED` documents, and 1024-dimensional chunk vectors.
2. Write a failing psycopg + pgvector semantic-search test: SQL must filter Board and document status; results include chunk/document identifiers, text, and source position.
3. Implement simple retrieval without reranking or hybrid search. Test result ordering, result limits, exclusion of other Boards and non-ready documents, and the no-evidence case.
4. Fast tests pass a fixed query vector. A separate smoke test may create a query vector with the agreed BGE-M3 revision; after merge, use Đạt's shared embedding module and remove the temporary duplicate production adapter.
5. Run focused and full-repository tests plus Ruff; record commands and results in the PR.

**Independent acceptance:** Retrieval finds the expected fixture chunks in the correct Board, includes only eligible documents, and returns verifiable source information. No FastAPI query route, LLM, or SSE is needed yet.

## 6. Avoiding conflicts during parallel work

- Each developer changes only their owned code areas. Coordinate before changing shared files such as `src/main.py`, `src/database.py`, README, or shared architecture.
- If both branches need dependencies, each may update `pyproject.toml`/`uv.lock` to run independently. During merge, resolve conflicts and rerun `uv lock` and `uv sync --locked`; do not manually select a stale lockfile.
- Test database fixtures may be duplicated initially so branches run independently. After merging, consolidate to one fixture; do not keep two test schemas with different meanings.
- Do not turn a test schema, temporary job name, or temporary route into a production contract just because the tests pass.

## 7. Merge gate owned by Bảo

1. Both branches pass `uv run pytest -q`, `uv run ruff check .`, and `uv run ruff format --check .`; each developer has tested against a real PostgreSQL instance at least once.
2. Review both PRs into `ai/staging`. Recheck the Decision log, especially model revision, vector normalization/distance, Board filtering, and source metadata.
3. Merge ingestion first so retrieval can use its shared embedding module; update/rebase retrieval and remove its temporary model-loading code if applicable.
4. Run **AI-only E2E** against a clean test database: TXT → ingestion → chunks/vectors → question → retrieval → expected passage and source. Also test another Board and a document that is not ready to confirm they are excluded.
5. Only after this gate, discuss backend Prisma migrations, BullMQ jobs, FastAPI ↔ Express API, generation/LLM, and response protocols. Passing AI-only E2E **does not prove** compatibility with the real backend schema. Once the backend migration exists, rerun tests against that schema and add deletion/reprocessing cases to confirm old chunks cannot leak into results.

## 8. Out of scope for these branches

BullMQ consumer runtime, uploads/R2, Tika/PDF/DOCX, FastAPI query route, internal authentication, LLM answers, SSE, Langfuse, reranking, Agents, and Prisma migrations. These belong to later phases and must not be added just to make the first demo look complete.

## 9. Decision log — fill in before dependent implementation

| Item | Result agreed by both developers |
| --- | --- |
| Shared SHA of `ai/staging` | _Not recorded_ |
| BGE-M3 model revision | _Not recorded_ |
| Dense-vector normalization and distance metric | _Not recorded_ |
| Test Document/Chunk data and searchable states | _Not recorded_ |
| Minimum TXT source metadata; PDF/DOCX rules deferred | _Not recorded_ |
| Sample TXT, question, and expected result | _Not recorded_ |
| Branch/PR names | _Not recorded_ |

**SRS traceability:** UC-KB-22 and sections 4.5.5, 6.8.2–6.8.3 require Board-scoped retrieval, eligible sources only, real citations, and no unsupported claims when evidence is missing. AI performance targets in section 6.1.4 will be measured once answer generation exists; they are not standalone acceptance criteria for these first two modules.
