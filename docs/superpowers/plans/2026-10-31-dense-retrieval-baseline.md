# Dense Retrieval Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement milestone 1 dense retrieval with BGE-M3 query embeddings and pgvector Top-k results scoped to one Board and `READY` documents.

**Architecture:** Add a vector-search core and a question wrapper under `src/retrieval/`. Reuse `BgeM3Embedder`, cache its default instance per process, and reuse `DocumentStorage.get_connection()`; explicitly bind query values with `pgvector.Vector`.

**Tech Stack:** Python 3.12, psycopg 3, pgvector, pytest, Ruff, existing BGE-M3 SentenceTransformers adapter.

**Spec:** `docs/superpowers/specs/2026-10-31-dense-retrieval-baseline-design.md`

## Global Constraints

- Work directly in the current `ai` checkout and branch; do not create a worktree.
- Preserve existing uncommitted user files; stage only the retrieval files and approved design/plan documents for the feature commit.
- Keep `board_id` and `status = 'READY'` predicates in the SQL query before ordering and `LIMIT`.
- Use the existing BGE-M3 model ID, pinned revision, 1024 dimensions, and normalized question embedding.
- Reuse the existing dev/test schema; do not modify production schema, Prisma migrations, or shared ingestion code.
- Return Top-k even when scores are weak; return an empty list only when no eligible chunks exist.
- Keep model compatibility metadata checks, hybrid search, RRF, reranking, LLM, API routes, and Agent integration out of milestone 1.

## Review Focus

- Whitespace-only questions must fail before loading or calling the model (`test_search_by_text_rejects_whitespace_question`).
- Vectors with the wrong dimension, non-finite elements, or all-zero elements must fail before opening a connection (`test_search_by_vector_rejects_invalid_vector`).
- Invalid Board UUIDs and non-positive `top_k` values must fail before opening a connection (`test_search_by_vector_rejects_invalid_scope_or_limit`).
- A closer chunk in Board B must not displace Board A's best chunk when querying Board A with `top_k=1` (`test_top_k_one_filters_board_before_limit_and_binds_pgvector_type`).
- psycopg must bind the query as pgvector's `vector` type, verified by executing the search against PostgreSQL/pgvector (`test_query_vector_binding_and_board_ready_filtering`).

---

### Task 1: Add typed results and fixed-vector search

**Files:**
- Create: `src/retrieval/__init__.py`
- Create: `src/retrieval/schemas.py`
- Create: `src/retrieval/dense.py`
- Create: `tests/retrieval/conftest.py`
- Test: `tests/retrieval/test_dense_search.py`

**Interfaces:**
- Produces `RetrievedChunk` in `src/retrieval/schemas.py` with fields `chunk_id: str`, `document_id: str`, `chunk_index: int`, `file_name: str`, `storage_key: str`, `content: str`, `page_number: int | None`, `start_offset: int | None`, `end_offset: int | None`, and `cosine_distance: float`.
- Produces `search_by_vector(query_vector: Sequence[float], board_id: str | UUID, *, top_k: int = 5, storage: DocumentStorage | None = None) -> list[RetrievedChunk]`.
- Uses `DocumentStorage()` when `storage` is omitted; callers can inject a configured storage for tests.

- [x] **Step 1: Sync the locked Python environment**

Run: `uv sync --locked`
Expected: the project `.venv` contains the exact dependencies in `uv.lock`.

- [x] **Step 2: Write failing validation and SQL unit tests**

Add tests named `test_search_by_vector_rejects_invalid_vector`, `test_search_by_vector_rejects_invalid_scope_or_limit`, `test_search_by_vector_uses_board_ready_cosine_query`, `test_search_by_vector_maps_source_fields_and_distance`, and `test_top_k_one_filters_board_before_limit_and_binds_pgvector_type`. The last test seeds an isolated PostgreSQL/pgvector fixture with Boards A/B and ready/non-ready documents; Board B has the exact query vector while Board A has the best eligible non-exact vector. Assert Board A's best result still wins at `top_k=1`. Assert invalid inputs do not call `get_connection`; assert the query joins `document_chunks` to `documents`, filters Board and `READY`, orders by `<=>`, and binds `top_k` after the filters. The fixture skips unless `TEST_DATABASE_URL` is explicitly set and cleans up only the document IDs it creates.

- [x] **Step 3: Run the focused tests and confirm they fail for the missing API**

Run: `\.venv\Scripts\python.exe -m pytest tests/retrieval/test_dense_search.py -q -m "not database"`
Expected: FAIL because the retrieval module and types are not implemented.

- [x] **Step 4: Implement the result type and vector-search core**

Validate UUID, positive `top_k`, and a vector of exactly 1024 finite floats with at least one non-zero value before connecting. Use `Vector(query_vector)` as the parameter for `dc.embedding <=> %(query_vector)s::vector`; select the distance and source fields without selecting the embedding. Order by cosine distance, then chunk ID for stable ties, and apply `LIMIT %(top_k)s` in the same query as the Board/status filters.

- [x] **Step 5: Run the focused tests and confirm they pass**

Run: `\.venv\Scripts\python.exe -m pytest tests/retrieval/test_dense_search.py -q -m "not database"`
Expected: PASS, including no-connection assertions for invalid inputs.

---

### Task 2: Add question embedding wrapper and process-level embedder reuse

**Files:**
- Modify: `src/retrieval/dense.py`
- Modify: `src/retrieval/__init__.py`
- Test: `tests/retrieval/test_dense_search.py`

**Interfaces:**
- Consumes `RetrievedChunk` and `search_by_vector` from Task 1.
- Produces `get_default_embedder() -> BgeM3Embedder`, cached with `lru_cache(maxsize=1)`.
- Produces `search_by_text(question: str, board_id: str | UUID, *, top_k: int = 5, embedder: Embedder | None = None, storage: DocumentStorage | None = None) -> list[RetrievedChunk]`.

- [x] **Step 1: Write failing wrapper and cache tests**

Add `test_search_by_text_rejects_whitespace_question`, `test_search_by_text_embeds_once_and_delegates_vector`, and `test_default_embedder_instance_is_reused`. Use a fake embedder returning one fixed 1024-dimensional normalized vector and a mocked vector-search operation; assert the question is passed once, the Board/limit/storage are forwarded, and the cached default factory returns the same object.

- [x] **Step 2: Run wrapper tests and confirm they fail for the missing API**

Run: `\.venv\Scripts\python.exe -m pytest tests/retrieval/test_dense_search.py -q -k 'text or embedder' -m "not database"`
Expected: FAIL because the wrapper and cached default are not implemented.

- [x] **Step 3: Implement the wrapper and cached default embedder**

Reject whitespace-only input before obtaining the embedder. Default to the cached `BgeM3Embedder`; call `embed_texts([question])` once and pass its sole vector to `search_by_vector`. Keep embedder injection for fast tests.

- [x] **Step 4: Run the focused tests and confirm they pass**

Run: `\.venv\Scripts\python.exe -m pytest tests/retrieval/test_dense_search.py -q -m "not database"`
Expected: PASS for both vector core and wrapper tests.

---

### Task 3: Prove Board/status isolation and pgvector binding against the test database

**Files:**
- Modify: `tests/retrieval/test_dense_search.py`

**Interfaces:**
- Uses `DocumentStorage(database_url=TEST_DATABASE_URL)` and the schema from `scripts/bootstrap_dev_db.sql`.
- Seeds only rows with unique UUIDs and removes only those test documents during fixture cleanup.

- [x] **Step 1: Run the live pgvector test written before implementation**

The fixture and `test_top_k_one_filters_board_before_limit_and_binds_pgvector_type` are added in Task 1 before the retrieval implementation. Seed two Boards with normalized 1024-dimensional vectors. Make Board B's vector identical to the query, while Board A has a best vector with non-zero cosine distance; also seed `PROCESSING` and `FAILED` documents in Board A. Assert Board A with `top_k=1` returns its own best `READY` chunk and Board B/non-ready chunks never appear. This simultaneously proves explicit pgvector binding and filtering before the limit.

- [x] **Step 2: Run the live tests against the configured test database**

Run: `\.venv\Scripts\python.exe -m pytest tests/retrieval/test_dense_search.py -q -m database`
Expected: PASS against an explicitly configured `TEST_DATABASE_URL` with pgvector and the dev/test tables. If the variable is absent, pytest reports a skip; do not report the integration gate as passed.

- [x] **Step 3: Run the full retrieval test file**

Run: `\.venv\Scripts\python.exe -m pytest tests/retrieval/test_dense_search.py -q --basetemp .tmp-pytest-run`
Expected: all 14 retrieval unit and pgvector integration tests pass. The repository-wide suite was also attempted; record separately any pre-existing external service/model failures that prevent a complete run.

- [x] **Step 4: Run Ruff lint**

Run: `\.venv\Scripts\ruff.exe check .`
Expected: no lint errors.

- [x] **Step 5: Check Ruff formatting**

Run: `\.venv\Scripts\ruff.exe format --check .`
Expected: all Python files already formatted.

- [x] **Step 6: Commit the milestone**

Stage only `src/retrieval/`, `tests/retrieval/`, the approved design, and this plan. Leave the pre-existing modified README, prompt, architecture, parallel plan, and assets unstaged.

```bash
git add src/retrieval tests/retrieval docs/superpowers/specs/2026-10-31-dense-retrieval-baseline-design.md docs/superpowers/plans/2026-10-31-dense-retrieval-baseline.md
git commit -m "feat(retrieval): add scoped dense vector search"
```
