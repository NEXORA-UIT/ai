# Milestone 2 Metadata Filtering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Exclude chunks whose document embedding metadata does not match the query embedding model.

**Architecture:** Extend the existing dense SQL, keeping Board and READY predicates in the same query. Select documents.board_id into each RetrievedChunk so later stages can verify the evidence scope. The text wrapper derives the expected fingerprint from its embedder; the vector core accepts an explicit fingerprint with pinned BGE-M3 defaults.

**Tech Stack:** Python 3.12, psycopg 3, PostgreSQL JSONB, pgvector, pytest, Ruff.

**Spec:** docs/superpowers/specs/2026-11-01-rag-retrieval-m2-m8-design.md and docs/superpowers/specs/2026-10-31-dense-retrieval-baseline-design.md

## Global Constraints

- Preserve Milestone 1 Board and READY SQL predicates before ORDER BY and LIMIT.
- Default fingerprint: src/embedding/config.py BGE-M3 model ID and pinned revision, 1024 dimensions, normalized true.
- Missing metadata excludes a document; do not silently accept legacy unknown vectors.
- Keep ingestion workflow and production schema unchanged. Only retrieval and test fixture files may change.
- Test with python -m pytest; one task, one commit after verification.

## Review Focus

- Document with missing model_id or model_revision is excluded (test_missing_metadata_excluded).
- Same model ID with wrong revision is excluded (test_wrong_revision_excluded).
- Wrong dimension or normalized=false is excluded (test_wrong_vector_metadata_excluded).
- Other Board exact-vector decoy cannot displace the eligible Board result at top_k=1 (test_metadata_filter_before_limit).
- Returned board_id comes from documents.board_id rather than being copied from the caller's argument (test_dense_result_has_database_board_id).
- Injected query embedder must drive the expected fingerprint, not a stale global constant (test_text_wrapper_uses_embedder_fingerprint).

---

### Task 1: Apply embedding compatibility inside dense SQL

**Files:**
- Create: src/retrieval/metadata.py
- Modify: src/retrieval/dense.py
- Modify: src/retrieval/schemas.py
- Modify: src/retrieval/__init__.py
- Modify: tests/retrieval/conftest.py
- Test: tests/retrieval/test_metadata_filtering.py
- Modify: tests/retrieval/test_dense_search.py and tests/retrieval/test_ingestion_retrieval_e2e.py only where their fake metadata must match the new contract.

**Interfaces:**
- Produce EmbeddingFingerprint(model_id: str, model_revision: str, dimension: int = 1024, normalized: bool = True) as a frozen dataclass.
- Add required board_id: str to RetrievedChunk; select d.board_id in dense SQL and map the returned value. Do not infer it from the function argument. Update existing fake result constructors in retrieval tests without changing the other result fields.
- Extend search_by_vector(..., fingerprint: EmbeddingFingerprint | None = None) -> list[RetrievedChunk]; None uses pinned BGE-M3 constants.
- search_by_text() obtains model_name, revision, and dimension from its chosen embedder and passes the fingerprint to search_by_vector(). Its fake embedders must provide these properties.
- SQL matches d.metadata model_id, model_revision, dimension and normalized using parameterized JSONB predicates in the same WHERE clause as Board and READY. Compare JSONB numeric/boolean values as typed JSON, not string guesses.

- [ ] **Step 1: Write failing fast and database tests.** Assert all Review Focus cases, including a live database assertion that the returned Board ID equals documents.board_id, plus one eligible result and an empty result when every document is incompatible. Adapt the retrieval fixture to seed explicit metadata without touching ingestion.
- [ ] **Step 2: Confirm failure.** Run python -m pytest tests/retrieval/test_metadata_filtering.py -q. Expected: missing fingerprint API or SQL predicates fail.
- [ ] **Step 3: Implement the frozen fingerprint and parameterized SQL predicates.** Validate nonempty ID/revision and positive dimension before opening a connection; preserve stable distance and chunk-ID ordering.
- [ ] **Step 4: Verify focused and regression tests.** Run python -m pytest tests/retrieval -q. Expected: all existing and new retrieval tests pass; live database cases must actually run when TEST_DATABASE_URL is configured.
- [ ] **Step 5: Verify lint and commit.** Run python -m ruff check src/retrieval tests/retrieval. Commit only this task's retrieval and test files with message feat(retrieval): filter incompatible embedding metadata.

**Acceptance:** An eligible Board A READY chunk wins at top_k=1 even when a closer Board B chunk, a non-READY chunk, or an incompatible model/revision chunk exists.
