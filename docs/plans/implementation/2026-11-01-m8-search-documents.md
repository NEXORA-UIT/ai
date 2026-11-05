# Milestone 8 Integration-Ready Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Expose one stable search_documents() Python function that a future Agent adapter can call with an authorized Board scope.

**Architecture:** The facade validates inputs, calls Milestone 3 hybrid candidates, Milestone 4 RRF and Milestone 5 reranker, then projects internal source records into typed public evidence without storage_key. It does not generate an answer or perform HTTP/Express authorization. Dependencies remain injectable for fast tests.

**Tech Stack:** Python 3.12, existing retrieval/reranking modules, pytest, Ruff.

**Spec:** docs/plans/specs/2026-11-01-rag-retrieval-m2-m8-design.md

## Global Constraints

- Required board_id is a scope, never an authorization claim. Express must authorize before future integration.
- Preserve Board/READY/model SQL filtering through underlying retrieval functions.
- Default final top_k=5; validate positive integer <=5. Return fewer when corpus has fewer eligible chunks.
- Validate candidate_k as an integer from top_k through 20; default 20 until CPU reranking latency is measured in M7. Pass it as each channel's fetch limit and as the RRF output limit before reranking. Reject 21 rather than silently truncating.
- Public SearchResult and SearchChunk must contain no internal storage_key. Keep it only inside retrieval/reranking records; a future authorized backend resolves document_id when needed.
- Do not change ingestion workflow, Agent code, FastAPI routes, Express, or production migrations.
- One task, one verified commit.

## Review Focus

- Board B exact dense/lexical decoys never appear for Board A (test_facade_board_isolation).
- Non-READY and incompatible-model chunks never appear (test_facade_eligibility).
- Duplicate dense/sparse hit appears once after fusion (test_facade_deduplication).
- Empty corpus returns an empty typed result without reranker loading (test_facade_empty).
- Whitespace, invalid UUID, and invalid top_k fail before embedding/DB access (test_facade_input_validation).
- candidate_k=21 fails before embedding/DB access, matching the M5 cap (test_facade_candidate_cap).
- Serialized SearchResult has no storage_key or internal storage path, while retaining document/chunk IDs and locations (test_facade_public_payload).

---

### Task 1: Publish the search_documents() facade

**Files:**
- Create: src/retrieval/search_documents.py
- Modify: src/retrieval/schemas.py
- Modify: src/retrieval/__init__.py
- Test: tests/retrieval/test_search_documents.py
- Test: tests/retrieval/test_search_documents_database.py
- Modify: README.md with one standalone Python usage example.

**Interfaces:**
- Produce SearchChunk(chunk_id: str, document_id: str, chunk_index: int, file_name: str, content: str, page_number: int | None, start_offset: int | None, end_offset: int | None, rerank_score: float, rrf_score: float). This is the public projection of RerankedChunk and intentionally excludes storage_key.
- Produce SearchResult(question: str, board_id: str, chunks: tuple[SearchChunk, ...]).
- Produce search_documents(question: str, board_id: str | UUID, *, top_k: int = 5, candidate_k: int = 20, embedder: Embedder | None = None, storage: DocumentStorage | None = None, reranker: Reranker | None = None) -> SearchResult.
- Reuse search_hybrid_candidates(), fuse_rrf(), rerank_chunks(); do not reimplement their SQL/ranking.

- [x] **Step 1: Write failing facade tests.** Assert all Review Focus cases, exact argument forwarding, stable final source fields, and top_k <= candidate_k <= 20. Database test uses Board A/B plus READY/non-READY/model-mismatch decoys and runs with TEST_DATABASE_URL.
- [x] **Step 2: Confirm failure.** Run python -m pytest tests/retrieval/test_search_documents.py -q. Expected: missing public facade fails.
- [x] **Step 3: Implement facade and export.** Keep one query embedding, call fuse_rrf(..., top_k=candidate_k) before reranking, and build SearchChunk explicitly from each final hit. Expose source metadata without raw vectors, storage_key, or storage credentials.
- [x] **Step 4: Verify and commit.** Run python -m pytest tests/retrieval tests/reranking -q and python -m ruff check src/retrieval src/reranking tests/retrieval tests/reranking. Confirm database test actually executes when TEST_DATABASE_URL is configured. Commit focused files with message feat(retrieval): expose search_documents facade.

**Acceptance:** A single Python call returns final cited-source-ready chunks for the supplied Board. Future Agent and transport work may import it without changing retrieval internals.
