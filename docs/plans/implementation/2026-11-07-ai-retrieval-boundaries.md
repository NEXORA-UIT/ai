# AI Retrieval Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep Board isolation and ranking identical across callers while separating PostgreSQL reads from ingestion writes.

**Architecture:** A PostgreSQL reader owns the shared eligibility predicate, row projection and connection cleanup. API and evaluation reuse the same retrieval stages, with evaluation retaining the choice of pipeline and its reporting.

**Tech Stack:** Python 3.12, psycopg, pgvector, pytest.

**Spec:** [AI maintainability review](../../refactor/ai-maintainability-review.md)

## Global Constraints

- Current AI feature branch only. Preserve routes, storage injection, CLI, model fingerprints and ranking defaults.
- No schema migration, external commits or shared database operations.

## Review Focus

- Board, READY and fingerprint scope must be identical before either query applies its limit.
- Sparse fetch must not select vectors and must keep the over-cap signal.
- Connection cleanup must happen when a read raises.
- Shared stages must retain candidate pool, shortlist, provenance, and evaluation timings.
- Empty evidence must not load the reranker.

### Task 1: Consolidate PostgreSQL reads

- [x] Reproduce the missing read boundary and verify connection closure on errors through both entrypoints. Preserve the existing SQL scope and typed binding assertions.
- [x] Introduce `src/persistence/postgres.py` for pgvector connection creation and `src/persistence/chunk_reader.py` for dense and corpus reads. Let `DocumentStorage.get_connection()` delegate to the common connector. Keep its injection and error contract.
- [x] Let dense and sparse functions delegate reads to the adapter. Keep the existing public result types and order. Focused run: 106 passed, 8 deselected. Offline run: 380 passed, 31 deselected.

### Task 2: Share runtime stages with evaluation

- [x] Verify both flows use the same fingerprint, query embedding, candidate reads, RRF and rerank budget with injected dependencies.
- [x] Expose reusable stage results and timings in the retrieval package. Adapt evaluation to call them for A/B/C and record the effective rerank configuration and explicit warmup state in reports.
- [x] Add an import-order regression test and separate shared ranking stages from the public facade module. Final offline run: 381 passed, 31 deselected. Ruff and diff checks passed.
- [x] Verify live PostgreSQL with the isolated test fixtures during the [entrypoint cleanup](2026-11-07-ai-entrypoint-cleanup.md). The full local integration run passed, including the database-backed scope and ranking checks.
