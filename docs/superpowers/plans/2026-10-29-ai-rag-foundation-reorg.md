# Nexora AI RAG Foundation Reorganization Implementation Plan

> Historical implementation plan. See the [README](../../../README.md) for the current package layout; later cleanup consolidated the placeholder folders and renamed `agent/` to `agents/`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver an accurately named AI service repo, explicit RAG/Agent module boundary, SRS-aligned RAG architecture handoff, and passing Python/BE foundation checks.

**Architecture:** Preserve the Python FastAPI process and BullMQ worker. Express remains the public/auth/schema owner and BullMQ producer; Python is the only RAG queue consumer and uses Tika, R2, BGE-M3, and psycopg according to the SRS. RAG and Agent are separate Python packages; only the health/Phase 0 behavior exists after this change.

**Tech Stack:** Python 3.12, uv, FastAPI, BullMQ Python SDK, Pydantic, psycopg 3 + pgvector, pytest, Ruff, Express/TypeScript, BullMQ, Prisma, Apache Tika.

**Spec:** `docs/superpowers/specs/2026-10-29-ai-rag-foundation-design.md`; user-provided SRS sections listed there.

## Global Constraints

- Do not implement full ingestion, retrieval, generation, Agent, or streaming behavior in this foundation.
- Keep BullMQ and Apache Tika per SRS; do not add Celery/RQ, a PostgreSQL job queue, LangGraph, an ORM, or Python migrations.
- Keep Express auth/RBAC, public API, Prisma migration, and R2 upload ownership; keep its email/maintenance workers.
- Python Worker owns RAG ingestion and its processing states; revision guards protect all chunk/state writes.
- `BAAI/bge-m3` dense vector target is 1024 dimensions; model hosting remains open and both Python processes share one runtime.
- Update Pydantic schemas and the BE JSON snapshot together when internal contract shape changes.
- Commit separately in the BE and AI repositories; do not push or merge.

## Review Focus

- A stale document revision must never publish chunks or mark the current revision `READY`; implementation is deferred but explicitly required in the handoff.
- Page-less TXT/DOCX citations must remain valid via optional `pageNumber` and `sourceLocator`.
- BullMQ ingestion queue must have only the Python consumer; unrelated Node email/maintenance consumers remain.
- The API and worker must use the same BGE-M3 model/configuration; no second model load per process is assumed.
- Phase 0 must continue to run locally without Docker packaging for the Python service.

---

### Task 1: Create branches and preserve the existing Phase 0 baseline

**Files:** BE repository only; no code changes.

**Interfaces:** Start `backend/feat/rag` at the cached `origin/staging` commit and preserve the three existing Phase 0 commits in order. The AI working copy's Git metadata was empty when commits were regrouped, so `feat/rag` now starts as a new local root history grouped by implementation task.

- [x] Create BE `feat/rag` from cached `origin/staging` (fetch was unavailable in the sandbox; cached ref matched the current base).
- [x] Cherry-pick Phase 0 smoke, pgvector, and schema snapshot commits in parent order.
- [x] Create AI `feat/rag` as a new local root branch because the previous `.git` history was no longer present.
- [x] Verify both repository heads and working trees before editing.

### Task 2: Document the agreed architecture and public contract

**Files:**
- Modify AI `README.md`, `docs/architecture/rag-overview.md`, `docs/architecture/contracts-v1.md`, and create `docs/architecture/foundation-handoff.md`.
- Create BE `docs/CORE/rag-foundation.md`; link it from `docs/CORE/README.md`.
- Modify BE `docs/api/openapi/components/knowledge-base.yaml` and `docs/api/openapi/knowledge-base.yaml` so citations support page-less sources and the query response documents SSE events.

**Interfaces:** Public OpenAPI remains owned by Express. Python internal queue/HTTP schemas remain Pydantic-owned. The handoff documents `document.ingest.v1`, owner/state boundaries, Tika, read-only R2 access, pgvector, revision guards, the `answer.delta`/`citation`/`done`/`error` SSE events, and open implementation decisions.

- [x] Write architecture docs with the selected design and explicit non-goals.
- [x] Update the citation OpenAPI schema for page-less sources and document query SSE without changing a running endpoint.
- [x] Run the BE RAG contract test, OpenAPI loader check, and typecheck; the full BE suite later passed with local service access enabled.
- [x] Commit BE and AI docs separately.

### Task 3: Reorganize the AI package and finalize shared DTOs

**Files:**
- Move source code to `src/nexora_ai/`; update imports in `tests/` and `scripts/`.
- Create AI `src/nexora_ai/rag/{__init__.py,ingestion/__init__.py,retrieval/__init__.py,generation/__init__.py}` and `src/nexora_ai/agent/__init__.py`.
- Modify AI `pyproject.toml`/`uv.lock` (distribution `nexora-ai`), README, scripts, docs, `src/nexora_ai/contracts/retrieval.py`, and `src/nexora_ai/contracts/schema.py`.
- Create AI `tests/unit/test_module_boundaries.py`; modify `tests/contracts/test_retrieval.py` and generated `schemas/v1/*.schema.json`.
- Modify BE `contracts/rag/v1/*.schema.json` and `tests/rag/phase0-contract.test.ts`.

**Interfaces:** `CitationV1` requires `documentName`, `documentId`, `chunkId`, and `snippet`; `pageNumber` and `sourceLocator` are optional. SSE payloads are `RagAnswerDeltaEventV1`, `RagCitationEventV1`, `RagAnswerDoneEventV1`, and `RagErrorEventV1` for `answer.delta`, `citation`, `done`, and `error` respectively. Error codes are `RAG_QUERY_FAILED`, `MODEL_UNAVAILABLE`, and `REQUEST_TIMEOUT`.

- [x] Add failing tests for the import namespace, required citation metadata, page-less citations, and SSE events.
- [x] Run the new tests and confirm they fail for the missing package/fields/models.
- [x] Move imports/entrypoints and distribution metadata to `nexora_ai`/`nexora-ai`, and create the RAG/Agent package boundaries.
- [x] Run the boundary and citation tests to confirm they pass; export schemas and copy them to BE.
- [x] Run the complete Python suite with local socket access enabled (41 passed, 1 skipped) and the BE RAG contract tests.
- [x] Commit the AI package and contract changes, plus the BE schema snapshot/test changes.

### Task 4: Rename the local checkout and perform final verification

**Files:** Local workspace checkout now lives in `ai/`; no source changes are needed for the folder rename.

- [x] Rename the local checkout to `ai/` after confirming the target did not exist.
- [x] Run `uv sync --locked`, Python tests (41 passed, 1 skipped), Ruff lint/format, and compare all 11 shared schema snapshots.
- [x] Run the full BE suite (122 passed) and `pnpm typecheck`.
- [x] Verify FastAPI `/health`, Express `/`, and the Node producer → BullMQ → Python Worker Phase 0 smoke.
- [x] Connect to the configured PostgreSQL instance and verify both the pgvector extension and a vector distance operation (1 integration test passed).
- [ ] Create the replacement GitHub repository `NEXORA-UIT/ai`; GitHub currently returns 404, and this session has no repository-creation API or `gh` CLI. Local task commits are ready but unpushed.
- [x] Commit final docs and verification notes in their owning repository.
