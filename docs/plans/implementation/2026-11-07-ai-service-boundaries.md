# AI Service Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** Keep HTTP concerns at the API boundary and share provider and ingestion infrastructure.

**Architecture:** Answer orchestration returns a domain result and raises typed errors. DeepSeek transport owns request and envelope parsing, while the answer and judge validate their own schemas. Ingestion streams its digest and centralizes failure recording.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, HTTPX, pytest.

**Spec:** [AI maintainability review](../../refactor/ai-maintainability-review.md)

## Global Constraints

- AI feature branch only. Preserve existing DTOs, status codes, citations and lazy model loading.
- Preserve file validation, processing ownership and current file retention behavior.
- Keep the existing dependency lock and leave the changes in the working tree.

## Review Focus

- Empty evidence skips provider initialization; foreign evidence fails closed.
- Missing credentials and invalid provider responses keep their existing public error contract.
- Invalid provider envelope shapes never expose raw payloads.
- Document hash does not read the full file into memory.
- Failure recording cannot mask the original ingestion error.

## Tasks

- [x] Move request/response DTOs to `src/api/schemas.py`. Add `src/generation/service.py` for retrieval, optional provider and grounding. Focused answer, search, generation and faithfulness run: 48 passed, 1 deselected.
- [x] Extract the HTTPX JSON envelope and configuration handling from answer and judge into one provider transport. Preserve their separate payload and error rules. Malformed choice reproduces a raw `AttributeError` before the fix; provider and faithfulness run: 15 passed, 1 deselected.
- [x] Use streaming SHA-256 and consolidate FAILED recording. Preserve transaction scope and the original exception if failure recording fails.
- [x] Separate local upload staging from HTTP handling and pass validated metadata through ingestion. Recheck the size and digest of bytes sent to Tika. Reject files changed, removed or replaced during ingestion. Use the same local environment loader in the CLI and tests.
- [x] Run the offline suite with an empty Hugging Face cache. Final run: 388 passed, 31 deselected. Ruff and diff checks passed.
- [x] Verify live Tika and PostgreSQL during the [entrypoint cleanup](2026-11-07-ai-entrypoint-cleanup.md). The smoke test ingested a real document and retrieved its source after embedding and reranking.
