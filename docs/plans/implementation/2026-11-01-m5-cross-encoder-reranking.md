# Milestone 5 Cross-Encoder Reranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Rescore fused retrieval candidates against the full question and return the most relevant 3–5 chunks.

**Architecture:** A small reranker protocol isolates model inference from ordering logic. The default adapter uses a single lazy cross-encoder instance per process; a pure orchestrator scores up to 20 fused candidates and selects up to five source chunks.

**Tech Stack:** Python 3.12, existing SentenceTransformers CrossEncoder, BAAI/bge-reranker-v2-m3, pytest, Ruff.

**Spec:** docs/plans/specs/2026-11-01-rag-retrieval-m2-m8-design.md

## Global Constraints

- D2 approved: BAAI/bge-reranker-v2-m3 on CPU for this milestone. Pin revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e as the plan's reproducibility default.
- Inputs come only from Milestone 4 fused candidates; no extra database query or ingestion change.
- Scores are raw model outputs used for ordering, not probabilities.
- Reuse one lazy model instance per process and inject a fake scorer in fast tests.
- The model repository contains a large weights file; real-model smoke and CPU latency measurements run separately from fast tests.
- Candidate pool defaults to and is capped at 20 until Milestone 7 measures CPU latency; final top_k defaults to 5 and permits 1–5 when fewer results are wanted. Reject more than 20 fused candidates rather than silently truncating them.
- Each task gets one commit after focused checks.

## Review Focus

- A low RRF rank but high cross-encoder score can move to first place (test_reranker_changes_order).
- Duplicate chunk IDs are not reintroduced (test_reranker_keeps_unique_ids).
- Empty input returns empty without loading the model (test_reranker_empty_skips_model).
- Model output count mismatch or NaN/Infinity fails clearly (test_reranker_rejects_bad_scores).
- Equal model scores use RRF rank then chunk ID as tie-breakers (test_reranker_stable_ties).
- A direct call with 21 candidates is rejected before model loading; the public M8 facade has the same 20-candidate cap (test_reranker_candidate_cap).

---

### Task 1: Add an injectable cross-encoder adapter

**Files:**
- Create: src/reranking/cross_encoder.py
- Create: src/reranking/config.py
- Create: src/reranking/__init__.py
- Reuse the existing sentence-transformers dependency; the proposed BGE adapter needs no new package.
- Test: tests/reranking/test_cross_encoder.py

**Interfaces:**
- Produce Reranker protocol score_pairs(question: str, passages: Sequence[str]) -> list[float].
- Produce BgeReranker with model_id=BAAI/bge-reranker-v2-m3, model_revision=953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e, device=cpu and explicit batch_size; default instance is cached lazily.
- Pin the approved model ID and revision in src/reranking/config.py; assert they are forwarded to CrossEncoder in tests.

- [ ] **Step 1: Write failing adapter tests.** Assert pinned revision and CPU device are passed to CrossEncoder, one model initialization across calls, pair order, batch boundaries, empty input without load, and clear inference error.
- [ ] **Step 2: Confirm failure.** Run python -m pytest tests/reranking/test_cross_encoder.py -q. Expected: missing adapter fails.
- [ ] **Step 3: Implement the approved adapter.** Use the cross-encoder API to score question/passage pairs; no normalization of raw scores.
- [ ] **Step 4: Verify and commit.** Run python -m pytest tests/reranking -q and python -m ruff check src/reranking tests/reranking. Commit focused files with message feat(reranking): add cross encoder adapter.

### Task 2: Select final chunks by reranker scores

**Files:**
- Create: src/reranking/select.py
- Modify: src/retrieval/schemas.py
- Test: tests/reranking/test_select.py
- Test: tests/retrieval/test_reranking_e2e.py

**Interfaces:**
- Produce RerankedChunk(source: SourceChunk, rerank_score: float, rrf_score: float).
- Produce rerank_chunks(question: str, candidates: Sequence[FusedChunk], *, top_k: int = 5, reranker: Reranker | None = None) -> list[RerankedChunk].
- Require 1 <= top_k <= 5 and at most 20 candidates, excluding bool for integer limits. Preserve all source IDs/positions, database-backed board_id, internal storage_key, and RRF scores from the input; rerank_score is finite raw model output. Public projection happens in M8.

- [ ] **Step 1: Write failing selection tests.** Assert all Review Focus cases, top_k bounds and the 20-candidate cap. Use a deterministic fake scorer. Add a PostgreSQL retrieval-to-rerank test with fake embeddings and scorer, preserving Board/status/model isolation and database-backed board_id.
- [ ] **Step 2: Confirm failure.** Run python -m pytest tests/reranking/test_select.py tests/retrieval/test_reranking_e2e.py -q. Expected: missing selector fails.
- [ ] **Step 3: Implement selection.** Score each unique candidate once, validate output, sort score DESC then RRF DESC then chunk ID ASC, and truncate after sorting.
- [ ] **Step 4: Verify and commit.** Run python -m pytest tests/reranking tests/retrieval -q and python -m ruff check src/reranking src/retrieval tests/reranking tests/retrieval. Commit focused files with message feat(reranking): select final evidence chunks.

**Acceptance:** The final evidence list is at most top_k, keeps verifiable source metadata, and reflects cross-encoder relevance ordering.
