# Milestone 3 Hybrid Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Combine Milestone 2 dense candidates with CPU-only Python BM25 candidates from the same Board's compatible READY documents.

**Architecture:** PostgreSQL selects source fields for every eligible chunk in one Board, capped at 5,001 rows for an explicit 5,000-chunk limit. Python applies the same deterministic Unicode tokenizer to corpus and question, uses rank_bm25.BM25Okapi to score lexical matches, then returns top candidates. A hybrid coordinator embeds the question once and runs dense and BM25 channels with the same Board and model fingerprint.

**Tech Stack:** Python 3.12, psycopg 3, pgvector, rank-bm25 0.2.2, NumPy, pytest, Ruff.

**Spec:** docs/superpowers/specs/2026-11-01-rag-retrieval-m2-m8-design.md

## Global Constraints

- D1 approved: rank_bm25 0.2.2 BM25Okapi. No LLM, GPU, PostgreSQL full-text search, new DB index, or ingestion workflow change.
- SQL filters board_id, READY, and embedding fingerprint before ORDER BY dc.id and LIMIT 5,001. Python ranks only the resulting eligible corpus.
- A Board with more than max_corpus_chunks=5000 fails with CorpusTooLargeError; never silently rank a truncated corpus. Milestone 7 measures latency and memory before choosing an optimization.
- Tokenize corpus and query identically with NFKC normalization and casefold. Keep complete hyphenated technical identifiers such as uc-kb-03 as additional tokens alongside Unicode word components; retain Vietnamese diacritics. When a query contains such an identifier, sort exact whole-identifier matches before BM25 score so a tiny corpus with zero IDF cannot tie UC-KB-03 with UC-KB-04. BM25 scores remain raw rank signals, not probabilities.
- Filter out chunks with no query-token overlap, regardless of whether BM25Okapi returns a zero or negative score. Empty query tokens or an empty/tokenless corpus return an empty list without constructing BM25Okapi.
- Each task gets one commit after focused and regression checks.

## Review Focus

- Board B exact-term decoy cannot displace Board A's best hit at top_k=1 (test_bm25_board_filter_before_limit).
- PROCESSING, FAILED and wrong-model chunks never enter the BM25 corpus (test_bm25_eligible_corpus_only).
- Empty/punctuation-only query and empty/tokenless corpus return empty without a divide-by-zero error (test_bm25_empty_inputs).
- Matching chunk with score zero in a tiny corpus remains eligible while unmatched chunks are excluded (test_bm25_zero_idf_overlap).
- In a two-chunk corpus, query UC-KB-03 ranks its exact identifier above UC-KB-04 even when BM25 IDF is zero; the tokenizer retains uc-kb-03 as one token as well as its components (test_bm25_exact_technical_identifier).
- Corpus size 5,001 raises a typed error before returning partial results (test_bm25_corpus_limit).
- Hybrid query embedding occurs once, with both channels receiving the same scope (test_hybrid_embeds_once).

---

### Task 1: Add Board-scoped Python BM25 retrieval

**Files:**
- Create: src/retrieval/bm25.py
- Create: src/retrieval/sparse.py
- Modify: src/retrieval/schemas.py
- Modify: src/retrieval/__init__.py
- Modify: pyproject.toml and regenerate uv.lock for rank-bm25==0.2.2.
- Test: tests/retrieval/test_bm25.py
- Test: tests/retrieval/test_sparse_search.py

**Interfaces:**
- Produce SourceChunk with required database-backed board_id and the nine source fields named in the shared design, plus SourceChunk.from_dense(hit: RetrievedChunk) -> SourceChunk. The dense field is added in M2; keep all other RetrievedChunk fields.
- Produce SparseHit(source: SourceChunk, sparse_score: float).
- Produce tokenize_lexical(text: str) -> list[str] in bm25.py; use it on every document and query. Keep whole-identifier tokens distinguishable from their components for exact-match ordering.
- Produce CorpusTooLargeError and search_sparse(question: str, board_id: str | UUID, *, top_k: int = 20, max_corpus_chunks: int = 5000, fingerprint: EmbeddingFingerprint | None = None, storage: DocumentStorage | None = None) -> list[SparseHit].
- SQL selects chunk/document source fields including d.board_id from document_chunks JOIN documents, filtering Board, READY and typed JSONB model fingerprint. ORDER BY dc.id ASC LIMIT max_corpus_chunks + 1. Do not select embedding vectors or copy board_id from the caller into SourceChunk. BM25Okapi receives nonempty token lists; get_scores(query_tokens) yields one score per corpus item. Stable ranking is exact whole-identifier match count DESC, then BM25 score DESC, then chunk_id ASC; without a query identifier, use BM25 score and chunk_id.

- [ ] **Step 1: Write failing BM25 unit and live database tests.** Cover every Review Focus case owned here, Vietnamese/English casefold matching, UC-KB-03 versus UC-KB-04 exact-ID ranking, invalid top_k/max_corpus_chunks, and stable score ties. Use TEST_DATABASE_URL with isolated rows for SQL scope tests; assert sparse source board_id is read from the row.
- [ ] **Step 2: Confirm failure.** Run python -m pytest tests/retrieval/test_bm25.py tests/retrieval/test_sparse_search.py -q. Expected: missing BM25 API fails.
- [ ] **Step 3: Add dependency and implement tokenization, scoped corpus load and ranking.** Pin rank-bm25==0.2.2, update the repository lock, and keep the query text out of SQL; validate inputs before opening a connection.
- [ ] **Step 4: Verify and commit.** Run python -m pytest tests/retrieval -q and python -m ruff check src/retrieval tests/retrieval. Confirm live DB cases executed rather than skipped. Commit focused files with message feat(retrieval): add board-scoped bm25 search.

### Task 2: Coordinate dense and BM25 candidate searches

**Files:**
- Create: src/retrieval/hybrid.py
- Modify: src/retrieval/schemas.py
- Modify: src/retrieval/__init__.py
- Test: tests/retrieval/test_hybrid_search.py

**Interfaces:**
- Produce HybridCandidates(dense: list[RetrievedChunk], sparse: list[SparseHit]).
- Produce search_hybrid_candidates(question: str, board_id: str | UUID, *, candidate_k: int = 20, embedder: Embedder | None = None, storage: DocumentStorage | None = None) -> HybridCandidates.
- Reuse Milestone 2 fingerprint derivation and cached default embedder; validate question, Board and candidate_k (positive integer <=20) before calling either channel. The sparse channel uses its 5,000-chunk corpus cap.

- [ ] **Step 1: Write failing coordinator tests.** Assert embed once, both channels receive the same Board/fingerprint/candidate_k, independent empty-channel behavior, and no partial success when one channel raises CorpusTooLargeError or a database error.
- [ ] **Step 2: Confirm failure.** Run python -m pytest tests/retrieval/test_hybrid_search.py -q. Expected: missing coordinator fails.
- [ ] **Step 3: Implement coordinator.** Keep orchestration synchronous and dependency-injectable; avoid duplicate BGE model initialization.
- [ ] **Step 4: Verify and commit.** Run python -m pytest tests/retrieval -q and python -m ruff check src/retrieval tests/retrieval. Commit focused files with message feat(retrieval): combine dense and bm25 candidates.

**Acceptance:** HybridCandidates contains two correctly scoped ranked lists. A corpus above the explicit cap fails, and this milestone does not yet fuse scores or generate an answer.
