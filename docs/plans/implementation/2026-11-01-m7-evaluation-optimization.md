# Milestone 7 Evaluation and Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Measure standard Recall@k, HitRate@k, MRR, citation accuracy, faithfulness and latency on reproducible datasets before setting quality gates.

**Architecture:** Pure metric functions and synthetic fixtures check correctness. A separate, curated non-sensitive PDF/DOCX corpus with manually labeled relevant chunks measures retrieval quality. One runner compares Dense-only (A), Hybrid+RRF (B), and Hybrid+RRF+Reranker (C) on the same locked corpus and queries. Citation and latency checks can use fake answer clients; claim faithfulness uses the selected DeepSeek judge behind an injected transport.

**Tech Stack:** Python 3.12, pytest, JSONL, perf_counter, existing retrieval/reranking stack, HTTPX, Langfuse OTLP/HTTP JSON and the approved DeepSeek judge.

**Spec:** docs/plans/specs/2026-11-01-rag-retrieval-m2-m8-design.md and docs/architecture/rag-architecture.md

## Global Constraints

- D4 approved: publish a measured baseline first. Do not set hard Recall@k, HitRate@k, MRR, faithfulness, citation-accuracy or latency pass/fail thresholds in this milestone.
- D3 selected in the design spec: DeepSeek `deepseek-flash` for generation and the M7 claim judge. Tasks 1–3 use offline deterministic clients; Task 4 uses injected mock transports in tests and does not make live model calls.
- Synthetic fixtures are for deterministic correctness tests. Real-document quality evaluation uses a separately versioned, non-sensitive PDF/DOCX sample and human-labeled relevant chunks; no production documents, credentials or private user data enter either dataset. Use the existing ingestion output to freeze the real-document chunk snapshot; do not modify ingestion.
- For each answerable query with relevant set R and unique top-k retrieved IDs T, Recall@k = |R intersect T| / |R|, HitRate@k = 1 if the intersection is nonempty else 0, and reciprocal rank = 1 / rank of the first relevant hit within the returned top-k list (0 if absent). Aggregate each as a macro mean over all answerable queries. Report N/A and count 0 if none are answerable. Deduplicate retrieved IDs before scoring.
- Failed retrieval on an answerable query contributes 0 to Recall@k, HitRate@k and MRR, remains in the denominator, and appears in the failure report. Unanswerable queries are excluded from these three denominators and reported separately; never silently drop failed or missing cases.
- A/B/C use the same golden dataset version, source chunk snapshot, embedding model/revision, chunking, query set and final top_k. Keep the shared reranker candidate_k at 20. Record per-variant scores, deltas versus A, sample counts, BM25 corpus size, failures and p50/p95 latency; measure CPU reranking separately before considering a larger pool.
- Report p50/p95 for dense, BM25, RRF, rerank, generation and total where those stages execute. SRS first-character <=2s and full-answer <=15s remain reference service targets until transport/streaming exists.
- The seeder accepts only TEST_DATABASE_URL with an explicit CLI opt-in and expected database name ending in _test. Before any write, compare that expected name with the connected current_database(); reject missing/mismatched names and any prod/staging/dev target. Never fall back to DATABASE_URL, TRUNCATE, DROP, or clean rows outside the fixed dataset IDs. Run all inserts/deletes in a transaction.
- No new service, ingestion workflow change or backend change. Each task ends in one commit after focused checks.

## Review Focus

- With three relevant chunks and one found at top-k, Recall@k is 1/3 while HitRate@k is 1 (test_recall_vs_hit_rate).
- MRR uses the first relevant hit's one-based rank, even when later relevant chunks appear (test_mrr_first_relevant).
- Answerable retrieval errors stay in all three metric denominators as zeroes (test_answerable_failure_counts_as_zero).
- A/B/C receive the same dataset version, query order, embedding fingerprint, chunking snapshot and top_k (test_variants_share_golden_snapshot).
- Seeder refuses an absent opt-in, non-test URL or connected database name mismatch before DML (test_seeder_rejects_unsafe_target).
- Synthetic correctness reports cannot be labeled real-document quality results (test_report_dataset_kind).
- Citation to the right document but wrong chunk or page is inaccurate (test_citation_location_accuracy).
- p50/p95 handles one sample and stable ties (test_latency_percentiles).

---

### Task 1: Add safe synthetic corpus and standard retrieval metrics

**Files:**
- Create: evaluation/golden/synthetic/corpus.jsonl with fixed Board/document/chunk UUIDs, synthetic text, READY/status and BGE-M3 metadata.
- Create: evaluation/golden/synthetic/retrieval.jsonl
- Create: evaluation/seed_corpus.py
- Create: evaluation/schemas.py
- Create: evaluation/metrics.py
- Modify: .gitignore to ignore generated .tmp-evaluation/ reports.
- Test: tests/evaluation/test_metrics.py
- Test: tests/evaluation/test_seed_corpus.py

**Interfaces:**
- GoldenCase(question: str, board_id: str, expected_chunk_ids: tuple[str, ...], answerable: bool, expected_facts: tuple[str, ...], expected_citation_ids: tuple[str, ...]). Require nonempty, unique expected IDs for answerable cases and no expected IDs for unanswerable cases.
- seed_corpus(*, test_database_url: str, expected_database_name: str, allow_test_seed: bool, dataset_path: Path, embedder: Embedder) -> None. Require explicit opt-in and the exact TEST_DATABASE_URL value; inspect current_database() before DML, require expected name to end in _test and match the connected name, and touch only fixed fixture IDs. Fast tests inject fixed vectors; opt-in runs compute real BGE-M3 vectors. Do not call or modify ingestion.
- recall_at_k(cases: Sequence[GoldenCase], results: Sequence[Sequence[str] | RetrievalFailure], k: int) -> float | None is macro average of per-query relevant-chunk fractions.
- hit_rate_at_k(cases: Sequence[GoldenCase], results: Sequence[Sequence[str] | RetrievalFailure], k: int) -> float | None is the macro average of any-hit indicators.
- mean_reciprocal_rank(cases: Sequence[GoldenCase], results: Sequence[Sequence[str] | RetrievalFailure]) -> float | None uses the first relevant rank. A failure contributes 0 for an answerable case in all metrics.

- [ ] **Step 1: Write synthetic fixtures and failing metric/seeder tests.** Cover all metric Review Focus cases, multiple relevant chunks, one/no hit, duplicates, unanswerable-only input, answerable exceptions, Board A/B decoys, READY/non-READY/model mismatch, Vietnamese/English and exact technical IDs. Assert the seeder rejects unsafe targets before INSERT/DELETE and never falls back to DATABASE_URL.
- [ ] **Step 2: Confirm failure.** Run python -m pytest tests/evaluation/test_metrics.py tests/evaluation/test_seed_corpus.py -q. Expected: missing metrics/seeder fails.
- [ ] **Step 3: Implement schema validation, metrics and guarded seeder.** Use transaction-scoped fixture IDs and report explicit counts. Keep synthetic test results labeled correctness-only.
- [ ] **Step 4: Verify and commit.** Run python -m pytest tests/evaluation -q and python -m ruff check evaluation tests/evaluation. Commit focused files with message feat(evaluation): add safe corpus and retrieval metrics.

### Task 2: Benchmark three retrieval variants on a real-document dataset

**Files:**
- Create: evaluation/golden/real_documents/ with versioned non-sensitive PDF/DOCX samples, source/provenance manifest, frozen chunk snapshot and manually reviewed relevant IDs/queries.
- Create: evaluation/run_retrieval.py
- Test: tests/evaluation/test_run_retrieval.py
- Test: tests/evaluation/test_real_document_manifest.py

**Interfaces:**
- run_retrieval(dataset_path: Path, variants: Mapping[str, RetrievalCallable], *, k: int = 5, candidate_k: int = 20, output_dir: Path) -> EvaluationReport. Require exactly A=Dense-only, B=Hybrid+RRF, C=Hybrid+RRF+Reranker; adapt the existing M2/M4/M5 functions without duplicating their ranking logic.
- Lock one dataset version, chunk snapshot, embedding model/revision, chunking and ordered queries for all variants. Run each at the same final top_k, with candidate_k=20 for B/C; output per-case ranks, all three metrics, failure counts, p50/p95 stage/total latency, BM25 corpus size and A-to-B/B-to-C deltas.
- The real-document snapshot is derived once from the existing ingestion process and checked in with manually labeled ground truth. Load that fixed snapshot through Task 1's guarded seeder into TEST_DATABASE_URL with real BGE-M3 embeddings before the opt-in quality run; fake vectors are for correctness tests only. This is distinct from fast synthetic tests; no ingestion code change or production data access.

- [ ] **Step 1: Write failing manifest and runner tests.** Assert the three variants share the exact snapshot/configuration, answerable failures score zero, missing cases fail report validation, CPU reranking candidate_k never exceeds 20, and synthetic results are not presented as real-document quality.
- [ ] **Step 2: Confirm failure.** Run python -m pytest tests/evaluation/test_run_retrieval.py tests/evaluation/test_real_document_manifest.py -q. Expected: missing runner/manifest validation fails.
- [ ] **Step 3: Add curated fixtures and comparison runner.** Keep source PDFs/DOCX non-sensitive with provenance/license, freeze their chunk IDs and human labels, load the snapshot only through the guarded seeder, and write per-case JSON plus Markdown comparison to output_dir. Record actual model/chunking versions; do not infer improvement from synthetic correctness fixtures.
- [ ] **Step 4: Verify and commit.** Run python -m pytest tests/evaluation -q and python -m ruff check evaluation tests/evaluation. Run the opt-in real-document comparison with the guarded TEST_DATABASE_URL; confirm A/B/C and failure counts appear in the report. Commit focused files with message feat(evaluation): compare retrieval variants on real documents.

### Task 3: Add deterministic citation and latency baseline

**Files:**
- Create: evaluation/answer_metrics.py
- Create: evaluation/run_answers.py
- Test: tests/evaluation/test_answer_metrics.py
- Test: tests/evaluation/test_run_answers.py

**Interfaces:**
- citation_accuracy(expected: GoldenCase, answer: AnswerResult, corpus: Mapping[str, SourceChunk]) -> float compares chunk ID, document ID, page and offsets to the canonical corpus.
- latency_summary(samples_ms: Sequence[float]) -> LatencySummary with p50, p95 and sample count.
- run_answers(dataset_path: Path, answer: Callable[..., AnswerResult], *, output_dir: Path) -> EvaluationReport. An injected fake answer client supports offline citation/refusal correctness checks before D3.

- [ ] **Step 1: Write failing citation and latency tests.** Assert wrong location, invented ID, correct multi-source citation, empty citation on answerable output, one-sample percentile, and failed calls being reported.
- [ ] **Step 2: Confirm failure.** Run python -m pytest tests/evaluation/test_answer_metrics.py tests/evaluation/test_run_answers.py -q. Expected: missing answer metrics fail.
- [ ] **Step 3: Implement deterministic metrics and runner.** Label fake-client reports synthetic correctness, record refusal counts and per-stage timing, and never claim real-model faithfulness from them.
- [ ] **Step 4: Verify and commit.** Run python -m pytest tests/evaluation -q and python -m ruff check evaluation tests/evaluation. Commit focused files with message feat(evaluation): report citations and latency.

### Task 4: Add faithfulness judging and Langfuse using D3

**Files:**
- Create: evaluation/faithfulness.py
- Create: evaluation/langfuse_trace.py
- Modify: evaluation/run_answers.py
- Modify: .env.example for optional Langfuse credentials.
- Do not add a Langfuse SDK or service: reuse HTTPX and send OTLP/HTTP JSON directly to the current Langfuse v4 trace endpoint.
- Do not create answers.jsonl until the user supplies real PDF/DOCX files and reviews claim-level annotations.
- Test: tests/evaluation/test_faithfulness.py
- Test: tests/evaluation/test_langfuse_trace.py

**Interfaces:**
- FaithfulnessJudge protocol judge(answer: str, evidence: Sequence[SourceChunk]) -> FaithfulnessResult with claim verdicts and supporting chunk IDs.
- faithfulness_score(answer: AnswerResult, evidence: Sequence[SourceChunk], judge: FaithfulnessJudge) -> FaithfulnessResult.
- Langfuse tracing is best effort; telemetry outage cannot change a search or answer result. Reports retain trace IDs and raw judge verdicts for human audit.

- [x] **Step 1: Record D3 provider/model and claim rubric in the design spec before implementation.**
- [x] **Step 2: Write failing tests.** Cover supported/unsupported claims, refusals, fake judge errors, offline Langfuse behavior, report verdicts, trace IDs, and cross-Board evidence rejection before judge calls.
- [x] **Step 3: Confirm failure.** The initial focused run failed with the expected missing evaluation.faithfulness module.
- [x] **Step 4: Implement judge adapter, traces and report fields.** Keep deterministic citation checks distinct from judge-based faithfulness. Langfuse receives only aggregate evaluator metadata and scores; local reports retain raw claim verdicts. Use cited, canonical evidence by default and reject any supplied cross-Board evidence before the judge call.
- [x] **Step 5: Verify and commit.** Run python -m pytest tests/evaluation -q and python -m ruff check evaluation tests/evaluation. Do not make live provider calls in tests. Commit focused files with message feat(evaluation): measure faithfulness with traces.

**Acceptance:** Synthetic fixtures verify metric/seeder correctness; real-document A/B/C runs report comparable retrieval quality and latency on one golden snapshot. Task 4 provides the real judge and trace adapters, but real-document faithfulness remains unmeasured until user-provided PDF/DOCX files and human-reviewed claim annotations exist. Optimization follows measured failures as separate tasks and commits.

### Task 5: Keep variant metrics on the same final top-k

**Files:**
- Modify: evaluation/run_retrieval.py
- Test: tests/evaluation/test_run_retrieval.py

**Interfaces:**
- Before storing or scoring each variant result, deduplicate IDs in rank order and cap the public final result at the shared `k`. This keeps Recall@k, HitRate@k and MRR comparable when the Hybrid+RRF adapter internally materializes up to candidate_k before its final metric cutoff.

- [x] **Step 1: Write a failing regression test.** For `k=5`, let variant B return a relevant chunk at rank 6 while A/C return only the first five decoys; assert all three metrics use only the shared final top-k and per-case outputs contain no more than k unique IDs.
- [x] **Step 2: Confirm failure.** Run python -m pytest tests/evaluation/test_run_retrieval.py -q. Expected: current MRR gives B a nonzero score from rank 6.
- [x] **Step 3: Apply the shared cutoff.** Deduplicate and cap every successful variant result to k before storing rows and computing any metric.
- [x] **Step 4: Verify and commit.** Run python -m pytest tests/evaluation/test_run_retrieval.py tests/evaluation/test_metrics.py -q and Ruff for the changed files. Commit with message fix(evaluation): cap retrieval metrics at final top-k.

### Task 6: Match the seeder embedder to the declared dataset model

**Files:**
- Modify: evaluation/seed_corpus.py
- Test: tests/evaluation/test_seed_corpus.py

**Interfaces:**
- Read the declared model ID and revision from the versioned evaluation manifest and compare them with the injected embedder's model_name, revision, and dimension before opening the database or generating vectors.
- The synthetic fixture intentionally includes a per-document incompatible-model decoy. Compare against the manifest's primary embedding fingerprint, not every document row, so that the negative eligibility fixture remains usable.

- [x] **Step 1: Write failing tests.** An embedder with a different model ID or revision must be rejected before database construction or embedding; the matching embedder must remain accepted.
- [x] **Step 2: Confirm failure.** Run python -m pytest tests/evaluation/test_seed_corpus.py -q. Expected: mismatched embedder currently reaches storage/embedding.
- [x] **Step 3: Validate before side effects.** Check the manifest fingerprint against the embedder before creating DocumentStorage, connecting, or producing vectors.
- [x] **Step 4: Verify and commit.** Run python -m pytest tests/evaluation/test_seed_corpus.py -q and Ruff for the changed files; run the guarded PostgreSQL seeder integration test. Commit with message fix(evaluation): validate seeder embedding fingerprint.
