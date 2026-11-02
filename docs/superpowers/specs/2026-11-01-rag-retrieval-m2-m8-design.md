# RAG Retrieval Milestones 2–8 — Draft Design

**Status:** D1–D4 approved on 2026-11-02; D3 provider selected as DeepSeek AI.

## Intent and current state

Extend the standalone AI retrieval vertical slice through metadata compatibility, lexical retrieval, RRF, reranking, grounded answers, evaluation, and a Python search_documents() entrypoint. Milestone 1 already provides search_by_vector() and search_by_text() in src/retrieval/dense.py. Both scope SQL by board_id and READY before LIMIT. RetrievedChunk already carries source IDs, text, location, and cosine distance.

Ingestion already persists document metadata keys model_id, model_revision, dimension, and normalized. The shared BGE-M3 constants are in src/embedding/config.py. Retrieval plans consume these fields without changing ingestion behavior.

## Boundary and invariant

- Python 3.12; current project pyproject.toml and uv.lock remain the runtime baseline.
- Work directly in the current ai checkout. Each implementation task ends with one focused commit after its checks.
- Every database retrieval query filters board_id and READY inside SQL before ranking and LIMIT, and selects the database-backed board_id into its internal source record. Express remains responsible for authorizing that board_id when integrated.
- Model compatibility is checked against document metadata; missing or mismatched metadata is ineligible.
- No change to ingestion workflow, BullMQ, R2, Express, Prisma production migrations, or Agent code in these plans.
- No search index or schema change is needed for the initial Python BM25 implementation. scripts/bootstrap_dev_db.sql remains a local dev/test fixture; Backend owns production migrations.
- M6 rejects any supplied evidence whose database-backed board_id differs from the requested Board before building context or calling an LLM. Empty evidence refuses deterministically. For weak evidence the generation prompt requests refusal; non-refusal output requires at least one valid citation, and the validator rejects invented IDs. Treat document text as untrusted data, never as instructions. Milestone 7 measures whether actual answers remain faithful. No retrieval score is treated as a calibrated probability.
- Internal source records may keep storage_key for backend use. Public Citation and SearchResult payloads omit it; document_id is the lookup key for a later authorized source resolver.
- Use injectable model and LLM interfaces so fast tests do not download model weights or make paid API calls.

## Planned path

| Milestone | Increment | Output |
| --- | --- | --- |
| 2 | Metadata compatibility | Dense results only from the query embedder's model/revision/dimension/normalization |
| 3 | Python BM25 retrieval | Board and READY scoped BM25 results plus a two-list candidate search |
| 4 | RRF | Stable deduplicated fused chunk ranking |
| 5 | Cross-encoder | Relevance scores and final top 3–5 chunks |
| 6 | Grounding and citation | Answer with validated source references or explicit refusal |
| 7 | Evaluation | Golden corpus, retrieval/answer metrics, latency report, Langfuse traces |
| 8 | Integration ready | Stable search_documents() Python entrypoint for a future Agent adapter |

## Proposed cross-milestone interfaces

- Metadata: EmbeddingFingerprint(model_id: str, model_revision: str, dimension: int = 1024, normalized: bool = True).
- RetrievedChunk gains board_id selected from documents.board_id in M2. SourceChunk carries that database-backed board_id plus chunk_id, document_id, chunk_index, file_name, internal storage_key, content, page_number, start_offset, and end_offset. SourceChunk.from_dense(hit: RetrievedChunk) copies those fields without a distance; the sparse SQL also selects documents.board_id.
- Sparse: SparseHit(source: SourceChunk, sparse_score: float). The sparse channel never fabricates a cosine distance.
- Hybrid: HybridCandidates(dense: list[RetrievedChunk], sparse: list[SparseHit]).
- RRF: FusedChunk(source: SourceChunk, rrf_score: float, channels: tuple[str, ...]).
- Rerank: RerankedChunk(source: SourceChunk, rerank_score: float, rrf_score: float).
- Answer: AnswerResult(text: str, citations: tuple[Citation, ...], refused: bool). Citation contains public source IDs/locations but no storage_key.
- Agent-facing retrieval: SearchResult(chunks: tuple[SearchChunk, ...], board_id: str, question: str). SearchChunk is a projection of RerankedChunk with public source fields and scores, excluding storage_key.

These are planned contracts, not current code. A later plan may add fields only by updating this design and its dependent plans first.

## Decision record and implementation defaults

- D1 approved: Use Python rank_bm25 0.2.2 and BM25Okapi. PostgreSQL selects only eligible chunks from the supplied Board; Python tokenizes the entire eligible corpus and ranks it. No LLM, GPU, PostgreSQL text-search extension, new index, or ingestion change is needed. Fetch at most 5,001 rows to detect a 5,000-chunk cap; exceedance fails explicitly instead of silently ranking a partial corpus. This baseline trades query-time memory/latency for a simple local implementation; Milestone 7 measures the cost.
- D2 approved: BAAI/bge-reranker-v2-m3 as a multilingual cross-encoder on CPU. This plan pins revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e for reproducibility. Reuse one lazy process instance; candidate pool defaults to and is capped at 20 until CPU latency is measured; final top_k 5 (allow 1–5 when fewer are requested). Real-model smoke tests are opt-in because model weights are large.
- D3 approved: The user selected DeepSeek AI for generation and the M7 judge. Pin the API model ID `deepseek-flash`, which DeepSeek currently documents as its Flash model, with the OpenAI-compatible base URL `https://api.deepseek.com`. Use the existing `httpx` dependency, a 30-second explicit timeout, and Chat Completions JSON Output (`response_format={"type":"json_object"}`) with the M6 response shape `{ "answer": string, "cited_chunk_ids": string[] }`. Keep the API key in `DEEPSEEK_API_KEY`; tests use injected mock transports and make no live calls. Cost if wrong: the vendor may change the model behind the stable `deepseek-flash` alias, changing behavior without an ID change.
- M7 claim-level faithfulness rubric: split the generated answer into atomic factual claims; a claim is supported only when the supplied evidence directly entails it. Citation presence alone is not support. Refusals have no faithfulness denominator. Score is supported factual claims divided by all factual claims; zero factual claims is N/A. Preserve per-claim verdicts and supporting chunk IDs for human review. The judge uses the same pinned `deepseek-flash` alias and a 30-second timeout; fake-judge tests do not count as model faithfulness evidence.
- D4 approved: Report a first measured baseline before setting hard Recall@k, HitRate@k, MRR, faithfulness, citation-accuracy and latency thresholds. Compare Dense-only, Hybrid+RRF, and Hybrid+RRF+Reranker on one fixed golden dataset, embedding model, chunking and top_k. Synthetic fixtures check correctness; a separate non-sensitive real PDF/DOCX dataset measures quality. Record SRS service latency targets separately until transport/streaming is implemented.
- RRF uses rank positions from each channel, k = 60, deduplication by chunk_id, and chunk_id as final stable tie-breaker.
- Milestone 8 returns retrieved evidence only. Express/FastAPI transport, authorization, and Agent invocation are separate integration work.

## Dependency order and ownership

2 is a prerequisite for 3; 3 feeds 4; 4 feeds 5; 5 feeds 6 and 8; 6 feeds answer evaluation in 7. The metric helpers in 7 can be implemented before 6, but the full answer evaluation needs 6. Each milestone plan is a separate reviewable unit. All tests use Python module commands in the project virtual environment; live PostgreSQL gates require TEST_DATABASE_URL.

## Decision status

1. D1: Python rank_bm25 approved; supersedes the earlier PostgreSQL full-text selection.
2. D2: BGE Reranker v2 M3 on CPU approved; plan revision pinned above.
3. D3: DeepSeek AI selected; use `deepseek-flash`, 30-second timeout, and JSON Output for grounded generation and claim judging.
4. D4: Baseline measurement before quality gates approved.

References: [rank-bm25 0.2.2](https://pypi.org/project/rank-bm25/0.2.2/) expects a pre-tokenized corpus and query; [BAAI's model repository](https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/main) documents SentenceTransformers CrossEncoder, and the [pinned revision](https://huggingface.co/BAAI/bge-reranker-v2-m3/commit/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e) is recorded above. The next plan review should focus on BM25 corpus size and CPU reranking latency in the target environment.

Provider references: [DeepSeek API quick start](https://api-docs.deepseek.com/quick_start/agent_integrations/) documents the OpenAI-compatible base URL and current `deepseek-flash` API model; [DeepSeek JSON Output](https://api-docs.deepseek.com/guides/json_mode/) documents the `response_format` contract.
