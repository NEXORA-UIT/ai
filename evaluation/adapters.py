"""Benchmark variants backed by the same stages used by the AI API."""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from uuid import UUID

from evaluation.run_retrieval import (
    RetrievalCallable,
    RetrievalOutput,
    RetrievalRequest,
)
from src.embedding.embedder import Embedder
from src.persistence.postgres import ConnectionProvider
from src.reranking.cross_encoder import Reranker
from src.retrieval.hybrid import search_candidates
from src.retrieval.ranking import fuse_candidates, rerank_fused_candidates


def build_retrieval_variants(
    *,
    embedder: Embedder | None = None,
    storage: ConnectionProvider | None = None,
    reranker: Reranker | None = None,
) -> Mapping[str, RetrievalCallable]:
    """Select stage combinations without reimplementing their logic."""

    def dense_only(request: RetrievalRequest) -> RetrievalOutput:
        result = search_candidates(
            request.case.question,
            request.case.board_id,
            candidate_k=request.k,
            include_sparse=False,
            embedder=embedder,
            storage=storage,
        )
        return RetrievalOutput(
            chunk_ids=tuple(hit.chunk_id for hit in result.candidates.dense),
            stage_latencies_ms=result.stage_latencies_ms,
        )

    def hybrid(request: RetrievalRequest, *, rerank: bool) -> RetrievalOutput:
        result = search_candidates(
            request.case.question,
            request.case.board_id,
            candidate_k=request.candidate_k,
            embedder=embedder,
            storage=storage,
        )
        timings = dict(result.stage_latencies_ms)
        started = perf_counter()
        fused = fuse_candidates(result.candidates, candidate_k=request.candidate_k)
        timings["rrf"] = (perf_counter() - started) * 1000
        if rerank:
            started = perf_counter()
            selected = rerank_fused_candidates(
                request.case.question,
                UUID(request.case.board_id),
                fused,
                top_k=request.k,
                reranker=reranker,
            )
            timings["rerank"] = (perf_counter() - started) * 1000
            chunk_ids = tuple(hit.source.chunk_id for hit in selected)
        else:
            chunk_ids = tuple(hit.source.chunk_id for hit in fused)
        return RetrievalOutput(
            chunk_ids=chunk_ids,
            stage_latencies_ms=timings,
            bm25_corpus_size=result.bm25_corpus_size,
        )

    return {
        "A": dense_only,
        "B": lambda request: hybrid(request, rerank=False),
        "C": lambda request: hybrid(request, rerank=True),
    }
