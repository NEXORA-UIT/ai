"""Benchmark variants backed by the same stages used by the AI API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from evaluation.run_retrieval import (
    RetrievalCallable,
    RetrievalOutput,
    RetrievalRequest,
)
from src.persistence.postgres import ConnectionProvider
from src.reranking.cross_encoder import Reranker
from src.retrieval.hybrid import search_candidates
from src.retrieval.workflow import retrieve_hybrid

if TYPE_CHECKING:
    from src.embedding.embedder import Embedder


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
        result = retrieve_hybrid(
            request.case.question,
            request.case.board_id,
            top_k=request.k,
            candidate_k=request.candidate_k,
            run_rerank=rerank,
            embedder=embedder,
            storage=storage,
            reranker=reranker,
        )
        hits = result.reranked if rerank else result.fused
        return RetrievalOutput(
            chunk_ids=tuple(hit.source.chunk_id for hit in hits),
            stage_latencies_ms=result.stage_latencies_ms,
            bm25_corpus_size=result.bm25_corpus_size,
        )

    return {
        "A": dense_only,
        "B": lambda request: hybrid(request, rerank=False),
        "C": lambda request: hybrid(request, rerank=True),
    }
