"""Shared hybrid retrieval workflow for production and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import UUID

from src.retrieval.hybrid import search_candidates
from src.retrieval.ranking import fuse_candidates, rerank_fused_candidates
from src.retrieval.schemas import FusedChunk, RerankedChunk

if TYPE_CHECKING:
    from src.embedding.embedder import Embedder
    from src.persistence.postgres import ConnectionProvider
    from src.reranking.cross_encoder import Reranker


@dataclass(frozen=True)
class HybridSearchResult:
    fused: list[FusedChunk]
    reranked: list[RerankedChunk]
    stage_latencies_ms: dict[str, float]
    bm25_corpus_size: int | None


def retrieve_hybrid(
    question: str,
    board_id: str | UUID,
    *,
    top_k: int,
    candidate_k: int,
    run_rerank: bool = True,
    embedder: Embedder | None = None,
    storage: ConnectionProvider | None = None,
    reranker: Reranker | None = None,
) -> HybridSearchResult:
    """Run the hybrid stages used by search, answering and evaluation."""
    result = search_candidates(
        question,
        board_id,
        candidate_k=candidate_k,
        embedder=embedder,
        storage=storage,
    )
    timings = dict(result.stage_latencies_ms)
    started = perf_counter()
    fused = fuse_candidates(result.candidates, candidate_k=candidate_k)
    timings["rrf"] = (perf_counter() - started) * 1000
    reranked: list[RerankedChunk] = []
    if run_rerank:
        started = perf_counter()
        reranked = rerank_fused_candidates(
            question, UUID(str(board_id)), fused, top_k=top_k, reranker=reranker
        )
        timings["rerank"] = (perf_counter() - started) * 1000
    return HybridSearchResult(fused, reranked, timings, result.bm25_corpus_size)
