"""Shared fusion and reranking stages for search and evaluation."""

import logging
from time import perf_counter
from uuid import UUID

from src.reranking.config import get_rerank_candidate_limit
from src.reranking.cross_encoder import Reranker
from src.reranking.select import rerank_chunks
from src.retrieval.fusion import fuse_rrf
from src.retrieval.schemas import FusedChunk, HybridCandidates, RerankedChunk

logger = logging.getLogger("uvicorn.error")


def fuse_candidates(
    candidates: HybridCandidates, *, candidate_k: int
) -> list[FusedChunk]:
    """Fuse the full candidate pool before applying a reranking budget."""
    return fuse_rrf(candidates, top_k=candidate_k)


def rerank_fused_candidates(
    question: str,
    board_id: UUID,
    fused: list[FusedChunk],
    *,
    top_k: int,
    reranker: Reranker | None = None,
) -> list[RerankedChunk]:
    """Apply the shared reranking budget and verify the selected evidence scope."""
    shortlist = fused[:get_rerank_candidate_limit(top_k=top_k)]
    rerank_started = perf_counter()
    reranked = rerank_chunks(
        question,
        shortlist,
        top_k=top_k,
        reranker=reranker,
    )
    logger.info(
        "RAG retrieval rerank_ms=%.0f "
        "candidates=%d reranked=%d results=%d",
        (perf_counter() - rerank_started) * 1000,
        len(fused),
        len(shortlist),
        len(reranked),
    )
    if any(hit.source.board_id != str(board_id) for hit in reranked):
        raise ValueError("retrieval returned a chunk outside the requested Board scope")
    return reranked
