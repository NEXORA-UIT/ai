"""Stable Board-scoped retrieval facade for future Agent adapters."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import UUID

from src.persistence.postgres import ConnectionProvider
from src.reranking.cross_encoder import Reranker
from src.retrieval.config import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_SEARCH_TOP_K,
    MAX_CANDIDATE_K,
    MAX_SEARCH_TOP_K,
)
from src.retrieval.hybrid import search_hybrid_candidates
from src.retrieval.ranking import fuse_candidates, rerank_fused_candidates
from src.retrieval.schemas import RerankedChunk, SearchChunk, SearchResult
from src.validation import BOARD_ID_INVALID_UUID, QUESTION_TEXT_REQUIRED

if TYPE_CHECKING:
    from src.embedding.embedder import Embedder

logger = logging.getLogger("uvicorn.error")


def _validate_inputs(
    question: str, board_id: str | UUID, top_k: int, candidate_k: int
) -> UUID:
    if not isinstance(question, str) or not question.strip():
        raise ValueError(QUESTION_TEXT_REQUIRED)
    try:
        board_uuid = UUID(str(board_id))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(BOARD_ID_INVALID_UUID) from exc
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 1 <= top_k <= MAX_SEARCH_TOP_K
    ):
        raise ValueError(f"top_k must be an integer from 1 to {MAX_SEARCH_TOP_K}")
    if (
        isinstance(candidate_k, bool)
        or not isinstance(candidate_k, int)
        or top_k > candidate_k
        or candidate_k > MAX_CANDIDATE_K
    ):
        raise ValueError(
            f"candidate_k must be an integer from top_k to {MAX_CANDIDATE_K}"
        )
    return board_uuid


def retrieve_reranked_chunks(
    question: str,
    board_id: str | UUID,
    *,
    top_k: int = DEFAULT_SEARCH_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    embedder: Embedder | None = None,
    storage: ConnectionProvider | None = None,
    reranker: Reranker | None = None,
) -> list[RerankedChunk]:
    """Return scoped evidence with database-backed provenance for generation."""
    board_uuid = _validate_inputs(question, board_id, top_k, candidate_k)
    started = perf_counter()
    candidates = search_hybrid_candidates(
        question,
        board_uuid,
        candidate_k=candidate_k,
        embedder=embedder,
        storage=storage,
    )
    candidate_ms = (perf_counter() - started) * 1000
    fused = fuse_candidates(candidates, candidate_k=candidate_k)
    reranked = rerank_fused_candidates(
        question, board_uuid, fused, top_k=top_k, reranker=reranker
    )
    logger.info("RAG retrieval candidates_ms=%.0f", candidate_ms)
    return reranked


def search_documents(
    question: str,
    board_id: str | UUID,
    *,
    top_k: int = DEFAULT_SEARCH_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    embedder: Embedder | None = None,
    storage: ConnectionProvider | None = None,
    reranker: Reranker | None = None,
) -> SearchResult:
    """Retrieve and rerank public chunks for one caller-authorized Board scope.

    The Board ID is a retrieval scope, not an authorization check. A future
    public API must authorize the caller for this Board before calling here.
    """
    board_uuid = _validate_inputs(question, board_id, top_k, candidate_k)
    reranked = retrieve_reranked_chunks(
        question,
        board_uuid,
        top_k=top_k,
        candidate_k=candidate_k,
        embedder=embedder,
        storage=storage,
        reranker=reranker,
    )
    public_chunks = tuple(
        SearchChunk(
            chunk_id=hit.source.chunk_id,
            document_id=hit.source.document_id,
            chunk_index=hit.source.chunk_index,
            file_name=hit.source.file_name,
            content=hit.source.content,
            page_number=hit.source.page_number,
            start_offset=hit.source.start_offset,
            end_offset=hit.source.end_offset,
            rerank_score=hit.rerank_score,
            rrf_score=hit.rrf_score,
        )
        for hit in reranked
    )
    return SearchResult(
        question=question,
        board_id=str(board_uuid),
        chunks=public_chunks,
    )
