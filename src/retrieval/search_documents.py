"""Stable Board-scoped retrieval facade for future Agent adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from src.ingestion.persistence.storage import DocumentStorage
from src.reranking.cross_encoder import Reranker
from src.reranking.select import rerank_chunks
from src.retrieval.config import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_SEARCH_TOP_K,
    MAX_CANDIDATE_K,
    MAX_SEARCH_TOP_K,
)
from src.retrieval.fusion import fuse_rrf
from src.retrieval.hybrid import search_hybrid_candidates
from src.retrieval.schemas import SearchChunk, SearchResult

if TYPE_CHECKING:
    from src.embedding.embedder import Embedder


def _validate_inputs(
    question: str, board_id: str | UUID, top_k: int, candidate_k: int
) -> UUID:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must contain non-whitespace text")
    try:
        board_uuid = UUID(str(board_id))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("board_id must be a valid UUID") from exc
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


def search_documents(
    question: str,
    board_id: str | UUID,
    *,
    top_k: int = DEFAULT_SEARCH_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    embedder: Embedder | None = None,
    storage: DocumentStorage | None = None,
    reranker: Reranker | None = None,
) -> SearchResult:
    """Retrieve and rerank final evidence for one caller-authorized Board scope.

    The Board ID is a retrieval scope, not an authorization check. A future
    public API must authorize the caller for this Board before calling here.
    """
    board_uuid = _validate_inputs(question, board_id, top_k, candidate_k)
    candidates = search_hybrid_candidates(
        question,
        board_uuid,
        candidate_k=candidate_k,
        embedder=embedder,
        storage=storage,
    )
    fused = fuse_rrf(candidates, top_k=candidate_k)
    reranked = rerank_chunks(
        question,
        fused,
        top_k=top_k,
        reranker=reranker,
    )
    if any(hit.source.board_id != str(board_uuid) for hit in reranked):
        raise ValueError("retrieval returned a chunk outside the requested Board scope")
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
