"""Coordinate dense and sparse candidate retrieval for one Board query."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import UUID

from src.persistence.postgres import ConnectionProvider
from src.retrieval.config import DEFAULT_CANDIDATE_K, MAX_CANDIDATE_K
from src.retrieval.dense import embed_query, get_default_embedder, search_by_vector
from src.retrieval.schemas import HybridCandidates
from src.retrieval.sparse import search_sparse_with_stats
from src.validation import BOARD_ID_INVALID_UUID, QUESTION_TEXT_REQUIRED

if TYPE_CHECKING:
    from src.embedding.embedder import Embedder

logger = logging.getLogger("uvicorn.error")


def _validate_inputs(question: str, board_id: str | UUID, candidate_k: int) -> UUID:
    if not isinstance(question, str) or not question.strip():
        raise ValueError(QUESTION_TEXT_REQUIRED)
    try:
        board_uuid = UUID(str(board_id))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(BOARD_ID_INVALID_UUID) from exc
    if (
        isinstance(candidate_k, bool)
        or not isinstance(candidate_k, int)
        or not 1 <= candidate_k <= MAX_CANDIDATE_K
    ):
        raise ValueError(f"candidate_k must be an integer from 1 to {MAX_CANDIDATE_K}")
    return board_uuid


@dataclass(frozen=True)
class CandidateSearch:
    candidates: HybridCandidates
    stage_latencies_ms: dict[str, float]
    bm25_corpus_size: int | None = None


def search_candidates(
    question: str,
    board_id: str | UUID,
    *,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    include_sparse: bool = True,
    embedder: Embedder | None = None,
    storage: ConnectionProvider | None = None,
) -> CandidateSearch:
    """Embed once and retrieve candidates and measurements for enabled channels."""
    board_uuid = _validate_inputs(question, board_id, candidate_k)
    selected_embedder = embedder if embedder is not None else get_default_embedder()
    started = perf_counter()
    vector, fingerprint = embed_query(question, selected_embedder)
    embedding_ms = (perf_counter() - started) * 1000

    started = perf_counter()
    dense_hits = search_by_vector(
        vector,
        board_uuid,
        top_k=candidate_k,
        fingerprint=fingerprint,
        storage=storage,
    )
    timings = {"embedding": embedding_ms, "dense": (perf_counter() - started) * 1000}
    if not include_sparse:
        return CandidateSearch(HybridCandidates(dense=dense_hits, sparse=[]), timings)

    started = perf_counter()
    sparse = search_sparse_with_stats(
        question,
        board_uuid,
        top_k=candidate_k,
        fingerprint=fingerprint,
        storage=storage,
    )
    timings["bm25"] = (perf_counter() - started) * 1000
    return CandidateSearch(
        HybridCandidates(dense=dense_hits, sparse=sparse.hits),
        timings,
        bm25_corpus_size=sparse.corpus_size,
    )


def search_hybrid_candidates(
    question: str,
    board_id: str | UUID,
    *,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    embedder: Embedder | None = None,
    storage: ConnectionProvider | None = None,
) -> HybridCandidates:
    """Retrieve both candidate lists for standalone and API callers."""
    result = search_candidates(
        question, board_id, candidate_k=candidate_k, embedder=embedder, storage=storage
    )
    timings = result.stage_latencies_ms
    logger.info(
        "RAG candidates embedding_ms=%.0f dense_ms=%.0f sparse_ms=%.0f",
        timings["embedding"],
        timings["dense"],
        timings["bm25"],
    )
    return result.candidates
