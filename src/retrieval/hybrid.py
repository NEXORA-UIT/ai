"""Coordinate dense and sparse candidate retrieval for one Board query."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from src.embedding.config import DEFAULT_NORMALIZE_EMBEDDINGS
from src.ingestion.persistence.storage import DocumentStorage
from src.retrieval.config import DEFAULT_CANDIDATE_K, MAX_CANDIDATE_K
from src.retrieval.dense import get_default_embedder, search_by_vector
from src.retrieval.metadata import EmbeddingFingerprint
from src.retrieval.schemas import HybridCandidates
from src.retrieval.sparse import search_sparse

if TYPE_CHECKING:
    from src.embedding.embedder import Embedder


def _validate_inputs(
    question: str, board_id: str | UUID, candidate_k: int
) -> UUID:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must contain non-whitespace text")
    try:
        board_uuid = UUID(str(board_id))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("board_id must be a valid UUID") from exc
    if (
        isinstance(candidate_k, bool)
        or not isinstance(candidate_k, int)
        or not 1 <= candidate_k <= MAX_CANDIDATE_K
    ):
        raise ValueError(
            f"candidate_k must be an integer from 1 to {MAX_CANDIDATE_K}"
        )
    return board_uuid


def search_hybrid_candidates(
    question: str,
    board_id: str | UUID,
    *,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    embedder: Embedder | None = None,
    storage: DocumentStorage | None = None,
) -> HybridCandidates:
    """Embed the question once and retrieve Board-scoped candidates by both channels."""
    board_uuid = _validate_inputs(question, board_id, candidate_k)
    selected_embedder = embedder if embedder is not None else get_default_embedder()
    fingerprint = EmbeddingFingerprint(
        model_id=selected_embedder.model_name,
        model_revision=selected_embedder.revision,
        dimension=selected_embedder.dimension,
        normalized=DEFAULT_NORMALIZE_EMBEDDINGS,
    )
    vectors = selected_embedder.embed_texts([question])
    if len(vectors) != 1:
        raise ValueError("embedder must return exactly one vector for one question")

    dense_hits = search_by_vector(
        vectors[0],
        board_uuid,
        top_k=candidate_k,
        fingerprint=fingerprint,
        storage=storage,
    )
    sparse_hits = search_sparse(
        question,
        board_uuid,
        top_k=candidate_k,
        fingerprint=fingerprint,
        storage=storage,
    )
    return HybridCandidates(dense=dense_hits, sparse=sparse_hits)
