"""Board-scoped dense vector search using PostgreSQL and pgvector."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING
from uuid import UUID

from src.embedding.config import (
    DEFAULT_DENSE_DIMENSION as _DENSE_DIMENSION,
)
from src.embedding.config import (
    DEFAULT_NORMALIZE_EMBEDDINGS,
)
from src.embedding.runtime import get_default_embedder as get_default_embedder
from src.persistence.chunk_reader import read_dense_chunks
from src.persistence.postgres import ConnectionProvider
from src.retrieval.config import DEFAULT_SEARCH_TOP_K
from src.retrieval.metadata import (
    DEFAULT_EMBEDDING_FINGERPRINT,
    EmbeddingFingerprint,
)
from src.retrieval.schemas import RetrievedChunk
from src.validation import (
    BOARD_ID_INVALID_UUID,
    QUESTION_TEXT_REQUIRED,
    TOP_K_POSITIVE_INTEGER_REQUIRED,
)

if TYPE_CHECKING:
    from src.embedding.embedder import Embedder

def _validate_query_vector(query_vector: Sequence[float]) -> list[float]:
    """Normalize a query vector to floats and reject invalid dense vectors."""
    try:
        values = [float(value) for value in query_vector]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("query_vector must contain numeric values") from exc

    if len(values) != _DENSE_DIMENSION:
        raise ValueError(
            f"query_vector must have {_DENSE_DIMENSION} dimensions, got {len(values)}"
        )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("query_vector must contain only finite values")
    if not any(value != 0.0 for value in values):
        raise ValueError("query_vector must not be the zero vector")

    return values


def _validate_board_and_limit(board_id: str | UUID, top_k: int) -> UUID:
    """Validate the retrieval scope and result limit before expensive work."""
    try:
        board_uuid = UUID(str(board_id))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(BOARD_ID_INVALID_UUID) from exc

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError(TOP_K_POSITIVE_INTEGER_REQUIRED)

    return board_uuid


def search_by_vector(
    query_vector: Sequence[float],
    board_id: str | UUID,
    *,
    top_k: int = DEFAULT_SEARCH_TOP_K,
    fingerprint: EmbeddingFingerprint | None = None,
    storage: ConnectionProvider | None = None,
) -> list[RetrievedChunk]:
    """Return the closest READY chunks belonging to ``board_id``.

    The caller is responsible for Board authorization. In the integrated
    application, Express validates access before passing the authorized Board.
    """
    board_uuid = _validate_board_and_limit(board_id, top_k)
    vector_values = _validate_query_vector(query_vector)
    selected_fingerprint = (
        DEFAULT_EMBEDDING_FINGERPRINT if fingerprint is None else fingerprint
    )
    if selected_fingerprint.dimension != len(vector_values):
        raise ValueError(
            "fingerprint dimension must match the query vector dimension"
        )
    return read_dense_chunks(
        board_uuid, selected_fingerprint, vector_values, top_k=top_k, storage=storage
    )


def search_by_text(
    question: str,
    board_id: str | UUID,
    *,
    top_k: int = DEFAULT_SEARCH_TOP_K,
    embedder: Embedder | None = None,
    storage: ConnectionProvider | None = None,
) -> list[RetrievedChunk]:
    """Embed a question with BGE-M3 and return scoped dense matches."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError(QUESTION_TEXT_REQUIRED)

    _validate_board_and_limit(board_id, top_k)
    selected_embedder = embedder if embedder is not None else get_default_embedder()
    vector, fingerprint = embed_query(question, selected_embedder)
    return search_by_vector(
        vector, board_id, top_k=top_k, fingerprint=fingerprint, storage=storage
    )


def embed_query(
    question: str, embedder: Embedder
) -> tuple[list[float], EmbeddingFingerprint]:
    """Encode one query with the fingerprint of the model that produced it."""
    fingerprint = EmbeddingFingerprint(
        model_id=embedder.model_name,
        model_revision=embedder.revision,
        dimension=embedder.dimension,
        normalized=DEFAULT_NORMALIZE_EMBEDDINGS,
    )
    vectors = embedder.embed_texts([question])
    if len(vectors) != 1:
        raise ValueError("embedder must return exactly one vector for one question")

    return vectors[0], fingerprint
