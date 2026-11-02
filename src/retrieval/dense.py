"""Board-scoped dense vector search using PostgreSQL and pgvector."""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache
from uuid import UUID

from pgvector import Vector

from src.embedding.config import DEFAULT_NORMALIZE_EMBEDDINGS
from src.embedding.embedder import BgeM3Embedder, Embedder
from src.ingestion.storage import DocumentStorage
from src.retrieval.metadata import (
    DEFAULT_EMBEDDING_FINGERPRINT,
    EmbeddingFingerprint,
)
from src.retrieval.schemas import RetrievedChunk

_DENSE_DIMENSION = 1024


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
        raise ValueError("board_id must be a valid UUID") from exc

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    return board_uuid


@lru_cache(maxsize=1)
def get_default_embedder() -> BgeM3Embedder:
    """Return the process-wide lazy BGE-M3 embedder instance."""
    return BgeM3Embedder()


def search_by_vector(
    query_vector: Sequence[float],
    board_id: str | UUID,
    *,
    top_k: int = 5,
    fingerprint: EmbeddingFingerprint | None = None,
    storage: DocumentStorage | None = None,
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
    db = storage if storage is not None else DocumentStorage()
    connection = db.get_connection()

    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT dc.id,
                           dc.document_id,
                           d.board_id,
                           dc.chunk_index,
                           d.file_name,
                           d.storage_key,
                           dc.content,
                           dc.page_number,
                           dc.start_offset,
                           dc.end_offset,
                           dc.embedding <=> %(query_vector)s::vector AS cosine_distance
                    FROM document_chunks AS dc
                    JOIN documents AS d ON d.id = dc.document_id
                    WHERE d.board_id = %(board_id)s
                      AND d.status = 'READY'
                      AND d.metadata -> 'model_id' = to_jsonb(%(model_id)s::text)
                      AND d.metadata -> 'model_revision' =
                          to_jsonb(%(model_revision)s::text)
                      AND d.metadata -> 'dimension' = to_jsonb(%(dimension)s::integer)
                      AND d.metadata -> 'normalized' = to_jsonb(%(normalized)s::boolean)
                    ORDER BY cosine_distance ASC, dc.id ASC
                    LIMIT %(top_k)s;
                    """,
                    {
                        "query_vector": Vector(vector_values),
                        "board_id": board_uuid,
                        "model_id": selected_fingerprint.model_id,
                        "model_revision": selected_fingerprint.model_revision,
                        "dimension": selected_fingerprint.dimension,
                        "normalized": selected_fingerprint.normalized,
                        "top_k": top_k,
                    },
                )
                rows = cursor.fetchall()
    finally:
        connection.close()

    return [
        RetrievedChunk(
            chunk_id=str(row[0]),
            document_id=str(row[1]),
            board_id=str(row[2]),
            chunk_index=row[3],
            file_name=row[4],
            storage_key=row[5],
            content=row[6],
            page_number=row[7],
            start_offset=row[8],
            end_offset=row[9],
            cosine_distance=float(row[10]),
        )
        for row in rows
    ]


def search_by_text(
    question: str,
    board_id: str | UUID,
    *,
    top_k: int = 5,
    embedder: Embedder | None = None,
    storage: DocumentStorage | None = None,
) -> list[RetrievedChunk]:
    """Embed a question with BGE-M3 and return scoped dense matches."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must contain non-whitespace text")

    _validate_board_and_limit(board_id, top_k)
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

    return search_by_vector(
        vectors[0],
        board_id,
        top_k=top_k,
        fingerprint=fingerprint,
        storage=storage,
    )
