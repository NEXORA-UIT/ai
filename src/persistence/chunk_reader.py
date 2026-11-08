"""Read eligible source chunks for dense and lexical retrieval."""

from collections.abc import Sequence
from typing import Any
from uuid import UUID

from pgvector import Vector

from src.documents.status import DocumentStatus
from src.persistence.postgres import ConnectionProvider, connect_pgvector
from src.retrieval.metadata import EmbeddingFingerprint
from src.retrieval.schemas import RetrievedChunk, SourceChunk

_SOURCE_COLUMNS = """
    dc.id, dc.document_id, d.board_id, dc.chunk_index, d.file_name,
    d.storage_key, dc.content, dc.page_number, dc.start_offset, dc.end_offset
"""

_ELIGIBLE_CHUNKS = """
    FROM document_chunks AS dc
    JOIN documents AS d ON d.id = dc.document_id
    WHERE d.board_id = %(board_id)s
      AND d.status = %(ready_status)s
      AND d.metadata -> 'model_id' = to_jsonb(%(model_id)s::text)
      AND d.metadata -> 'model_revision' = to_jsonb(%(model_revision)s::text)
      AND d.metadata -> 'dimension' = to_jsonb(%(dimension)s::integer)
      AND d.metadata -> 'normalized' = to_jsonb(%(normalized)s::boolean)
"""

_DENSE_QUERY = f"""
    SELECT {_SOURCE_COLUMNS},
           dc.embedding <=> %(query_vector)s::vector AS cosine_distance
    {_ELIGIBLE_CHUNKS}
    ORDER BY cosine_distance ASC, dc.id ASC
    LIMIT %(top_k)s;
"""

_CORPUS_QUERY = f"""
    SELECT {_SOURCE_COLUMNS}
    {_ELIGIBLE_CHUNKS}
    ORDER BY dc.id ASC
    LIMIT %(corpus_limit)s;
"""


def _fetch(
    query: str,
    board_id: UUID,
    fingerprint: EmbeddingFingerprint,
    storage: ConnectionProvider | None,
    **params: object,
) -> list[tuple[Any, ...]]:
    connection = storage.get_connection() if storage is not None else connect_pgvector()
    try:
        with connection, connection.cursor() as cursor:
            cursor.execute(
                query,
                {
                    "board_id": board_id,
                    "ready_status": DocumentStatus.READY.value,
                    "model_id": fingerprint.model_id,
                    "model_revision": fingerprint.model_revision,
                    "dimension": fingerprint.dimension,
                    "normalized": fingerprint.normalized,
                    **params,
                },
            )
            return cursor.fetchall()
    finally:
        connection.close()


def _source_fields(row: Sequence[Any]) -> dict[str, Any]:
    (
        chunk_id,
        document_id,
        board_id,
        chunk_index,
        file_name,
        storage_key,
        content,
        page_number,
        start_offset,
        end_offset,
    ) = row
    return {
        "chunk_id": str(chunk_id),
        "document_id": str(document_id),
        "board_id": str(board_id),
        "chunk_index": chunk_index,
        "file_name": file_name,
        "storage_key": storage_key,
        "content": content,
        "page_number": page_number,
        "start_offset": start_offset,
        "end_offset": end_offset,
    }


def read_dense_chunks(
    board_id: UUID,
    fingerprint: EmbeddingFingerprint,
    vector: Sequence[float],
    *,
    top_k: int,
    storage: ConnectionProvider | None,
) -> list[RetrievedChunk]:
    rows = _fetch(
        _DENSE_QUERY,
        board_id,
        fingerprint,
        storage,
        query_vector=Vector(vector),
        top_k=top_k,
    )
    return [
        RetrievedChunk(**_source_fields(row[:-1]), cosine_distance=float(row[-1]))
        for row in rows
    ]


def read_corpus_chunks(
    board_id: UUID,
    fingerprint: EmbeddingFingerprint,
    *,
    limit: int,
    storage: ConnectionProvider | None,
) -> list[SourceChunk]:
    rows = _fetch(_CORPUS_QUERY, board_id, fingerprint, storage, corpus_limit=limit)
    return [SourceChunk(**_source_fields(row)) for row in rows]
