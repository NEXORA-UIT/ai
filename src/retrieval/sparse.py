"""Board-scoped Python BM25 candidate search over PostgreSQL chunks."""

from __future__ import annotations

from uuid import UUID

from src.ingestion.persistence.storage import DocumentStorage
from src.retrieval.bm25 import rank_sparse, tokenize_lexical
from src.retrieval.config import DEFAULT_CANDIDATE_K
from src.retrieval.metadata import (
    DEFAULT_EMBEDDING_FINGERPRINT,
    EmbeddingFingerprint,
)
from src.retrieval.schemas import SourceChunk, SparseHit, SparseSearchResult


class CorpusTooLargeError(RuntimeError):
    """Raised when a Board corpus exceeds the explicit in-memory BM25 cap."""


def _validate_inputs(
    question: str,
    board_id: str | UUID,
    top_k: int,
    max_corpus_chunks: int,
) -> tuple[UUID, list[str]]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must contain non-whitespace text")
    try:
        board_uuid = UUID(str(board_id))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("board_id must be a valid UUID") from exc
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if (
        isinstance(max_corpus_chunks, bool)
        or not isinstance(max_corpus_chunks, int)
        or max_corpus_chunks < 1
    ):
        raise ValueError("max_corpus_chunks must be a positive integer")
    return board_uuid, tokenize_lexical(question)


def search_sparse(
    question: str,
    board_id: str | UUID,
    *,
    top_k: int = DEFAULT_CANDIDATE_K,
    max_corpus_chunks: int = 5000,
    fingerprint: EmbeddingFingerprint | None = None,
    storage: DocumentStorage | None = None,
) -> list[SparseHit]:
    """Rank eligible chunks from one Board without loading embedding vectors."""
    return search_sparse_with_stats(
        question,
        board_id,
        top_k=top_k,
        max_corpus_chunks=max_corpus_chunks,
        fingerprint=fingerprint,
        storage=storage,
    ).hits


def search_sparse_with_stats(
    question: str,
    board_id: str | UUID,
    *,
    top_k: int = DEFAULT_CANDIDATE_K,
    max_corpus_chunks: int = 5000,
    fingerprint: EmbeddingFingerprint | None = None,
    storage: DocumentStorage | None = None,
) -> SparseSearchResult:
    """Return BM25 hits plus the count of eligible chunks loaded for ranking."""
    board_uuid, query_tokens = _validate_inputs(
        question, board_id, top_k, max_corpus_chunks
    )
    if not query_tokens:
        return SparseSearchResult(hits=[], corpus_size=None)

    selected_fingerprint = (
        DEFAULT_EMBEDDING_FINGERPRINT if fingerprint is None else fingerprint
    )
    if not isinstance(selected_fingerprint, EmbeddingFingerprint):
        raise ValueError("fingerprint must be an EmbeddingFingerprint")

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
                           dc.end_offset
                    FROM document_chunks AS dc
                    JOIN documents AS d ON d.id = dc.document_id
                    WHERE d.board_id = %(board_id)s
                      AND d.status = 'READY'
                      AND d.metadata -> 'model_id' = to_jsonb(%(model_id)s::text)
                      AND d.metadata -> 'model_revision' =
                          to_jsonb(%(model_revision)s::text)
                      AND d.metadata -> 'dimension' = to_jsonb(%(dimension)s::integer)
                      AND d.metadata -> 'normalized' = to_jsonb(%(normalized)s::boolean)
                    ORDER BY dc.id ASC
                    LIMIT %(corpus_limit)s;
                    """,
                    {
                        "board_id": board_uuid,
                        "model_id": selected_fingerprint.model_id,
                        "model_revision": selected_fingerprint.model_revision,
                        "dimension": selected_fingerprint.dimension,
                        "normalized": selected_fingerprint.normalized,
                        "corpus_limit": max_corpus_chunks + 1,
                    },
                )
                rows = cursor.fetchall()
    finally:
        connection.close()

    if len(rows) > max_corpus_chunks:
        raise CorpusTooLargeError(
            f"Board BM25 corpus exceeds the configured limit of "
            f"{max_corpus_chunks} chunks"
        )

    corpus = [
        SourceChunk(
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
        )
        for row in rows
    ]
    if not corpus:
        return SparseSearchResult(hits=[], corpus_size=0)
    return SparseSearchResult(
        hits=rank_sparse(question, corpus, top_k=top_k),
        corpus_size=len(corpus),
    )
