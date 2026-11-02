"""Tests for Board-scoped PostgreSQL BM25 candidate retrieval."""

from __future__ import annotations

from importlib import import_module
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from src.embedding.config import (
    DEFAULT_DENSE_DIMENSION,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    DEFAULT_NORMALIZE_EMBEDDINGS,
)

_BOARD_A = "11111111-1111-1111-1111-111111111111"


def _valid_fingerprint():
    fingerprint_type = import_module(
        "src.retrieval.metadata"
    ).EmbeddingFingerprint
    return fingerprint_type(
        model_id=DEFAULT_MODEL_ID,
        model_revision=DEFAULT_MODEL_REVISION,
        dimension=DEFAULT_DENSE_DIMENSION,
        normalized=DEFAULT_NORMALIZE_EMBEDDINGS,
    )


def _storage_with_rows(rows: list[tuple[object, ...]]):
    storage = MagicMock()
    connection = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.return_value = rows
    connection.__enter__.return_value = connection
    connection.cursor.return_value.__enter__.return_value = cursor
    storage.get_connection.return_value = connection
    return storage, connection, cursor


def _row(
    chunk_id: str,
    document_id: str,
    board_id: str,
    file_name: str,
    content: str,
) -> tuple[object, ...]:
    return (
        UUID(chunk_id),
        UUID(document_id),
        UUID(board_id),
        2,
        file_name,
        f"internal://{document_id}",
        content,
        3,
        10,
        10 + len(content),
    )


def test_search_sparse_uses_typed_scope_filters_and_maps_database_board_id() -> None:
    module = import_module("src.retrieval.sparse")
    chunk_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    document_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    board_id = "11111111-1111-1111-1111-111111111111"
    storage, connection, cursor = _storage_with_rows(
        [_row(chunk_id, document_id, board_id, "guide.txt", "UC-KB-03 setup")]
    )

    hits = module.search_sparse(
        "UC-KB-03",
        _BOARD_A,
        top_k=1,
        fingerprint=_valid_fingerprint(),
        storage=storage,
    )

    assert len(hits) == 1
    hit = hits[0]
    assert hit.source.chunk_id == chunk_id
    assert hit.source.document_id == document_id
    assert hit.source.board_id == board_id
    assert hit.source.file_name == "guide.txt"
    assert hit.source.content == "UC-KB-03 setup"
    assert isinstance(hit.sparse_score, float)

    query, params = cursor.execute.call_args.args
    normalized_query = " ".join(query.split()).casefold()
    where = normalized_query.index("where d.board_id")
    order = normalized_query.index("order by")
    limit = normalized_query.index("limit")
    conditions = (
        "d.status = 'ready'",
        "d.metadata -> 'model_id' = to_jsonb(%(model_id)s::text)",
        "d.metadata -> 'model_revision' = to_jsonb(%(model_revision)s::text)",
        "d.metadata -> 'dimension' = to_jsonb(%(dimension)s::integer)",
        "d.metadata -> 'normalized' = to_jsonb(%(normalized)s::boolean)",
    )
    assert all(condition in normalized_query for condition in conditions)
    assert where < order < limit
    assert "d.board_id" in normalized_query
    assert "dc.embedding" not in normalized_query
    assert "uc-kb-03" not in normalized_query
    assert params["board_id"] == UUID(_BOARD_A)
    assert params["model_id"] == DEFAULT_MODEL_ID
    assert params["model_revision"] == DEFAULT_MODEL_REVISION
    assert params["dimension"] == DEFAULT_DENSE_DIMENSION
    assert params["normalized"] is DEFAULT_NORMALIZE_EMBEDDINGS
    assert params["corpus_limit"] == 5001
    connection.close.assert_called_once_with()


@pytest.mark.parametrize(
    ("board_id", "top_k", "max_corpus_chunks"),
    [
        ("not-a-uuid", 5, 5000),
        (_BOARD_A, 0, 5000),
        (_BOARD_A, True, 5000),
        (_BOARD_A, 5, 0),
        (_BOARD_A, 5, True),
    ],
    ids=["invalid-board", "zero-top-k", "boolean-top-k", "zero-cap", "boolean-cap"],
)
def test_search_sparse_rejects_invalid_inputs_before_database(
    board_id: str, top_k: int, max_corpus_chunks: int
) -> None:
    storage = MagicMock()
    search_sparse = import_module("src.retrieval.sparse").search_sparse

    with pytest.raises(ValueError):
        search_sparse(
            "valid query",
            board_id,
            top_k=top_k,
            max_corpus_chunks=max_corpus_chunks,
            storage=storage,
        )

    storage.get_connection.assert_not_called()


def test_search_sparse_empty_query_skips_database() -> None:
    storage = MagicMock()
    search_sparse = import_module("src.retrieval.sparse").search_sparse

    assert search_sparse("... ---", _BOARD_A, storage=storage) == []
    storage.get_connection.assert_not_called()


def test_search_sparse_raises_instead_of_truncating_large_corpus() -> None:
    module = import_module("src.retrieval.sparse")
    too_large_error = getattr(module, "CorpusTooLargeError", None)
    assert too_large_error is not None, "CorpusTooLargeError is not implemented yet"
    rows = [
        _row(
            f"{index:08x}-0000-0000-0000-000000000000",
            f"{index:08x}-0000-0000-0000-000000000001",
            _BOARD_A,
            "guide.txt",
            "query token",
        )
        for index in range(1, 5002)
    ]
    storage, _, cursor = _storage_with_rows(rows)

    with pytest.raises(too_large_error, match="5000"):
        module.search_sparse("query", _BOARD_A, storage=storage)

    assert cursor.execute.call_args.args[1]["corpus_limit"] == 5001


@pytest.mark.integration
@pytest.mark.database
def test_bm25_board_filter_before_limit(retrieval_database) -> None:
    from src.retrieval.sparse import search_sparse

    storage, seed_document = retrieval_database
    board_a = str(UUID(_BOARD_A))
    board_b = "22222222-2222-2222-2222-222222222222"
    vector = [1.0] + [0.0] * 1023
    matching_metadata = {
        "model_id": DEFAULT_MODEL_ID,
        "model_revision": DEFAULT_MODEL_REVISION,
        "dimension": DEFAULT_DENSE_DIMENSION,
        "normalized": DEFAULT_NORMALIZE_EMBEDDINGS,
    }
    _, board_a_chunk_id = seed_document(
        board_a,
        "READY",
        "sundial-alpha",
        vector,
        matching_metadata,
        content="sundial alpha operations guide",
    )
    seed_document(
        board_a,
        "READY",
        "sundial-unrelated",
        vector,
        matching_metadata,
        content="unrelated research archive",
    )
    seed_document(
        board_b,
        "READY",
        "sundial-decoy",
        vector,
        matching_metadata,
        content="sundial target target target target",
    )
    seed_document(
        board_a, "PROCESSING", "sundial-processing", vector, matching_metadata
    )
    seed_document(
        board_a, "FAILED", "sundial-failed", vector, matching_metadata
    )
    seed_document(
        board_a,
        "READY",
        "sundial-wrong-model",
        vector,
        {**matching_metadata, "model_id": "different/model"},
    )

    hits = search_sparse("sundial target", board_a, top_k=1, storage=storage)

    assert len(hits) == 1
    assert hits[0].source.chunk_id == board_a_chunk_id
    assert hits[0].source.board_id == board_a
