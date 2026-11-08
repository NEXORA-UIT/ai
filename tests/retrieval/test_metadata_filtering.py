"""Dense retrieval compatibility filtering for embedding fingerprints."""

from __future__ import annotations

from importlib import import_module
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from pgvector import Vector

from src.embedding.config import (
    DEFAULT_DENSE_DIMENSION,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    DEFAULT_NORMALIZE_EMBEDDINGS,
)

_BOARD_A = "11111111-1111-1111-1111-111111111111"
_BOARD_B = "22222222-2222-2222-2222-222222222222"


def _valid_vector() -> list[float]:
    return [1.0] + [0.0] * 1023


def _fingerprint_type():
    module = import_module("src.retrieval.metadata")
    fingerprint_type = getattr(module, "EmbeddingFingerprint", None)
    assert fingerprint_type is not None, "EmbeddingFingerprint is not implemented yet"
    return fingerprint_type


def _compatible_metadata() -> dict[str, object]:
    return {
        "model_id": DEFAULT_MODEL_ID,
        "model_revision": DEFAULT_MODEL_REVISION,
        "dimension": DEFAULT_DENSE_DIMENSION,
        "normalized": DEFAULT_NORMALIZE_EMBEDDINGS,
    }


def test_embedding_fingerprint_rejects_invalid_metadata() -> None:
    fingerprint_type = _fingerprint_type()

    for values in (
        {"model_id": " ", "model_revision": "revision"},
        {"model_id": "model", "model_revision": ""},
        {"model_id": "model", "model_revision": "revision", "dimension": 0},
        {
            "model_id": "model",
            "model_revision": "revision",
            "normalized": 1,
        },
    ):
        with pytest.raises(ValueError):
            fingerprint_type(**values)


def test_search_by_vector_binds_typed_fingerprint_before_order_and_limit() -> None:
    module = import_module("src.retrieval.dense")
    fingerprint_type = _fingerprint_type()
    storage = MagicMock()
    connection = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    connection.__enter__.return_value = connection
    connection.cursor.return_value.__enter__.return_value = cursor
    storage.get_connection.return_value = connection
    fingerprint = fingerprint_type(
        model_id="test/model",
        model_revision="revision-123",
        dimension=1024,
        normalized=True,
    )

    assert module.search_by_vector(
        _valid_vector(), _BOARD_A, fingerprint=fingerprint, storage=storage
    ) == []

    query, params = cursor.execute.call_args.args
    normalized_query = " ".join(query.split()).casefold()
    where = normalized_query.index("where d.board_id")
    order = normalized_query.index("order by")
    limit = normalized_query.index("limit")
    metadata_conditions = (
        "d.metadata -> 'model_id' = to_jsonb(%(model_id)s::text)",
        "d.metadata -> 'model_revision' = to_jsonb(%(model_revision)s::text)",
        "d.metadata -> 'dimension' = to_jsonb(%(dimension)s::integer)",
        "d.metadata -> 'normalized' = to_jsonb(%(normalized)s::boolean)",
    )
    assert all(condition in normalized_query for condition in metadata_conditions)
    assert where < order < limit
    assert params["board_id"] == UUID(_BOARD_A)
    assert params["model_id"] == "test/model"
    assert params["model_revision"] == "revision-123"
    assert params["dimension"] == 1024
    assert params["normalized"] is True
    assert isinstance(params["query_vector"], Vector)


def test_search_by_text_derives_fingerprint_from_selected_embedder(monkeypatch) -> None:
    dense = import_module("src.retrieval.dense")
    fingerprint_type = _fingerprint_type()

    class FixedEmbedder:
        model_name = "test/query-model"
        revision = "query-revision"
        dimension = 1024

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            assert texts == ["test query"]
            return [_valid_vector()]

    captured: dict[str, object] = {}

    def fake_search_by_vector(
        query_vector: list[float],
        board_id: str,
        *,
        top_k: int,
        fingerprint: object,
        storage: object,
    ) -> list[object]:
        captured.update(
            query_vector=query_vector,
            board_id=board_id,
            top_k=top_k,
            fingerprint=fingerprint,
            storage=storage,
        )
        return []

    monkeypatch.setattr(dense, "search_by_vector", fake_search_by_vector)
    storage = object()

    assert dense.search_by_text(
        "test query",
        _BOARD_A,
        top_k=3,
        embedder=FixedEmbedder(),
        storage=storage,
    ) == []

    assert captured["fingerprint"] == fingerprint_type(
        model_id="test/query-model",
        model_revision="query-revision",
        dimension=1024,
        normalized=True,
    )
    assert captured["board_id"] == _BOARD_A
    assert captured["top_k"] == 3
    assert captured["storage"] is storage


@pytest.mark.integration
@pytest.mark.database
def test_metadata_filters_incompatible_chunks_before_limit(retrieval_database) -> None:
    from src.retrieval.dense import search_by_vector
    from src.retrieval.metadata import EmbeddingFingerprint

    storage, seed_document = retrieval_database
    query_vector = _valid_vector()
    fingerprint = EmbeddingFingerprint(
        model_id=DEFAULT_MODEL_ID,
        model_revision=DEFAULT_MODEL_REVISION,
        dimension=DEFAULT_DENSE_DIMENSION,
        normalized=DEFAULT_NORMALIZE_EMBEDDINGS,
    )
    eligible_metadata = _compatible_metadata()
    eligible_vector = [0.8, 0.6] + [0.0] * 1022

    _, eligible_chunk_id = seed_document(
        _BOARD_A, "READY", "compatible-board-a", eligible_vector, eligible_metadata
    )
    seed_document(
        _BOARD_A,
        "READY",
        "wrong-model",
        query_vector,
        {**eligible_metadata, "model_id": "another/model"},
    )
    seed_document(
        _BOARD_A,
        "READY",
        "wrong-revision",
        query_vector,
        {**eligible_metadata, "model_revision": "another-revision"},
    )
    seed_document(
        _BOARD_A,
        "READY",
        "wrong-dimension",
        query_vector,
        {**eligible_metadata, "dimension": 768},
    )
    seed_document(
        _BOARD_A,
        "READY",
        "wrong-normalization",
        query_vector,
        {**eligible_metadata, "normalized": False},
    )
    seed_document(
        _BOARD_A, "READY", "missing-metadata", query_vector, {}
    )
    seed_document(
        _BOARD_A,
        "READY",
        "wrong-json-types",
        query_vector,
        {**eligible_metadata, "dimension": "1024", "normalized": "true"},
    )
    seed_document(
        _BOARD_B, "READY", "board-b-decoy", query_vector, eligible_metadata
    )
    seed_document(
        _BOARD_A, "PROCESSING", "processing-decoy", query_vector, eligible_metadata
    )

    top_one = search_by_vector(
        query_vector, _BOARD_A, top_k=1, fingerprint=fingerprint, storage=storage
    )
    all_matches = search_by_vector(
        query_vector, _BOARD_A, top_k=10, fingerprint=fingerprint, storage=storage
    )

    assert len(top_one) == 1
    assert top_one[0].file_name == "compatible-board-a.txt", top_one[0].file_name
    assert top_one[0].chunk_id == eligible_chunk_id
    assert top_one[0].board_id == _BOARD_A
    assert [chunk.chunk_id for chunk in all_matches] == [eligible_chunk_id]
