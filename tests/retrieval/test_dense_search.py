"""Unit tests for Board-scoped dense vector retrieval."""

from __future__ import annotations

from importlib import import_module
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from pgvector import Vector


def _function(name: str):
    """Load a retrieval function as an assertion failure while it is absent."""
    try:
        module = import_module("src.retrieval.dense")
    except ModuleNotFoundError as exc:
        pytest.fail(f"retrieval module is not implemented yet: {exc}")

    function = getattr(module, name, None)
    assert callable(function), f"retrieval function {name} is not implemented yet"
    return function


def _storage_with_rows(rows: list[tuple[object, ...]]):
    storage = MagicMock()
    connection = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.return_value = rows
    connection.__enter__.return_value = connection
    connection.cursor.return_value.__enter__.return_value = cursor
    storage.get_connection.return_value = connection
    return storage, connection, cursor


def _valid_vector() -> list[float]:
    return [1.0] + [0.0] * 1023


@pytest.mark.parametrize(
    "query_vector",
    [
        [1.0] * 1023,
        [float("nan")] + [0.0] * 1023,
        [float("inf")] + [0.0] * 1023,
        [0.0] * 1024,
    ],
    ids=["wrong-dimension", "nan", "infinity", "zero"],
)
def test_search_by_vector_rejects_invalid_vector(query_vector: list[float]) -> None:
    storage = MagicMock()

    with pytest.raises(ValueError):
        _function("search_by_vector")(
            query_vector,
            "11111111-1111-1111-1111-111111111111",
            storage=storage,
        )

    storage.get_connection.assert_not_called()


@pytest.mark.parametrize(
    ("board_id", "top_k"),
    [
        ("not-a-uuid", 5),
        ("11111111-1111-1111-1111-111111111111", 0),
        ("11111111-1111-1111-1111-111111111111", -1),
        ("11111111-1111-1111-1111-111111111111", 1.5),
    ],
    ids=["bad-board-id", "zero-limit", "negative-limit", "fractional-limit"],
)
def test_search_by_vector_rejects_invalid_scope_or_limit(
    board_id: str, top_k: int
) -> None:
    storage = MagicMock()

    with pytest.raises(ValueError):
        _function("search_by_vector")(
            _valid_vector(), board_id, top_k=top_k, storage=storage
        )

    storage.get_connection.assert_not_called()


def test_search_by_vector_uses_board_ready_cosine_query() -> None:
    storage, connection, cursor = _storage_with_rows([])
    board_id = "11111111-1111-1111-1111-111111111111"

    result = _function("search_by_vector")(
        _valid_vector(), board_id, top_k=1, storage=storage
    )

    assert result == []
    storage.get_connection.assert_called_once_with()
    connection.cursor.assert_called_once_with()
    query, params = cursor.execute.call_args.args
    normalized_query = " ".join(query.split()).casefold()
    where = normalized_query.index("where d.board_id")
    order = normalized_query.index("order by")
    limit = normalized_query.index("limit")
    assert "join documents as d on d.id = dc.document_id" in normalized_query
    assert "d.status = %(ready_status)s" in normalized_query
    assert "d.metadata -> 'model_id'" in normalized_query
    assert "d.metadata -> 'model_revision'" in normalized_query
    assert "d.metadata -> 'dimension'" in normalized_query
    assert "d.metadata -> 'normalized'" in normalized_query
    assert "dc.embedding <=> %(query_vector)s::vector" in normalized_query
    assert where < order < limit
    assert params["board_id"] == UUID(board_id)
    assert params["ready_status"] == "READY"
    assert params["top_k"] == 1
    assert isinstance(params["query_vector"], Vector)


def test_search_by_vector_maps_source_fields_and_distance() -> None:
    row = (
        UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        UUID("11111111-1111-1111-1111-111111111111"),
        3,
        "handbook.txt",
        "local://handbook.txt",
        "The release date is 2026-12-01.",
        None,
        120,
        153,
        0.125,
    )
    storage, _, _ = _storage_with_rows([row])

    results = _function("search_by_vector")(
        _valid_vector(),
        "11111111-1111-1111-1111-111111111111",
        storage=storage,
    )

    assert len(results) == 1
    result = results[0]
    assert result.chunk_id == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    assert result.document_id == "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    assert result.board_id == "11111111-1111-1111-1111-111111111111"
    assert result.chunk_index == 3
    assert result.file_name == "handbook.txt"
    assert result.storage_key == "local://handbook.txt"
    assert result.content == "The release date is 2026-12-01."
    assert result.page_number is None
    assert result.start_offset == 120
    assert result.end_offset == 153
    assert result.cosine_distance == pytest.approx(0.125)


@pytest.mark.integration
@pytest.mark.database
def test_top_k_one_filters_board_before_limit_and_binds_pgvector_type(
    retrieval_database,
) -> None:
    storage, seed_document = retrieval_database
    board_a = "11111111-1111-1111-1111-111111111111"
    board_b = "22222222-2222-2222-2222-222222222222"
    query_vector = _valid_vector()
    best_a_vector = [0.8, 0.6] + [0.0] * 1022
    second_a_vector = [0.0, 1.0] + [0.0] * 1022

    _, best_a_chunk_id = seed_document(board_a, "READY", "best-a", best_a_vector)
    _, second_a_chunk_id = seed_document(board_a, "READY", "second-a", second_a_vector)
    seed_document(board_a, "PROCESSING", "processing-a", query_vector)
    seed_document(board_a, "FAILED", "failed-a", query_vector)
    seed_document(board_b, "READY", "closest-b", query_vector)

    results = _function("search_by_vector")(
        query_vector, board_a, top_k=1, storage=storage
    )

    assert len(results) == 1
    assert results[0].chunk_id == best_a_chunk_id
    assert results[0].cosine_distance == pytest.approx(0.2, abs=1e-5)

    board_a_results = _function("search_by_vector")(
        query_vector, board_a, top_k=5, storage=storage
    )
    assert {result.chunk_id for result in board_a_results} == {
        best_a_chunk_id,
        second_a_chunk_id,
    }


def test_search_by_text_rejects_whitespace_question() -> None:
    embedder = MagicMock()
    storage = MagicMock()

    with pytest.raises(ValueError):
        _function("search_by_text")(
            " \t\n ",
            "11111111-1111-1111-1111-111111111111",
            embedder=embedder,
            storage=storage,
        )

    embedder.embed_texts.assert_not_called()
    storage.get_connection.assert_not_called()


@pytest.mark.parametrize(
    ("board_id", "top_k"),
    [
        ("not-a-uuid", 5),
        ("11111111-1111-1111-1111-111111111111", 0),
        ("11111111-1111-1111-1111-111111111111", True),
    ],
    ids=["bad-board-id", "zero-limit", "boolean-limit"],
)
def test_search_by_text_rejects_invalid_scope_before_embedding(
    board_id: str, top_k: int
) -> None:
    embedder = MagicMock()
    embedder.embed_texts.return_value = [_valid_vector()]

    with pytest.raises(ValueError):
        _function("search_by_text")(
            "When is the release?",
            board_id,
            top_k=top_k,
            embedder=embedder,
            storage=MagicMock(),
        )

    embedder.embed_texts.assert_not_called()


def test_search_by_text_embeds_once_and_delegates_vector(monkeypatch) -> None:
    module = import_module("src.retrieval.dense")
    search_by_text = _function("search_by_text")
    vector = _valid_vector()

    class FixedEmbedder:
        model_name = "BAAI/bge-m3"
        revision = "test-revision"
        dimension = 1024

        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(texts)
            return [vector]

    embedder = FixedEmbedder()
    storage = object()
    expected_results = [object()]
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
        return expected_results

    monkeypatch.setattr(module, "search_by_vector", fake_search_by_vector)

    results = search_by_text(
        "When is the release?",
        "11111111-1111-1111-1111-111111111111",
        top_k=3,
        embedder=embedder,
        storage=storage,
    )

    assert embedder.calls == [["When is the release?"]]
    assert captured == {
        "query_vector": vector,
        "board_id": "11111111-1111-1111-1111-111111111111",
        "top_k": 3,
        "fingerprint": import_module("src.retrieval.metadata").EmbeddingFingerprint(
            model_id="BAAI/bge-m3",
            model_revision="test-revision",
            dimension=1024,
            normalized=True,
        ),
        "storage": storage,
    }
    assert results is expected_results


def test_default_embedder_instance_is_reused(monkeypatch) -> None:
    module = import_module("src.retrieval.dense")
    get_default_embedder = _function("get_default_embedder")
    get_default_embedder.cache_clear()
    created: list[object] = []

    def new_embedder() -> object:
        embedder = object()
        created.append(embedder)
        return embedder

    monkeypatch.setattr(module, "_create_default_embedder", new_embedder)

    try:
        first = get_default_embedder()
        second = get_default_embedder()
    finally:
        get_default_embedder.cache_clear()

    assert first is second
    assert len(created) == 1
