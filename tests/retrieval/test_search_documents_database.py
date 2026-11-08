"""End-to-end facade test for PostgreSQL board/status/model scoping."""

from __future__ import annotations

from uuid import UUID

import pytest

from src.embedding.config import (
    DEFAULT_DENSE_DIMENSION,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
)
from src.retrieval import search_documents

_BOARD_A = "11111111-1111-4111-8111-111111111111"
_BOARD_B = "22222222-2222-4222-8222-222222222222"
_QUERY_VECTOR = [1.0] + [0.0] * (DEFAULT_DENSE_DIMENSION - 1)
_BOARD_A_VECTOR = [0.8, 0.6] + [0.0] * (DEFAULT_DENSE_DIMENSION - 2)


class _FakeEmbedder:
    model_name = DEFAULT_MODEL_ID
    revision = DEFAULT_MODEL_REVISION
    dimension = DEFAULT_DENSE_DIMENSION

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [_QUERY_VECTOR]


class _FakeReranker:
    def __init__(self) -> None:
        self.passages: list[str] = []

    def score_pairs(self, question: str, passages: list[str]) -> list[float]:
        assert question == "xyzzyqwerty"
        self.passages = passages
        return [0.9 for _ in passages]


@pytest.mark.integration
@pytest.mark.database
def test_search_documents_board_ready_and_model_scope_before_limit(
    retrieval_database,
) -> None:
    storage, seed_document = retrieval_database
    board_a = str(UUID(_BOARD_A))
    board_b = str(UUID(_BOARD_B))
    _, eligible_chunk_id = seed_document(
        board_a,
        "READY",
        "search-facade-eligible",
        _BOARD_A_VECTOR,
        content="A distinct source with no query token.",
    )
    seed_document(
        board_b,
        "READY",
        "search-facade-cross-board-high-score",
        _QUERY_VECTOR,
        content="xyzzyqwerty higher similarity but belongs to Board B",
    )
    seed_document(
        board_a,
        "PROCESSING",
        "search-facade-processing",
        _QUERY_VECTOR,
        content="xyzzyqwerty non-ready decoy",
    )
    seed_document(
        board_a,
        "READY",
        "search-facade-incompatible-model",
        _QUERY_VECTOR,
        {
            "model_id": "different/model",
            "model_revision": DEFAULT_MODEL_REVISION,
            "dimension": DEFAULT_DENSE_DIMENSION,
            "normalized": True,
        },
        content="xyzzyqwerty incompatible-model decoy",
    )
    embedder = _FakeEmbedder()
    reranker = _FakeReranker()

    result = search_documents(
        "xyzzyqwerty",
        board_a,
        top_k=1,
        candidate_k=1,
        embedder=embedder,
        storage=storage,
        reranker=reranker,
    )

    assert embedder.calls == [["xyzzyqwerty"]]
    assert len(result.chunks) == 1
    assert result.chunks[0].chunk_id == eligible_chunk_id
    assert result.chunks[0].content == "A distinct source with no query token."
    assert reranker.passages == ["A distinct source with no query token."]
    assert "storage_key" not in repr(result)
