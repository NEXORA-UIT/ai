"""PostgreSQL-to-reranker integration tests with fake model interfaces."""

from __future__ import annotations

from uuid import UUID

import pytest

from src.embedding.config import (
    DEFAULT_DENSE_DIMENSION,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
)
from src.reranking.select import rerank_chunks
from src.retrieval.fusion import fuse_rrf
from src.retrieval.hybrid import search_hybrid_candidates

_BOARD_A = "11111111-1111-1111-1111-111111111111"
_QUERY_VECTOR = [1.0] + [0.0] * (DEFAULT_DENSE_DIMENSION - 1)


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
        assert question == "reranking control"
        self.passages = passages
        return [float(index) for index in range(len(passages))]


@pytest.mark.integration
@pytest.mark.database
def test_postgres_hybrid_retrieval_to_reranker_preserves_scope_and_sources(
    retrieval_database,
) -> None:
    storage, seed_document = retrieval_database
    board_a = str(UUID(_BOARD_A))
    board_b = "22222222-2222-2222-2222-222222222222"
    _, eligible_chunk_id = seed_document(
        board_a,
        "READY",
        "rerank-eligible",
        _QUERY_VECTOR,
        content="reranking control eligible source",
    )
    seed_document(
        board_b,
        "READY",
        "rerank-cross-board",
        _QUERY_VECTOR,
        content="reranking control cross-board decoy",
    )
    seed_document(
        board_a,
        "PROCESSING",
        "rerank-processing",
        _QUERY_VECTOR,
        content="reranking control processing decoy",
    )
    seed_document(
        board_a,
        "FAILED",
        "rerank-failed",
        _QUERY_VECTOR,
        content="reranking control failed decoy",
    )
    seed_document(
        board_a,
        "READY",
        "rerank-wrong-model",
        _QUERY_VECTOR,
        {
            "model_id": "different/model",
            "model_revision": DEFAULT_MODEL_REVISION,
            "dimension": DEFAULT_DENSE_DIMENSION,
            "normalized": True,
        },
        content="reranking control incompatible decoy",
    )
    embedder = _FakeEmbedder()
    reranker = _FakeReranker()

    candidates = search_hybrid_candidates(
        "reranking control",
        board_a,
        candidate_k=20,
        embedder=embedder,
        storage=storage,
    )
    fused = fuse_rrf(candidates)
    results = rerank_chunks(
        "reranking control", fused, reranker=reranker
    )

    assert embedder.calls == [["reranking control"]]
    assert len(results) == 1
    assert results[0].source.chunk_id == eligible_chunk_id
    assert results[0].source.board_id == board_a
    assert results[0].source.storage_key.startswith("local://retrieval-tests/")
    assert reranker.passages == ["reranking control eligible source"]
