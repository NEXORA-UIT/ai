"""Tests for one-embedding dense and sparse candidate coordination."""

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
from src.retrieval.metadata import EmbeddingFingerprint
from src.retrieval.schemas import RetrievedChunk, SourceChunk, SparseHit

_BOARD_A = "11111111-1111-1111-1111-111111111111"
_QUERY_VECTOR = [1.0] + [0.0] * (DEFAULT_DENSE_DIMENSION - 1)


def _embedder():
    embedder = MagicMock()
    embedder.model_name = DEFAULT_MODEL_ID
    embedder.revision = DEFAULT_MODEL_REVISION
    embedder.dimension = DEFAULT_DENSE_DIMENSION
    embedder.embed_texts.return_value = [_QUERY_VECTOR]
    return embedder


def _dense_hit(chunk_id: str = "dense") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="document-dense",
        board_id=_BOARD_A,
        chunk_index=0,
        file_name="dense.txt",
        storage_key="internal://dense",
        content="dense source",
        page_number=None,
        start_offset=0,
        end_offset=12,
        cosine_distance=0.0,
    )


def _sparse_hit(chunk_id: str = "sparse") -> SparseHit:
    return SparseHit(
        source=SourceChunk(
            board_id=_BOARD_A,
            chunk_id=chunk_id,
            document_id="document-sparse",
            chunk_index=0,
            file_name="sparse.txt",
            storage_key="internal://sparse",
            content="sparse source",
            page_number=None,
            start_offset=0,
            end_offset=14,
        ),
        sparse_score=0.0,
    )


def test_hybrid_embeds_once_and_forwards_shared_scope_and_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("src.retrieval.hybrid")
    embedder = _embedder()
    dense_hits = [_dense_hit()]
    sparse_hits = [_sparse_hit()]
    dense_search = MagicMock(return_value=dense_hits)
    sparse_search = MagicMock(return_value=sparse_hits)
    monkeypatch.setattr(module, "search_by_vector", dense_search)
    monkeypatch.setattr(module, "search_sparse", sparse_search)
    storage = object()

    candidates = module.search_hybrid_candidates(
        "hybrid question",
        _BOARD_A,
        candidate_k=7,
        embedder=embedder,
        storage=storage,
    )

    embedder.embed_texts.assert_called_once_with(["hybrid question"])
    dense_args, dense_kwargs = dense_search.call_args
    sparse_args, sparse_kwargs = sparse_search.call_args
    assert dense_args == (_QUERY_VECTOR, UUID(_BOARD_A))
    assert sparse_args == ("hybrid question", UUID(_BOARD_A))
    assert dense_kwargs["top_k"] == sparse_kwargs["top_k"] == 7
    assert dense_kwargs["fingerprint"] is sparse_kwargs["fingerprint"]
    assert dense_kwargs["fingerprint"] == EmbeddingFingerprint(
        model_id=DEFAULT_MODEL_ID,
        model_revision=DEFAULT_MODEL_REVISION,
        dimension=DEFAULT_DENSE_DIMENSION,
        normalized=DEFAULT_NORMALIZE_EMBEDDINGS,
    )
    assert dense_kwargs["storage"] is sparse_kwargs["storage"] is storage
    assert candidates.dense is dense_hits
    assert candidates.sparse is sparse_hits


@pytest.mark.parametrize(
    ("dense_hits", "sparse_hits"),
    [([], [_sparse_hit()]), ([_dense_hit()], [])],
    ids=["empty-dense-channel", "empty-sparse-channel"],
)
def test_hybrid_preserves_independently_empty_channels(
    dense_hits: list[RetrievedChunk],
    sparse_hits: list[SparseHit],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("src.retrieval.hybrid")
    embedder = _embedder()
    monkeypatch.setattr(
        module, "search_by_vector", MagicMock(return_value=dense_hits)
    )
    monkeypatch.setattr(
        module, "search_sparse", MagicMock(return_value=sparse_hits)
    )

    candidates = module.search_hybrid_candidates(
        "hybrid question", _BOARD_A, embedder=embedder, storage=object()
    )

    assert candidates.dense is dense_hits
    assert candidates.sparse is sparse_hits


@pytest.mark.parametrize("failing_channel", ["dense", "sparse"])
def test_hybrid_propagates_channel_errors_without_returning_partial_results(
    failing_channel: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = import_module("src.retrieval.hybrid")
    embedder = _embedder()
    failure = RuntimeError("retrieval channel failed")
    dense_search = MagicMock(return_value=[_dense_hit()])
    sparse_search = MagicMock(return_value=[_sparse_hit()])
    if failing_channel == "dense":
        dense_search.side_effect = failure
    else:
        sparse_search.side_effect = failure
    monkeypatch.setattr(module, "search_by_vector", dense_search)
    monkeypatch.setattr(module, "search_sparse", sparse_search)

    with pytest.raises(RuntimeError, match="retrieval channel failed"):
        module.search_hybrid_candidates(
            "hybrid question", _BOARD_A, embedder=embedder, storage=object()
        )

    if failing_channel == "dense":
        sparse_search.assert_not_called()
    else:
        dense_search.assert_called_once()


@pytest.mark.parametrize(
    ("question", "board_id", "candidate_k"),
    [
        ("  ", _BOARD_A, 5),
        ("valid", "not-a-uuid", 5),
        ("valid", _BOARD_A, 0),
        ("valid", _BOARD_A, True),
        ("valid", _BOARD_A, 21),
    ],
    ids=["blank-question", "bad-board", "zero-k", "boolean-k", "over-cap-k"],
)
def test_hybrid_validates_inputs_before_embedding_or_search(
    question: str,
    board_id: str,
    candidate_k: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("src.retrieval.hybrid")
    embedder = _embedder()
    dense_search = MagicMock()
    sparse_search = MagicMock()
    monkeypatch.setattr(module, "search_by_vector", dense_search)
    monkeypatch.setattr(module, "search_sparse", sparse_search)

    with pytest.raises(ValueError):
        module.search_hybrid_candidates(
            question,
            board_id,
            candidate_k=candidate_k,
            embedder=embedder,
            storage=object(),
        )

    embedder.embed_texts.assert_not_called()
    dense_search.assert_not_called()
    sparse_search.assert_not_called()


@pytest.mark.parametrize("vectors", [[], [_QUERY_VECTOR, _QUERY_VECTOR]])
def test_hybrid_rejects_wrong_embedding_batch_size_before_search(
    vectors: list[list[float]], monkeypatch: pytest.MonkeyPatch
) -> None:
    module = import_module("src.retrieval.hybrid")
    embedder = _embedder()
    embedder.embed_texts.return_value = vectors
    dense_search = MagicMock()
    sparse_search = MagicMock()
    monkeypatch.setattr(module, "search_by_vector", dense_search)
    monkeypatch.setattr(module, "search_sparse", sparse_search)

    with pytest.raises(ValueError, match="exactly one vector"):
        module.search_hybrid_candidates(
            "valid question", _BOARD_A, embedder=embedder, storage=object()
        )

    dense_search.assert_not_called()
    sparse_search.assert_not_called()


@pytest.mark.integration
@pytest.mark.database
def test_hybrid_search_returns_board_scoped_dense_and_sparse_candidates(
    retrieval_database,
) -> None:
    from src.retrieval.hybrid import search_hybrid_candidates

    storage, seed_document = retrieval_database
    board_a = str(UUID(_BOARD_A))
    board_b = "22222222-2222-2222-2222-222222222222"
    seed_document(
        board_a,
        "READY",
        "hybrid-match",
        _QUERY_VECTOR,
        content="hybrid query scoped source",
    )
    seed_document(
        board_b,
        "READY",
        "hybrid-cross-board-decoy",
        _QUERY_VECTOR,
        content="hybrid query cross board decoy",
    )
    embedder = _embedder()

    candidates = search_hybrid_candidates(
        "hybrid query",
        board_a,
        candidate_k=3,
        embedder=embedder,
        storage=storage,
    )

    embedder.embed_texts.assert_called_once_with(["hybrid query"])
    assert candidates.dense
    assert candidates.sparse
    assert all(hit.board_id == board_a for hit in candidates.dense)
    assert all(hit.source.board_id == board_a for hit in candidates.sparse)
