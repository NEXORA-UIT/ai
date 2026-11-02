"""Public facade tests with injected retrieval, fusion and reranking stages."""

from __future__ import annotations

from dataclasses import asdict
from importlib import import_module
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from src.retrieval import SearchChunk, SearchResult, search_documents
from src.retrieval.schemas import (
    FusedChunk,
    HybridCandidates,
    RerankedChunk,
    RetrievedChunk,
    SourceChunk,
    SparseHit,
)

_BOARD_A = "11111111-1111-4111-8111-111111111111"
_STORAGE_KEY = "private://bucket/internal/document.pdf"


def _source(chunk_id: str = "chunk-a") -> SourceChunk:
    return SourceChunk(
        board_id=_BOARD_A,
        chunk_id=chunk_id,
        document_id="document-a",
        chunk_index=2,
        file_name="policy.pdf",
        storage_key=_STORAGE_KEY,
        content="The approved policy source.",
        page_number=4,
        start_offset=100,
        end_offset=128,
    )


def _retrieved(chunk_id: str = "chunk-a") -> RetrievedChunk:
    source = _source(chunk_id)
    return RetrievedChunk(
        chunk_id=source.chunk_id,
        document_id=source.document_id,
        board_id=source.board_id,
        chunk_index=source.chunk_index,
        file_name=source.file_name,
        storage_key=source.storage_key,
        content=source.content,
        page_number=source.page_number,
        start_offset=source.start_offset,
        end_offset=source.end_offset,
        cosine_distance=0.1,
    )


def _reranked(chunk_id: str = "chunk-a") -> RerankedChunk:
    return RerankedChunk(source=_source(chunk_id), rerank_score=0.92, rrf_score=0.03)


def test_facade_forwards_shared_scope_and_candidate_pool_to_each_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("src.retrieval.search_documents")
    candidates = HybridCandidates(dense=[_retrieved()], sparse=[])
    fused = [FusedChunk(source=_source(), rrf_score=0.03, channels=("dense",))]
    reranked = [_reranked()]
    hybrid = MagicMock(return_value=candidates)
    fuse = MagicMock(return_value=fused)
    rerank = MagicMock(return_value=reranked)
    monkeypatch.setattr(module, "search_hybrid_candidates", hybrid)
    monkeypatch.setattr(module, "fuse_rrf", fuse)
    monkeypatch.setattr(module, "rerank_chunks", rerank)
    embedder = object()
    storage = object()
    reranker = object()

    result = search_documents(
        "  How does this policy work?  ",
        UUID(_BOARD_A),
        top_k=3,
        candidate_k=8,
        embedder=embedder,
        storage=storage,
        reranker=reranker,
    )

    hybrid.assert_called_once_with(
        "  How does this policy work?  ",
        UUID(_BOARD_A),
        candidate_k=8,
        embedder=embedder,
        storage=storage,
    )
    fuse.assert_called_once_with(candidates, top_k=8)
    rerank.assert_called_once_with(
        "  How does this policy work?  ", fused, top_k=3, reranker=reranker
    )
    assert result.question == "  How does this policy work?  "
    assert result.board_id == _BOARD_A
    assert result.chunks == (
        SearchChunk(
            chunk_id="chunk-a",
            document_id="document-a",
            chunk_index=2,
            file_name="policy.pdf",
            content="The approved policy source.",
            page_number=4,
            start_offset=100,
            end_offset=128,
            rerank_score=0.92,
            rrf_score=0.03,
        ),
    )


def test_public_payload_excludes_internal_storage_key() -> None:
    result = SearchResult(
        question="question",
        board_id=_BOARD_A,
        chunks=(
            SearchChunk(
                chunk_id="chunk-a",
                document_id="document-a",
                chunk_index=2,
                file_name="policy.pdf",
                content="The approved policy source.",
                page_number=4,
                start_offset=100,
                end_offset=128,
                rerank_score=0.92,
                rrf_score=0.03,
            ),
        ),
    )

    payload = asdict(result)

    assert "storage_key" not in str(payload)
    assert "private://" not in str(payload)
    assert payload["chunks"][0]["chunk_id"] == "chunk-a"
    assert payload["chunks"][0]["page_number"] == 4


def test_facade_rejects_final_results_outside_board_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("src.retrieval.search_documents")
    cross_board_source = SourceChunk(
        **{
            **_source().__dict__,
            "board_id": "22222222-2222-4222-8222-222222222222",
        }
    )
    monkeypatch.setattr(
        module,
        "search_hybrid_candidates",
        lambda *_args, **_kwargs: HybridCandidates([], []),
    )
    monkeypatch.setattr(module, "fuse_rrf", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        module,
        "rerank_chunks",
        lambda *_args, **_kwargs: [
            RerankedChunk(source=cross_board_source, rerank_score=0.5, rrf_score=0.01)
        ],
    )

    with pytest.raises(ValueError, match="outside the requested Board"):
        search_documents("question", _BOARD_A)


def test_empty_corpus_returns_typed_result_without_loading_reranker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("src.retrieval.search_documents")
    reranking = import_module("src.reranking.select")
    monkeypatch.setattr(
        module,
        "search_hybrid_candidates",
        lambda *_args, **_kwargs: HybridCandidates([], []),
    )
    monkeypatch.setattr(
        reranking,
        "get_default_reranker",
        MagicMock(side_effect=AssertionError("loaded")),
    )

    result = search_documents("valid question", _BOARD_A)

    assert result == SearchResult(
        question="valid question", board_id=_BOARD_A, chunks=()
    )


def test_facade_deduplicates_dense_and_sparse_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("src.retrieval.search_documents")
    source = _source()
    candidates = HybridCandidates(
        dense=[_retrieved()],
        sparse=[SparseHit(source=source, sparse_score=2.0)],
    )
    monkeypatch.setattr(
        module, "search_hybrid_candidates", lambda *_args, **_kwargs: candidates
    )

    class _Reranker:
        passages: list[str] = []

        def score_pairs(self, question: str, passages: list[str]) -> list[float]:
            self.passages = passages
            return [0.8 for _ in passages]

    reranker = _Reranker()
    result = search_documents(
        "policy question", _BOARD_A, candidate_k=5, reranker=reranker
    )

    assert len(result.chunks) == 1
    assert result.chunks[0].chunk_id == "chunk-a"
    assert reranker.passages == ["The approved policy source."]


@pytest.mark.parametrize(
    ("question", "board_id", "top_k", "candidate_k"),
    [
        (" \t ", _BOARD_A, 1, 1),
        ("question", "not-a-uuid", 1, 1),
        ("question", _BOARD_A, 0, 1),
        ("question", _BOARD_A, 6, 6),
        ("question", _BOARD_A, True, 1),
        ("question", _BOARD_A, 2, 1),
        ("question", _BOARD_A, 1, 21),
        ("question", _BOARD_A, 1, True),
    ],
)
def test_input_validation_precedes_retrieval(
    question: str,
    board_id: str,
    top_k: int,
    candidate_k: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("src.retrieval.search_documents")
    hybrid = MagicMock()
    monkeypatch.setattr(module, "search_hybrid_candidates", hybrid)

    with pytest.raises(ValueError):
        search_documents(question, board_id, top_k=top_k, candidate_k=candidate_k)

    hybrid.assert_not_called()


def test_candidate_limit_is_forwarded_to_rrf_before_final_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("src.retrieval.search_documents")
    candidates = HybridCandidates(dense=[], sparse=[])
    hybrid = MagicMock(return_value=candidates)
    fuse = MagicMock(return_value=[])
    rerank = MagicMock(return_value=[])
    monkeypatch.setattr(module, "search_hybrid_candidates", hybrid)
    monkeypatch.setattr(module, "fuse_rrf", fuse)
    monkeypatch.setattr(module, "rerank_chunks", rerank)

    search_documents("question", _BOARD_A, top_k=2, candidate_k=20)

    assert hybrid.call_args.kwargs["candidate_k"] == 20
    fuse.assert_called_once_with(candidates, top_k=20)
    rerank.assert_called_once_with("question", [], top_k=2, reranker=None)
