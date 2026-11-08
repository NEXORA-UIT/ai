"""Tests for final evidence selection by cross-encoder score."""

from __future__ import annotations

from importlib import import_module
from unittest.mock import MagicMock

import pytest

from src.retrieval.schemas import FusedChunk, SourceChunk


def _candidate(
    chunk_id: str,
    rrf_score: float,
    *,
    document_id: str = "document-1",
    content: str | None = None,
) -> FusedChunk:
    passage = content if content is not None else f"passage for {chunk_id}"
    return FusedChunk(
        source=SourceChunk(
            board_id="11111111-1111-1111-1111-111111111111",
            chunk_id=chunk_id,
            document_id=document_id,
            chunk_index=2,
            file_name=f"{chunk_id}.pdf",
            storage_key=f"internal://{chunk_id}",
            content=passage,
            page_number=4,
            start_offset=20,
            end_offset=20 + len(passage),
        ),
        rrf_score=rrf_score,
        channels=("dense", "sparse"),
    )


def test_reranker_changes_order_by_relevance_score() -> None:
    rerank_chunks = import_module("src.reranking.select").rerank_chunks
    candidates = [_candidate("low-rrf", 0.01), _candidate("high-relevance", 0.02)]
    reranker = MagicMock()
    reranker.score_pairs.return_value = [-2.5, 4.75]

    results = rerank_chunks("question", candidates, reranker=reranker)

    assert [result.source.chunk_id for result in results] == [
        "high-relevance",
        "low-rrf",
    ]
    assert [result.rerank_score for result in results] == [4.75, -2.5]
    assert [result.rrf_score for result in results] == [0.02, 0.01]
    reranker.score_pairs.assert_called_once_with(
        "question", ["passage for low-rrf", "passage for high-relevance"]
    )


def test_reranker_keeps_unique_chunk_ids_and_first_source_record() -> None:
    rerank_chunks = import_module("src.reranking.select").rerank_chunks
    first = _candidate("same", 0.04, content="canonical source text")
    duplicate = _candidate("same", 0.01, content="duplicate source text")
    reranker = MagicMock()
    reranker.score_pairs.return_value = [0.5]

    results = rerank_chunks("question", [first, duplicate], reranker=reranker)

    assert len(results) == 1
    assert results[0].source == first.source
    assert results[0].rrf_score == 0.04
    reranker.score_pairs.assert_called_once_with("question", ["canonical source text"])


def test_empty_candidates_return_without_loading_default_reranker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("src.reranking.select")
    get_default = MagicMock(side_effect=AssertionError("must stay lazy"))
    monkeypatch.setattr(module, "get_default_reranker", get_default)

    assert module.rerank_chunks("question", []) == []

    get_default.assert_not_called()


def test_equal_scores_use_rrf_rank_then_chunk_id() -> None:
    rerank_chunks = import_module("src.reranking.select").rerank_chunks
    candidates = [
        _candidate("chunk-b", 0.02),
        _candidate("chunk-z", 0.02),
        _candidate("chunk-a", 0.03),
    ]
    reranker = MagicMock()
    reranker.score_pairs.return_value = [1.0, 1.0, 1.0]

    results = rerank_chunks("question", candidates, reranker=reranker)

    assert [result.source.chunk_id for result in results] == [
        "chunk-a",
        "chunk-b",
        "chunk-z",
    ]


def test_top_k_is_applied_after_reranking() -> None:
    rerank_chunks = import_module("src.reranking.select").rerank_chunks
    candidates = [_candidate("first", 0.03), _candidate("second", 0.02)]
    reranker = MagicMock()
    reranker.score_pairs.return_value = [0.1, 0.9]

    results = rerank_chunks("question", candidates, top_k=1, reranker=reranker)

    assert [result.source.chunk_id for result in results] == ["second"]


@pytest.mark.parametrize(
    "scores",
    [[], [float("nan")], [float("inf")], ["not-a-score"]],
    ids=["count-mismatch", "nan", "infinity", "nonnumeric"],
)
def test_reranker_rejects_bad_scores(scores: list[object]) -> None:
    rerank_chunks = import_module("src.reranking.select").rerank_chunks
    reranker = MagicMock()
    reranker.score_pairs.return_value = scores

    with pytest.raises(ValueError, match="reranker"):
        rerank_chunks("question", [_candidate("one", 0.02)], reranker=reranker)


@pytest.mark.parametrize(
    ("top_k", "candidates"),
    [
        (0, [_candidate("one", 0.02)]),
        (6, [_candidate("one", 0.02)]),
        (True, [_candidate("one", 0.02)]),
        (5, [_candidate(f"chunk-{index}", 0.01) for index in range(21)]),
    ],
    ids=["zero-top-k", "over-five-top-k", "boolean-top-k", "candidate-cap"],
)
def test_reranker_rejects_invalid_limits_before_scoring(
    top_k: int, candidates: list[FusedChunk]
) -> None:
    rerank_chunks = import_module("src.reranking.select").rerank_chunks
    reranker = MagicMock()

    with pytest.raises(ValueError):
        rerank_chunks("question", candidates, top_k=top_k, reranker=reranker)

    reranker.score_pairs.assert_not_called()


def test_reranker_rejects_conflicting_document_ids_before_scoring() -> None:
    rerank_chunks = import_module("src.reranking.select").rerank_chunks
    reranker = MagicMock()

    with pytest.raises(ValueError, match="conflicting document IDs"):
        rerank_chunks(
            "question",
            [
                _candidate("same", 0.02, document_id="document-a"),
                _candidate("same", 0.01, document_id="document-b"),
            ],
            reranker=reranker,
        )

    reranker.score_pairs.assert_not_called()
