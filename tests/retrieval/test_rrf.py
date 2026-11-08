"""Tests for deterministic reciprocal-rank fusion."""

from __future__ import annotations

from importlib import import_module
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from src.embedding.config import (
    DEFAULT_DENSE_DIMENSION,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
)
from src.retrieval.schemas import (
    HybridCandidates,
    RetrievedChunk,
    SourceChunk,
    SparseHit,
)

_BOARD_A = "11111111-1111-1111-1111-111111111111"
_QUERY_VECTOR = [1.0] + [0.0] * (DEFAULT_DENSE_DIMENSION - 1)


def _dense_hit(
    chunk_id: str,
    *,
    document_id: str = "document-1",
    content: str = "same source text",
    distance: float = 0.99,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        board_id=_BOARD_A,
        chunk_index=0,
        file_name=f"{chunk_id}.txt",
        storage_key=f"internal://{chunk_id}",
        content=content,
        page_number=None,
        start_offset=0,
        end_offset=len(content),
        cosine_distance=distance,
    )


def _sparse_hit(
    chunk_id: str,
    *,
    document_id: str = "document-1",
    content: str = "same source text",
    score: float = 1000.0,
) -> SparseHit:
    return SparseHit(
        source=SourceChunk(
            board_id=_BOARD_A,
            chunk_id=chunk_id,
            document_id=document_id,
            chunk_index=0,
            file_name=f"{chunk_id}.txt",
            storage_key=f"internal://{chunk_id}",
            content=content,
            page_number=None,
            start_offset=0,
            end_offset=len(content),
        ),
        sparse_score=score,
    )


def test_rrf_deduplicates_chunk_and_adds_rank_one_contributions() -> None:
    fuse_rrf = import_module("src.retrieval.fusion").fuse_rrf
    dense = _dense_hit("same", distance=999.0)
    sparse = _sparse_hit("same", score=-999.0)

    fused = fuse_rrf(HybridCandidates(dense=[dense], sparse=[sparse]))

    assert len(fused) == 1
    assert fused[0].source == SourceChunk.from_dense(dense)
    assert fused[0].rrf_score == pytest.approx(2 / 61)
    assert fused[0].channels == ("dense", "sparse")


def test_rrf_keeps_same_text_chunks_with_distinct_ids() -> None:
    fuse_rrf = import_module("src.retrieval.fusion").fuse_rrf
    candidates = HybridCandidates(
        dense=[_dense_hit("chunk-a", content="identical text")],
        sparse=[_sparse_hit("chunk-b", content="identical text")],
    )

    fused = fuse_rrf(candidates)

    assert [hit.source.chunk_id for hit in fused] == ["chunk-a", "chunk-b"]


@pytest.mark.parametrize(
    ("dense", "sparse", "expected_ids"),
    [
        ([], [], []),
        ([], [_sparse_hit("sparse-only")], ["sparse-only"]),
        ([_dense_hit("dense-only")], [], ["dense-only"]),
    ],
    ids=["both-empty", "dense-empty", "sparse-empty"],
)
def test_rrf_handles_empty_channels(
    dense: list[RetrievedChunk],
    sparse: list[SparseHit],
    expected_ids: list[str],
) -> None:
    fuse_rrf = import_module("src.retrieval.fusion").fuse_rrf

    fused = fuse_rrf(HybridCandidates(dense=dense, sparse=sparse))

    assert [hit.source.chunk_id for hit in fused] == expected_ids


def test_rrf_duplicate_id_within_channel_contributes_once() -> None:
    fuse_rrf = import_module("src.retrieval.fusion").fuse_rrf
    candidates = HybridCandidates(
        dense=[_dense_hit("repeat"), _dense_hit("repeat")], sparse=[]
    )

    fused = fuse_rrf(candidates)

    assert len(fused) == 1
    assert fused[0].rrf_score == pytest.approx(1 / 61)
    assert fused[0].channels == ("dense",)


def test_rrf_ties_sort_by_chunk_id() -> None:
    fuse_rrf = import_module("src.retrieval.fusion").fuse_rrf
    candidates = HybridCandidates(
        dense=[_dense_hit("chunk-b"), _dense_hit("chunk-a")],
        sparse=[_sparse_hit("chunk-a"), _sparse_hit("chunk-b")],
    )

    fused = fuse_rrf(candidates)

    assert [hit.source.chunk_id for hit in fused] == ["chunk-a", "chunk-b"]


def test_rrf_uses_channel_rank_instead_of_raw_scores() -> None:
    fuse_rrf = import_module("src.retrieval.fusion").fuse_rrf
    candidates = HybridCandidates(
        dense=[_dense_hit("chunk-z", distance=0.0)],
        sparse=[_sparse_hit("chunk-a", score=1_000_000.0)],
    )

    fused = fuse_rrf(candidates)

    assert [hit.source.chunk_id for hit in fused] == ["chunk-a", "chunk-z"]
    assert fused[0].rrf_score == fused[1].rrf_score == pytest.approx(1 / 61)


def test_rrf_truncates_after_fusing_channel_ranks() -> None:
    fuse_rrf = import_module("src.retrieval.fusion").fuse_rrf
    candidates = HybridCandidates(
        dense=[_dense_hit("a"), _dense_hit("b")],
        sparse=[_sparse_hit("b"), _sparse_hit("a")],
    )

    fused = fuse_rrf(candidates, top_k=1)

    assert len(fused) == 1
    assert fused[0].source.chunk_id == "a"


def test_rrf_raises_when_chunk_id_has_conflicting_document_ids() -> None:
    fuse_rrf = import_module("src.retrieval.fusion").fuse_rrf
    candidates = HybridCandidates(
        dense=[_dense_hit("same", document_id="document-a")],
        sparse=[_sparse_hit("same", document_id="document-b")],
    )

    with pytest.raises(ValueError, match="conflicting document IDs"):
        fuse_rrf(candidates)


@pytest.mark.parametrize(
    ("rrf_k", "top_k"),
    [(0, None), (True, None), (1.5, None), (60, 0), (60, True), (60, 1.5)],
    ids=[
        "zero-rrf-k",
        "boolean-rrf-k",
        "float-rrf-k",
        "zero-top-k",
        "bool-top-k",
        "float-top-k",
    ],
)
def test_rrf_rejects_invalid_limits(rrf_k: int, top_k: int | None) -> None:
    fuse_rrf = import_module("src.retrieval.fusion").fuse_rrf

    with pytest.raises(ValueError):
        fuse_rrf(HybridCandidates(dense=[], sparse=[]), rrf_k=rrf_k, top_k=top_k)


@pytest.mark.integration
@pytest.mark.database
def test_hybrid_to_rrf_end_to_end_is_deduplicated_and_board_scoped(
    retrieval_database,
) -> None:
    from src.retrieval.fusion import fuse_rrf
    from src.retrieval.hybrid import search_hybrid_candidates

    storage, seed_document = retrieval_database
    board_a = str(UUID(_BOARD_A))
    board_b = "22222222-2222-2222-2222-222222222222"
    _, board_a_chunk_id = seed_document(
        board_a,
        "READY",
        "rrf-integration",
        _QUERY_VECTOR,
        content="hybrid retrieval integration source",
    )
    seed_document(
        board_b,
        "READY",
        "rrf-cross-board-decoy",
        _QUERY_VECTOR,
        content="hybrid retrieval integration decoy",
    )
    embedder = MagicMock()
    embedder.model_name = DEFAULT_MODEL_ID
    embedder.revision = DEFAULT_MODEL_REVISION
    embedder.dimension = DEFAULT_DENSE_DIMENSION
    embedder.embed_texts.return_value = [_QUERY_VECTOR]

    candidates = search_hybrid_candidates(
        "hybrid retrieval integration",
        board_a,
        candidate_k=5,
        embedder=embedder,
        storage=storage,
    )
    fused = fuse_rrf(candidates)

    assert len(fused) == 1
    assert fused[0].source.chunk_id == board_a_chunk_id
    assert fused[0].source.board_id == board_a
    assert fused[0].channels == ("dense", "sparse")
    assert fused[0].rrf_score == pytest.approx(2 / 61)
