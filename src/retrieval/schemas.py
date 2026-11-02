"""Result types returned by retrieval operations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    """A ranked chunk with enough source data to locate its document."""

    chunk_id: str
    document_id: str
    board_id: str
    chunk_index: int
    file_name: str
    storage_key: str
    content: str
    page_number: int | None
    start_offset: int | None
    end_offset: int | None
    cosine_distance: float


@dataclass(frozen=True)
class SourceChunk:
    """Database-backed source fields shared by dense and lexical candidates."""

    board_id: str
    chunk_id: str
    document_id: str
    chunk_index: int
    file_name: str
    storage_key: str
    content: str
    page_number: int | None
    start_offset: int | None
    end_offset: int | None

    @classmethod
    def from_dense(cls, hit: RetrievedChunk) -> SourceChunk:
        """Project a dense hit to its shared source record, omitting distance."""
        return cls(
            board_id=hit.board_id,
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            chunk_index=hit.chunk_index,
            file_name=hit.file_name,
            storage_key=hit.storage_key,
            content=hit.content,
            page_number=hit.page_number,
            start_offset=hit.start_offset,
            end_offset=hit.end_offset,
        )


@dataclass(frozen=True)
class SparseHit:
    """A lexical match and its source, with no synthetic vector score."""

    source: SourceChunk
    sparse_score: float


@dataclass(frozen=True)
class HybridCandidates:
    """Ranked dense and sparse candidates for a single question and Board."""

    dense: list[RetrievedChunk]
    sparse: list[SparseHit]
