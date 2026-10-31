"""Result types returned by retrieval operations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    """A ranked chunk with enough source data to locate its document."""

    chunk_id: str
    document_id: str
    chunk_index: int
    file_name: str
    storage_key: str
    content: str
    page_number: int | None
    start_offset: int | None
    end_offset: int | None
    cosine_distance: float
