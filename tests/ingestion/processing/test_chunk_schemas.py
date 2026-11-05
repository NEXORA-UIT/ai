"""Unit tests for Chunk and ChunkedDocument schemas."""

import pytest

from src.ingestion.schemas import Chunk, ChunkedDocument


def test_chunk_attributes_and_post_init_validation():
    chunk = Chunk(
        chunk_index=0,
        content="Nội dung kiểm thử.",
        token_count=5,
        start_offset=0,
        end_offset=18,
        page_number=1,
        metadata={"file_name": "test.pdf"},
    )
    assert chunk.chunk_index == 0
    assert chunk.char_count == 18
    assert chunk.page_number == 1
    assert chunk.token_count == 5
    assert chunk.start_offset == 0
    assert chunk.end_offset == 18


def test_chunk_allows_none_page_number():
    chunk = Chunk(
        chunk_index=1,
        content="Đoạn văn không phân trang.",
        token_count=6,
        start_offset=19,
        end_offset=45,
        page_number=None,
    )
    assert chunk.page_number is None


def test_chunk_rejects_negative_index():
    with pytest.raises(ValueError, match="chunk_index must be >= 0"):
        Chunk(
            chunk_index=-1,
            content="abc",
            token_count=1,
            start_offset=0,
            end_offset=3,
        )


def test_chunk_rejects_negative_start_offset():
    with pytest.raises(ValueError, match="start_offset must be >= 0"):
        Chunk(
            chunk_index=0,
            content="abc",
            token_count=1,
            start_offset=-1,
            end_offset=3,
        )


def test_chunk_rejects_inverted_offsets():
    with pytest.raises(ValueError, match="end_offset"):
        Chunk(
            chunk_index=0,
            content="abc",
            token_count=1,
            start_offset=10,
            end_offset=5,
        )


def test_chunked_document_attributes():
    chunk0 = Chunk(
        chunk_index=0,
        content="Đoạn 1",
        token_count=2,
        start_offset=0,
        end_offset=6,
    )
    chunk1 = Chunk(
        chunk_index=1,
        content="Đoạn 2",
        token_count=2,
        start_offset=8,
        end_offset=14,
    )
    doc = ChunkedDocument(
        document_name="test.txt",
        total_chunks=2,
        total_tokens=4,
        chunks=[chunk0, chunk1],
        chunking_version="1.0.0",
        config={"chunk_size": 512, "chunk_overlap": 64},
    )
    assert doc.document_name == "test.txt"
    assert doc.total_chunks == 2
    assert doc.total_tokens == 4
    assert len(doc.chunks) == 2
    assert doc.chunking_version == "1.0.0"
