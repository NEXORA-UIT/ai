"""Unit tests for EmbeddedChunk and EmbeddedDocument schemas."""

import pytest

from src.ingestion.schemas import Chunk, EmbeddedChunk, EmbeddedDocument


def _make_sample_chunk(index: int = 0, content: str = "Test chunk content") -> Chunk:
    return Chunk(
        chunk_index=index,
        content=content,
        token_count=5,
        start_offset=0,
        end_offset=len(content),
        page_number=1,
    )


def test_embedded_chunk_valid():
    chunk = _make_sample_chunk()
    vector = [0.1] * 1024
    embedded = EmbeddedChunk(chunk=chunk, dense_embedding=vector)
    assert embedded.chunk == chunk
    assert len(embedded.dense_embedding) == 1024
    assert embedded.dense_embedding[0] == 0.1


def test_embedded_chunk_invalid_dimension():
    chunk = _make_sample_chunk()
    vector = [0.1] * 512
    with pytest.raises(ValueError, match="dimension must be 1024"):
        EmbeddedChunk(chunk=chunk, dense_embedding=vector)


def test_embedded_document_creation():
    chunk = _make_sample_chunk()
    embedded_chunk = EmbeddedChunk(chunk=chunk, dense_embedding=[0.0] * 1024)
    doc = EmbeddedDocument(
        document_name="test.txt",
        total_chunks=1,
        total_tokens=5,
        chunks=[embedded_chunk],
        model_name="BAAI/bge-m3",
        model_revision="5617a9f61b028005a4858fdac845db406aefb181",
        dimension=1024,
        normalized=True,
    )
    assert doc.document_name == "test.txt"
    assert doc.total_chunks == 1
    assert doc.total_tokens == 5
    assert len(doc.chunks) == 1
    assert doc.model_name == "BAAI/bge-m3"
    assert doc.dimension == 1024
    assert doc.normalized is True
