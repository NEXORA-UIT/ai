"""Integration tests for chunker with real BGE-M3 tokenizer."""

import pytest

from src.ingestion.chunker import chunk_document
from src.ingestion.schemas import CleanedDocument
from src.ingestion.tokenizer import BgeM3Tokenizer


@pytest.mark.integration
def test_bge_m3_chunking_vietnamese_text():
    text = (
        "# Hệ Thống Quản Lý Dự Án Nexora\n\n"
        "Nexora là nền tảng quản lý dự án thông minh tích hợp AI Agent.\n\n"
        "1. Quản lý Workspace và phân quyền thành viên theo vai trò.\n"
        "2. Kanban Board hỗ trợ đồng bộ theo thời gian thực với WebSocket.\n"
        "3. Trợ lý RAG tra cứu tri thức nội bộ từ các tài liệu số.\n"
        "4. Cơ chế Human-in-the-loop bảo đảm quyền kiểm soát của người dùng.\n\n"
        "Khi thực hiện các thao tác quan trọng, AI Agent sẽ yêu cầu phê duyệt."
    )
    doc = CleanedDocument(
        cleaned_text=text,
        file_name="nexora_intro.txt",
        mime_type="text/plain",
    )
    tokenizer = BgeM3Tokenizer()

    # Chunk with 50 tokens max, overlap 10 tokens
    chunked = chunk_document(
        cleaned_doc=doc,
        token_counter=tokenizer,
        chunk_size=50,
        chunk_overlap=10,
    )

    assert chunked.total_chunks >= 2
    for chunk in chunked.chunks:
        # Assert each chunk strictly respects max tokens limit
        assert chunk.token_count <= 50
        # Assert exact character offset slicing invariant
        sliced = text[chunk.start_offset : chunk.end_offset]
        assert sliced == chunk.content
        # Assert no Vietnamese diacritic corruption
        assert "Nexora" in text


@pytest.mark.integration
def test_bge_m3_chunking_long_vietnamese_document():
    # Construct a multi-paragraph document with ~300 tokens
    paragraphs = [
        f"Đoạn văn số {i}: Nexora cung cấp các tính năng hỗ trợ quản lý dự án, "
        f"cho phép lập kế hoạch và theo dõi tiến độ một cách khoa học."
        for i in range(1, 10)
    ]
    full_text = "\n\n".join(paragraphs)
    doc = CleanedDocument(
        cleaned_text=full_text,
        file_name="long_doc.txt",
        mime_type="text/plain",
    )
    tokenizer = BgeM3Tokenizer()

    chunked = chunk_document(
        cleaned_doc=doc,
        token_counter=tokenizer,
        chunk_size=60,
        chunk_overlap=15,
    )

    assert chunked.total_chunks > 1
    # Verify sequential indices starting at 0
    for idx, chunk in enumerate(chunked.chunks):
        assert chunk.chunk_index == idx
        assert chunk.token_count <= 60
        assert full_text[chunk.start_offset : chunk.end_offset] == chunk.content
