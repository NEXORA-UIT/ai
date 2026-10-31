"""Unit tests for recursive text chunker."""

import pytest

from src.ingestion.chunker import chunk_document
from src.ingestion.schemas import CleanedDocument, PageSpan
from src.ingestion.tokenizer import WhitespaceTokenCounter


class TestChunker:
    def test_short_document_produces_single_chunk(self):
        doc = CleanedDocument(
            cleaned_text="Đoạn văn ngắn gọn kiểm thử.",
            file_name="test.txt",
            mime_type="text/plain",
        )
        counter = WhitespaceTokenCounter()
        result = chunk_document(
            doc, token_counter=counter, chunk_size=512, chunk_overlap=64
        )

        assert result.total_chunks == 1
        chunk = result.chunks[0]
        assert chunk.chunk_index == 0
        assert chunk.content == "Đoạn văn ngắn gọn kiểm thử."
        assert chunk.start_offset == 0
        assert chunk.end_offset == len("Đoạn văn ngắn gọn kiểm thử.")
        assert chunk.page_number is None
        assert chunk.token_count == 6

    def test_empty_document_produces_empty_chunks(self):
        doc = CleanedDocument(
            cleaned_text="",
            file_name="empty.txt",
            mime_type="text/plain",
        )
        counter = WhitespaceTokenCounter()
        result = chunk_document(doc, token_counter=counter)
        assert result.total_chunks == 0
        assert result.total_tokens == 0
        assert result.chunks == []

    def test_invalid_chunk_parameters_raise_value_error(self):
        doc = CleanedDocument(
            cleaned_text="Sample text",
            file_name="test.txt",
            mime_type="text/plain",
        )
        counter = WhitespaceTokenCounter()
        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            chunk_document(doc, token_counter=counter, chunk_size=0)

        with pytest.raises(ValueError, match="chunk_overlap must be >= 0"):
            chunk_document(doc, token_counter=counter, chunk_size=10, chunk_overlap=-1)

        with pytest.raises(ValueError, match=r"chunk_overlap .* must be < chunk_size"):
            chunk_document(doc, token_counter=counter, chunk_size=10, chunk_overlap=10)

    def test_recursive_split_along_paragraph_boundaries(self):
        p1 = "Đoạn 1 có bốn từ"
        p2 = "Đoạn 2 có bốn từ"
        p3 = "Đoạn 3 có bốn từ"
        text = f"{p1}\n\n{p2}\n\n{p3}"
        doc = CleanedDocument(
            cleaned_text=text,
            file_name="test.txt",
            mime_type="text/plain",
        )
        counter = WhitespaceTokenCounter()
        # Each paragraph has 5 words; chunk_size=6 fits exactly 1 paragraph
        result = chunk_document(
            doc, token_counter=counter, chunk_size=6, chunk_overlap=0
        )

        assert result.total_chunks == 3
        assert result.chunks[0].content == p1
        assert result.chunks[1].content == p2
        assert result.chunks[2].content == p3

    def test_multipage_pdf_resolves_page_numbers(self):
        text = "Trang 1: Tong quan he thong.\n\nTrang 2: Kien truc RAG."
        p2_pos = text.index("Trang 2")
        spans = [
            PageSpan(page_number=1, start_offset=0, end_offset=p2_pos),
            PageSpan(page_number=2, start_offset=p2_pos, end_offset=len(text)),
        ]
        doc = CleanedDocument(
            cleaned_text=text,
            file_name="multi.pdf",
            mime_type="application/pdf",
            page_spans=spans,
        )
        counter = WhitespaceTokenCounter()
        result = chunk_document(
            doc, token_counter=counter, chunk_size=6, chunk_overlap=0
        )

        assert result.total_chunks == 2
        assert result.chunks[0].page_number == 1
        assert result.chunks[1].page_number == 2

    def test_unpaginated_doc_leaves_page_number_none(self):
        text = "Phần 1 tài liệu Word.\n\nPhần 2 tài liệu Word."
        doc = CleanedDocument(
            cleaned_text=text,
            file_name="word.docx",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            page_spans=None,
        )
        counter = WhitespaceTokenCounter()
        result = chunk_document(
            doc, token_counter=counter, chunk_size=6, chunk_overlap=0
        )

        for chunk in result.chunks:
            assert chunk.page_number is None

    def test_offset_slicing_matches_content_exactly(self):
        text = (
            "Mục 1: Trích xuất bằng Apache Tika.\n\n"
            "Mục 2: Làm sạch văn bản chuẩn hóa NFC.\n\n"
            "Mục 3: Cắt đoạn văn bản theo tokenizer BGE-M3."
        )
        doc = CleanedDocument(
            cleaned_text=text,
            file_name="doc.txt",
            mime_type="text/plain",
        )
        counter = WhitespaceTokenCounter()
        result = chunk_document(
            doc, token_counter=counter, chunk_size=7, chunk_overlap=2
        )

        assert result.total_chunks > 1
        for chunk in result.chunks:
            # Crucial invariant: slicing cleaned_text matches chunk content
            sliced = text[chunk.start_offset : chunk.end_offset]
            assert sliced == chunk.content

    def test_overlap_preserves_boundary_context(self):
        text = (
            "Câu một của đoạn thứ nhất.\n\n"
            "Câu hai của đoạn thứ hai.\n\n"
            "Câu ba của đoạn thứ ba."
        )
        doc = CleanedDocument(
            cleaned_text=text,
            file_name="doc.txt",
            mime_type="text/plain",
        )
        counter = WhitespaceTokenCounter()
        # chunk_size=13 fits 2 paragraphs; overlap=6 retains 1 paragraph
        result = chunk_document(
            doc, token_counter=counter, chunk_size=13, chunk_overlap=6
        )

        assert result.total_chunks >= 2
        # Verify chunk 1 start_offset is strictly less than chunk 0 end_offset
        assert result.chunks[1].start_offset < result.chunks[0].end_offset

    def test_deterministic_chunking(self):
        text = (
            "Dữ liệu kiểm thử tính bất biến và xác định của chunker.\n\n"
            "Lần chạy thứ nhất và thứ hai phải sinh kết quả trùng khớp hoàn toàn."
        )
        doc = CleanedDocument(
            cleaned_text=text,
            file_name="det.txt",
            mime_type="text/plain",
        )
        counter = WhitespaceTokenCounter()
        res1 = chunk_document(doc, token_counter=counter, chunk_size=8, chunk_overlap=2)
        res2 = chunk_document(doc, token_counter=counter, chunk_size=8, chunk_overlap=2)

        assert res1.total_chunks == res2.total_chunks
        assert res1.total_tokens == res2.total_tokens
        for c1, c2 in zip(res1.chunks, res2.chunks, strict=True):
            assert c1.chunk_index == c2.chunk_index
            assert c1.content == c2.content
            assert c1.start_offset == c2.start_offset
            assert c1.end_offset == c2.end_offset
            assert c1.token_count == c2.token_count
