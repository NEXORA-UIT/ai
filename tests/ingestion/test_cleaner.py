"""Unit tests for text cleaning and page span calculation."""

import unicodedata

import pytest

from src.ingestion.cleaner import clean_document, clean_text, clean_text_fragment
from src.ingestion.exceptions import EmptyDocumentError
from src.ingestion.schemas import ExtractionResult, PageSpan


class TestCleanTextFragment:
    def test_unicode_nfc_normalization_vietnamese(self):
        # NFD form of 'tiếng Việt' has decomposed tone marks
        decomposed = unicodedata.normalize("NFD", "Tiếng Việt có dấu: ế, ồ, ừ, đ, ắ, ặ")
        assert unicodedata.is_normalized("NFD", decomposed)

        cleaned = clean_text_fragment(decomposed)
        assert unicodedata.is_normalized("NFC", cleaned)
        assert cleaned == "Tiếng Việt có dấu: ế, ồ, ừ, đ, ắ, ặ"

    def test_normalize_line_endings_crlf_and_cr(self):
        text = "Line 1\r\nLine 2\rLine 3\nLine 4"
        cleaned = clean_text_fragment(text)
        assert "\r" not in cleaned
        assert cleaned == "Line 1\nLine 2\nLine 3\nLine 4"

    def test_strip_nul_and_control_characters(self):
        # \x00 NUL, \x07 BELL, \x08 BACKSPACE, \x1b ESCAPE
        text = "Hello\x00 World\x07! Tab\tpreserved, newline\nkept.\x1bDone"
        cleaned = clean_text_fragment(text)
        assert "\x00" not in cleaned
        assert "\x07" not in cleaned
        assert "\x1b" not in cleaned
        assert cleaned == "Hello World! Tab\tpreserved, newline\nkept.Done"

    def test_strip_trailing_whitespace_per_line(self):
        text = "Heading   \t\nLine 1    \nLine 2\t\t\nLine 3"
        cleaned = clean_text_fragment(text)
        assert cleaned == "Heading\nLine 1\nLine 2\nLine 3"

    def test_condense_excessive_newlines(self):
        # 3 or more newlines become double newlines (paragraphs preserved)
        text = "Paragraph 1\n\n\n\n\nParagraph 2\n\nParagraph 3"
        cleaned = clean_text_fragment(text)
        assert cleaned == "Paragraph 1\n\nParagraph 2\n\nParagraph 3"

    def test_preserves_headings_and_lists(self):
        text = (
            "# Tiêu đề 1\n\n"
            "Mô tả dự án:\n"
            "- Mục 1: Khởi tạo dữ liệu\n"
            "- Mục 2: Xử lý Tika\n"
            "  * Nhánh con A\n"
            "  * Nhánh con B\n"
        )
        cleaned = clean_text_fragment(text)
        assert cleaned == text.strip()

    def test_idempotence(self):
        samples = [
            "Đoạn văn tiếng Việt có dấu.   \r\n\r\nDòng thứ hai.\t\t",
            "Special chars: \x00\x01\x02 test\n\n\n\nMultiple\nLines",
            "# Heading\n* Item 1\n* Item 2",
        ]
        for sample in samples:
            once = clean_text_fragment(sample)
            twice = clean_text_fragment(once)
            assert once == twice, f"Failed idempotence on sample: {sample!r}"


class TestCleanText:
    def test_empty_input_raises_empty_document_error(self):
        with pytest.raises(EmptyDocumentError, match="contains no readable text"):
            clean_text("")

        with pytest.raises(EmptyDocumentError, match="contains no readable text"):
            clean_text("   \n\t  \r\n  ")

    def test_control_chars_only_raises_empty_document_error(self):
        with pytest.raises(EmptyDocumentError, match="contains no readable text"):
            clean_text("\x00\x01\x07\x08")


class TestCleanDocument:
    def test_multi_page_pdf_with_form_feeds(self):
        raw_text = (
            "Trang 1: Giới thiệu hệ thống Nexora.  \r\n"
            "Các thành phần chính.\r\n"
            "\x0c"
            "Trang 2: Kiến trúc AI Ingestion.\n"
            "- Tika extraction\n"
            "- Text cleaning\n"
            "\x0c"
            "Trang 3: Kết luận và kế hoạch tiếp theo.\n"
        )

        extraction = ExtractionResult(
            raw_text=raw_text,
            file_name="nexora_arch.pdf",
            mime_type="application/pdf",
            metadata={"xmpTPg:NPages": "3"},
            page_count=3,
            has_form_feeds=True,
        )

        cleaned = clean_document(extraction)

        assert cleaned.file_name == "nexora_arch.pdf"
        assert cleaned.mime_type == "application/pdf"
        assert cleaned.cleaning_version == "1.0.0"
        assert cleaned.page_spans is not None
        assert len(cleaned.page_spans) == 3

        # Verify page span offsets match actual text slices
        span1 = cleaned.page_spans[0]
        span2 = cleaned.page_spans[1]
        span3 = cleaned.page_spans[2]

        assert span1.page_number == 1
        page1_slice = cleaned.cleaned_text[span1.start_offset : span1.end_offset]
        assert "Trang 1: Giới thiệu hệ thống Nexora." in page1_slice
        assert "Các thành phần chính." in page1_slice

        assert span2.page_number == 2
        page2_slice = cleaned.cleaned_text[span2.start_offset : span2.end_offset]
        assert "Trang 2: Kiến trúc AI Ingestion." in page2_slice
        assert "- Tika extraction" in page2_slice

        assert span3.page_number == 3
        page3_slice = cleaned.cleaned_text[span3.start_offset : span3.end_offset]
        assert "Trang 3: Kết luận và kế hoạch tiếp theo." in page3_slice

        # Verify no overlapping spans
        assert span1.end_offset < span2.start_offset
        assert span2.end_offset < span3.start_offset

    def test_document_without_form_feeds_leaves_page_spans_none(self):
        raw_text = "Nội dung tài liệu DOCX hoặc TXT không có ranh giới trang."
        extraction = ExtractionResult(
            raw_text=raw_text,
            file_name="notes.txt",
            mime_type="text/plain",
            metadata={},
            page_count=None,
            has_form_feeds=False,
        )

        cleaned = clean_document(extraction)
        assert cleaned.page_spans is None
        assert cleaned.cleaned_text == raw_text

    def test_single_page_pdf_without_form_feeds_sets_single_span(self):
        raw_text = "Tài liệu PDF 1 trang duy nhất."
        extraction = ExtractionResult(
            raw_text=raw_text,
            file_name="onepage.pdf",
            mime_type="application/pdf",
            metadata={"xmpTPg:NPages": "1"},
            page_count=1,
            has_form_feeds=False,
        )

        cleaned = clean_document(extraction)
        assert cleaned.page_spans == [
            PageSpan(
                page_number=1, start_offset=0, end_offset=len(cleaned.cleaned_text)
            )
        ]

    def test_clean_document_rejects_empty_pages(self):
        raw_text = "   \x0c   \x0c   "
        extraction = ExtractionResult(
            raw_text=raw_text,
            file_name="empty.pdf",
            mime_type="application/pdf",
            metadata={},
            page_count=2,
            has_form_feeds=True,
        )

        with pytest.raises(EmptyDocumentError):
            clean_document(extraction)
