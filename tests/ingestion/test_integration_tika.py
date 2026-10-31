"""Integration tests with live Apache Tika Server.

Tests marked with @pytest.mark.integration require a running Apache Tika server.
If Tika is not accessible, tests are cleanly SKIPPED with an explicit message.
"""

import os

import httpx
import pytest

from src.ingestion.exceptions import EmptyDocumentError
from src.ingestion.pipeline import extract_and_clean
from tests.ingestion.fixtures import (
    create_sample_docx,
    create_sample_empty_text_pdf,
    create_sample_multipage_pdf,
    create_sample_txt,
)

TIKA_ENDPOINT = os.environ.get("TIKA_SERVER_ENDPOINT", "http://127.0.0.1:9998")


def is_tika_available() -> bool:
    """Check if Apache Tika server is responding at TIKA_ENDPOINT."""
    try:
        res = httpx.get(f"{TIKA_ENDPOINT.rstrip('/')}/tika", timeout=1.5)
        return res.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def require_tika():
    """Skip integration test suite if Apache Tika is not running."""
    if not is_tika_available():
        pytest.skip(
            f"Apache Tika server not available at {TIKA_ENDPOINT}. "
            "Start Tika server (e.g. java -jar tika-server.jar) to run live tests."
        )


@pytest.mark.integration
class TestTikaLiveIntegration:
    def test_extract_vietnamese_txt(self, require_tika, tmp_path):
        txt_path = tmp_path / "sample_vn.txt"
        create_sample_txt(txt_path)

        extraction, cleaned = extract_and_clean(txt_path, tika_endpoint=TIKA_ENDPOINT)

        assert "text/plain" in extraction.mime_type
        assert "Hệ Thống Quản Lý Dự Án Nexora" in cleaned.cleaned_text
        assert "Quản lý Workspace và Board Kanban" in cleaned.cleaned_text
        assert "Trợ lý RAG tra cứu tài liệu nội bộ" in cleaned.cleaned_text

        # Verify statements appear in correct order
        pos1 = cleaned.cleaned_text.index("Quản lý Workspace và Board Kanban")
        pos2 = cleaned.cleaned_text.index("Trợ lý RAG tra cứu tài liệu nội bộ")
        pos3 = cleaned.cleaned_text.index("Cơ chế Human-in-the-loop phê duyệt thay đổi")
        assert pos1 < pos2 < pos3

    def test_extract_docx_with_headings_and_lists(self, require_tika, tmp_path):
        docx_path = tmp_path / "sample.docx"
        create_sample_docx(docx_path)

        extraction, cleaned = extract_and_clean(docx_path, tika_endpoint=TIKA_ENDPOINT)

        assert (
            "wordprocessingml.document" in extraction.mime_type
            or "zip" in extraction.mime_type
        )
        assert "Kế hoạch triển khai Ingestion" in cleaned.cleaned_text
        assert "Mục 1: Trích xuất bằng Apache Tika" in cleaned.cleaned_text
        assert "Mục 2: Làm sạch văn bản Unicode NFC" in cleaned.cleaned_text

    def test_extract_multipage_pdf_and_page_spans(self, require_tika, tmp_path):
        pdf_path = tmp_path / "multipage.pdf"
        create_sample_multipage_pdf(pdf_path)

        extraction, cleaned = extract_and_clean(pdf_path, tika_endpoint=TIKA_ENDPOINT)

        assert "application/pdf" in extraction.mime_type
        assert "Trang 1: Tong quan he thong Nexora" in cleaned.cleaned_text
        assert "Trang 2: Kien truc AI Ingestion pipeline" in cleaned.cleaned_text

        # Verify sequential ordering
        pos_p1 = cleaned.cleaned_text.index("Trang 1")
        pos_p2 = cleaned.cleaned_text.index("Trang 2")
        assert pos_p1 < pos_p2

        # If Tika preserved form feeds, verify accurate page spans
        if cleaned.page_spans:
            assert len(cleaned.page_spans) == 2
            span1 = cleaned.page_spans[0]
            span2 = cleaned.page_spans[1]

            assert span1.page_number == 1
            assert span2.page_number == 2
            p1_text = cleaned.cleaned_text[span1.start_offset : span1.end_offset]
            p2_text = cleaned.cleaned_text[span2.start_offset : span2.end_offset]
            assert "Trang 1" in p1_text
            assert "Trang 2" in p2_text

    def test_non_text_pdf_raises_empty_document_error_without_ocr(
        self, require_tika, tmp_path
    ):
        empty_pdf = tmp_path / "blank.pdf"
        create_sample_empty_text_pdf(empty_pdf)

        with pytest.raises(EmptyDocumentError):
            extract_and_clean(empty_pdf, tika_endpoint=TIKA_ENDPOINT)
