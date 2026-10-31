"""Unit tests for TikaClient with mocked HTTP responses."""

from unittest.mock import MagicMock

import httpx
import pytest

from src.ingestion.exceptions import (
    TikaConnectionError,
    TikaExtractionError,
    TikaTimeoutError,
)
from src.ingestion.tika_client import TikaClient


class TestTikaClient:
    def test_successful_rmeta_extraction(self, monkeypatch):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "X-TIKA:content": "Sample extracted text.\n\x0cPage 2 text.",
                "Content-Type": "application/pdf",
                "xmpTPg:NPages": "2",
                "dc:title": "Test Title",
            }
        ]

        sent_headers = {}

        def mock_put(url, content, headers):
            nonlocal sent_headers
            sent_headers = headers
            return mock_response

        monkeypatch.setattr(
            httpx.Client, "put", lambda self, url, **kwargs: mock_put(url, **kwargs)
        )

        client = TikaClient(endpoint="http://tika-test:9998")
        result = client.extract_bytes(
            b"%PDF-1.7...", file_name="doc.pdf", mime_type="application/pdf"
        )

        assert result.file_name == "doc.pdf"
        assert result.mime_type == "application/pdf"
        assert "Sample extracted text" in result.raw_text
        assert result.page_count == 2
        assert result.has_form_feeds is True
        assert result.metadata["dc:title"] == "Test Title"

        # Verify OCR was explicitly disabled in request header
        assert sent_headers.get("X-Tika-PDFOcrStrategy") == "no_ocr"

    def test_connection_error_raises_tika_connection_error(self, monkeypatch):
        def mock_put(self, url, **kwargs):
            raise httpx.ConnectError("Failed to connect")

        monkeypatch.setattr(httpx.Client, "put", mock_put)

        client = TikaClient(endpoint="http://127.0.0.1:9998")
        with pytest.raises(TikaConnectionError, match="Cannot connect to Apache Tika"):
            client.extract_bytes(b"content", file_name="test.txt")

    def test_timeout_raises_tika_timeout_error(self, monkeypatch):
        def mock_put(self, url, **kwargs):
            raise httpx.TimeoutException("Timed out")

        monkeypatch.setattr(httpx.Client, "put", mock_put)

        client = TikaClient(endpoint="http://127.0.0.1:9998", timeout=5.0)
        with pytest.raises(TikaTimeoutError, match="timed out after 5.0s"):
            client.extract_bytes(b"content", file_name="slow.pdf")

    def test_http_422_raises_tika_extraction_error(self, monkeypatch):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 422
        mock_response.text = "Unprocessable Entity: Corrupt file"

        monkeypatch.setattr(
            httpx.Client, "put", lambda self, url, **kwargs: mock_response
        )

        client = TikaClient(endpoint="http://127.0.0.1:9998")
        with pytest.raises(TikaExtractionError, match="HTTP 422"):
            client.extract_bytes(b"corrupt", file_name="corrupt.pdf")

    def test_invalid_json_raises_tika_extraction_error(self, monkeypatch):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("No JSON object")

        monkeypatch.setattr(
            httpx.Client, "put", lambda self, url, **kwargs: mock_response
        )

        client = TikaClient(endpoint="http://127.0.0.1:9998")
        with pytest.raises(TikaExtractionError, match="Invalid JSON response"):
            client.extract_bytes(b"content", file_name="bad_json.pdf")

    def test_empty_json_list_raises_tika_extraction_error(self, monkeypatch):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = []

        monkeypatch.setattr(
            httpx.Client, "put", lambda self, url, **kwargs: mock_response
        )

        client = TikaClient(endpoint="http://127.0.0.1:9998")
        with pytest.raises(TikaExtractionError, match="unexpected response format"):
            client.extract_bytes(b"content", file_name="empty_meta.pdf")

    def test_fallback_when_rmeta_returns_404(self, monkeypatch):
        rmeta_resp = MagicMock(spec=httpx.Response)
        rmeta_resp.status_code = 404

        tika_text_resp = MagicMock(spec=httpx.Response)
        tika_text_resp.status_code = 200
        tika_text_resp.text = "Fallback extracted text from /tika"

        meta_resp = MagicMock(spec=httpx.Response)
        meta_resp.status_code = 200
        meta_resp.json.return_value = {"Content-Type": "text/plain", "Page-Count": "1"}

        def mock_put(self, url, **kwargs):
            if "/rmeta/text" in url:
                return rmeta_resp
            if "/tika" in url:
                return tika_text_resp
            if "/meta" in url:
                return meta_resp
            raise ValueError(f"Unexpected url: {url}")

        monkeypatch.setattr(httpx.Client, "put", mock_put)

        client = TikaClient(endpoint="http://127.0.0.1:9998")
        result = client.extract_bytes(
            b"text content", file_name="fallback.txt", mime_type="text/plain"
        )

        assert result.raw_text == "Fallback extracted text from /tika"
        assert result.page_count == 1
        assert result.mime_type == "text/plain"
