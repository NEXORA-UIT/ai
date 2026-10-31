"""HTTP client for Apache Tika Server extraction."""

import os
from pathlib import Path
from typing import Any

import httpx

from src.ingestion.exceptions import (
    TikaConnectionError,
    TikaExtractionError,
    TikaTimeoutError,
)
from src.ingestion.file_validator import DEFAULT_MAX_FILE_SIZE_BYTES, validate_file
from src.ingestion.schemas import ExtractionResult

DEFAULT_TIKA_ENDPOINT = "http://127.0.0.1:9998"
DEFAULT_TIMEOUT_SECONDS = 30.0


class TikaClient:
    """Client for Apache Tika Server REST API.

    Extracts plain text and metadata without OCR.

    Attributes:
        endpoint: Base URL of Tika server (e.g. http://127.0.0.1:9998).
        timeout: HTTP request timeout in seconds.
        max_file_size_bytes: Maximum allowed file size for input validation.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    ) -> None:
        self.endpoint = (
            endpoint or os.environ.get("TIKA_SERVER_ENDPOINT") or DEFAULT_TIKA_ENDPOINT
        ).rstrip("/")
        self.timeout = timeout
        self.max_file_size_bytes = max_file_size_bytes

    def extract_file(self, file_path: Path | str) -> ExtractionResult:
        """Validate and extract text and metadata from a local file.

        Args:
            file_path: Local file path (.txt, .pdf, .docx).

        Returns:
            ExtractionResult containing raw text, metadata, and page indicators.

        Raises:
            FileValidationError: If file fails validation checks.
            TikaConnectionError: If connection to Tika server fails.
            TikaTimeoutError: If Tika request times out.
            TikaExtractionError: If Tika returns an error status or invalid output.
        """
        path, mime_type, file_size = validate_file(
            file_path, max_size_bytes=self.max_file_size_bytes
        )

        with open(path, "rb") as f:
            content = f.read()

        return self.extract_bytes(
            content=content,
            file_name=path.name,
            mime_type=mime_type,
        )

    def extract_bytes(
        self,
        content: bytes,
        file_name: str,
        mime_type: str = "application/octet-stream",
    ) -> ExtractionResult:
        """Send document bytes to Apache Tika for extraction.

        Uses the recursive metadata endpoint `/rmeta/text` with explicit
        non-OCR strategy (`X-Tika-PDFOcrStrategy: no_ocr`). If `/rmeta/text`
        is not supported, falls back to `/tika` and `/meta`.

        Args:
            content: Document file bytes.
            file_name: Original file name.
            mime_type: Content type of the file.

        Returns:
            ExtractionResult.
        """
        headers = {
            "Accept": "application/json",
            "Content-Type": mime_type,
            # Explicitly disable OCR in Tika PDFParser
            "X-Tika-PDFOcrStrategy": "no_ocr",
        }

        url = f"{self.endpoint}/rmeta/text"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.put(url, content=content, headers=headers)

                # Fallback to /tika and /meta if /rmeta/text returns 404
                if res.status_code == 404:
                    return self._extract_fallback(client, content, file_name, mime_type)

                if res.status_code != 200:
                    raise TikaExtractionError(
                        f"Tika extraction failed for '{file_name}' with HTTP "
                        f"{res.status_code}: {res.text[:200]}"
                    )

                try:
                    data = res.json()
                except Exception as e:
                    raise TikaExtractionError(
                        f"Invalid JSON response from Tika for '{file_name}': {e}"
                    ) from e

        except httpx.ConnectError as e:
            raise TikaConnectionError(
                f"Cannot connect to Apache Tika server at {self.endpoint}. "
                "Please verify that Tika is running."
            ) from e
        except httpx.TimeoutException as e:
            raise TikaTimeoutError(
                f"Request to Apache Tika at {self.endpoint} timed out after "
                f"{self.timeout}s for file '{file_name}'."
            ) from e
        except httpx.RequestError as e:
            raise TikaConnectionError(
                f"Network error communicating with Apache Tika: {e}"
            ) from e

        return self._parse_rmeta_response(data, file_name, mime_type)

    def _extract_fallback(
        self,
        client: httpx.Client,
        content: bytes,
        file_name: str,
        mime_type: str,
    ) -> ExtractionResult:
        """Fallback extraction using separate /tika and /meta endpoints."""
        text_url = f"{self.endpoint}/tika"
        meta_url = f"{self.endpoint}/meta"

        text_headers = {
            "Accept": "text/plain",
            "Content-Type": mime_type,
            "X-Tika-PDFOcrStrategy": "no_ocr",
        }
        meta_headers = {
            "Accept": "application/json",
            "Content-Type": mime_type,
        }

        res_text = client.put(text_url, content=content, headers=text_headers)
        if res_text.status_code != 200:
            raise TikaExtractionError(
                f"Tika text extraction failed for '{file_name}' with HTTP "
                f"{res_text.status_code}: {res_text.text[:200]}"
            )
        raw_text = res_text.text

        metadata: dict[str, Any] = {}
        res_meta = client.put(meta_url, content=content, headers=meta_headers)
        if res_meta.status_code == 200:
            try:
                meta_json = res_meta.json()
                if isinstance(meta_json, dict):
                    metadata = meta_json
            except Exception:
                pass

        page_count = self._extract_page_count(metadata)
        has_form_feeds = "\x0c" in raw_text

        return ExtractionResult(
            raw_text=raw_text,
            file_name=file_name,
            mime_type=mime_type,
            metadata=metadata,
            page_count=page_count,
            has_form_feeds=has_form_feeds,
        )

    def _parse_rmeta_response(
        self,
        data: Any,
        file_name: str,
        fallback_mime: str,
    ) -> ExtractionResult:
        """Parse the /rmeta/text JSON array response."""
        if not isinstance(data, list) or len(data) == 0:
            raise TikaExtractionError(
                f"Tika returned unexpected response format for '{file_name}' "
                f"(expected non-empty list, got {type(data).__name__})."
            )

        container = data[0]
        if not isinstance(container, dict):
            raise TikaExtractionError(
                f"Tika container entry is not a dict: {type(container).__name__}"
            )

        raw_text = container.get("X-TIKA:content", "")
        if not isinstance(raw_text, str):
            raw_text = str(raw_text)

        metadata = {k: v for k, v in container.items() if k != "X-TIKA:content"}
        mime_type = metadata.get("Content-Type") or fallback_mime
        if isinstance(mime_type, list) and mime_type:
            mime_type = str(mime_type[0])

        page_count = self._extract_page_count(metadata)
        has_form_feeds = "\x0c" in raw_text

        return ExtractionResult(
            raw_text=raw_text,
            file_name=file_name,
            mime_type=str(mime_type),
            metadata=metadata,
            page_count=page_count,
            has_form_feeds=has_form_feeds,
        )

    @staticmethod
    def _extract_page_count(metadata: dict[str, Any]) -> int | None:
        """Extract page count integer from standard metadata keys."""
        candidates = [
            metadata.get("xmpTPg:NPages"),
            metadata.get("pdf:docinfo:pages"),
            metadata.get("Page-Count"),
        ]
        for val in candidates:
            if val is not None:
                try:
                    num = int(val)
                    if num > 0:
                        return num
                except (ValueError, TypeError):
                    continue
        return None
