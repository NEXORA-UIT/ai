"""Minimal, safe, and idempotent text cleaning and page span calculation."""

import re
import unicodedata

from src.ingestion.exceptions import EmptyDocumentError
from src.ingestion.schemas import CleanedDocument, ExtractionResult, PageSpan

CLEANING_VERSION = "1.0.0"


def clean_text_fragment(text: str) -> str:
    """Normalize and clean a fragment of text.

    Applies:
    1. Unicode NFC normalization (preserves Vietnamese accents).
    2. Conversion of CRLF and CR to LF.
    3. Removal of NUL byte and non-printable control characters (preserves \\t, \\n).
    4. Stripping trailing whitespace per line.
    5. Condensing 3 or more consecutive newlines to 2 (preserves paragraphs).
    6. Stripping leading/trailing whitespace of the fragment.

    Args:
        text: Input string.

    Returns:
        Cleaned, normalized string.
    """
    if not text:
        return ""

    # 1. Unicode NFC normalization
    normalized = unicodedata.normalize("NFC", text)

    # 2. Normalize line endings to standard LF
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Strip NUL and non-printable control characters, preserving \t and \n
    cleaned_chars = []
    for ch in normalized:
        if ch in ("\t", "\n"):
            cleaned_chars.append(ch)
        elif ord(ch) >= 32 and unicodedata.category(ch) != "Cc":
            cleaned_chars.append(ch)
    normalized = "".join(cleaned_chars)

    # 4. Strip trailing whitespace per line
    lines = [line.rstrip(" \t") for line in normalized.split("\n")]
    normalized = "\n".join(lines)

    # 5. Condense 3+ consecutive newlines to double newline
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    # 6. Strip leading and trailing whitespace
    return normalized.strip()


def clean_text(raw_text: str) -> str:
    """Clean full document text and enforce non-empty constraint.

    Args:
        raw_text: Uncleaned raw text.

    Returns:
        Cleaned text.

    Raises:
        EmptyDocumentError: If text is empty or contains only whitespace.
    """
    if not raw_text or raw_text.isspace():
        raise EmptyDocumentError("Document contains no readable text.")

    cleaned = clean_text_fragment(raw_text)
    if not cleaned:
        raise EmptyDocumentError("Document contains no readable text after cleaning.")

    return cleaned


def clean_document(extraction: ExtractionResult) -> CleanedDocument:
    """Clean an extraction result and calculate page spans if page markers exist.

    If the extraction contains Form Feed (\\x0c) page boundaries (digital PDFs),
    each page is cleaned individually and assembled with verified character offsets.
    If no reliable page markers exist (e.g. TXT, DOCX), page_spans is set to None.

    Args:
        extraction: Raw ExtractionResult from Tika.

    Returns:
        CleanedDocument ready for chunking.

    Raises:
        EmptyDocumentError: If document contains no usable text after cleaning.
    """
    raw_text = extraction.raw_text

    if not raw_text or raw_text.isspace():
        raise EmptyDocumentError(
            f"Document '{extraction.file_name}' contains no readable text."
        )

    # Multi-page PDF extraction with Form Feed page separators
    if "\x0c" in raw_text:
        raw_pages = raw_text.split("\x0c")
        cleaned_page_texts: list[str] = []

        for raw_page in raw_pages:
            cleaned_p = clean_text_fragment(raw_page)
            if cleaned_p:
                cleaned_page_texts.append(cleaned_p)

        if not cleaned_page_texts:
            raise EmptyDocumentError(
                f"Document '{extraction.file_name}' contains no readable text."
            )

        full_cleaned_text = "\n\n".join(cleaned_page_texts)

        # Calculate exact [start_offset, end_offset) spans on full_cleaned_text
        page_spans: list[PageSpan] = []
        current_offset = 0
        for idx, page_str in enumerate(cleaned_page_texts, start=1):
            start = current_offset
            end = start + len(page_str)
            page_spans.append(
                PageSpan(page_number=idx, start_offset=start, end_offset=end)
            )
            current_offset = end + len("\n\n")

        return CleanedDocument(
            cleaned_text=full_cleaned_text,
            file_name=extraction.file_name,
            mime_type=extraction.mime_type,
            metadata=extraction.metadata,
            cleaning_version=CLEANING_VERSION,
            page_spans=page_spans,
        )

    # Documents without Form Feed separators (e.g. TXT, DOCX, or unpaginated text)
    cleaned_text = clean_text(raw_text)

    page_spans = None
    if extraction.page_count == 1:
        page_spans = [
            PageSpan(page_number=1, start_offset=0, end_offset=len(cleaned_text))
        ]

    return CleanedDocument(
        cleaned_text=cleaned_text,
        file_name=extraction.file_name,
        mime_type=extraction.mime_type,
        metadata=extraction.metadata,
        cleaning_version=CLEANING_VERSION,
        page_spans=page_spans,
    )
