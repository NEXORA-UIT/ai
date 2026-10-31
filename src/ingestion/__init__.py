"""Ingestion package: extraction, cleaning, and preparation of documents."""

from src.ingestion.cleaner import clean_document, clean_text, clean_text_fragment
from src.ingestion.exceptions import (
    CleaningError,
    EmptyDocumentError,
    FileEmptyError,
    FileTooLargeError,
    FileValidationError,
    IngestionError,
    TikaConnectionError,
    TikaError,
    TikaExtractionError,
    TikaTimeoutError,
    UnsupportedFileFormatError,
)
from src.ingestion.file_validator import validate_file
from src.ingestion.pipeline import extract_and_clean
from src.ingestion.schemas import CleanedDocument, ExtractionResult, PageSpan
from src.ingestion.tika_client import TikaClient

__all__ = [
    "CleaningError",
    "CleanedDocument",
    "EmptyDocumentError",
    "ExtractionResult",
    "FileEmptyError",
    "FileTooLargeError",
    "FileValidationError",
    "IngestionError",
    "PageSpan",
    "TikaClient",
    "TikaConnectionError",
    "TikaError",
    "TikaExtractionError",
    "TikaTimeoutError",
    "UnsupportedFileFormatError",
    "clean_document",
    "clean_text",
    "clean_text_fragment",
    "extract_and_clean",
    "validate_file",
]
