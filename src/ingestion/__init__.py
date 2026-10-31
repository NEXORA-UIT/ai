"""Ingestion package: extraction, cleaning, and preparation of documents."""

from src.ingestion.chunker import chunk_document
from src.ingestion.cleaner import clean_document, clean_text, clean_text_fragment
from src.ingestion.exceptions import (
    ChunkingError,
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
    TokenizerLoadError,
    UnsupportedFileFormatError,
)
from src.ingestion.file_validator import validate_file
from src.ingestion.pipeline import (
    extract_and_clean,
    extract_clean_and_chunk,
    extract_clean_chunk_and_embed,
)
from src.ingestion.schemas import (
    Chunk,
    ChunkedDocument,
    CleanedDocument,
    EmbeddedChunk,
    EmbeddedDocument,
    ExtractionResult,
    PageSpan,
)
from src.ingestion.tika_client import TikaClient

__all__ = [
    "Chunk",
    "ChunkedDocument",
    "ChunkingError",
    "CleanedDocument",
    "CleaningError",
    "EmbeddedChunk",
    "EmbeddedDocument",
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
    "TokenizerLoadError",
    "UnsupportedFileFormatError",
    "chunk_document",
    "clean_document",
    "clean_text",
    "clean_text_fragment",
    "extract_and_clean",
    "extract_clean_and_chunk",
    "extract_clean_chunk_and_embed",
    "validate_file",
]
