"""Ingestion package with lazy public exports."""

from importlib import import_module
from typing import Any

__all__ = [
    "BoardMismatchError",
    "Chunk",
    "ChunkedDocument",
    "ChunkingError",
    "CleanedDocument",
    "CleaningError",
    "DatabaseConnectionError",
    "DocumentProcessingConflictError",
    "DocumentRecord",
    "DocumentStorage",
    "DocumentStatus",
    "EmbeddedChunk",
    "EmbeddedDocument",
    "EmptyDocumentError",
    "ExtractionResult",
    "FileEmptyError",
    "FileTooLargeError",
    "FileValidationError",
    "FileValidationResult",
    "IngestionError",
    "PageSpan",
    "ProcessingOwnershipLostError",
    "StorageError",
    "StorageValidationError",
    "StoredChunk",
    "StoredDocument",
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
    "ingest_file_to_storage",
    "sanitize_error_message",
    "validate_file",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "BoardMismatchError": ("src.ingestion.exceptions", "BoardMismatchError"),
    "Chunk": ("src.ingestion.schemas", "Chunk"),
    "ChunkedDocument": ("src.ingestion.schemas", "ChunkedDocument"),
    "ChunkingError": ("src.ingestion.exceptions", "ChunkingError"),
    "CleanedDocument": ("src.ingestion.schemas", "CleanedDocument"),
    "CleaningError": ("src.ingestion.exceptions", "CleaningError"),
    "DatabaseConnectionError": ("src.ingestion.exceptions", "DatabaseConnectionError"),
    "DocumentProcessingConflictError": (
        "src.ingestion.exceptions",
        "DocumentProcessingConflictError",
    ),
    "DocumentRecord": ("src.persistence.document_storage", "DocumentRecord"),
    "DocumentStorage": ("src.persistence.document_storage", "DocumentStorage"),
    "DocumentStatus": ("src.ingestion.status", "DocumentStatus"),
    "EmbeddedChunk": ("src.ingestion.schemas", "EmbeddedChunk"),
    "EmbeddedDocument": ("src.ingestion.schemas", "EmbeddedDocument"),
    "EmptyDocumentError": ("src.ingestion.exceptions", "EmptyDocumentError"),
    "ExtractionResult": ("src.ingestion.schemas", "ExtractionResult"),
    "FileEmptyError": ("src.ingestion.exceptions", "FileEmptyError"),
    "FileTooLargeError": ("src.ingestion.exceptions", "FileTooLargeError"),
    "FileValidationError": ("src.ingestion.exceptions", "FileValidationError"),
    "FileValidationResult": ("src.ingestion.schemas", "FileValidationResult"),
    "IngestionError": ("src.ingestion.exceptions", "IngestionError"),
    "PageSpan": ("src.ingestion.schemas", "PageSpan"),
    "ProcessingOwnershipLostError": (
        "src.ingestion.exceptions",
        "ProcessingOwnershipLostError",
    ),
    "StorageError": ("src.ingestion.exceptions", "StorageError"),
    "StorageValidationError": ("src.ingestion.exceptions", "StorageValidationError"),
    "StoredChunk": ("src.persistence.document_storage", "StoredChunk"),
    "StoredDocument": ("src.persistence.document_storage", "StoredDocument"),
    "TikaClient": ("src.ingestion.extraction.tika_client", "TikaClient"),
    "TikaConnectionError": ("src.ingestion.exceptions", "TikaConnectionError"),
    "TikaError": ("src.ingestion.exceptions", "TikaError"),
    "TikaExtractionError": ("src.ingestion.exceptions", "TikaExtractionError"),
    "TikaTimeoutError": ("src.ingestion.exceptions", "TikaTimeoutError"),
    "TokenizerLoadError": ("src.ingestion.exceptions", "TokenizerLoadError"),
    "UnsupportedFileFormatError": (
        "src.ingestion.exceptions",
        "UnsupportedFileFormatError",
    ),
    "chunk_document": ("src.chunking.chunker", "chunk_document"),
    "clean_document": ("src.ingestion.cleaning.cleaner", "clean_document"),
    "clean_text": ("src.ingestion.cleaning.cleaner", "clean_text"),
    "clean_text_fragment": (
        "src.ingestion.cleaning.cleaner",
        "clean_text_fragment",
    ),
    "extract_and_clean": ("src.ingestion.pipeline", "extract_and_clean"),
    "extract_clean_and_chunk": (
        "src.ingestion.pipeline",
        "extract_clean_and_chunk",
    ),
    "extract_clean_chunk_and_embed": (
        "src.ingestion.pipeline",
        "extract_clean_chunk_and_embed",
    ),
    "ingest_file_to_storage": (
        "src.ingestion.pipeline",
        "ingest_file_to_storage",
    ),
    "sanitize_error_message": (
        "src.persistence.document_storage",
        "sanitize_error_message",
    ),
    "validate_file": ("src.ingestion.extraction.file_validator", "validate_file"),
}


def __getattr__(name: str) -> Any:
    """Load a public API item only when it is accessed."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
