"""Exceptions for document extraction and cleaning."""


class IngestionError(Exception):
    """Base exception for all ingestion errors."""


class FileValidationError(IngestionError):
    """Raised when an input file fails validation checks."""


class FileEmptyError(FileValidationError):
    """Raised when an input file is 0 bytes."""


class FileTooLargeError(FileValidationError):
    """Raised when an input file exceeds the allowed size limit."""


class UnsupportedFileFormatError(FileValidationError):
    """Raised when a file has an unsupported format or mismatched magic bytes."""


class TikaError(IngestionError):
    """Base exception for Apache Tika client errors."""


class TikaConnectionError(TikaError):
    """Raised when connection to Apache Tika server fails."""


class TikaTimeoutError(TikaError):
    """Raised when a request to Apache Tika times out."""


class TikaExtractionError(TikaError):
    """Raised when Apache Tika returns an error or unparseable response."""


class CleaningError(IngestionError):
    """Base exception for document cleaning errors."""


class EmptyDocumentError(CleaningError):
    """Raised when document contains no usable text after cleaning."""


class ChunkingError(IngestionError):
    """Raised when document chunking fails or cannot respect constraints."""


class TokenizerLoadError(IngestionError):
    """Raised when designated tokenizer cannot be loaded."""
