"""Exceptions raised by PostgreSQL document persistence."""


class StorageError(Exception):
    """Base exception for document and chunk storage errors."""


class DatabaseConnectionError(StorageError):
    """Raised when connecting or communicating with PostgreSQL fails."""


class DocumentProcessingConflictError(StorageError):
    """Raised when document is currently PROCESSING and reclaim was not requested."""


class BoardMismatchError(StorageError):
    """Raised when an existing document belongs to a different board."""


class ProcessingOwnershipLostError(StorageError):
    """Raised when processing token ownership was lost before completion."""


class StorageValidationError(StorageError):
    """Raised when pre-save validation of documents, chunks, or vectors fails."""
