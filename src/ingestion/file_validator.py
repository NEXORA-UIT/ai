"""Validation rules and content sniffing for input documents."""

import mimetypes
from pathlib import Path

from src.ingestion.exceptions import (
    FileEmptyError,
    FileTooLargeError,
    FileValidationError,
    UnsupportedFileFormatError,
)

DEFAULT_MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB, matching Nexora SRS limit

SUPPORTED_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
}


def validate_file(
    file_path: Path | str,
    max_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> tuple[Path, str, int]:
    """Validate file existence, size, extension, and magic bytes.

    Args:
        file_path: Path to the target file.
        max_size_bytes: Maximum allowed file size in bytes (default 25 MB).

    Returns:
        tuple[Path, str, int]: (resolved_path, mime_type, file_size_bytes).

    Raises:
        FileValidationError: If file does not exist or is not a regular file.
        FileEmptyError: If file is 0 bytes.
        FileTooLargeError: If file size exceeds max_size_bytes.
        UnsupportedFileFormatError: If extension is unsupported or magic bytes mismatch.
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileValidationError(f"File not found: {path}")

    if not path.is_file():
        raise FileValidationError(f"Path is not a regular file: {path}")

    file_size = path.stat().st_size
    if file_size == 0:
        raise FileEmptyError(f"File is empty (0 bytes): {path.name}")

    if file_size > max_size_bytes:
        raise FileTooLargeError(
            f"File size ({file_size} bytes) exceeds limit ({max_size_bytes} bytes): "
            f"{path.name}"
        )

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_MIME_TYPES:
        supported = ", ".join(SUPPORTED_MIME_TYPES.keys())
        raise UnsupportedFileFormatError(
            f"Unsupported file format '{suffix}'. Supported formats: {supported}"
        )

    # Magic byte verification
    with open(path, "rb") as f:
        header = f.read(16)

    if suffix == ".pdf":
        if not header.startswith(b"%PDF-"):
            raise UnsupportedFileFormatError(
                f"File '{path.name}' has .pdf extension but lacks '%PDF-' magic header."
            )
        mime_type = "application/pdf"
    elif suffix == ".docx":
        # Standard OpenXML DOCX files are packaged in a ZIP container starting
        # with PK\x03\x04 magic bytes. Deep XML structure validation is handled
        # downstream by Apache Tika.
        if not header.startswith(b"PK\x03\x04"):
            raise UnsupportedFileFormatError(
                f"File '{path.name}' has .docx extension but lacks standard ZIP "
                f"magic header (PK\\x03\\x04)."
            )
        mime_type = (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    elif suffix == ".txt":
        # Scope is UTF-8 and ASCII plain text. Files containing binary NUL bytes
        # in the first 16 bytes are rejected to prevent binary files from being
        # processed as text (UTF-16 with NUL bytes is outside current scope).
        if b"\x00" in header:
            raise UnsupportedFileFormatError(
                f"File '{path.name}' has .txt extension but contains binary NUL bytes."
            )
        inferred = mimetypes.guess_type(path.name)[0]
        mime_type = inferred or "text/plain"
    else:
        mime_type = SUPPORTED_MIME_TYPES[suffix]

    return path, mime_type, file_size
