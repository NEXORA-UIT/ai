"""Stage a bounded upload and publish it only after local validation."""

import os
import re
from contextlib import suppress
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import BinaryIO
from uuid import UUID

from src.ingestion.exceptions import FileTooLargeError
from src.ingestion.extraction.file_validator import ValidatedFile, validate_file

_MAX_FILENAME_LENGTH = 255
_COPY_CHUNK_BYTES = 1024 * 1024
_DEFAULT_FILENAME = "upload.txt"
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")


def _safe_filename(filename: str | None) -> str:
    normalized = (filename or _DEFAULT_FILENAME).replace("\\", "/")
    basename = _CONTROL_CHARACTERS.sub("", PurePosixPath(normalized).name)
    basename = basename.strip().strip(".") or _DEFAULT_FILENAME
    suffix = Path(basename).suffix
    if len(basename) > _MAX_FILENAME_LENGTH:
        stem = basename[: -len(suffix)] if suffix else basename
        basename = f"{stem[: _MAX_FILENAME_LENGTH - len(suffix)]}{suffix}"
    return basename


def stage_upload(
    source: BinaryIO,
    filename: str | None,
    *,
    root: Path,
    document_id: UUID,
    max_size_bytes: int,
) -> ValidatedFile:
    basename = _safe_filename(filename)
    destination = root / str(document_id)
    path = destination / basename
    root.mkdir(parents=True, exist_ok=True)

    staging_path: Path | None = None
    destination_created = False
    try:
        with NamedTemporaryFile(
            dir=root, suffix=Path(basename).suffix, delete=False
        ) as staged:
            staging_path = Path(staged.name)
            size = 0
            while chunk := source.read(_COPY_CHUNK_BYTES):
                size += len(chunk)
                if size > max_size_bytes:
                    raise FileTooLargeError("Upload exceeds the configured size limit")
                staged.write(chunk)
        _, mime_type, size = validate_file(staging_path, max_size_bytes=max_size_bytes)
        destination.mkdir()
        destination_created = True
        os.replace(staging_path, path)
        return ValidatedFile(path, mime_type, size)
    except Exception:
        if destination_created:
            with suppress(OSError):
                destination.rmdir()
        raise
    finally:
        if staging_path is not None:
            with suppress(OSError):
                staging_path.unlink(missing_ok=True)
