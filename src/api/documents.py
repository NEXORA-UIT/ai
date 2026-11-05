"""Swagger multipart endpoint for synchronous document ingestion."""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path, PurePosixPath
from typing import NoReturn

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from src.api.config import (
    DEFAULT_UPLOAD_DIRECTORY,
    MAX_STORED_FILE_NAME_LENGTH,
    UPLOAD_COPY_CHUNK_BYTES,
    UPLOAD_DIRECTORY_ENV,
)
from src.api.schemas import DocumentUploadResponse
from src.api.security import require_service_key
from src.ingestion.exceptions import (
    DatabaseConnectionError,
    DocumentProcessingConflictError,
    FileEmptyError,
    FileTooLargeError,
    FileValidationError,
    IngestionError,
    TikaError,
    UnsupportedFileFormatError,
)
from src.ingestion.extraction.file_validator import (
    DEFAULT_MAX_FILE_SIZE_BYTES,
    validate_file,
)
from src.ingestion.pipeline import ingest_file_to_storage

router = APIRouter(prefix="/v1", tags=["Documents"])

MAX_UPLOAD_SIZE_BYTES = DEFAULT_MAX_FILE_SIZE_BYTES
_DEFAULT_UPLOAD_FILENAME = "upload.txt"
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")
_UPLOAD_FILE_FIELD = File(..., description="PDF, DOCX, or UTF-8 text file")
_BOARD_ID_FIELD = Form(..., description="UUID of the caller-authorized Board")
_UPLOADED_BY_FIELD = Form(..., description="UUID of the uploading user")


def _upload_directory() -> Path:
    configured = os.environ.get(UPLOAD_DIRECTORY_ENV)
    directory = Path(configured) if configured else DEFAULT_UPLOAD_DIRECTORY
    return directory if directory.is_absolute() else Path.cwd() / directory


def _safe_filename(filename: str | None) -> str:
    normalized = (filename or _DEFAULT_UPLOAD_FILENAME).replace("\\", "/")
    basename = PurePosixPath(normalized).name
    basename = _CONTROL_CHARACTERS.sub("", basename).strip().strip(".")
    if not basename:
        basename = _DEFAULT_UPLOAD_FILENAME

    suffix = Path(basename).suffix
    if len(basename) > MAX_STORED_FILE_NAME_LENGTH:
        stem = basename[: -len(suffix)] if suffix else basename
        stem_limit = MAX_STORED_FILE_NAME_LENGTH - len(suffix)
        basename = f"{stem[:stem_limit]}{suffix}"
    return basename


def _validate_upload(path: Path) -> None:
    try:
        validate_file(path, max_size_bytes=MAX_UPLOAD_SIZE_BYTES)
    except FileTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="Uploaded file exceeds the size limit",
        ) from exc
    except FileEmptyError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Uploaded file is empty",
        ) from exc
    except UnsupportedFileFormatError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Uploaded file format is not supported",
        ) from exc
    except FileValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Uploaded file failed validation",
        ) from exc


def _copy_upload(upload: UploadFile, staging_path: Path) -> None:
    total_bytes = 0
    with staging_path.open("xb") as destination:
        while True:
            chunk = upload.file.read(UPLOAD_COPY_CHUNK_BYTES)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail="Uploaded file exceeds the size limit",
                )
            destination.write(chunk)


def _store_upload(upload: UploadFile, document_id: uuid.UUID) -> Path:
    root = _upload_directory()
    destination_directory = root / str(document_id)
    filename = _safe_filename(upload.filename)
    suffix = Path(filename).suffix
    staging_path = root / f".{document_id}.staging{suffix}"
    final_path = destination_directory / filename

    try:
        root.mkdir(parents=True, exist_ok=True)
        _copy_upload(upload, staging_path)
        _validate_upload(staging_path)
        destination_directory.mkdir(parents=True, exist_ok=False)
        os.replace(staging_path, final_path)
        return final_path
    except Exception:
        staging_path.unlink(missing_ok=True)
        raise


def _raise_ingestion_http_error(exc: Exception) -> NoReturn:
    if isinstance(exc, FileTooLargeError):
        code, detail = 413, "Uploaded file exceeds the size limit"
    elif isinstance(exc, FileEmptyError):
        code, detail = 422, "Uploaded file is empty"
    elif isinstance(exc, UnsupportedFileFormatError):
        code, detail = 415, "Uploaded file format is not supported"
    elif isinstance(exc, FileValidationError):
        code, detail = 422, "Uploaded file failed validation"
    elif isinstance(exc, DocumentProcessingConflictError):
        code, detail = 409, "Document is already being processed"
    elif isinstance(exc, DatabaseConnectionError):
        code, detail = 503, "AI database is unavailable"
    elif isinstance(exc, TikaError):
        code, detail = 502, "Document extraction failed"
    elif isinstance(exc, IngestionError):
        code, detail = 422, "Document could not be processed"
    else:
        code, detail = 500, "Document ingestion failed"
    raise HTTPException(status_code=code, detail=detail) from exc


@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document",
    description=(
        "Upload a PDF, DOCX, or UTF-8 text file and wait for synchronous ingestion. "
        "The caller must already be authorized for the supplied Board."
    ),
    responses={
        401: {"description": "Missing or invalid service key"},
        409: {"description": "Document is already being processed"},
        413: {"description": "Uploaded file exceeds the configured limit"},
        415: {"description": "Unsupported or invalid file format"},
        422: {"description": "Invalid form fields or unreadable document"},
        500: {"description": "Ingestion failed"},
        502: {"description": "Document extraction service failed"},
        503: {"description": "Service API key or database is unavailable"},
    },
    dependencies=[Depends(require_service_key)],
)
def upload_document(
    file: UploadFile = _UPLOAD_FILE_FIELD,
    board_id: uuid.UUID = _BOARD_ID_FIELD,
    uploaded_by: uuid.UUID = _UPLOADED_BY_FIELD,
) -> DocumentUploadResponse:
    """Persist the upload and run the existing ingestion pipeline synchronously."""
    document_id = uuid.uuid4()
    try:
        file_path = _store_upload(file, document_id)
    except HTTPException:
        raise
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not store uploaded file",
        ) from exc

    try:
        persisted_id, embedded_document = ingest_file_to_storage(
            file_path=file_path,
            board_id=board_id,
            uploaded_by=uploaded_by,
            document_id=document_id,
        )
        persisted_uuid = uuid.UUID(str(persisted_id))
        if persisted_uuid != document_id:
            raise RuntimeError("Ingestion returned a different document identifier")
    except Exception as exc:
        _raise_ingestion_http_error(exc)

    return DocumentUploadResponse(
        document_id=document_id,
        status="READY",
        chunk_count=embedded_document.total_chunks,
        total_tokens=embedded_document.total_tokens,
    )
