"""Swagger multipart endpoint for synchronous document ingestion."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import NoReturn

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from src.api.config import (
    API_V1_PREFIX,
    DATABASE_UNAVAILABLE_DETAIL,
    DEFAULT_UPLOAD_DIRECTORY,
    DOCUMENTS_ROUTE_PATH,
    UPLOAD_DIRECTORY_ENV,
    UPLOAD_FILE_TOO_LARGE_DETAIL,
)
from src.api.schemas import DocumentUploadResponse
from src.api.security import require_service_key
from src.documents.status import DocumentStatus
from src.ingestion.exceptions import (
    FileEmptyError,
    FileTooLargeError,
    FileValidationError,
    IngestionError,
    TikaError,
    UnsupportedFileFormatError,
)
from src.ingestion.extraction.file_validator import (
    DEFAULT_MAX_FILE_SIZE_BYTES,
)
from src.ingestion.pipeline import ingest_file_to_storage
from src.ingestion.upload import stage_upload
from src.persistence.errors import (
    DatabaseConnectionError,
    DocumentProcessingConflictError,
    StorageError,
)

router = APIRouter(prefix=API_V1_PREFIX, tags=["Documents"])

MAX_UPLOAD_SIZE_BYTES = DEFAULT_MAX_FILE_SIZE_BYTES
_UPLOAD_FILE_FIELD = File(..., description="PDF, DOCX, or UTF-8 text file")
_BOARD_ID_FIELD = Form(..., description="UUID of the caller-authorized Board")
_UPLOADED_BY_FIELD = Form(..., description="UUID of the uploading user")


def _upload_directory() -> Path:
    configured = os.environ.get(UPLOAD_DIRECTORY_ENV)
    directory = Path(configured) if configured else DEFAULT_UPLOAD_DIRECTORY
    return directory if directory.is_absolute() else Path.cwd() / directory


def _raise_ingestion_http_error(exc: Exception) -> NoReturn:
    if isinstance(exc, FileTooLargeError):
        code, detail = (
            status.HTTP_413_CONTENT_TOO_LARGE,
            UPLOAD_FILE_TOO_LARGE_DETAIL,
        )
    elif isinstance(exc, FileEmptyError):
        code, detail = (
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "Uploaded file is empty",
        )
    elif isinstance(exc, UnsupportedFileFormatError):
        code, detail = (
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "Uploaded file format is not supported",
        )
    elif isinstance(exc, FileValidationError):
        code, detail = (
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "Uploaded file failed validation",
        )
    elif isinstance(exc, DocumentProcessingConflictError):
        code, detail = (
            status.HTTP_409_CONFLICT,
            "Document is already being processed",
        )
    elif isinstance(exc, DatabaseConnectionError):
        code, detail = (
            status.HTTP_503_SERVICE_UNAVAILABLE,
            DATABASE_UNAVAILABLE_DETAIL,
        )
    elif isinstance(exc, TikaError):
        code, detail = (status.HTTP_502_BAD_GATEWAY, "Document extraction failed")
    elif isinstance(exc, (IngestionError, StorageError)):
        code, detail = (
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "Document could not be processed",
        )
    else:
        code, detail = (
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Document ingestion failed",
        )
    raise HTTPException(status_code=code, detail=detail) from exc


@router.post(
    DOCUMENTS_ROUTE_PATH,
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document",
    description=(
        "Upload a PDF, DOCX, or UTF-8 text file and wait for synchronous ingestion. "
        "The caller must already be authorized for the supplied Board."
    ),
    responses={
        status.HTTP_401_UNAUTHORIZED: {"description": "Missing or invalid service key"},
        status.HTTP_409_CONFLICT: {
            "description": "Document is already being processed"
        },
        status.HTTP_413_CONTENT_TOO_LARGE: {
            "description": "Uploaded file exceeds the configured limit"
        },
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {
            "description": "Unsupported or invalid file format"
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "Invalid form fields or unreadable document"
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Ingestion failed"},
        status.HTTP_502_BAD_GATEWAY: {
            "description": "Document extraction service failed"
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Service API key or database is unavailable"
        },
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
        staged = stage_upload(
            file.file,
            file.filename,
            root=_upload_directory(),
            document_id=document_id,
            max_size_bytes=MAX_UPLOAD_SIZE_BYTES,
        )
    except FileValidationError as exc:
        _raise_ingestion_http_error(exc)
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not store uploaded file",
        ) from exc

    try:
        persisted_id, embedded_document = ingest_file_to_storage(
            file_path=staged,
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
        status=DocumentStatus.READY,
        chunk_count=embedded_document.total_chunks,
        total_tokens=embedded_document.total_tokens,
    )
