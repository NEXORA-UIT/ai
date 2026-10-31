"""PostgreSQL and pgvector storage implementation for document ingestion.

Manages the persistence of Document and DocumentChunk records with
strict transaction boundaries and processing token ownership for safe
concurrency and idempotent re-processing.
"""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import psycopg
from pgvector.psycopg import register_vector

from src.ingestion.exceptions import (
    BoardMismatchError,
    DatabaseConnectionError,
    DocumentProcessingConflictError,
    ProcessingOwnershipLostError,
    StorageError,
    StorageValidationError,
)

if TYPE_CHECKING:
    from src.ingestion.schemas import (
        EmbeddedDocument,
        FileValidationResult,
    )

# Vector specifications
DEFAULT_DENSE_DIMENSION: int = 1024
L2_NORM_TOLERANCE: float = 1e-4

_URL_PASSWORD_RE = re.compile(
    r"(?P<prefix>(?:postgres(?:ql)?|https?|ftp)://[^:]+:)(?P<pass>[^@]+)(?P<suffix>@)"
)
_DATABASE_URL_RE = re.compile(r"(?:postgres(?:ql)?|postgresql\+psycopg)://[^\s'\"]+")
_SQL_STATEMENT_RE = re.compile(
    r"(?i)\b(SELECT|INSERT INTO|UPDATE|DELETE FROM|"
    r"CREATE TABLE|ALTER TABLE|DROP TABLE)\b[^;]*"
)
_SENSITIVE_PARAM_RE = re.compile(
    r"(?i)(password|secret|token|api_key|credential)\s*[:=]\s*['\"]?[^\s'\"]+['\"]?"
)


def sanitize_error_message(message: str | Exception, max_length: int = 250) -> str:
    """Sanitize error messages to prevent leaking secrets or raw SQL."""
    if isinstance(message, Exception):
        msg = str(message) or type(message).__name__
    else:
        msg = str(message)

    # 1. Redact credentials in connection URLs
    msg = _URL_PASSWORD_RE.sub(r"\g<prefix>***\g<suffix>", msg)
    msg = _DATABASE_URL_RE.sub("[REDACTED_DATABASE_URL]", msg)

    # 2. Redact sensitive key=value pairs
    msg = _SENSITIVE_PARAM_RE.sub(r"\1=[REDACTED]", msg)

    # 3. Redact embedded raw SQL statements
    msg = _SQL_STATEMENT_RE.sub("[REDACTED_SQL]", msg)

    # 4. Collapse whitespace
    msg = " ".join(msg.split())

    # 5. Enforce bounded length
    if len(msg) > max_length:
        msg = msg[: max_length - 3] + "..."

    return msg


def _to_float_list(val: Any) -> list[float]:
    """Convert pgvector Vector, numpy ndarray, or iterable to list of floats."""
    if hasattr(val, "to_list"):
        return [float(x) for x in val.to_list()]
    if hasattr(val, "tolist"):
        return [float(x) for x in val.tolist()]
    return [float(x) for x in val]


@dataclass(frozen=True)
class DocumentRecord:
    """Represents a row from the documents table."""

    id: str
    board_id: str
    uploaded_by: str
    file_name: str
    storage_key: str
    mime_type: str
    file_size: int
    checksum: str
    status: str
    processing_token: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class StoredChunk:
    """Represents a row from the document_chunks table."""

    id: str
    document_id: str
    chunk_index: int
    content: str
    page_number: int | None
    start_offset: int | None
    end_offset: int | None
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass(frozen=True)
class StoredDocument:
    """Complete document record paired with all its stored chunks."""

    document: DocumentRecord
    chunks: list[StoredChunk]


class DocumentStorage:
    """Handles PostgreSQL/pgvector persistence for documents and chunks.

    Separates database storage from CPU/network heavy tasks (Tika, BGE-M3).
    """

    def __init__(
        self,
        connection_factory: Callable[[], psycopg.Connection] | None = None,
        database_url: str | None = None,
    ) -> None:
        self._connection_factory = connection_factory
        self._database_url = database_url

    def get_connection(self) -> psycopg.Connection:
        """Create and return a new psycopg connection with pgvector registered."""
        try:
            if self._connection_factory is not None:
                conn = self._connection_factory()
            else:
                url = self._database_url or os.environ.get("DATABASE_URL")
                if not url:
                    raise DatabaseConnectionError("DATABASE_URL is not configured.")
                conn = psycopg.connect(url, connect_timeout=10)
            if isinstance(conn, psycopg.Connection):
                register_vector(conn)
            return conn
        except psycopg.Error as exc:
            raise DatabaseConnectionError(
                f"Failed to connect to PostgreSQL: {exc}"
            ) from exc

    def claim_or_create_document(
        self,
        board_id: str | uuid.UUID,
        uploaded_by: str | uuid.UUID,
        file_name: str,
        storage_key: str,
        mime_type: str,
        file_size: int,
        checksum: str,
        document_id: str | uuid.UUID | None = None,
        reclaim: bool = False,
    ) -> tuple[str, uuid.UUID]:
        """Claim a new or existing document and transition it to PROCESSING.

        Uses a short transaction with row-level locking or conflict-safe insert.

        Args:
            board_id: UUID of the owning Board.
            uploaded_by: UUID of the uploading User.
            file_name: Base name of the file.
            storage_key: Storage key (e.g. local://...).
            mime_type: MIME type of the file.
            file_size: Size in bytes (>= 0).
            checksum: SHA-256 hex digest.
            document_id: Optional existing/target document UUID.
            reclaim: Reclaim ownership from an interrupted PROCESSING run.

        Returns:
            tuple[str, uuid.UUID]: (document_id, new_processing_token)

        Raises:
            BoardMismatchError: If document exists on a different board.
            DocumentProcessingConflictError: If already PROCESSING and reclaim=False.
            StorageError: On database or validation errors.
        """
        board_str = str(uuid.UUID(str(board_id)))
        uploader_str = str(uuid.UUID(str(uploaded_by)))
        target_id_str = str(
            uuid.UUID(str(document_id)) if document_id else uuid.uuid4()
        )
        new_token = uuid.uuid4()

        if file_size < 0:
            raise StorageValidationError("file_size must be non-negative.")

        conn = self.get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    # Attempt conflict-safe insert for non-existent document
                    cur.execute(
                        """
                        INSERT INTO documents (
                            id, board_id, uploaded_by, file_name, storage_key,
                            mime_type, file_size, checksum, status,
                            processing_token, metadata
                        ) VALUES (
                            %(id)s, %(board_id)s, %(uploaded_by)s, %(file_name)s,
                            %(storage_key)s, %(mime_type)s, %(file_size)s,
                            %(checksum)s, 'PROCESSING', %(token)s, '{}'::jsonb
                        )
                        ON CONFLICT (id) DO NOTHING
                        RETURNING id;
                        """,
                        {
                            "id": target_id_str,
                            "board_id": board_str,
                            "uploaded_by": uploader_str,
                            "file_name": file_name,
                            "storage_key": storage_key,
                            "mime_type": mime_type,
                            "file_size": file_size,
                            "checksum": checksum,
                            "token": new_token,
                        },
                    )
                    inserted = cur.fetchone()
                    if inserted is not None:
                        return target_id_str, new_token

                    # Document already exists, lock and inspect row
                    cur.execute(
                        """
                        SELECT id, board_id, uploaded_by, status, processing_token
                        FROM documents
                        WHERE id = %(id)s
                        FOR UPDATE;
                        """,
                        {"id": target_id_str},
                    )
                    row = cur.fetchone()
                    if row is None:
                        raise StorageError(
                            f"Document {target_id_str} was concurrently deleted."
                        )

                    existing_board = str(row[1])
                    existing_status = str(row[3])

                    if existing_board != board_str:
                        raise BoardMismatchError(
                            f"Document {target_id_str} belongs to board "
                            f"{existing_board}, cannot claim for board {board_str}."
                        )

                    if existing_status == "PROCESSING" and not reclaim:
                        raise DocumentProcessingConflictError(
                            f"Document {target_id_str} is currently in "
                            f"PROCESSING status. Pass reclaim=True to take ownership."
                        )

                    # Update status to PROCESSING (preserves initial uploaded_by)
                    cur.execute(
                        """
                        UPDATE documents
                        SET status = 'PROCESSING',
                            processing_token = %(token)s,
                            file_name = %(file_name)s,
                            storage_key = %(storage_key)s,
                            mime_type = %(mime_type)s,
                            file_size = %(file_size)s,
                            checksum = %(checksum)s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = %(id)s;
                        """,
                        {
                            "id": target_id_str,
                            "token": new_token,
                            "file_name": file_name,
                            "storage_key": storage_key,
                            "mime_type": mime_type,
                            "file_size": file_size,
                            "checksum": checksum,
                        },
                    )
                    return target_id_str, new_token
        except psycopg.Error as exc:
            raise DatabaseConnectionError(
                f"Database error during document claim: {exc}"
            ) from exc
        finally:
            conn.close()

    def validate_pre_save(self, embedded_doc: EmbeddedDocument) -> None:
        """Validate chunks, vectors, and metadata before initiating save."""
        if not embedded_doc.chunks:
            raise StorageValidationError("Embedded document contains 0 chunks.")

        if embedded_doc.total_chunks != len(embedded_doc.chunks):
            raise StorageValidationError(
                f"Chunk count mismatch: total_chunks={embedded_doc.total_chunks} "
                f"but len(chunks)={len(embedded_doc.chunks)}."
            )

        indices = [c.chunk.chunk_index for c in embedded_doc.chunks]
        if indices != list(range(len(embedded_doc.chunks))):
            raise StorageValidationError(
                f"Chunk indices must be continuous from 0 to N-1, got: {indices}"
            )

        for ec in embedded_doc.chunks:
            if not ec.chunk.content or not ec.chunk.content.strip():
                raise StorageValidationError(
                    f"Chunk at index {ec.chunk.chunk_index} has empty content."
                )

            vec_len = len(ec.dense_embedding)
            if vec_len != DEFAULT_DENSE_DIMENSION:
                raise StorageValidationError(
                    f"Chunk {ec.chunk.chunk_index} dimension {vec_len} "
                    f"!= expected {DEFAULT_DENSE_DIMENSION}."
                )

            if not all(math.isfinite(x) for x in ec.dense_embedding):
                raise StorageValidationError(
                    f"Chunk {ec.chunk.chunk_index} contains NaN or Inf in vector."
                )

            if all(x == 0.0 for x in ec.dense_embedding):
                raise StorageValidationError(
                    f"Chunk {ec.chunk.chunk_index} contains all-zero vector."
                )

            norm = math.sqrt(sum(x * x for x in ec.dense_embedding))
            if abs(norm - 1.0) > L2_NORM_TOLERANCE:
                raise StorageValidationError(
                    f"Chunk {ec.chunk.chunk_index} vector L2 norm {norm:.6f} "
                    f"deviates from 1.0 (tol: {L2_NORM_TOLERANCE})."
                )

            start = ec.chunk.start_offset
            end = ec.chunk.end_offset
            valid_offsets = (start is None and end is None) or (
                start is not None and end is not None and start >= 0 and end > start
            )
            if not valid_offsets:
                raise StorageValidationError(
                    f"Chunk {ec.chunk.chunk_index} offsets invalid: [{start}, {end})."
                )

            page = ec.chunk.page_number
            if page is not None and page <= 0:
                raise StorageValidationError(
                    f"Chunk {ec.chunk.chunk_index} page_number must be > 0, got {page}."
                )

    def save_embedded_document(
        self,
        document_id: str | uuid.UUID,
        processing_token: uuid.UUID,
        embedded_doc: EmbeddedDocument,
        file_validation: FileValidationResult,
        storage_key: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Atomically persist chunks, clear old chunks, update metadata, mark READY.

        Guaranteed by processing_token ownership verification under row lock.

        Args:
            document_id: UUID of document being persisted.
            processing_token: Ownership token obtained at claim.
            embedded_doc: Embedded document artifacts.
            file_validation: Validated file attributes.
            storage_key: Transparent source identifier.
            extra_metadata: Additional metadata to merge into Document.metadata.

        Raises:
            StorageValidationError: If pre-save validation fails.
            ProcessingOwnershipLostError: If token does not match or not PROCESSING.
            DatabaseConnectionError: If connection or SQL execution fails.
        """
        self.validate_pre_save(embedded_doc)

        doc_id_str = str(uuid.UUID(str(document_id)))
        token_str = str(uuid.UUID(str(processing_token)))

        conn = self.get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, status, processing_token, metadata
                        FROM documents
                        WHERE id = %(id)s
                        FOR UPDATE;
                        """,
                        {"id": doc_id_str},
                    )
                    row = cur.fetchone()
                    if row is None:
                        raise ProcessingOwnershipLostError(
                            f"Document {doc_id_str} was deleted before persist."
                        )

                    current_status = str(row[1])
                    current_token = str(row[2]) if row[2] else None

                    if current_status != "PROCESSING":
                        raise ProcessingOwnershipLostError(
                            f"Document {doc_id_str} status is {current_status}, "
                            f"expected PROCESSING."
                        )

                    if current_token != token_str:
                        raise ProcessingOwnershipLostError(
                            f"Processing token mismatch for document {doc_id_str}. "
                            f"Ownership was lost or reclaimed by another process."
                        )

                    # Prepare merged metadata, clearing previous error information
                    existing_meta = dict(row[3] or {})
                    err_keys = (
                        "error_stage",
                        "error_type",
                        "error_message",
                        "failed_at",
                    )
                    for err_k in err_keys:
                        existing_meta.pop(err_k, None)

                    merged_metadata = {
                        **existing_meta,
                        "model_id": embedded_doc.model_name,
                        "model_revision": embedded_doc.model_revision,
                        "dimension": embedded_doc.dimension,
                        "normalized": embedded_doc.normalized,
                        "cleaning_version": embedded_doc.cleaning_version,
                        "chunking_version": embedded_doc.chunking_version,
                        "total_tokens": embedded_doc.total_tokens,
                        "total_chunks": embedded_doc.total_chunks,
                        **(extra_metadata or {}),
                    }

                    # Atomic replacement: Delete old chunks first
                    cur.execute(
                        "DELETE FROM document_chunks WHERE document_id = %(id)s;",
                        {"id": doc_id_str},
                    )

                    # Batch insert all new chunks
                    chunk_records = [
                        (
                            str(uuid.uuid4()),
                            doc_id_str,
                            ec.chunk.chunk_index,
                            ec.chunk.content,
                            ec.chunk.page_number,
                            ec.chunk.start_offset,
                            ec.chunk.end_offset,
                            ec.dense_embedding,
                            json.dumps(
                                {
                                    "token_count": ec.chunk.token_count,
                                    **(ec.chunk.metadata or {}),
                                }
                            ),
                        )
                        for ec in embedded_doc.chunks
                    ]

                    cur.executemany(
                        """
                        INSERT INTO document_chunks (
                            id, document_id, chunk_index, content, page_number,
                            start_offset, end_offset, embedding, metadata
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb
                        );
                        """,
                        chunk_records,
                    )

                    # Finalize status to READY and clear processing_token
                    cur.execute(
                        """
                        UPDATE documents
                        SET status = 'READY',
                            processing_token = NULL,
                            file_name = %(file_name)s,
                            storage_key = %(storage_key)s,
                            mime_type = %(mime_type)s,
                            file_size = %(file_size)s,
                            checksum = %(checksum)s,
                            metadata = %(metadata)s::jsonb,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = %(id)s;
                        """,
                        {
                            "id": doc_id_str,
                            "file_name": file_validation.file_name,
                            "storage_key": storage_key,
                            "mime_type": file_validation.mime_type,
                            "file_size": file_validation.file_size_bytes,
                            "checksum": file_validation.sha256,
                            "metadata": json.dumps(merged_metadata),
                        },
                    )
        except psycopg.Error as exc:
            raise DatabaseConnectionError(
                f"Failed to persist document and chunks: {exc}"
            ) from exc
        finally:
            conn.close()

    def mark_document_failed(
        self,
        document_id: str | uuid.UUID,
        processing_token: uuid.UUID,
        error_stage: str,
        error_type: str,
        error_message: str,
    ) -> bool:
        """Mark document as FAILED and record safe error details.

        Returns:
            bool: True if status updated to FAILED, False if token mismatch.
        """
        doc_id_str = str(uuid.UUID(str(document_id)))
        token_str = str(uuid.UUID(str(processing_token)))

        conn = self.get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, status, processing_token, metadata
                        FROM documents
                        WHERE id = %(id)s
                        FOR UPDATE;
                        """,
                        {"id": doc_id_str},
                    )
                    row = cur.fetchone()
                    if row is None:
                        return False

                    current_status = str(row[1])
                    current_token = str(row[2]) if row[2] else None

                    if current_status != "PROCESSING" or current_token != token_str:
                        return False

                    existing_meta = dict(row[3] or {})
                    existing_meta["error_stage"] = error_stage
                    existing_meta["error_type"] = error_type
                    existing_meta["error_message"] = sanitize_error_message(
                        error_message
                    )
                    existing_meta["failed_at"] = datetime.now(UTC).isoformat()

                    cur.execute(
                        """
                        UPDATE documents
                        SET status = 'FAILED',
                            processing_token = NULL,
                            metadata = %(metadata)s::jsonb,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = %(id)s;
                        """,
                        {
                            "id": doc_id_str,
                            "metadata": json.dumps(existing_meta),
                        },
                    )
                    return True
        except psycopg.Error as exc:
            raise DatabaseConnectionError(
                f"Failed to update document failure status: {exc}"
            ) from exc
        finally:
            conn.close()

    def get_document_with_chunks(
        self, document_id: str | uuid.UUID
    ) -> StoredDocument | None:
        """Retrieve stored document and all its chunks in index order."""
        doc_id_str = str(uuid.UUID(str(document_id)))
        conn = self.get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, board_id, uploaded_by, file_name, storage_key,
                               mime_type, file_size, checksum, status,
                               processing_token, metadata, created_at, updated_at
                        FROM documents
                        WHERE id = %(id)s;
                        """,
                        {"id": doc_id_str},
                    )
                    doc_row = cur.fetchone()
                    if doc_row is None:
                        return None

                    doc_record = DocumentRecord(
                        id=str(doc_row[0]),
                        board_id=str(doc_row[1]),
                        uploaded_by=str(doc_row[2]),
                        file_name=doc_row[3],
                        storage_key=doc_row[4],
                        mime_type=doc_row[5],
                        file_size=doc_row[6],
                        checksum=doc_row[7],
                        status=doc_row[8],
                        processing_token=str(doc_row[9]) if doc_row[9] else None,
                        metadata=doc_row[10] or {},
                        created_at=doc_row[11],
                        updated_at=doc_row[12],
                    )

                    cur.execute(
                        """
                        SELECT id, document_id, chunk_index, content, page_number,
                               start_offset, end_offset, embedding, metadata, created_at
                        FROM document_chunks
                        WHERE document_id = %(id)s
                        ORDER BY chunk_index ASC;
                        """,
                        {"id": doc_id_str},
                    )
                    chunk_rows = cur.fetchall()
                    stored_chunks = [
                        StoredChunk(
                            id=str(r[0]),
                            document_id=str(r[1]),
                            chunk_index=r[2],
                            content=r[3],
                            page_number=r[4],
                            start_offset=r[5],
                            end_offset=r[6],
                            embedding=_to_float_list(r[7]),
                            metadata=r[8] or {},
                            created_at=r[9],
                        )
                        for r in chunk_rows
                    ]

                    return StoredDocument(document=doc_record, chunks=stored_chunks)
        finally:
            conn.close()
