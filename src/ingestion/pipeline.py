"""High-level Python callable interface for document extraction and cleaning."""

import hashlib
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from src.ingestion.chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_document,
)
from src.ingestion.cleaner import clean_document
from src.ingestion.file_validator import validate_file
from src.ingestion.schemas import (
    ChunkedDocument,
    CleanedDocument,
    EmbeddedDocument,
    ExtractionResult,
    FileValidationResult,
)
from src.ingestion.storage import DocumentStorage, sanitize_error_message
from src.ingestion.tika_client import (
    DEFAULT_MAX_FILE_SIZE_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    TikaClient,
)
from src.ingestion.tokenizer import TokenCounter

if TYPE_CHECKING:
    from src.embedding.embedder import Embedder


def extract_and_clean(
    file_path: Path | str,
    tika_endpoint: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> tuple[ExtractionResult, CleanedDocument]:
    """Execute complete extraction and text cleaning on a local file.

    Args:
        file_path: Path to target file (.txt, .pdf, .docx).
        tika_endpoint: Optional override for Apache Tika endpoint.
        timeout: HTTP timeout in seconds.
        max_file_size_bytes: Maximum allowed file size limit.

    Returns:
        tuple[ExtractionResult, CleanedDocument]:
            - ExtractionResult: Raw extracted text and metadata directly from Tika.
            - CleanedDocument: Cleaned, normalized text with verified page spans.

    Raises:
        FileValidationError: If file fails validation.
        TikaError: If extraction fails.
        EmptyDocumentError: If document contains no readable text after cleaning.
    """
    client = TikaClient(
        endpoint=tika_endpoint,
        timeout=timeout,
        max_file_size_bytes=max_file_size_bytes,
    )
    extraction = client.extract_file(file_path)
    cleaned = clean_document(extraction)
    return extraction, cleaned


def extract_clean_and_chunk(
    file_path: Path | str,
    token_counter: TokenCounter | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    tika_endpoint: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> tuple[ExtractionResult, CleanedDocument, ChunkedDocument]:
    """Execute complete extraction, text cleaning, and chunking on a local file.

    Args:
        file_path: Path to target file (.txt, .pdf, .docx).
        token_counter: TokenCounter instance (defaults to BgeM3Tokenizer).
        chunk_size: Target maximum tokens per chunk (default 512).
        chunk_overlap: Overlap tokens between adjacent chunks (default 64).
        tika_endpoint: Optional override for Apache Tika endpoint.
        timeout: HTTP timeout in seconds.
        max_file_size_bytes: Maximum allowed file size limit.

    Returns:
        tuple[ExtractionResult, CleanedDocument, ChunkedDocument]:
            - ExtractionResult: Raw output from Tika.
            - CleanedDocument: Cleaned text and metadata.
            - ChunkedDocument: Ordered chunks with verified offsets and token counts.
    """
    extraction, cleaned = extract_and_clean(
        file_path=file_path,
        tika_endpoint=tika_endpoint,
        timeout=timeout,
        max_file_size_bytes=max_file_size_bytes,
    )
    chunked = chunk_document(
        cleaned_doc=cleaned,
        token_counter=token_counter,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return extraction, cleaned, chunked


def extract_clean_chunk_and_embed(
    file_path: Path | str,
    embedder: "Embedder | None" = None,
    token_counter: TokenCounter | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    batch_size: int = 8,
    tika_endpoint: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> tuple[ExtractionResult, CleanedDocument, ChunkedDocument, EmbeddedDocument]:
    """Execute complete extraction, cleaning, chunking, and embedding on a local file.

    Args:
        file_path: Path to target file (.txt, .pdf, .docx).
        embedder: Optional Embedder instance (defaults to lazy BgeM3Embedder on CPU).
        token_counter: TokenCounter instance (defaults to BgeM3Tokenizer).
        chunk_size: Target maximum tokens per chunk (default 512).
        chunk_overlap: Overlap tokens between adjacent chunks (default 64).
        batch_size: Inference batch size for embedding (default 8).
        tika_endpoint: Optional override for Apache Tika endpoint.
        timeout: HTTP timeout in seconds.
        max_file_size_bytes: Maximum allowed file size limit.

    Returns:
        tuple[ExtractionResult, CleanedDocument, ChunkedDocument, EmbeddedDocument]:
            All intermediate and final pipeline stage artifacts.
    """
    extraction, cleaned, chunked = extract_clean_and_chunk(
        file_path=file_path,
        token_counter=token_counter,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tika_endpoint=tika_endpoint,
        timeout=timeout,
        max_file_size_bytes=max_file_size_bytes,
    )

    if embedder is None:
        from src.embedding.embedder import BgeM3Embedder

        embedder = BgeM3Embedder(device="cpu", batch_size=batch_size)

    embedded_chunks = embedder.embed_chunks(chunked.chunks, batch_size=batch_size)
    embedded_doc = EmbeddedDocument(
        document_name=chunked.document_name,
        total_chunks=len(embedded_chunks),
        total_tokens=chunked.total_tokens,
        chunks=embedded_chunks,
        model_name=embedder.model_name,
        model_revision=embedder.revision or "",
        dimension=embedder.dimension,
        normalized=True,
        chunking_version=chunked.chunking_version,
        cleaning_version=cleaned.cleaning_version,
    )

    return extraction, cleaned, chunked, embedded_doc


def ingest_file_to_storage(
    file_path: Path | str,
    board_id: str | uuid.UUID,
    uploaded_by: str | uuid.UUID,
    storage: DocumentStorage | None = None,
    document_id: str | uuid.UUID | None = None,
    reclaim: bool = False,
    embedder: "Embedder | None" = None,
    token_counter: TokenCounter | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    batch_size: int = 8,
    tika_endpoint: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> tuple[str, EmbeddedDocument]:
    """Execute complete ingestion pipeline and persist to PostgreSQL/pgvector.

    Follows strict 4-phase transaction lifecycle:
      Phase 1: Claim document in short transaction (returns processing_token)
      Phase 2: Extraction, cleaning, chunking, and embedding (OUTSIDE transaction)
      Phase 3: Verify token and atomically persist chunks + mark READY
      Phase 4: On error, mark document FAILED in isolated transaction if token owned.

    Args:
        file_path: Path to target file (.txt, .pdf, .docx).
        board_id: UUID of owning Board.
        uploaded_by: UUID of uploading User.
        storage: DocumentStorage instance (defaults to default DocumentStorage()).
        document_id: Optional UUID of existing/target document for reprocess.
        reclaim: Allow reclaiming ownership of a PROCESSING document.
        embedder: Optional Embedder instance.
        token_counter: Optional TokenCounter instance.
        chunk_size: Target content tokens per chunk.
        chunk_overlap: Overlap tokens between adjacent chunks.
        batch_size: Inference batch size.
        tika_endpoint: Optional override for Tika server endpoint.
        timeout: HTTP timeout in seconds.
        max_file_size_bytes: Maximum allowed file size limit.

    Returns:
        tuple[str, EmbeddedDocument]: (persisted_document_id, embedded_doc)
    """
    path, mime_type, file_size = validate_file(
        file_path, max_size_bytes=max_file_size_bytes
    )
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    file_validation = FileValidationResult(
        file_name=path.name,
        file_size_bytes=file_size,
        sha256=sha256,
        mime_type=mime_type,
    )
    storage_key = f"local://{path.resolve().as_posix()}"

    if storage is None:
        storage = DocumentStorage()

    # Phase 1: Claim/create document under short transaction
    doc_id, token = storage.claim_or_create_document(
        board_id=board_id,
        uploaded_by=uploaded_by,
        file_name=file_validation.file_name,
        storage_key=storage_key,
        mime_type=file_validation.mime_type,
        file_size=file_validation.file_size_bytes,
        checksum=file_validation.sha256,
        document_id=document_id,
        reclaim=reclaim,
    )

    # Phase 2: Heavy computation completely OUTSIDE database transaction
    try:
        _, _, _, embedded_doc = extract_clean_chunk_and_embed(
            file_path=path,
            embedder=embedder,
            token_counter=token_counter,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
            tika_endpoint=tika_endpoint,
            timeout=timeout,
            max_file_size_bytes=max_file_size_bytes,
        )
    except Exception as exc:
        try:
            storage.mark_document_failed(
                document_id=doc_id,
                processing_token=token,
                error_stage="extraction_or_embedding",
                error_type=type(exc).__name__,
                error_message=sanitize_error_message(exc),
            )
        except Exception:
            # Preserve original exception even if failure logging fails
            pass
        raise exc

    # Phase 3: Atomically persist chunks and transition to READY
    try:
        storage.save_embedded_document(
            document_id=doc_id,
            processing_token=token,
            embedded_doc=embedded_doc,
            file_validation=file_validation,
            storage_key=storage_key,
        )
    except Exception as exc:
        try:
            storage.mark_document_failed(
                document_id=doc_id,
                processing_token=token,
                error_stage="storage_persist",
                error_type=type(exc).__name__,
                error_message=sanitize_error_message(exc),
            )
        except Exception:
            pass
        raise exc

    return doc_id, embedded_doc
