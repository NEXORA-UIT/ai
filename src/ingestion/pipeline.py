"""High-level Python callable interface for document extraction and cleaning."""

from pathlib import Path
from typing import TYPE_CHECKING

from src.ingestion.chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_document,
)
from src.ingestion.cleaner import clean_document
from src.ingestion.schemas import (
    ChunkedDocument,
    CleanedDocument,
    EmbeddedDocument,
    ExtractionResult,
)
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
