"""Schemas and data models for document extraction and cleaning."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PageSpan:
    """Character offset span for a single page in cleaned text.

    Attributes:
        page_number: 1-indexed page number.
        start_offset: Inclusive character start index on cleaned_text.
        end_offset: Exclusive character end index on cleaned_text.
    """

    page_number: int
    start_offset: int
    end_offset: int

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError(f"page_number must be >= 1, got {self.page_number}")
        if self.start_offset < 0:
            raise ValueError(f"start_offset must be >= 0, got {self.start_offset}")
        if self.end_offset < self.start_offset:
            raise ValueError(
                f"end_offset ({self.end_offset}) cannot be less than "
                f"start_offset ({self.start_offset})"
            )


@dataclass(frozen=True)
class ExtractionResult:
    """Raw output from document extractor (e.g. Apache Tika).

    Attributes:
        raw_text: Uncleaned text directly extracted from the file.
        file_name: Original file name.
        mime_type: Detected or declared MIME type.
        metadata: Source metadata dictionary extracted from the document.
        page_count: Declared or detected page count, if available.
        has_form_feeds: True if Form Feed (\\x0c) page separator was detected.
    """

    raw_text: str
    file_name: str
    mime_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    page_count: int | None = None
    has_form_feeds: bool = False


@dataclass(frozen=True)
class CleanedDocument:
    """Processed document ready for chunking.

    Attributes:
        cleaned_text: Normalized, cleaned text.
        file_name: Original file name.
        mime_type: Detected MIME type.
        metadata: Preserved document metadata.
        cleaning_version: Semantic version of the cleaning algorithm used.
        page_spans: Optional list of verified page spans on cleaned_text.
            None if page boundaries cannot be reliably verified.
        char_count: Total character length of cleaned_text.
        line_count: Total line count of cleaned_text.
    """

    cleaned_text: str
    file_name: str
    mime_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    cleaning_version: str = "1.0.0"
    page_spans: list[PageSpan] | None = None
    char_count: int = field(init=False)
    line_count: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "char_count", len(self.cleaned_text))
        object.__setattr__(
            self,
            "line_count",
            len(self.cleaned_text.splitlines()) if self.cleaned_text else 0,
        )


@dataclass(frozen=True)
class Chunk:
    """A semantic text chunk ready for embedding and retrieval.

    Attributes:
        chunk_index: 0-indexed sequence order of this chunk in the document.
        content: Text content of the chunk.
        token_count: Number of tokens measured by the designated tokenizer.
        start_offset: Inclusive character start offset on cleaned_text.
        end_offset: Exclusive character end offset on cleaned_text [start, end).
        page_number: 1-indexed source page number if verifiable, else None.
        metadata: Chunk and source metadata dictionary.
        char_count: Length of chunk content string (auto-computed).
    """

    chunk_index: int
    content: str
    token_count: int
    start_offset: int
    end_offset: int
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        if self.chunk_index < 0:
            raise ValueError(f"chunk_index must be >= 0, got {self.chunk_index}")
        if self.start_offset < 0:
            raise ValueError(f"start_offset must be >= 0, got {self.start_offset}")
        if self.end_offset < self.start_offset:
            raise ValueError(
                f"end_offset ({self.end_offset}) cannot be less than "
                f"start_offset ({self.start_offset})"
            )
        object.__setattr__(self, "char_count", len(self.content))


@dataclass(frozen=True)
class ChunkedDocument:
    """Complete collection of chunks produced from a cleaned document.

    Attributes:
        document_name: Original file/document name.
        total_chunks: Total number of chunks.
        total_tokens: Sum of tokens across all chunks.
        chunks: Ordered list of Chunk objects.
        chunking_version: Semantic version of chunking algorithm.
        config: Chunking configuration parameters used.
    """

    document_name: str
    total_chunks: int
    total_tokens: int
    chunks: list[Chunk]
    chunking_version: str = "1.0.0"
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddedChunk:
    """A semantic text chunk paired with its dense vector embedding.

    Attributes:
        chunk: The underlying Chunk instance.
        dense_embedding: Dense vector embedding (1024 float elements, L2 normalized).
    """

    chunk: Chunk
    dense_embedding: list[float]

    def __post_init__(self) -> None:
        if len(self.dense_embedding) != 1024:
            raise ValueError(
                f"dense_embedding dimension must be 1024, "
                f"got {len(self.dense_embedding)}"
            )


@dataclass(frozen=True)
class EmbeddedDocument:
    """Complete collection of embedded chunks produced from a document.

    Attributes:
        document_name: Original file/document name.
        total_chunks: Number of chunks embedded.
        total_tokens: Sum of tokens across all chunks.
        chunks: Ordered list of EmbeddedChunk objects.
        model_name: Identifier of the embedding model used.
        model_revision: Revision commit hash of the model.
        dimension: Embedding dimensionality (default: 1024).
        normalized: Whether embeddings are L2 normalized (default: True).
        chunking_version: Semantic version of chunking algorithm used.
        cleaning_version: Semantic version of cleaning algorithm used.
    """

    document_name: str
    total_chunks: int
    total_tokens: int
    chunks: list[EmbeddedChunk]
    model_name: str
    model_revision: str
    dimension: int = 1024
    normalized: bool = True
    chunking_version: str = "1.0.0"
    cleaning_version: str = "1.0.0"
