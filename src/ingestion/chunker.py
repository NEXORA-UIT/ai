"""Recursive text chunking engine with BGE-M3 tokenizer awareness."""

from typing import Any

from src.ingestion.exceptions import ChunkingError
from src.ingestion.schemas import Chunk, ChunkedDocument, CleanedDocument, PageSpan
from src.ingestion.tokenizer import BgeM3Tokenizer, TokenCounter

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
CHUNKING_VERSION = "1.0.0"

# Separator priority hierarchy
SEPARATORS = ["\n\n", "\n", ". ", " "]


def chunk_document(
    cleaned_doc: CleanedDocument,
    token_counter: TokenCounter | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    chunking_version: str = CHUNKING_VERSION,
) -> ChunkedDocument:
    """Split a CleanedDocument into semantic chunks using recursive splitting.

    Args:
        cleaned_doc: Validated CleanedDocument instance.
        token_counter: TokenCounter implementation (defaults to BgeM3Tokenizer).
        chunk_size: Maximum token count per chunk (default 512).
        chunk_overlap: Target overlap token count between adjacent chunks (default 64).
        chunking_version: Version identifier for the chunking logic.

    Returns:
        ChunkedDocument with ordered Chunk objects and metadata.

    Raises:
        ValueError: If chunk_size < 1, chunk_overlap < 0, or overlap >= size.
        ChunkingError: If a character/segment cannot be split to fit chunk_size.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
        )

    if token_counter is None:
        token_counter = BgeM3Tokenizer()

    text = cleaned_doc.cleaned_text
    tokenizer_name = token_counter.name
    tokenizer_revision = getattr(token_counter, "revision", None)

    config: dict[str, Any] = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "tokenizer_name": tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "chunking_version": chunking_version,
    }

    if not text or text.isspace():
        return ChunkedDocument(
            document_name=cleaned_doc.file_name,
            total_chunks=0,
            total_tokens=0,
            chunks=[],
            chunking_version=chunking_version,
            config=config,
        )

    # Find atomic segments that each fit within chunk_size tokens
    segments = _find_atomic_segments(
        text=text,
        start=0,
        end=len(text),
        token_counter=token_counter,
        max_tokens=chunk_size,
        sep_idx=0,
    )

    if not segments:
        return ChunkedDocument(
            document_name=cleaned_doc.file_name,
            total_chunks=0,
            total_tokens=0,
            chunks=[],
            chunking_version=chunking_version,
            config=config,
        )

    # Assemble segments into chunks with overlap
    chunk_spans = _assemble_chunks_with_overlap(
        text=text,
        segments=segments,
        token_counter=token_counter,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[Chunk] = []
    total_tokens = 0

    for idx, (start_offset, end_offset) in enumerate(chunk_spans):
        content = text[start_offset:end_offset]
        token_count = token_counter.count_tokens(content)

        if token_count > chunk_size:
            raise ChunkingError(
                f"Chunk #{idx} token count ({token_count}) exceeded "
                f"chunk_size ({chunk_size})."
            )

        total_tokens += token_count

        page_number = _resolve_page_number(
            start_offset=start_offset, page_spans=cleaned_doc.page_spans
        )

        chunk_meta = dict(cleaned_doc.metadata)
        chunk_meta.update(
            {
                "file_name": cleaned_doc.file_name,
                "mime_type": cleaned_doc.mime_type,
                "cleaning_version": cleaned_doc.cleaning_version,
                "tokenizer_name": tokenizer_name,
                "tokenizer_revision": tokenizer_revision,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "chunking_version": chunking_version,
            }
        )

        chunks.append(
            Chunk(
                chunk_index=idx,
                content=content,
                token_count=token_count,
                start_offset=start_offset,
                end_offset=end_offset,
                page_number=page_number,
                metadata=chunk_meta,
            )
        )

    return ChunkedDocument(
        document_name=cleaned_doc.file_name,
        total_chunks=len(chunks),
        total_tokens=total_tokens,
        chunks=chunks,
        chunking_version=chunking_version,
        config=config,
    )


def _find_atomic_segments(
    text: str,
    start: int,
    end: int,
    token_counter: TokenCounter,
    max_tokens: int,
    sep_idx: int = 0,
) -> list[tuple[int, int]]:
    """Recursively split text into non-overlapping atomic segments <= max_tokens."""
    sub = text[start:end]
    if not sub or sub.isspace():
        return []

    tokens = token_counter.count_tokens(sub)
    if tokens <= max_tokens:
        l_offset = len(sub) - len(sub.lstrip())
        r_offset = len(sub) - len(sub.rstrip())
        seg_start = start + l_offset
        seg_end = end - r_offset
        if seg_end > seg_start:
            return [(seg_start, seg_end)]
        return []

    if sep_idx >= len(SEPARATORS):
        # When all separators are exhausted, split by character midpoint
        if end - start <= 1:
            raise ChunkingError(
                f"Cannot split text segment '{sub}' ({len(sub)} chars) to fit "
                f"token budget ({tokens} tokens > max {max_tokens} tokens)."
            )

        mid = (start + end) // 2
        if mid == start or mid == end:
            raise ChunkingError(
                f"Cannot split text segment '{sub}' ({len(sub)} chars) to fit "
                f"token budget ({tokens} tokens > max {max_tokens} tokens)."
            )

        left = _find_atomic_segments(
            text, start, mid, token_counter, max_tokens, sep_idx
        )
        right = _find_atomic_segments(
            text, mid, end, token_counter, max_tokens, sep_idx
        )
        return left + right

    sep = SEPARATORS[sep_idx]
    if sep not in sub:
        return _find_atomic_segments(
            text, start, end, token_counter, max_tokens, sep_idx + 1
        )

    segments: list[tuple[int, int]] = []
    curr = start
    while curr < end:
        next_pos = text.find(sep, curr, end)
        if next_pos == -1:
            part_end = end
            step = end - curr
        else:
            if sep == ". ":
                part_end = next_pos + 1
                step = (next_pos - curr) + 2
            else:
                part_end = next_pos
                step = (next_pos - curr) + len(sep)

        if part_end > curr:
            segments.extend(
                _find_atomic_segments(
                    text, curr, part_end, token_counter, max_tokens, sep_idx + 1
                )
            )

        if next_pos == -1:
            break
        curr = curr + step

    return segments


def _assemble_chunks_with_overlap(
    text: str,
    segments: list[tuple[int, int]],
    token_counter: TokenCounter,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[int, int]]:
    """Group atomic segments into chunks respecting chunk_size and chunk_overlap.

    Guarantees that every subsequent chunk introduces new content and strictly
    advances end_offset beyond the previous chunk. Overlap is reduced or dropped
    if keeping it would prevent adding new segments.
    """
    chunks: list[tuple[int, int]] = []
    n = len(segments)
    prev_end_seg = 0
    start_seg = 0

    while start_seg < n:
        end_seg = start_seg
        while end_seg < n:
            candidate_start = segments[start_seg][0]
            candidate_end = segments[end_seg][1]
            candidate_text = text[candidate_start:candidate_end]
            if token_counter.count_tokens(candidate_text) <= chunk_size:
                end_seg += 1
            else:
                break

        if end_seg == start_seg:
            end_seg = start_seg + 1

        # If keeping the chosen overlap resulted in no new segment being added,
        # drop overlap completely and start directly at the first un-emitted segment
        if chunks and end_seg <= prev_end_seg:
            start_seg = prev_end_seg
            continue

        chunk_start = segments[start_seg][0]
        chunk_end = segments[end_seg - 1][1]
        chunks.append((chunk_start, chunk_end))
        prev_end_seg = end_seg

        if end_seg >= n:
            break

        if chunk_overlap <= 0:
            start_seg = end_seg
        else:
            # Backtrack from end_seg - 1 to find overlap segments fitting chunk_overlap
            cand = end_seg - 1
            while cand > start_seg:
                overlap_text = text[segments[cand][0] : chunk_end]
                if token_counter.count_tokens(overlap_text) <= chunk_overlap:
                    cand -= 1
                else:
                    cand += 1
                    break

            # Crucial check: verify that cand + segments[end_seg] <= chunk_size.
            # If not, keeping cand as overlap would cause the next chunk to be
            # unable to add the new segment segments[end_seg].
            # Advance cand (reducing overlap) until segments[cand..end_seg] fits
            # or cand reaches end_seg (dropping overlap entirely).
            while cand < end_seg:
                combined_text = text[segments[cand][0] : segments[end_seg][1]]
                if token_counter.count_tokens(combined_text) <= chunk_size:
                    break
                cand += 1

            start_seg = cand

    return chunks


def _resolve_page_number(
    start_offset: int, page_spans: list[PageSpan] | None
) -> int | None:
    """Find 1-indexed page number containing start_offset, or None if unpaginated.

    Never guesses or falls back to the last page for offsets outside known page spans.
    """
    if not page_spans:
        return None

    for span in page_spans:
        if span.start_offset <= start_offset < span.end_offset:
            return span.page_number

    return None
