"""Regression tests for chunking edge cases and reported findings."""

import pytest

from src.ingestion.chunker import chunk_document
from src.ingestion.exceptions import ChunkingError
from src.ingestion.schemas import CleanedDocument, PageSpan
from src.ingestion.tokenizer import WhitespaceTokenCounter


class SingleCharExpensiveTokenCounter:
    """Mock counter where every character produces 10 tokens."""

    @property
    def name(self) -> str:
        return "expensive_char"

    @property
    def revision(self) -> str | None:
        return None

    def count_tokens(self, text: str, add_special_tokens: bool = False) -> int:
        return len(text) * 10

    def encode_ids(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text) * 10))


class DeterministicSequenceTokenCounter:
    """Mock counter mapping specific words to token weights."""

    def __init__(self, weights: dict[str, int]) -> None:
        self.weights = weights

    @property
    def name(self) -> str:
        return "deterministic_seq"

    @property
    def revision(self) -> str | None:
        return None

    def count_tokens(self, text: str, add_special_tokens: bool = False) -> int:
        tokens = 0
        for word in text.split():
            tokens += self.weights.get(word, 1)
        return tokens

    def encode_ids(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(self.count_tokens(text, add_special_tokens)))


class TestChunkerRegressions:
    def test_regression_single_char_exceeding_budget_raises_clear_error(self):
        """Single char > max_tokens raises ChunkingError instead of infinite loop."""
        doc = CleanedDocument(
            cleaned_text="X",
            file_name="single_char.txt",
            mime_type="text/plain",
        )
        counter = SingleCharExpensiveTokenCounter()  # "X" has 10 tokens

        with pytest.raises(
            ChunkingError, match=r"Cannot split text segment .* to fit token budget"
        ):
            # Budget is 5 tokens, but "X" alone takes 10 tokens
            chunk_document(doc, token_counter=counter, chunk_size=5, chunk_overlap=1)

    def test_regression_overlap_must_not_produce_subset_chunk(self):
        """Overlap must not emit a chunk that is a pure subset of the prior chunk."""
        # A has 5 tokens, B has 4 tokens, C has 7 tokens
        # chunk_size = 10, chunk_overlap = 4
        # Chunk 1: A + B (9 tokens <= 10)
        # B is candidate overlap (4 tokens <= 4).
        # B + C (4 + 7 = 11 > 10) cannot fit together.
        # Must NOT emit Chunk 2 with only [B]! Overlap must be dropped to take C!
        weights = {"AAA": 5, "BBB": 4, "CCC": 7}
        counter = DeterministicSequenceTokenCounter(weights)
        text = "AAA\n\nBBB\n\nCCC"
        doc = CleanedDocument(
            cleaned_text=text,
            file_name="overlap_test.txt",
            mime_type="text/plain",
        )

        result = chunk_document(
            doc, token_counter=counter, chunk_size=10, chunk_overlap=4
        )

        # Verify no chunk is a subset of the previous chunk
        for i in range(1, len(result.chunks)):
            prev = result.chunks[i - 1]
            curr = result.chunks[i]
            # Every subsequent chunk must strictly advance end_offset
            assert curr.end_offset > prev.end_offset, (
                f"Chunk #{curr.chunk_index} end_offset ({curr.end_offset}) "
                f"did not advance beyond Chunk #{prev.chunk_index} ({prev.end_offset})"
            )
            # The current chunk must not be a subset of the previous chunk
            assert not (
                curr.start_offset >= prev.start_offset
                and curr.end_offset <= prev.end_offset
            ), f"Chunk #{curr.chunk_index} is a subset of Chunk #{prev.chunk_index}"

    def test_regression_page_mapping_returns_none_for_gaps_without_fallback(
        self,
    ):
        """Offsets outside valid spans must return None, not guess the last page."""
        text = "Page 1 text.\n\nGap text.\n\nPage 2 text."
        p2_start = text.index("Page 2")
        spans = [
            PageSpan(page_number=1, start_offset=0, end_offset=len("Page 1 text.")),
            PageSpan(page_number=2, start_offset=p2_start, end_offset=len(text)),
        ]
        doc = CleanedDocument(
            cleaned_text=text,
            file_name="gaps.pdf",
            mime_type="application/pdf",
            page_spans=spans,
        )
        counter = WhitespaceTokenCounter()
        # Create chunk specifically starting in the gap
        result = chunk_document(
            doc, token_counter=counter, chunk_size=3, chunk_overlap=0
        )

        # Find the chunk covering "Gap text."
        gap_chunk = next((c for c in result.chunks if "Gap" in c.content), None)
        assert gap_chunk is not None
        # Gap is not inside Page 1 [0, 12) or Page 2 [25, 37), so must be None
        assert gap_chunk.page_number is None

    def test_regression_whitespace_token_counter_consistency(self):
        """WhitespaceTokenCounter count_tokens and len(encode_ids) must match."""
        counter = WhitespaceTokenCounter()
        text = "Một hai ba bốn năm"

        count_raw = counter.count_tokens(text, add_special_tokens=False)
        ids_raw = counter.encode_ids(text, add_special_tokens=False)
        assert count_raw == len(ids_raw) == 5

        count_special = counter.count_tokens(text, add_special_tokens=True)
        ids_special = counter.encode_ids(text, add_special_tokens=True)
        assert count_special == len(ids_special) == 7

    def test_long_continuous_text_without_spaces_splits_correctly(self):
        """Unbroken long string splits by character midpoint without losing content."""
        unbroken_text = "ABCDEFGHIJ" * 20  # 200 chars
        doc = CleanedDocument(
            cleaned_text=unbroken_text,
            file_name="unbroken.txt",
            mime_type="text/plain",
        )
        # chunk_size = 100 tokens (10 chars per chunk) using mock counter
        result = chunk_document(
            doc,
            token_counter=SingleCharExpensiveTokenCounter(),
            chunk_size=100,
            chunk_overlap=20,
        )

        assert result.total_chunks > 1
        for chunk in result.chunks:
            assert chunk.token_count <= 100
            assert unbroken_text[chunk.start_offset : chunk.end_offset] == chunk.content
        assert result.chunks[0].start_offset == 0
        assert result.chunks[-1].end_offset == len(unbroken_text)

    def test_overlap_zero_with_large_segment(self):
        """Chunk overlap = 0 and segment nearly filling budget."""
        p1 = "w " * 8 + "w"  # 9 words
        p2 = "w " * 8 + "w"  # 9 words
        p3 = "w " * 8 + "w"  # 9 words
        text = f"{p1}\n\n{p2}\n\n{p3}"
        doc = CleanedDocument(
            cleaned_text=text,
            file_name="large_segs.txt",
            mime_type="text/plain",
        )
        counter = WhitespaceTokenCounter()
        result = chunk_document(
            doc, token_counter=counter, chunk_size=10, chunk_overlap=0
        )

        assert result.total_chunks == 3
        for i in range(1, len(result.chunks)):
            assert result.chunks[i].end_offset > result.chunks[i - 1].end_offset
            assert result.chunks[i].start_offset >= result.chunks[i - 1].end_offset

    def test_chunk_crossing_multiple_pages_assigns_start_page(self):
        """Chunk crossing Page 1 and Page 2 is assigned Page 1 (its starting page)."""
        text = "Start on Page 1.\n\nContinuation on Page 2."
        p2_start = text.index("Continuation")
        spans = [
            PageSpan(page_number=1, start_offset=0, end_offset=p2_start),
            PageSpan(page_number=2, start_offset=p2_start, end_offset=len(text)),
        ]
        doc = CleanedDocument(
            cleaned_text=text,
            file_name="crossing.pdf",
            mime_type="application/pdf",
            page_spans=spans,
        )
        counter = WhitespaceTokenCounter()
        result = chunk_document(
            doc, token_counter=counter, chunk_size=20, chunk_overlap=0
        )

        assert result.total_chunks == 1
        assert result.chunks[0].page_number == 1
