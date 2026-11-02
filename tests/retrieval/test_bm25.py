"""Unit tests for deterministic lexical tokenization and BM25 ranking."""

from __future__ import annotations

from importlib import import_module


def _source_chunk(chunk_id: str, content: str):
    schemas = import_module("src.retrieval.schemas")
    return schemas.SourceChunk(
        board_id="11111111-1111-1111-1111-111111111111",
        chunk_id=chunk_id,
        document_id=f"document-{chunk_id}",
        chunk_index=0,
        file_name=f"{chunk_id}.txt",
        storage_key=f"internal://{chunk_id}",
        content=content,
        page_number=None,
        start_offset=0,
        end_offset=len(content),
    )


def test_tokenize_lexical_keeps_full_technical_identifier_and_unicode_words() -> None:
    tokenize = import_module("src.retrieval.bm25").tokenize_lexical

    tokens = tokenize("UC-KB-03 supports Tiếng Việt")

    assert tokens == ["uc-kb-03", "uc", "kb", "03", "supports", "tiếng", "việt"]


def test_source_chunk_from_dense_copies_board_scoped_source_fields() -> None:
    schemas = import_module("src.retrieval.schemas")
    dense_hit = schemas.RetrievedChunk(
        chunk_id="chunk",
        document_id="document",
        board_id="board",
        chunk_index=2,
        file_name="guide.pdf",
        storage_key="internal://guide",
        content="source text",
        page_number=3,
        start_offset=10,
        end_offset=21,
        cosine_distance=0.25,
    )

    source = schemas.SourceChunk.from_dense(dense_hit)

    assert source == schemas.SourceChunk(
        board_id="board",
        chunk_id="chunk",
        document_id="document",
        chunk_index=2,
        file_name="guide.pdf",
        storage_key="internal://guide",
        content="source text",
        page_number=3,
        start_offset=10,
        end_offset=21,
    )


def test_rank_sparse_prefers_exact_identifier_when_bm25_idf_is_zero() -> None:
    rank_sparse = import_module("src.retrieval.bm25").rank_sparse
    near_identifier = _source_chunk(
        "a-near", "Procedure UC-KB-04 contains a related technical identifier"
    )
    exact_identifier = _source_chunk(
        "z-exact", "Procedure UC-KB-03 contains the requested technical identifier"
    )

    hits = rank_sparse(
        "UC-KB-03", [near_identifier, exact_identifier], top_k=2
    )

    assert [hit.source.chunk_id for hit in hits] == ["z-exact", "a-near"]
    # Both documents have the same score for the query's component tokens;
    # exact full-identifier matching must break the tie before chunk_id.
    assert hits[0].sparse_score == hits[1].sparse_score


def test_rank_sparse_excludes_chunks_without_query_token_overlap() -> None:
    rank_sparse = import_module("src.retrieval.bm25").rank_sparse
    matching = _source_chunk("match", "A cobalt lighthouse guides the ship")
    unrelated = _source_chunk("unrelated", "A maple tree grows by the road")

    hits = rank_sparse("lighthouse", [unrelated, matching], top_k=2)

    assert [hit.source.chunk_id for hit in hits] == ["match"]


def test_rank_sparse_keeps_zero_idf_matches_and_stable_ties() -> None:
    rank_sparse = import_module("src.retrieval.bm25").rank_sparse
    first = _source_chunk("b", "shared token appears here")
    second = _source_chunk("a", "shared token appears here")
    unmatched_one = _source_chunk("c", "unrelated words over there")
    unmatched_two = _source_chunk("d", "another unrelated passage")
    hits = rank_sparse(
        "shared", [first, unmatched_one, second, unmatched_two], top_k=5
    )

    assert [hit.source.chunk_id for hit in hits] == ["a", "b"]
    assert all(hit.sparse_score == 0.0 for hit in hits)


def test_rank_sparse_empty_query_or_corpus_returns_empty() -> None:
    rank_sparse = import_module("src.retrieval.bm25").rank_sparse

    assert rank_sparse("... ---", [_source_chunk("a", "text")], top_k=5) == []
    assert rank_sparse("text", [], top_k=5) == []
    assert rank_sparse("text", [_source_chunk("empty", "... ---")], top_k=5) == []
