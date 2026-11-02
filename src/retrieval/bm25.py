"""Deterministic tokenization and CPU BM25 ranking for retrieval chunks."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from src.retrieval.schemas import SourceChunk, SparseHit

_TOKEN_PATTERN = re.compile(r"[\w]+(?:-[\w]+)+|[\w]+", re.UNICODE)
_WORD_PATTERN = re.compile(r"[\w]+", re.UNICODE)


def tokenize_lexical(text: str) -> list[str]:
    """Normalize text and retain both full hyphenated IDs and their parts."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")

    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(normalized):
        token = match.group()
        if "-" in token:
            tokens.append(token)
            tokens.extend(_WORD_PATTERN.findall(token))
        else:
            tokens.append(token)
    return tokens


def _validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")


def rank_sparse(
    question: str,
    corpus: Sequence[SourceChunk],
    *,
    top_k: int = 20,
) -> list[SparseHit]:
    """Rank token-overlapping chunks with BM25 and exact-ID tie breaking."""
    _validate_top_k(top_k)
    if not isinstance(question, str):
        raise ValueError("question must be a string")

    query_tokens = tokenize_lexical(question)
    if not query_tokens or not corpus:
        return []

    tokenized_chunks = [
        (source, tokenize_lexical(source.content))
        for source in corpus
    ]
    tokenized_chunks = [item for item in tokenized_chunks if item[1]]
    if not tokenized_chunks:
        return []

    query_terms = set(query_tokens)
    indexed_chunks = [
        (source, tokens)
        for source, tokens in tokenized_chunks
        if query_terms.intersection(tokens)
    ]
    if not indexed_chunks:
        return []

    bm25 = BM25Okapi([tokens for _, tokens in tokenized_chunks])
    scores = bm25.get_scores(query_tokens)
    score_by_chunk_id = {
        source.chunk_id: float(scores[index])
        for index, (source, _) in enumerate(tokenized_chunks)
    }

    query_identifiers = Counter(
        token for token in query_tokens if "-" in token
    )

    def exact_identifier_matches(tokens: list[str]) -> int:
        chunk_identifiers = Counter(token for token in tokens if "-" in token)
        return sum(
            min(count, chunk_identifiers[identifier])
            for identifier, count in query_identifiers.items()
        )

    ranked = sorted(
        indexed_chunks,
        key=lambda item: (
            -exact_identifier_matches(item[1]),
            -score_by_chunk_id[item[0].chunk_id],
            item[0].chunk_id,
        ),
    )
    return [
        SparseHit(
            source=source,
            sparse_score=score_by_chunk_id[source.chunk_id],
        )
        for source, _ in ranked[:top_k]
    ]
