"""Select final source chunks by cross-encoder relevance scores."""

from __future__ import annotations

import math
from collections.abc import Sequence

from src.reranking.cross_encoder import Reranker, get_default_reranker
from src.retrieval.schemas import FusedChunk, RerankedChunk

_MAX_CANDIDATES = 20
_MAX_TOP_K = 5


def _validate_limits(top_k: int, candidate_count: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 5:
        raise ValueError("top_k must be an integer from 1 to 5")
    if candidate_count > _MAX_CANDIDATES:
        raise ValueError("reranker accepts at most 20 fused candidates")


def rerank_chunks(
    question: str,
    candidates: Sequence[FusedChunk],
    *,
    top_k: int = 5,
    reranker: Reranker | None = None,
) -> list[RerankedChunk]:
    """Rescore unique fused chunks and return the highest ranked evidence."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must contain non-whitespace text")
    _validate_limits(top_k, len(candidates))

    unique_candidates: list[FusedChunk] = []
    source_by_id: dict[str, FusedChunk] = {}
    for candidate in candidates:
        if not isinstance(candidate, FusedChunk):
            raise TypeError("candidates must contain FusedChunk instances")
        chunk_id = candidate.source.chunk_id
        existing = source_by_id.get(chunk_id)
        if existing is not None:
            if candidate.source.document_id != existing.source.document_id:
                raise ValueError(
                    f"chunk_id {chunk_id} has conflicting document IDs"
                )
            continue
        source_by_id[chunk_id] = candidate
        unique_candidates.append(candidate)

    if not unique_candidates:
        return []

    selected_reranker = reranker if reranker is not None else get_default_reranker()
    raw_scores = selected_reranker.score_pairs(
        question,
        [candidate.source.content for candidate in unique_candidates],
    )

    try:
        scores = [float(score) for score in raw_scores]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("reranker returned non-numeric scores") from exc
    if len(scores) != len(unique_candidates):
        raise ValueError("reranker score count does not match candidate count")
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("reranker scores must all be finite")

    ranked = sorted(
        zip(unique_candidates, scores, strict=True),
        key=lambda item: (
            -item[1],
            -item[0].rrf_score,
            item[0].source.chunk_id,
        ),
    )
    return [
        RerankedChunk(
            source=candidate.source,
            rerank_score=score,
            rrf_score=candidate.rrf_score,
        )
        for candidate, score in ranked[:top_k]
    ]
