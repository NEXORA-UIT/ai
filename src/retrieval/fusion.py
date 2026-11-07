"""Deterministic Reciprocal Rank Fusion for dense and sparse candidates."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.retrieval.config import DEFAULT_RRF_K
from src.retrieval.schemas import (
    FusedChunk,
    HybridCandidates,
    RetrievedChunk,
    SourceChunk,
    SparseHit,
)


@dataclass
class _FusedAccumulator:
    source: SourceChunk
    score: float = 0.0
    channels: list[str] = field(default_factory=list)


def _validate_limits(rrf_k: int, top_k: int | None) -> None:
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 1:
        raise ValueError("rrf_k must be a positive integer")
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1
    ):
        raise ValueError("top_k must be a positive integer or None")


def _source_for_channel(
    channel: str, candidate: RetrievedChunk | SparseHit
) -> SourceChunk:
    if channel == "dense":
        if not isinstance(candidate, RetrievedChunk):
            raise TypeError("dense candidates must be RetrievedChunk instances")
        return SourceChunk.from_dense(candidate)
    if not isinstance(candidate, SparseHit):
        raise TypeError("sparse candidates must be SparseHit instances")
    return candidate.source


def fuse_rrf(
    candidates: HybridCandidates,
    *,
    rrf_k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
) -> list[FusedChunk]:
    """Fuse one-based channel ranks, deduplicated by chunk ID."""
    if not isinstance(candidates, HybridCandidates):
        raise TypeError("candidates must be HybridCandidates")
    _validate_limits(rrf_k, top_k)

    accumulated: dict[str, _FusedAccumulator] = {}
    for channel, channel_hits in (
        ("dense", candidates.dense),
        ("sparse", candidates.sparse),
    ):
        seen_in_channel: set[str] = set()
        for rank, candidate in enumerate(channel_hits, start=1):
            source = _source_for_channel(channel, candidate)
            chunk_id = source.chunk_id
            existing = accumulated.get(chunk_id)
            if (
                existing is not None
                and source.document_id != existing.source.document_id
            ):
                raise ValueError(
                    f"chunk_id {chunk_id} has conflicting document IDs"
                )
            if chunk_id in seen_in_channel:
                continue
            seen_in_channel.add(chunk_id)

            contribution = 1.0 / (rrf_k + rank)
            if existing is None:
                accumulated[chunk_id] = _FusedAccumulator(
                    source=source,
                    score=contribution,
                    channels=[channel],
                )
            else:
                existing.score += contribution
                if channel not in existing.channels:
                    existing.channels.append(channel)

    ranked = sorted(
        accumulated.values(),
        key=lambda item: (-item.score, item.source.chunk_id),
    )
    if top_k is not None:
        ranked = ranked[:top_k]
    return [
        FusedChunk(
            source=item.source,
            rrf_score=item.score,
            channels=tuple(item.channels),
        )
        for item in ranked
    ]
