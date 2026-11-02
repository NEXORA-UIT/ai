"""Dense and semantic retrieval operations."""

from src.retrieval.dense import get_default_embedder, search_by_text, search_by_vector
from src.retrieval.hybrid import search_hybrid_candidates
from src.retrieval.metadata import EmbeddingFingerprint
from src.retrieval.schemas import (
    HybridCandidates,
    RetrievedChunk,
    SourceChunk,
    SparseHit,
)
from src.retrieval.sparse import CorpusTooLargeError, search_sparse

__all__ = [
    "RetrievedChunk",
    "SourceChunk",
    "SparseHit",
    "HybridCandidates",
    "EmbeddingFingerprint",
    "CorpusTooLargeError",
    "get_default_embedder",
    "search_by_text",
    "search_by_vector",
    "search_sparse",
    "search_hybrid_candidates",
]
