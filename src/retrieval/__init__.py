"""Dense and semantic retrieval operations."""

from src.retrieval.dense import get_default_embedder, search_by_text, search_by_vector
from src.retrieval.fusion import fuse_rrf
from src.retrieval.hybrid import search_hybrid_candidates
from src.retrieval.metadata import EmbeddingFingerprint
from src.retrieval.schemas import (
    FusedChunk,
    HybridCandidates,
    RetrievedChunk,
    SearchChunk,
    SearchResult,
    SourceChunk,
    SparseHit,
)
from src.retrieval.search_documents import search_documents
from src.retrieval.sparse import CorpusTooLargeError, search_sparse

__all__ = [
    "RetrievedChunk",
    "SourceChunk",
    "SparseHit",
    "HybridCandidates",
    "FusedChunk",
    "SearchChunk",
    "SearchResult",
    "EmbeddingFingerprint",
    "CorpusTooLargeError",
    "get_default_embedder",
    "search_by_text",
    "search_by_vector",
    "search_sparse",
    "search_hybrid_candidates",
    "fuse_rrf",
    "search_documents",
]
