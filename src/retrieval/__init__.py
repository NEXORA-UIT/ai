"""Dense and semantic retrieval operations with lazy public exports."""

from importlib import import_module
from typing import Any

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

_EXPORTS: dict[str, tuple[str, str]] = {
    "RetrievedChunk": ("schemas", "RetrievedChunk"),
    "SourceChunk": ("schemas", "SourceChunk"),
    "SparseHit": ("schemas", "SparseHit"),
    "HybridCandidates": ("schemas", "HybridCandidates"),
    "FusedChunk": ("schemas", "FusedChunk"),
    "SearchChunk": ("schemas", "SearchChunk"),
    "SearchResult": ("schemas", "SearchResult"),
    "EmbeddingFingerprint": ("metadata", "EmbeddingFingerprint"),
    "CorpusTooLargeError": ("sparse", "CorpusTooLargeError"),
    "get_default_embedder": ("dense", "get_default_embedder"),
    "search_by_text": ("dense", "search_by_text"),
    "search_by_vector": ("dense", "search_by_vector"),
    "search_sparse": ("sparse", "search_sparse"),
    "search_hybrid_candidates": ("hybrid", "search_hybrid_candidates"),
    "fuse_rrf": ("fusion", "fuse_rrf"),
    "search_documents": ("search_documents", "search_documents"),
}


def __getattr__(name: str) -> Any:
    """Load a public API item only when it is accessed."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(f".{module_name}", __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
