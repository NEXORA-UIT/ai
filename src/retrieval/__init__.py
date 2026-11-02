"""Dense and semantic retrieval operations."""

from src.retrieval.dense import get_default_embedder, search_by_text, search_by_vector
from src.retrieval.metadata import EmbeddingFingerprint
from src.retrieval.schemas import RetrievedChunk

__all__ = [
    "RetrievedChunk",
    "EmbeddingFingerprint",
    "get_default_embedder",
    "search_by_text",
    "search_by_vector",
]
