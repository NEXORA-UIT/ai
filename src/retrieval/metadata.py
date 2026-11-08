"""Embedding metadata used to keep dense retrieval model-compatible."""

from __future__ import annotations

from dataclasses import dataclass

from src.embedding.config import (
    DEFAULT_DENSE_DIMENSION,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    DEFAULT_NORMALIZE_EMBEDDINGS,
)


@dataclass(frozen=True)
class EmbeddingFingerprint:
    """Describe the model contract used to create a document embedding."""

    model_id: str
    model_revision: str
    dimension: int = DEFAULT_DENSE_DIMENSION
    normalized: bool = DEFAULT_NORMALIZE_EMBEDDINGS

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must contain non-whitespace text")
        if not isinstance(self.model_revision, str) or not self.model_revision.strip():
            raise ValueError("model_revision must contain non-whitespace text")
        if (
            isinstance(self.dimension, bool)
            or not isinstance(self.dimension, int)
            or self.dimension < 1
        ):
            raise ValueError("dimension must be a positive integer")
        if not isinstance(self.normalized, bool):
            raise ValueError("normalized must be a boolean")


DEFAULT_EMBEDDING_FINGERPRINT = EmbeddingFingerprint(
    model_id=DEFAULT_MODEL_ID,
    model_revision=DEFAULT_MODEL_REVISION,
)
