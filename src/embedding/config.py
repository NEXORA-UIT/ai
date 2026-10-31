"""Configuration constants for BGE-M3 embedding."""

from src.ingestion.tokenizer import (
    DEFAULT_BGE_M3_MODEL_ID,
    DEFAULT_BGE_M3_REVISION,
)

# Re-use pinned identifiers from tokenizer to guarantee zero divergence
DEFAULT_MODEL_ID: str = DEFAULT_BGE_M3_MODEL_ID
DEFAULT_MODEL_REVISION: str = DEFAULT_BGE_M3_REVISION

# Dense embedding specifications
DEFAULT_DENSE_DIMENSION: int = 1024
DEFAULT_NORMALIZE_EMBEDDINGS: bool = True
DEFAULT_DEVICE: str = "cpu"
DEFAULT_BATCH_SIZE: int = 8

# Model architecture token limit for BAAI/bge-m3
MAX_SEQUENCE_LENGTH: int = 8192

# Numerical tolerance for L2 unit norm validation in float32
L2_NORM_TOLERANCE: float = 1e-4
