"""Configuration constants for BGE-M3 embedding."""

DEFAULT_MODEL_ID: str = "BAAI/bge-m3"
DEFAULT_MODEL_REVISION: str = "5617a9f61b028005a4858fdac845db406aefb181"

# Dense embedding specifications
DEFAULT_DENSE_DIMENSION: int = 1024
DEFAULT_NORMALIZE_EMBEDDINGS: bool = True
DEFAULT_DEVICE: str = "cpu"
DEFAULT_BATCH_SIZE: int = 8

# Model architecture token limit for BAAI/bge-m3
MAX_SEQUENCE_LENGTH: int = 8192

# Numerical tolerance for L2 unit norm validation in float32
L2_NORM_TOLERANCE: float = 1e-4
