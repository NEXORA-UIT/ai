"""Dense vector embedding module for Nexora AI service."""

from src.embedding.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DENSE_DIMENSION,
    DEFAULT_DEVICE,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    DEFAULT_NORMALIZE_EMBEDDINGS,
    L2_NORM_TOLERANCE,
    MAX_SEQUENCE_LENGTH,
)
from src.embedding.embedder import BgeM3Embedder, Embedder
from src.embedding.exceptions import (
    DeviceUnavailableError,
    EmbeddingError,
    InferenceError,
    ModelLoadError,
    TokenBudgetExceededError,
    VectorValidationError,
)

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DENSE_DIMENSION",
    "DEFAULT_DEVICE",
    "DEFAULT_MODEL_ID",
    "DEFAULT_MODEL_REVISION",
    "DEFAULT_NORMALIZE_EMBEDDINGS",
    "L2_NORM_TOLERANCE",
    "MAX_SEQUENCE_LENGTH",
    "BgeM3Embedder",
    "DeviceUnavailableError",
    "Embedder",
    "EmbeddingError",
    "InferenceError",
    "ModelLoadError",
    "TokenBudgetExceededError",
    "VectorValidationError",
]
