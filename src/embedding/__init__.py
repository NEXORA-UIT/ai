"""Dense vector embedding module with lazy model-runtime exports."""

from importlib import import_module
from typing import Any

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

_EXPORTS: dict[str, tuple[str, str]] = {
    "DEFAULT_BATCH_SIZE": ("config", "DEFAULT_BATCH_SIZE"),
    "DEFAULT_DENSE_DIMENSION": ("config", "DEFAULT_DENSE_DIMENSION"),
    "DEFAULT_DEVICE": ("config", "DEFAULT_DEVICE"),
    "DEFAULT_MODEL_ID": ("config", "DEFAULT_MODEL_ID"),
    "DEFAULT_MODEL_REVISION": ("config", "DEFAULT_MODEL_REVISION"),
    "DEFAULT_NORMALIZE_EMBEDDINGS": ("config", "DEFAULT_NORMALIZE_EMBEDDINGS"),
    "L2_NORM_TOLERANCE": ("config", "L2_NORM_TOLERANCE"),
    "MAX_SEQUENCE_LENGTH": ("config", "MAX_SEQUENCE_LENGTH"),
    "BgeM3Embedder": ("embedder", "BgeM3Embedder"),
    "Embedder": ("embedder", "Embedder"),
    "DeviceUnavailableError": ("exceptions", "DeviceUnavailableError"),
    "EmbeddingError": ("exceptions", "EmbeddingError"),
    "InferenceError": ("exceptions", "InferenceError"),
    "ModelLoadError": ("exceptions", "ModelLoadError"),
    "TokenBudgetExceededError": ("exceptions", "TokenBudgetExceededError"),
    "VectorValidationError": ("exceptions", "VectorValidationError"),
}


def __getattr__(name: str) -> Any:
    """Load config or model objects only when callers request them."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(f".{module_name}", __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
