"""Exception classes for embedding operations."""


class EmbeddingError(Exception):
    """Base exception for all embedding-related errors."""


class ModelLoadError(EmbeddingError):
    """Raised when an embedding model or weights cannot be loaded."""


class DeviceUnavailableError(EmbeddingError):
    """Raised when an explicitly requested device (e.g. CUDA) is not available."""


class TokenBudgetExceededError(EmbeddingError):
    """Raised when an input text exceeds the maximum sequence token budget."""


class VectorValidationError(EmbeddingError):
    """Raised when an output embedding vector fails mathematical/format validation."""


class InferenceError(EmbeddingError):
    """Raised when an error occurs during model encoding or forward pass."""
