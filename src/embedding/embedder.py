"""Embedder protocol and BAAI/bge-m3 sentence-transformers implementation."""

from typing import Any, Protocol

import numpy as np
import torch

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
from src.embedding.exceptions import (
    DeviceUnavailableError,
    InferenceError,
    ModelLoadError,
    TokenBudgetExceededError,
    VectorValidationError,
)
from src.ingestion.schemas import Chunk, EmbeddedChunk


class Embedder(Protocol):
    """Protocol for dense vector embedding components."""

    @property
    def model_name(self) -> str:
        """Identifier of the underlying model."""
        ...

    @property
    def revision(self) -> str | None:
        """Model commit revision hash."""
        ...

    @property
    def dimension(self) -> int:
        """Dimensionality of the dense output vectors."""
        ...

    def embed_texts(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """Encode raw text strings into L2-normalized dense vectors."""
        ...

    def embed_chunks(
        self, chunks: list[Chunk], batch_size: int | None = None
    ) -> list[EmbeddedChunk]:
        """Encode Chunk objects and pair them with their dense vector embeddings."""
        ...


class BgeM3Embedder:
    """Dense embedding model using BAAI/bge-m3 via sentence-transformers.

    Features:
    - Lazy loading: `embed_texts([])` returns `[]` without loading weights.
    - Explicit device check: rejects CUDA when unavailable without silent CPU fallback.
    - Token budget verification before inference with padding/truncation disabled.
    - Strict output validation: dimensionality, non-zero, finiteness, L2 normalization.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str = DEFAULT_MODEL_REVISION,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        model: Any = None,
        max_seq_length: int = MAX_SEQUENCE_LENGTH,
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise DeviceUnavailableError(
                "CUDA was explicitly requested, but torch.cuda.is_available() is "
                "False. Nexora does not fall back silently to CPU."
            )
        if device not in ("cpu", "cuda"):
            raise ValueError(
                f"Unsupported device: '{device}'. Must be 'cpu' or 'cuda'."
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if max_seq_length < 1:
            raise ValueError(f"max_seq_length must be >= 1, got {max_seq_length}")

        self._model_id = model_id
        self._revision = revision
        self._device = device
        self._batch_size = batch_size
        self._max_seq_length = max_seq_length
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def dimension(self) -> int:
        return DEFAULT_DENSE_DIMENSION

    @property
    def device(self) -> str:
        return self._device

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _get_model(self) -> Any:
        """Lazily initialize the SentenceTransformer model if not already loaded."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    model_name_or_path=self._model_id,
                    revision=self._revision,
                    device=self._device,
                )
            except Exception as exc:
                raise ModelLoadError(
                    f"Failed to load embedding model '{self._model_id}' "
                    f"(revision: {self._revision}): {exc}"
                ) from exc
        return self._model

    def embed_texts(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """Encode raw text strings into L2-normalized dense vectors.

        Args:
            texts: List of text strings to embed.
            batch_size: Optional batch size override (defaults to self.batch_size).

        Returns:
            list[list[float]]: Dense vectors in the same order as input texts.

        Raises:
            TypeError: If texts or any element is not a string.
            ValueError: If batch_size < 1 or any text is empty/whitespace.
            TokenBudgetExceededError: If any text exceeds maximum sequence length.
            InferenceError: If model fails during forward pass.
            VectorValidationError: If outputs fail format or normalization checks.
        """
        if not isinstance(texts, list):
            raise TypeError(f"texts must be a list, got {type(texts).__name__}")

        # Quick return without loading model if list is empty
        if len(texts) == 0:
            return []

        eff_batch_size = self._batch_size if batch_size is None else batch_size
        if eff_batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {eff_batch_size}")

        # Validate each input string
        for idx, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"Input text at index {idx} must be a string, "
                    f"got {type(text).__name__}"
                )
            if not text or not text.strip():
                raise ValueError(
                    f"Input text at index {idx} is empty or whitespace-only. "
                    "Embedder requires non-empty text."
                )

        model = self._get_model()

        # Validate model tokenizer and sequence limit
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "encode"):
            raise ModelLoadError(
                "Embedding model does not provide a valid tokenizer for token "
                "budget validation."
            )

        model_limit = getattr(model, "max_seq_length", None)
        if not isinstance(model_limit, int) or model_limit < 1:
            raise ModelLoadError(
                "Embedding model does not provide a valid integer max_seq_length "
                f"(got {model_limit!r})."
            )

        effective_limit = min(self._max_seq_length, model_limit)

        # Validate token count per input text using model tokenizer
        for idx, text in enumerate(texts):
            # Count tokens including special tokens (<s>, </s>) without truncation
            token_ids = tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=False,
                padding=False,
            )
            if len(token_ids) > effective_limit:
                raise TokenBudgetExceededError(
                    f"Input text at index {idx} exceeds maximum token budget: "
                    f"{len(token_ids)} tokens > {effective_limit} max allowed tokens "
                    f"(configured: {self._max_seq_length}, model limit: {model_limit})."
                )

        # Batch inference
        results: list[list[float]] = []
        for start_idx in range(0, len(texts), eff_batch_size):
            batch_texts = texts[start_idx : start_idx + eff_batch_size]
            try:
                raw_output = model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    normalize_embeddings=DEFAULT_NORMALIZE_EMBEDDINGS,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            except Exception as exc:
                raise InferenceError(
                    f"Inference failed during batch encoding: {exc}"
                ) from exc

            # Validate output batch shape and numeric format
            try:
                batch_vectors = np.asarray(raw_output, dtype=np.float32)
            except (ValueError, TypeError) as exc:
                raise VectorValidationError(
                    f"Inference output cannot be converted to numeric array: {exc}"
                ) from exc

            expected_shape = (len(batch_texts), DEFAULT_DENSE_DIMENSION)
            if batch_vectors.ndim != 2 or batch_vectors.shape != expected_shape:
                raise VectorValidationError(
                    f"Inference output must have shape {expected_shape}, "
                    f"got ndim={batch_vectors.ndim}, shape={batch_vectors.shape}."
                )

            if not np.isfinite(batch_vectors).all():
                raise VectorValidationError(
                    f"Batch at index {start_idx} contains NaN or Infinity."
                )

            for local_idx in range(len(batch_texts)):
                global_idx = start_idx + local_idx
                vec = batch_vectors[local_idx]
                if np.all(vec == 0):
                    raise VectorValidationError(
                        f"Vector at index {global_idx} is an all-zero vector."
                    )
                norm = float(np.linalg.norm(vec))
                if abs(norm - 1.0) > L2_NORM_TOLERANCE:
                    raise VectorValidationError(
                        f"Vector at index {global_idx} has invalid L2 norm {norm:.6f} "
                        f"(expected 1.0 +- {L2_NORM_TOLERANCE})."
                    )
                results.append([float(x) for x in vec])

        if len(results) != len(texts):
            raise VectorValidationError(
                f"Expected {len(texts)} total vectors, got {len(results)}."
            )

        return results

    def embed_chunks(
        self, chunks: list[Chunk], batch_size: int | None = None
    ) -> list[EmbeddedChunk]:
        """Encode Chunk objects and pair them with their dense vector embeddings.

        Args:
            chunks: List of Chunk objects to embed.
            batch_size: Optional batch size override.

        Returns:
            list[EmbeddedChunk]: Ordered list of EmbeddedChunk instances.
        """
        if not isinstance(chunks, list):
            raise TypeError(f"chunks must be a list, got {type(chunks).__name__}")
        if len(chunks) == 0:
            return []

        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise TypeError(
                    f"Element at index {idx} is not a Chunk, got {type(chunk).__name__}"
                )

        contents = [chunk.content for chunk in chunks]
        vectors = self.embed_texts(contents, batch_size=batch_size)

        return [
            EmbeddedChunk(chunk=chunk, dense_embedding=vector)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
