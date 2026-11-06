"""Lazy adapter for the approved BGE multilingual cross-encoder."""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

from src.reranking.config import (
    DEFAULT_RERANKER_BATCH_SIZE,
    DEFAULT_RERANKER_DEVICE,
    DEFAULT_RERANKER_MODEL_ID,
    DEFAULT_RERANKER_MODEL_REVISION,
)
from src.validation import QUESTION_TEXT_REQUIRED


class Reranker(Protocol):
    """Score passages against one question with raw relevance values."""

    def score_pairs(self, question: str, passages: Sequence[str]) -> list[float]:
        """Return one raw cross-encoder score per passage, in input order."""
        ...


class RerankerError(RuntimeError):
    """Base error raised by the BGE cross-encoder adapter."""


class RerankerLoadError(RerankerError):
    """Raised when the pinned model cannot be initialized."""


class RerankerInferenceError(RerankerError):
    """Raised when model inference fails or returns invalid scores."""


def _create_cross_encoder(*, model_id: str, revision: str, device: str):
    """Import and initialize SentenceTransformers only when first needed."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        model_name_or_path=model_id,
        revision=revision,
        device=device,
    )


class BgeReranker:
    """Lazily load BGE reranker weights and score in explicit batches."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_RERANKER_MODEL_ID,
        model_revision: str = DEFAULT_RERANKER_MODEL_REVISION,
        device: str = DEFAULT_RERANKER_DEVICE,
        batch_size: int = DEFAULT_RERANKER_BATCH_SIZE,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must contain non-whitespace text")
        if not isinstance(model_revision, str) or not model_revision.strip():
            raise ValueError("model_revision must contain non-whitespace text")
        if device != "cpu":
            raise ValueError("the approved BGE reranker runs on CPU")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise ValueError("batch_size must be a positive integer")
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer")

        self.model_id = model_id
        self.model_revision = model_revision
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                self._model = _create_cross_encoder(
                    model_id=self.model_id,
                    revision=self.model_revision,
                    device=self.device,
                )
            except Exception as exc:
                raise RerankerLoadError(
                    "BGE cross-encoder loading failed"
                ) from exc
        return self._model

    def score_pairs(self, question: str, passages: Sequence[str]) -> list[float]:
        """Score each question/passage pair without normalizing model outputs."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError(QUESTION_TEXT_REQUIRED)
        try:
            passage_list = list(passages)
        except TypeError as exc:
            raise ValueError("passages must be a sequence of strings") from exc
        if not all(isinstance(passage, str) for passage in passage_list):
            raise ValueError("each passage must be a string")
        if not passage_list:
            return []

        model = self._get_model()
        scores: list[float] = []
        for start in range(0, len(passage_list), self.batch_size):
            batch = passage_list[start : start + self.batch_size]
            pairs = [[question, passage] for passage in batch]
            try:
                batch_scores = model.predict(
                    pairs,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                )
            except Exception as exc:
                raise RerankerInferenceError(
                    "BGE cross-encoder scoring failed"
                ) from exc

            try:
                values = [float(score) for score in batch_scores]
            except (TypeError, ValueError, OverflowError) as exc:
                raise RerankerInferenceError(
                    "BGE cross-encoder returned non-numeric scores"
                ) from exc
            if len(values) != len(batch):
                raise RerankerInferenceError(
                    "BGE cross-encoder returned a score count that does not "
                    "match the input batch"
                )
            if not all(math.isfinite(score) for score in values):
                raise RerankerInferenceError(
                    "BGE cross-encoder returned a non-finite score"
                )
            scores.extend(values)
        return scores


@lru_cache(maxsize=1)
def get_default_reranker() -> BgeReranker:
    """Return the process-wide lazy instance of the approved BGE reranker."""
    return BgeReranker()
