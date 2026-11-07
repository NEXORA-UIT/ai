"""Board-scoped retrieval and grounded answer orchestration."""

from __future__ import annotations

import logging
import os
from time import perf_counter
from uuid import UUID

from src.answering.answer import (
    AnswerProviderError,
    EvidenceScopeError,
    answer_with_sources,
)
from src.answering.schemas import AnswerResult, GroundedPrompt
from src.persistence.errors import DatabaseConnectionError
from src.retrieval.schemas import RerankedChunk

logger = logging.getLogger("uvicorn.error").getChild("nexora.generation")


class AnswerRetrievalError(RuntimeError):
    """The answer use case could not retrieve evidence."""


class AnswerConfigurationError(RuntimeError):
    """Evidence is available but the answer provider is not configured."""


class AnswerGenerationError(RuntimeError):
    """The grounding pipeline failed outside of the provider call."""


class _LazyDeepSeekClient:
    def complete(self, prompt: GroundedPrompt) -> dict[str, object]:
        from src.providers.deepseek.client import DeepSeekAnswerClient

        with DeepSeekAnswerClient() as client:
            return client.complete(prompt)


def _retrieve_evidence(
    question: str, board_id: UUID, *, top_k: int, candidate_k: int
) -> list[RerankedChunk]:
    from src.retrieval.search_documents import retrieve_reranked_chunks

    return retrieve_reranked_chunks(
        question, board_id, top_k=top_k, candidate_k=candidate_k
    )


def answer_question(
    question: str, board_id: UUID, *, top_k: int, candidate_k: int
) -> AnswerResult:
    started = perf_counter()
    try:
        evidence = _retrieve_evidence(
            question, board_id, top_k=top_k, candidate_k=candidate_k
        )
    except DatabaseConnectionError:
        raise
    except Exception as exc:
        raise AnswerRetrievalError("Answer retrieval failed") from exc

    logger.info(
        "RAG answer retrieval_ms=%.0f evidence_count=%d",
        (perf_counter() - started) * 1000,
        len(evidence),
    )
    if evidence and not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        raise AnswerConfigurationError("DeepSeek API key is not configured")

    started = perf_counter()
    try:
        result = answer_with_sources(
            question, board_id, evidence, client=_LazyDeepSeekClient()
        )
    except (EvidenceScopeError, AnswerProviderError):
        raise
    except Exception as exc:
        raise AnswerGenerationError("Answer generation failed") from exc

    logger.info(
        "RAG answer generation_ms=%.0f refused=%s",
        (perf_counter() - started) * 1000,
        result.refused,
    )
    return result
