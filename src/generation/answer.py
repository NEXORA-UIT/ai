"""Validate Board-scoped evidence and citations around answer generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from uuid import UUID

from src.generation.context import build_grounded_prompt
from src.generation.schemas import AnswerClient, AnswerResult, Citation
from src.retrieval.schemas import RerankedChunk

_REFUSAL_TEXT = "I cannot answer this question from the available sources."
_DEFAULT_CONTEXT_CHARS = 12000


class EvidenceScopeError(ValueError):
    """Raised when supplied evidence does not belong to the requested Board."""


class AnswerProviderError(RuntimeError):
    """Raised when the injected answer client fails to complete a request."""


def _refusal() -> AnswerResult:
    return AnswerResult(text=_REFUSAL_TEXT, citations=(), refused=True)


def _normalize_board_id(board_id: str | UUID) -> UUID:
    try:
        return UUID(str(board_id))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("board_id must be a valid UUID") from exc


def _validate_context_budget(max_context_chars: int) -> None:
    if (
        isinstance(max_context_chars, bool)
        or not isinstance(max_context_chars, int)
        or max_context_chars < 1
    ):
        raise ValueError("max_context_chars must be a positive integer")


def _citation_for(chunk: RerankedChunk) -> Citation:
    source = chunk.source
    return Citation(
        chunk_id=source.chunk_id,
        document_id=source.document_id,
        file_name=source.file_name,
        page_number=source.page_number,
        start_offset=source.start_offset,
        end_offset=source.end_offset,
    )


def _validated_result(
    response: object, included_chunks: Sequence[RerankedChunk]
) -> AnswerResult:
    if not isinstance(response, Mapping):
        return _refusal()
    answer = response.get("answer")
    cited_ids = response.get("cited_chunk_ids")
    if not isinstance(answer, str) or not answer.strip():
        return _refusal()
    if not isinstance(cited_ids, (list, tuple)) or not cited_ids:
        return _refusal()
    if not all(isinstance(chunk_id, str) and chunk_id for chunk_id in cited_ids):
        return _refusal()

    sources_by_id = {chunk.source.chunk_id: chunk for chunk in included_chunks}
    unique_ids = tuple(dict.fromkeys(cited_ids))
    if any(chunk_id not in sources_by_id for chunk_id in unique_ids):
        return _refusal()

    citations = tuple(
        _citation_for(sources_by_id[chunk_id]) for chunk_id in unique_ids
    )
    return AnswerResult(text=answer.strip(), citations=citations, refused=False)


def answer_with_sources(
    question: str,
    board_id: str | UUID,
    chunks: Sequence[RerankedChunk],
    *,
    client: AnswerClient,
    max_context_chars: int = _DEFAULT_CONTEXT_CHARS,
) -> AnswerResult:
    """Answer from supplied scoped evidence, refusing invalid or absent sources."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must contain non-whitespace text")
    board_uuid = _normalize_board_id(board_id)
    _validate_context_budget(max_context_chars)

    for chunk in chunks:
        try:
            source_board = UUID(str(chunk.source.board_id))
        except (TypeError, ValueError, AttributeError) as exc:
            raise EvidenceScopeError("evidence has an invalid Board ID") from exc
        if source_board != board_uuid:
            raise EvidenceScopeError(
                "supplied evidence belongs to a different Board"
            )

    if not chunks:
        return _refusal()

    prompt, included_chunks = build_grounded_prompt(
        question, chunks, max_context_chars=max_context_chars
    )
    if not included_chunks:
        return _refusal()

    try:
        response = client.complete(prompt)
    except Exception as exc:
        raise AnswerProviderError("answer provider request failed") from exc
    return _validated_result(response, included_chunks)
