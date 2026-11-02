"""Deterministic source citation and latency metric tests."""

from __future__ import annotations

import pytest

from evaluation.answer_metrics import citation_accuracy, latency_summary
from evaluation.latency import LatencySummary
from evaluation.schemas import GoldenCase
from src.generation.schemas import AnswerResult, Citation
from src.retrieval.schemas import SourceChunk

_BOARD_ID = "11111111-1111-4111-8111-111111111111"


def _case(
    expected_citations: tuple[str, ...], *, answerable: bool = True
) -> GoldenCase:
    return GoldenCase(
        question="Which approved step applies?",
        board_id=_BOARD_ID,
        expected_chunk_ids=expected_citations if answerable else (),
        answerable=answerable,
        expected_facts=("approved step",) if answerable else (),
        expected_citation_ids=expected_citations,
    )


def _source(chunk_id: str, document_id: str, page: int) -> SourceChunk:
    return SourceChunk(
        board_id=_BOARD_ID,
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        file_name=f"{document_id}.pdf",
        storage_key=f"internal://{document_id}",
        content="synthetic approved source text",
        page_number=page,
        start_offset=0,
        end_offset=32,
    )


def _citation(source: SourceChunk, **overrides: object) -> Citation:
    values = {
        "chunk_id": source.chunk_id,
        "document_id": source.document_id,
        "file_name": source.file_name,
        "page_number": source.page_number,
        "start_offset": source.start_offset,
        "end_offset": source.end_offset,
    }
    values.update(overrides)
    return Citation(**values)  # type: ignore[arg-type]


def test_citation_accuracy_requires_expected_chunk_and_exact_location() -> None:
    source_a = _source("chunk-a", "doc-a", 1)
    source_b = _source("chunk-b", "doc-b", 4)
    case = _case(("chunk-a", "chunk-b"))
    corpus = {source.chunk_id: source for source in (source_a, source_b)}
    correct = AnswerResult(
        text="Use both approved steps.",
        citations=(_citation(source_a), _citation(source_b)),
        refused=False,
    )

    assert citation_accuracy(case, correct, corpus) == 1.0

    wrong_location = AnswerResult(
        text="Use the approved step.",
        citations=(_citation(source_a, page_number=2), _citation(source_b)),
        refused=False,
    )
    assert citation_accuracy(case, wrong_location, corpus) == 0.5


def test_invented_or_wrong_document_citation_is_inaccurate() -> None:
    source = _source("expected", "doc-1", 1)
    case = _case(("expected",))
    invented = AnswerResult(
        text="Answer",
        citations=(_citation(source, chunk_id="invented"),),
        refused=False,
    )
    wrong_document = AnswerResult(
        text="Answer",
        citations=(_citation(source, document_id="other-doc"),),
        refused=False,
    )
    wrong_offsets = AnswerResult(
        text="Answer",
        citations=(_citation(source, end_offset=999),),
        refused=False,
    )

    assert citation_accuracy(case, invented, {"expected": source}) == 0.0
    assert citation_accuracy(case, wrong_document, {"expected": source}) == 0.0
    assert citation_accuracy(case, wrong_offsets, {"expected": source}) == 0.0


def test_empty_citations_on_answerable_answer_score_zero() -> None:
    case = _case(("expected",))
    answer = AnswerResult(text="Unsupported answer", citations=(), refused=False)

    assert citation_accuracy(case, answer, {}) == 0.0


def test_unanswerable_case_is_correct_only_when_refused_without_citations() -> None:
    case = _case((), answerable=False)
    refused = AnswerResult(
        text="I cannot answer from the evidence.", citations=(), refused=True
    )
    unsupported = AnswerResult(text="Unfounded answer", citations=(), refused=False)

    assert citation_accuracy(case, refused, {}) == 1.0
    assert citation_accuracy(case, unsupported, {}) == 0.0


def test_latency_summary_handles_one_sample_and_validates_inputs() -> None:
    assert latency_summary([123.5]) == LatencySummary(
        p50_ms=123.5, p95_ms=123.5, sample_count=1
    )
    assert latency_summary([0, 10, 20]).p50_ms == 10
    with pytest.raises(ValueError, match="finite and nonnegative"):
        latency_summary([1.0, float("nan")])
    with pytest.raises(ValueError, match="at least one"):
        latency_summary([])
