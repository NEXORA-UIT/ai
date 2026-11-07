"""Correctness tests for standard retrieval metrics and golden cases."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.schemas import GoldenCase, RetrievalFailure, load_golden_cases

_BOARD_A = "11111111-1111-4111-8111-111111111111"


def _case(
    relevant: tuple[str, ...], *, answerable: bool = True
) -> GoldenCase:
    return GoldenCase(
        question="synthetic evaluation query",
        board_id=_BOARD_A,
        expected_chunk_ids=relevant,
        answerable=answerable,
        expected_facts=(),
        expected_citation_ids=relevant,
    )


def test_recall_and_hit_rate_are_separate_metrics() -> None:
    metrics = __import__("evaluation.metrics", fromlist=["recall_at_k"])
    cases = [_case(("rel-1", "rel-2", "rel-3"))]

    assert metrics.recall_at_k(cases, [["rel-1"]], k=5) == pytest.approx(1 / 3)
    assert metrics.hit_rate_at_k(cases, [["rel-1"]], k=5) == 1.0


def test_mrr_uses_the_first_relevant_hit() -> None:
    metrics = __import__("evaluation.metrics", fromlist=["mean_reciprocal_rank"])
    cases = [_case(("rel-first", "rel-later"))]

    assert (
        metrics.mean_reciprocal_rank(
            cases, [["irrelevant", "rel-later", "irrelevant-2", "rel-first"]]
        )
        == pytest.approx(1 / 2)
    )


def test_answerable_failures_stay_in_metric_denominators_as_zeroes() -> None:
    metrics = __import__("evaluation.metrics", fromlist=["recall_at_k"])
    cases = [_case(("rel-1",)), _case(("rel-2",))]
    results = [RetrievalFailure("database timeout"), ["rel-2"]]

    assert metrics.recall_at_k(cases, results, k=5) == pytest.approx(0.5)
    assert metrics.hit_rate_at_k(cases, results, k=5) == pytest.approx(0.5)
    assert metrics.mean_reciprocal_rank(cases, results) == pytest.approx(0.5)


def test_retrieved_ids_are_deduplicated_before_cutting_top_k() -> None:
    metrics = __import__("evaluation.metrics", fromlist=["recall_at_k"])
    cases = [_case(("relevant",))]

    assert metrics.recall_at_k(
        cases, [["irrelevant", "irrelevant", "relevant"]], k=2
    ) == 1.0
    assert metrics.mean_reciprocal_rank(
        cases, [["irrelevant", "irrelevant", "relevant"]]
    ) == pytest.approx(0.5)


def test_unanswerable_only_cases_have_no_retrieval_denominator() -> None:
    metrics = __import__("evaluation.metrics", fromlist=["recall_at_k"])
    cases = [_case((), answerable=False)]

    assert metrics.recall_at_k(cases, [["any-result"]], k=5) is None
    assert metrics.hit_rate_at_k(cases, [["any-result"]], k=5) is None
    assert metrics.mean_reciprocal_rank(cases, [["any-result"]]) is None


@pytest.mark.parametrize(
    "expected_ids",
    [("same", "same"), ()],
    ids=["duplicate-answerable-id", "empty-answerable-ids"],
)
def test_golden_case_rejects_invalid_answerable_relevance(
    expected_ids: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        _case(expected_ids)


def test_unanswerable_case_rejects_relevant_ids() -> None:
    with pytest.raises(ValueError):
        _case(("must-not-be-relevant",), answerable=False)


def test_metric_input_lengths_and_top_k_are_validated() -> None:
    metrics = __import__("evaluation.metrics", fromlist=["recall_at_k"])
    cases = [_case(("relevant",))]

    with pytest.raises(ValueError, match="same number"):
        metrics.recall_at_k(cases, [], k=5)
    with pytest.raises(ValueError, match="positive integer"):
        metrics.recall_at_k(cases, [["relevant"]], k=True)


def test_synthetic_golden_dataset_covers_english_vietnamese_and_ids() -> None:
    dataset = Path("evaluation/golden/synthetic/retrieval.jsonl")

    cases = load_golden_cases(dataset)

    questions = [case.question for case in cases]
    assert any("UC-KB-03" in question for question in questions)
    assert any("Hướng dẫn" in question for question in questions)
    assert any("UC-KB-04" in question for question in questions)
