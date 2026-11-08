"""Standard, deterministic retrieval metrics for a fixed golden dataset."""

from __future__ import annotations

from collections.abc import Sequence

from evaluation.schemas import GoldenCase, RetrievalFailure

RetrievalResult = Sequence[str] | RetrievalFailure


def _validate_inputs(
    cases: Sequence[GoldenCase], results: Sequence[RetrievalResult], k: int | None
) -> None:
    if len(cases) != len(results):
        raise ValueError("cases and results must have the same number of entries")
    if k is not None and (
        isinstance(k, bool) or not isinstance(k, int) or k < 1
    ):
        raise ValueError("k must be a positive integer")
    for result in results:
        if isinstance(result, RetrievalFailure):
            continue
        if isinstance(result, str) or not isinstance(result, Sequence):
            raise ValueError("each retrieval result must be an ID sequence or failure")
        if not all(isinstance(chunk_id, str) and chunk_id for chunk_id in result):
            raise ValueError("retrieved chunk IDs must be nonempty strings")


def _unique_ids(result: RetrievalResult) -> tuple[str, ...]:
    if isinstance(result, RetrievalFailure):
        return ()
    return tuple(dict.fromkeys(result))


def _macro_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def recall_at_k(
    cases: Sequence[GoldenCase], results: Sequence[RetrievalResult], k: int
) -> float | None:
    """Macro-average relevant unique IDs retrieved in each answerable top-k."""
    _validate_inputs(cases, results, k)
    values: list[float] = []
    for case, result in zip(cases, results, strict=True):
        if not case.answerable:
            continue
        if isinstance(result, RetrievalFailure):
            values.append(0.0)
            continue
        retrieved = set(_unique_ids(result)[:k])
        relevant = set(case.expected_chunk_ids)
        values.append(len(retrieved & relevant) / len(relevant))
    return _macro_mean(values)


def hit_rate_at_k(
    cases: Sequence[GoldenCase], results: Sequence[RetrievalResult], k: int
) -> float | None:
    """Macro-average whether any relevant unique ID occurs in top-k."""
    _validate_inputs(cases, results, k)
    values: list[float] = []
    for case, result in zip(cases, results, strict=True):
        if not case.answerable:
            continue
        if isinstance(result, RetrievalFailure):
            values.append(0.0)
            continue
        retrieved = set(_unique_ids(result)[:k])
        values.append(float(bool(retrieved & set(case.expected_chunk_ids))))
    return _macro_mean(values)


def mean_reciprocal_rank(
    cases: Sequence[GoldenCase], results: Sequence[RetrievalResult]
) -> float | None:
    """Macro-average the reciprocal rank of the first relevant unique hit."""
    _validate_inputs(cases, results, None)
    values: list[float] = []
    for case, result in zip(cases, results, strict=True):
        if not case.answerable:
            continue
        if isinstance(result, RetrievalFailure):
            values.append(0.0)
            continue
        relevant = set(case.expected_chunk_ids)
        reciprocal_rank = 0.0
        for rank, chunk_id in enumerate(_unique_ids(result), start=1):
            if chunk_id in relevant:
                reciprocal_rank = 1.0 / rank
                break
        values.append(reciprocal_rank)
    return _macro_mean(values)
