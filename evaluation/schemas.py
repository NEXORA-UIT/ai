"""Data contracts for reproducible retrieval evaluation cases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from uuid import UUID


class DatasetKind(StrEnum):
    """Separate synthetic correctness checks from real-document quality data."""

    SYNTHETIC_CORRECTNESS = "synthetic_correctness"
    REAL_DOCUMENT_QUALITY = "real_document_quality"


@dataclass(frozen=True)
class GoldenCase:
    """One ordered query with answerability and human-labeled evidence."""

    question: str
    board_id: str
    expected_chunk_ids: tuple[str, ...]
    answerable: bool
    expected_facts: tuple[str, ...]
    expected_citation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must contain non-whitespace text")
        try:
            normalized_board = str(UUID(self.board_id))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("board_id must be a valid UUID") from exc
        object.__setattr__(self, "board_id", normalized_board)
        if not isinstance(self.answerable, bool):
            raise ValueError("answerable must be a boolean")
        _validate_ids(self.expected_chunk_ids, "expected_chunk_ids")
        _validate_ids(self.expected_citation_ids, "expected_citation_ids")
        if self.answerable and not self.expected_chunk_ids:
            raise ValueError("answerable cases must have relevant chunk IDs")
        if not self.answerable and self.expected_chunk_ids:
            raise ValueError("unanswerable cases must not have relevant chunk IDs")
        if not set(self.expected_citation_ids).issubset(self.expected_chunk_ids):
            raise ValueError("expected citations must refer to relevant chunk IDs")
        if not all(
            isinstance(fact, str) and fact.strip() for fact in self.expected_facts
        ):
            raise ValueError("expected_facts must contain nonempty strings")


@dataclass(frozen=True)
class RetrievalFailure:
    """A retrieval error that stays visible in answerable metric denominators."""

    error: str

    def __post_init__(self) -> None:
        if not isinstance(self.error, str) or not self.error.strip():
            raise ValueError("error must contain non-whitespace text")


def _validate_ids(ids: tuple[str, ...], field: str) -> None:
    if not isinstance(ids, tuple):
        raise ValueError(f"{field} must be a tuple")
    if not all(isinstance(item, str) and item.strip() for item in ids):
        raise ValueError(f"{field} must contain nonempty strings")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{field} must contain unique IDs")


def load_golden_cases(dataset_path: Path) -> tuple[GoldenCase, ...]:
    """Read ordered JSONL golden cases with a useful line-number error."""
    cases: list[GoldenCase] = []
    for line_number, line in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            cases.append(
                GoldenCase(
                    question=row["question"],
                    board_id=row["board_id"],
                    expected_chunk_ids=tuple(row["expected_chunk_ids"]),
                    answerable=row["answerable"],
                    expected_facts=tuple(row["expected_facts"]),
                    expected_citation_ids=tuple(row["expected_citation_ids"]),
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid golden case at {dataset_path}:{line_number}"
            ) from exc
    if not cases:
        raise ValueError(f"golden dataset is empty: {dataset_path}")
    return tuple(cases)
