"""Deterministic citation accuracy and shared latency evaluation helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from evaluation.latency import LatencySummary, latency_summary
from evaluation.manifest import load_evaluation_dataset
from evaluation.schemas import GoldenCase
from src.generation.schemas import AnswerResult, Citation
from src.retrieval.schemas import SourceChunk

__all__ = [
    "LatencySummary",
    "citation_accuracy",
    "latency_summary",
    "load_canonical_source_chunks",
]


def citation_accuracy(
    expected: GoldenCase,
    answer: AnswerResult,
    corpus: Mapping[str, SourceChunk],
) -> float:
    """Score citations against both human labels and canonical source locations.

    Each correctly cited expected chunk counts once. The denominator is the
    larger of the expected and emitted citation counts, so missing, invented,
    misplaced, wrong-document and duplicate citations all reduce the score.
    Correct refusal on an unanswerable case scores 1; unsupported output scores
    0. This score is citation correctness, not claim faithfulness.
    """
    if not isinstance(expected, GoldenCase):
        raise TypeError("expected must be a GoldenCase")
    if not isinstance(answer, AnswerResult):
        raise TypeError("answer must be an AnswerResult")
    if not isinstance(corpus, Mapping):
        raise TypeError("corpus must map chunk IDs to canonical SourceChunk records")
    if not expected.answerable:
        return float(answer.refused and not answer.citations)
    if answer.refused or not expected.expected_citation_ids:
        return 0.0

    expected_ids = set(expected.expected_citation_ids)
    matched_ids: set[str] = set()
    for citation in answer.citations:
        if not isinstance(citation, Citation):
            continue
        if citation.chunk_id not in expected_ids:
            continue
        source = corpus.get(citation.chunk_id)
        if source is None or not isinstance(source, SourceChunk):
            continue
        if (
            citation.document_id == source.document_id
            and citation.page_number == source.page_number
            and citation.start_offset == source.start_offset
            and citation.end_offset == source.end_offset
        ):
            matched_ids.add(citation.chunk_id)

    denominator = max(len(expected_ids), len(answer.citations))
    if denominator == 0:
        return 0.0
    return len(matched_ids) / denominator


def load_canonical_source_chunks(dataset_path: Path) -> dict[str, SourceChunk]:
    """Load the frozen JSONL snapshot as internal canonical citation records."""
    dataset = load_evaluation_dataset(dataset_path)
    corpus: dict[str, SourceChunk] = {}
    for line_number, line in enumerate(
        dataset.chunk_snapshot_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            document = json.loads(line)
            chunks = document["chunks"]
            if not isinstance(chunks, list):
                raise ValueError("chunks must be a list")
            for chunk in chunks:
                source = SourceChunk(
                    board_id=str(document["board_id"]),
                    chunk_id=str(chunk["chunk_id"]),
                    document_id=str(document["document_id"]),
                    chunk_index=chunk["chunk_index"],
                    file_name=document["file_name"],
                    storage_key=document["storage_key"],
                    content=chunk["content"],
                    page_number=chunk.get("page_number"),
                    start_offset=chunk.get("start_offset"),
                    end_offset=chunk.get("end_offset"),
                )
                if source.chunk_id in corpus:
                    raise ValueError("duplicate chunk ID")
                corpus[source.chunk_id] = source
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid source chunk snapshot at "
                f"{dataset.chunk_snapshot_path}:{line_number}"
            ) from exc
    if not corpus:
        raise ValueError("canonical source chunk snapshot is empty")
    return corpus
