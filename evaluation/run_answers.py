"""Measure citation correctness, refusal counts and answer latency."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

from evaluation.answer_metrics import citation_accuracy, load_canonical_source_chunks
from evaluation.latency import LatencySummary, latency_summary
from evaluation.manifest import load_evaluation_dataset
from evaluation.schemas import DatasetKind, GoldenCase
from src.generation.schemas import AnswerResult


@dataclass(frozen=True)
class AnswerCaseResult:
    """One answer run, retaining typed result or a safe failure marker."""

    case_index: int
    question: str
    answerable: bool
    result: AnswerResult | None
    citation_accuracy: float | None
    failure: str | None
    stage_latencies_ms: Mapping[str, float]


@dataclass(frozen=True)
class AnswerEvaluationReport:
    """Deterministic answer report with no judge-based faithfulness claim."""

    dataset_kind: DatasetKind
    dataset_version: str
    answer_client_kind: Literal["fake_correctness", "real_provider"]
    answerable_count: int
    unanswerable_count: int
    citation_accuracy: float | None
    failure_count: int
    refusal_count: int
    correct_unanswerable_refusals: int
    latency: Mapping[str, LatencySummary]
    case_results: tuple[AnswerCaseResult, ...]
    faithfulness: None = None

    @property
    def is_real_document_quality(self) -> bool:
        return (
            self.dataset_kind is DatasetKind.REAL_DOCUMENT_QUALITY
            and self.answer_client_kind == "real_provider"
        )


def _safe_failure(exc: Exception) -> str:
    """Keep an error visible without recording provider credentials or text."""
    return f"{type(exc).__name__}: answer evaluation failed"


def _serialize_answer(answer: AnswerResult | None) -> object:
    if answer is None:
        return None
    return {
        "text": answer.text,
        "refused": answer.refused,
        "citations": [
            {
                "chunk_id": citation.chunk_id,
                "document_id": citation.document_id,
                "file_name": citation.file_name,
                "page_number": citation.page_number,
                "start_offset": citation.start_offset,
                "end_offset": citation.end_offset,
            }
            for citation in answer.citations
        ],
    }


def _payload(report: AnswerEvaluationReport) -> dict[str, object]:
    return {
        "dataset_kind": report.dataset_kind.value,
        "dataset_version": report.dataset_version,
        "is_real_document_quality": report.is_real_document_quality,
        "answer_client_kind": report.answer_client_kind,
        "answerable_count": report.answerable_count,
        "unanswerable_count": report.unanswerable_count,
        "citation_accuracy": report.citation_accuracy,
        "failure_count": report.failure_count,
        "refusal_count": report.refusal_count,
        "correct_unanswerable_refusals": report.correct_unanswerable_refusals,
        "faithfulness": report.faithfulness,
        "faithfulness_status": "not_measured_by_deterministic_citation_runner",
        "latency": {
            stage: asdict(summary) for stage, summary in report.latency.items()
        },
        "cases": [
            {
                "case_index": row.case_index,
                "question": row.question,
                "answerable": row.answerable,
                "answer": _serialize_answer(row.result),
                "citation_accuracy": row.citation_accuracy,
                "failure": row.failure,
                "stage_latencies_ms": dict(row.stage_latencies_ms),
            }
            for row in report.case_results
        ],
    }


def _render_markdown(report: AnswerEvaluationReport) -> str:
    if report.dataset_kind is DatasetKind.SYNTHETIC_CORRECTNESS:
        warning = "Synthetic correctness only; not real-document answer quality."
    elif report.answer_client_kind == "fake_correctness":
        warning = "Fake answer client; not real-document answer quality."
    else:
        warning = "Real-provider citation baseline; faithfulness was not measured."
    citation = (
        "N/A" if report.citation_accuracy is None else f"{report.citation_accuracy:.3f}"
    )
    lines = [
        f"# Answer evaluation: {report.dataset_version}",
        "",
        f"**Dataset:** `{report.dataset_kind.value}` — {warning}",
        f"**Citation accuracy:** {citation}",
        f"**Refusals:** {report.refusal_count}  |  "
        "**Correct unanswerable refusals:** "
        f"{report.correct_unanswerable_refusals}  |  "
        f"**Failures:** {report.failure_count}",
        "**Faithfulness:** not measured; use the approved judge in a later run.",
        "",
        "## Latency p50 / p95 (ms)",
        "",
    ]
    for stage, summary in report.latency.items():
        lines.append(
            f"- **{stage}:** {summary.p50_ms:.2f} / {summary.p95_ms:.2f} "
            f"({summary.sample_count} samples)"
        )
    lines.extend(["", "## Cases", ""])
    for row in report.case_results:
        outcome = f"failure: {row.failure}" if row.failure else "completed"
        lines.append(
            f"- Case {row.case_index + 1}: {row.question} — {outcome}; "
            f"citation accuracy: {row.citation_accuracy}"
        )
    return "\n".join(lines) + "\n"


def run_answers(
    dataset_path: Path,
    answer: Callable[[GoldenCase], AnswerResult],
    *,
    output_dir: Path,
    answer_client_kind: Literal[
        "fake_correctness", "real_provider"
    ] = "fake_correctness",
) -> AnswerEvaluationReport:
    """Run an injected answer service over the exact ordered golden cases."""
    if not callable(answer):
        raise ValueError("answer must be callable")
    if answer_client_kind not in {"fake_correctness", "real_provider"}:
        raise ValueError("answer_client_kind is invalid")
    dataset = load_evaluation_dataset(dataset_path)
    corpus = load_canonical_source_chunks(dataset_path)
    rows: list[AnswerCaseResult] = []
    answerable_scores: list[float] = []
    generation_samples: list[float] = []
    citation_samples: list[float] = []
    total_samples: list[float] = []

    for case_index, case in enumerate(dataset.cases):
        total_started = time.perf_counter()
        generation_started = time.perf_counter()
        try:
            result = answer(case)
            generation_ms = max(0.0, (time.perf_counter() - generation_started) * 1000)
            if not isinstance(result, AnswerResult):
                raise TypeError("answer callable must return AnswerResult")
            generation_samples.append(generation_ms)
            citation_started = time.perf_counter()
            score = citation_accuracy(case, result, corpus)
            citation_ms = max(0.0, (time.perf_counter() - citation_started) * 1000)
            citation_samples.append(citation_ms)
            if case.answerable:
                answerable_scores.append(score)
            rows.append(
                AnswerCaseResult(
                    case_index=case_index,
                    question=case.question,
                    answerable=case.answerable,
                    result=result,
                    citation_accuracy=score,
                    failure=None,
                    stage_latencies_ms={
                        "generation": generation_ms,
                        "citation_validation": citation_ms,
                        "total": max(0.0, (time.perf_counter() - total_started) * 1000),
                    },
                )
            )
        except Exception as exc:
            generation_ms = max(0.0, (time.perf_counter() - generation_started) * 1000)
            generation_samples.append(generation_ms)
            if case.answerable:
                answerable_scores.append(0.0)
            rows.append(
                AnswerCaseResult(
                    case_index=case_index,
                    question=case.question,
                    answerable=case.answerable,
                    result=None,
                    citation_accuracy=(0.0 if case.answerable else None),
                    failure=_safe_failure(exc),
                    stage_latencies_ms={"generation": generation_ms},
                )
            )
        total_ms = max(0.0, (time.perf_counter() - total_started) * 1000)
        total_samples.append(total_ms)
        if rows[-1].stage_latencies_ms.get("total") is None:
            rows[-1] = replace(
                rows[-1],
                stage_latencies_ms={
                    **rows[-1].stage_latencies_ms,
                    "total": total_ms,
                },
            )

    answerable_count = sum(case.answerable for case in dataset.cases)
    unanswerable_count = len(dataset.cases) - answerable_count
    refusal_count = sum(row.result is not None and row.result.refused for row in rows)
    correct_unanswerable_refusals = sum(
        not row.answerable
        and row.result is not None
        and row.result.refused
        and not row.result.citations
        for row in rows
    )
    latency_samples: dict[str, list[float]] = {
        "generation": generation_samples,
        "total": total_samples,
    }
    if citation_samples:
        latency_samples["citation_validation"] = citation_samples
    report = AnswerEvaluationReport(
        dataset_kind=dataset.kind,
        dataset_version=dataset.version,
        answer_client_kind=answer_client_kind,
        answerable_count=answerable_count,
        unanswerable_count=unanswerable_count,
        citation_accuracy=(
            sum(answerable_scores) / len(answerable_scores)
            if answerable_scores
            else None
        ),
        failure_count=sum(row.failure is not None for row in rows),
        refusal_count=refusal_count,
        correct_unanswerable_refusals=correct_unanswerable_refusals,
        latency={
            stage: latency_summary(samples)
            for stage, samples in latency_samples.items()
            if samples
        },
        case_results=tuple(rows),
    )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "answer-report.json").write_text(
        json.dumps(_payload(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (target / "answer-report.md").write_text(_render_markdown(report), encoding="utf-8")
    return report
