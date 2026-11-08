"""Compare dense, hybrid+RRF, and hybrid+RRF+reranker on one locked corpus."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Protocol

from evaluation.latency import LatencySummary, latency_summary
from evaluation.manifest import EvaluationDataset, load_evaluation_dataset
from evaluation.metrics import hit_rate_at_k, mean_reciprocal_rank, recall_at_k
from evaluation.schemas import DatasetKind, GoldenCase, RetrievalFailure

VARIANTS: dict[str, str] = {
    "A": "Dense-only",
    "B": "Hybrid+RRF",
    "C": "Hybrid+RRF+Reranker",
}
_MAX_FINAL_K = 5
_MAX_CANDIDATE_K = 20


@dataclass(frozen=True)
class RetrievalRequest:
    """One case plus the immutable dataset and candidate settings for a variant."""

    variant: str
    case_index: int
    case: GoldenCase
    k: int
    candidate_k: int
    dataset_version: str
    chunk_snapshot_sha256: str
    query_snapshot_sha256: str
    embedding_fingerprint: str
    chunking_fingerprint: str


@dataclass(frozen=True)
class RetrievalOutput:
    """Ranked candidate IDs plus measured component-stage information."""

    chunk_ids: tuple[str, ...]
    stage_latencies_ms: Mapping[str, float] = field(default_factory=dict)
    bm25_corpus_size: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.chunk_ids, str) or not isinstance(self.chunk_ids, Sequence):
            raise ValueError("chunk_ids must be an ID sequence")
        if not all(
            isinstance(chunk_id, str) and chunk_id for chunk_id in self.chunk_ids
        ):
            raise ValueError("chunk_ids must contain nonempty strings")
        if not isinstance(self.stage_latencies_ms, Mapping):
            raise ValueError("stage_latencies_ms must be a mapping")
        for stage, value in self.stage_latencies_ms.items():
            if not isinstance(stage, str) or not stage.strip():
                raise ValueError("latency stage names must be nonempty strings")
            if isinstance(value, bool):
                raise ValueError("stage latencies must be finite nonnegative numbers")
            try:
                latency = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "stage latencies must be finite nonnegative numbers"
                ) from exc
            if not math.isfinite(latency) or latency < 0:
                raise ValueError("stage latencies must be finite nonnegative numbers")
        if self.bm25_corpus_size is not None and (
            isinstance(self.bm25_corpus_size, bool)
            or not isinstance(self.bm25_corpus_size, int)
            or self.bm25_corpus_size < 0
        ):
            raise ValueError("bm25_corpus_size must be a nonnegative integer or None")


class RetrievalCallable(Protocol):
    """Adapter around the existing M2/M3/M4/M5 retrieval implementations."""

    def __call__(self, request: RetrievalRequest) -> RetrievalOutput | Sequence[str]:
        """Return the ranked unique or repeated candidate IDs for one case."""
        ...


@dataclass(frozen=True)
class CaseResult:
    """One query result, including failures and per-stage timings."""

    case_index: int
    question: str
    answerable: bool
    expected_chunk_ids: tuple[str, ...]
    result: tuple[str, ...] | RetrievalFailure
    stage_latencies_ms: Mapping[str, float]
    total_latency_ms: float
    bm25_corpus_size: int | None

    @property
    def case_key(self) -> str:
        return f"{self.case_index}:{self.question}"

    @property
    def relevant_ranks(self) -> dict[str, int | None]:
        if isinstance(self.result, RetrievalFailure):
            return {chunk_id: None for chunk_id in self.expected_chunk_ids}
        unique = tuple(dict.fromkeys(self.result))
        positions = {chunk_id: index for index, chunk_id in enumerate(unique, start=1)}
        return {
            chunk_id: positions.get(chunk_id) for chunk_id in self.expected_chunk_ids
        }


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_k: float | None
    hit_rate_at_k: float | None
    mrr: float | None
    answerable_count: int
    unanswerable_count: int


@dataclass(frozen=True)
class VariantReport:
    variant: str
    label: str
    candidate_k: int
    metrics: RetrievalMetrics
    case_results: tuple[CaseResult, ...]
    failure_count: int
    latency: Mapping[str, LatencySummary]
    bm25_corpus_size_mean: float | None
    delta_from_previous: Mapping[str, float | None] | None = None

    def validate_case_ids(self, case_ids: Sequence[str]) -> None:
        actual = tuple(result.case_key for result in self.case_results)
        if tuple(case_ids) != actual:
            raise ValueError("variant report must contain the same cases in order")


@dataclass(frozen=True)
class RerankerProfile:
    model_id: str
    model_revision: str
    candidate_limit: int
    max_length: int | None
    explicit_warmup: bool


@dataclass(frozen=True)
class EvaluationReport:
    """Validated A/B/C report tied to one manifest and frozen golden snapshot."""

    dataset_kind: DatasetKind
    dataset_version: str
    embedding_fingerprint: str
    chunking_fingerprint: str
    chunk_snapshot_sha256: str
    query_snapshot_sha256: str
    k: int
    candidate_k: int
    source_document_count: int
    variants: Mapping[str, VariantReport]
    reranker_profile: RerankerProfile | None = None

    def __post_init__(self) -> None:
        if set(self.variants) != set(VARIANTS):
            raise ValueError("report must contain exactly variants A, B and C")
        case_ids: tuple[str, ...] | None = None
        for name in VARIANTS:
            report = self.variants[name]
            if report.variant != name or report.label != VARIANTS[name]:
                raise ValueError("variant names and labels must match the benchmark")
            current_ids = tuple(row.case_key for row in report.case_results)
            if case_ids is None:
                case_ids = current_ids
            elif current_ids != case_ids:
                raise ValueError("all variants must contain the same cases in order")

    @property
    def is_real_document_quality(self) -> bool:
        return self.dataset_kind is DatasetKind.REAL_DOCUMENT_QUALITY

    def validate_case_ids(self, case_ids: Sequence[str]) -> None:
        expected = tuple(case_ids)
        for variant in self.variants.values():
            variant.validate_case_ids(expected)


def _validate_run_inputs(
    variants: Mapping[str, RetrievalCallable], k: int, candidate_k: int
) -> None:
    if set(variants) != set(VARIANTS):
        raise ValueError("variants must contain exactly A, B and C")
    if any(not callable(callback) for callback in variants.values()):
        raise ValueError("every retrieval variant must be callable")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= _MAX_FINAL_K:
        raise ValueError("k must be an integer from 1 to 5")
    if (
        isinstance(candidate_k, bool)
        or not isinstance(candidate_k, int)
        or not k <= candidate_k <= _MAX_CANDIDATE_K
    ):
        raise ValueError("candidate_k must be an integer from k through 20")


def _to_output(value: RetrievalOutput | Sequence[str]) -> RetrievalOutput:
    if isinstance(value, RetrievalOutput):
        return value
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError("retrieval callable must return IDs or RetrievalOutput")
    return RetrievalOutput(chunk_ids=tuple(value))


def _safe_failure(exc: Exception) -> RetrievalFailure:
    """Keep failed cases visible without leaking provider/DB exception details."""
    return RetrievalFailure(error=f"{type(exc).__name__}: retrieval failed")


def _variant_report(
    name: str,
    candidate_k: int,
    cases: Sequence[GoldenCase],
    case_results: tuple[CaseResult, ...],
    k: int,
) -> VariantReport:
    metric_results = [row.result for row in case_results]
    latency_samples: dict[str, list[float]] = {"total": []}
    for row in case_results:
        latency_samples["total"].append(row.total_latency_ms)
        for stage, value in row.stage_latencies_ms.items():
            latency_samples.setdefault(stage, []).append(value)
    latency = {
        stage: latency_summary(values) for stage, values in latency_samples.items()
    }
    bm25_sizes = [
        row.bm25_corpus_size for row in case_results if row.bm25_corpus_size is not None
    ]
    answerable_count = sum(case.answerable for case in cases)
    return VariantReport(
        variant=name,
        label=VARIANTS[name],
        candidate_k=candidate_k,
        metrics=RetrievalMetrics(
            recall_at_k=recall_at_k(cases, metric_results, k),
            hit_rate_at_k=hit_rate_at_k(cases, metric_results, k),
            mrr=mean_reciprocal_rank(cases, metric_results),
            answerable_count=answerable_count,
            unanswerable_count=len(cases) - answerable_count,
        ),
        case_results=case_results,
        failure_count=sum(
            isinstance(row.result, RetrievalFailure) for row in case_results
        ),
        latency=latency,
        bm25_corpus_size_mean=(
            sum(bm25_sizes) / len(bm25_sizes) if bm25_sizes else None
        ),
    )


def _delta(current: VariantReport, previous: VariantReport) -> dict[str, float | None]:
    metrics = {}
    for field_name in ("recall_at_k", "hit_rate_at_k", "mrr"):
        current_value = getattr(current.metrics, field_name)
        previous_value = getattr(previous.metrics, field_name)
        metrics[field_name] = (
            None
            if current_value is None or previous_value is None
            else current_value - previous_value
        )
    return metrics


def _serialize_result(result: tuple[str, ...] | RetrievalFailure) -> object:
    if isinstance(result, RetrievalFailure):
        return {"failure": result.error}
    return list(result)


def _report_payload(report: EvaluationReport) -> dict[str, object]:
    return {
        "dataset_kind": report.dataset_kind.value,
        "dataset_version": report.dataset_version,
        "is_real_document_quality": report.is_real_document_quality,
        "embedding_fingerprint": report.embedding_fingerprint,
        "chunking_fingerprint": report.chunking_fingerprint,
        "chunk_snapshot_sha256": report.chunk_snapshot_sha256,
        "query_snapshot_sha256": report.query_snapshot_sha256,
        "source_document_count": report.source_document_count,
        "k": report.k,
        "candidate_k": report.candidate_k,
        "reranker_profile": (
            asdict(report.reranker_profile) if report.reranker_profile else None
        ),
        "variants": {
            name: {
                "label": variant.label,
                "candidate_k": variant.candidate_k,
                "metrics": asdict(variant.metrics),
                "failure_count": variant.failure_count,
                "latency": {
                    stage: asdict(summary) for stage, summary in variant.latency.items()
                },
                "bm25_corpus_size_mean": variant.bm25_corpus_size_mean,
                "delta_from_previous": variant.delta_from_previous,
                "cases": [
                    {
                        "case_index": row.case_index,
                        "question": row.question,
                        "answerable": row.answerable,
                        "expected_chunk_ids": list(row.expected_chunk_ids),
                        "result": _serialize_result(row.result),
                        "relevant_ranks": row.relevant_ranks,
                        "stage_latencies_ms": dict(row.stage_latencies_ms),
                        "total_latency_ms": row.total_latency_ms,
                        "bm25_corpus_size": row.bm25_corpus_size,
                    }
                    for row in variant.case_results
                ],
            }
            for name, variant in report.variants.items()
        },
    }


def _render_markdown(report: EvaluationReport) -> str:
    if report.dataset_kind is DatasetKind.SYNTHETIC_CORRECTNESS:
        heading_note = (
            "Synthetic correctness only; these results are not real-document "
            "quality evaluation."
        )
    else:
        heading_note = "Real-document quality evaluation on a reviewed, frozen corpus."
    lines = [
        f"# Retrieval comparison: {report.dataset_version}",
        "",
        f"**Dataset:** `{report.dataset_kind.value}` — {heading_note}",
        f"**Final top_k:** {report.k}  |  **Candidate pool:** {report.candidate_k}",
        f"**Chunk snapshot SHA-256:** `{report.chunk_snapshot_sha256}`",
        f"**Query snapshot SHA-256:** `{report.query_snapshot_sha256}`",
        f"**Source documents:** {report.source_document_count}",
    ]
    if profile := report.reranker_profile:
        limit = profile.max_length or "model default"
        lines += [
            f"**Reranker:** `{profile.model_id}@{profile.model_revision}`",
            f"**Rerank candidates:** {profile.candidate_limit} "
            f"| **Max tokens:** {limit}",
            f"**Explicit model warmup:** {'yes' if profile.explicit_warmup else 'no'}",
        ]
    lines += [
        "",
        "| Variant | Recall@k | HitRate@k | MRR | Failures | p50 total ms | "
        "p95 total ms | BM25 corpus mean | Δ from previous |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, variant in report.variants.items():
        total_latency = variant.latency["total"]
        delta = variant.delta_from_previous
        delta_text = (
            "—"
            if delta is None
            else ", ".join(
                f"{key} {value:+.3f}" if value is not None else f"{key} N/A"
                for key, value in delta.items()
            )
        )
        lines.append(
            "| {name}: {label} | {recall} | {hit_rate} | {mrr} | {failures} | "
            "{p50:.2f} | {p95:.2f} | {corpus} | {delta} |".format(
                name=name,
                label=variant.label,
                recall=_format_metric(variant.metrics.recall_at_k),
                hit_rate=_format_metric(variant.metrics.hit_rate_at_k),
                mrr=_format_metric(variant.metrics.mrr),
                failures=variant.failure_count,
                p50=total_latency.p50_ms,
                p95=total_latency.p95_ms,
                corpus=(
                    "N/A"
                    if variant.bm25_corpus_size_mean is None
                    else f"{variant.bm25_corpus_size_mean:.1f}"
                ),
                delta=delta_text,
            )
        )
    lines.extend(["", "## Stage latency (p50 / p95 ms)", ""])
    for name, variant in report.variants.items():
        stages = ", ".join(
            f"{stage}: {summary.p50_ms:.2f}/{summary.p95_ms:.2f}"
            for stage, summary in variant.latency.items()
        )
        lines.append(f"- **{name} {variant.label}:** {stages}")
    lines.extend(["", "## Per-query ranks and failures", ""])
    for name, variant in report.variants.items():
        lines.append(f"### {name}: {variant.label}")
        for row in variant.case_results:
            if isinstance(row.result, RetrievalFailure):
                result_text = f"FAILED ({row.result.error})"
            else:
                result_text = (
                    ", ".join(
                        f"{chunk_id} rank {rank if rank is not None else '—'}"
                        for chunk_id, rank in row.relevant_ranks.items()
                    )
                    or "no relevant chunks"
                )
            lines.append(f"- Case {row.case_index + 1} — {row.question}: {result_text}")
    return "\n".join(lines) + "\n"


def _format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def run_retrieval(
    dataset_path: Path,
    variants: Mapping[str, RetrievalCallable],
    *,
    k: int = 5,
    candidate_k: int = 20,
    output_dir: Path,
    reranker_profile: RerankerProfile | None = None,
) -> EvaluationReport:
    """Run all three variants on exactly the same ordered cases and snapshot."""
    _validate_run_inputs(variants, k, candidate_k)
    dataset: EvaluationDataset = load_evaluation_dataset(dataset_path)
    report_variants: dict[str, VariantReport] = {}
    for name in VARIANTS:
        callback = variants[name]
        variant_candidate_k = k if name == "A" else candidate_k
        rows: list[CaseResult] = []
        for case_index, case in enumerate(dataset.cases):
            request = RetrievalRequest(
                variant=name,
                case_index=case_index,
                case=case,
                k=k,
                candidate_k=variant_candidate_k,
                dataset_version=dataset.version,
                chunk_snapshot_sha256=dataset.chunk_snapshot_sha256,
                query_snapshot_sha256=dataset.query_snapshot_sha256,
                embedding_fingerprint=dataset.embedding_fingerprint,
                chunking_fingerprint=dataset.chunking_version,
            )
            started = time.perf_counter()
            try:
                output = _to_output(callback(request))
                elapsed_ms = (time.perf_counter() - started) * 1000
                result: tuple[str, ...] | RetrievalFailure = tuple(
                    dict.fromkeys(output.chunk_ids)
                )[:k]
                stage_latencies = {
                    stage: float(value)
                    for stage, value in output.stage_latencies_ms.items()
                }
                bm25_size = output.bm25_corpus_size
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000
                result = _safe_failure(exc)
                stage_latencies = {}
                bm25_size = None
            rows.append(
                CaseResult(
                    case_index=case_index,
                    question=case.question,
                    answerable=case.answerable,
                    expected_chunk_ids=case.expected_chunk_ids,
                    result=result,
                    stage_latencies_ms=stage_latencies,
                    total_latency_ms=max(0.0, elapsed_ms),
                    bm25_corpus_size=bm25_size,
                )
            )
        report_variants[name] = _variant_report(
            name, variant_candidate_k, dataset.cases, tuple(rows), k
        )

    report_variants["B"] = replace(
        report_variants["B"],
        delta_from_previous=_delta(report_variants["B"], report_variants["A"]),
    )
    report_variants["C"] = replace(
        report_variants["C"],
        delta_from_previous=_delta(report_variants["C"], report_variants["B"]),
    )
    report = EvaluationReport(
        dataset_kind=dataset.kind,
        dataset_version=dataset.version,
        embedding_fingerprint=dataset.embedding_fingerprint,
        chunking_fingerprint=dataset.chunking_version,
        chunk_snapshot_sha256=dataset.chunk_snapshot_sha256,
        query_snapshot_sha256=dataset.query_snapshot_sha256,
        k=k,
        candidate_k=candidate_k,
        source_document_count=dataset.source_document_count,
        variants=report_variants,
        reranker_profile=reranker_profile,
    )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "retrieval-report.json").write_text(
        json.dumps(_report_payload(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (target / "retrieval-report.md").write_text(
        _render_markdown(report), encoding="utf-8"
    )
    return report


# Keep imports local to the public execution path to avoid a project dependency
# on dataclasses.replace for the serialized schema.
