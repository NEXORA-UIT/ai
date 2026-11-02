"""Offline fake-answer runner tests; no model claims or API calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evaluation.answer_metrics import load_canonical_source_chunks
from evaluation.faithfulness import FaithfulnessClaim, FaithfulnessResult
from evaluation.run_answers import run_answers
from evaluation.schemas import DatasetKind
from src.generation.schemas import AnswerResult, Citation
from src.retrieval.schemas import SourceChunk

_DATASET = Path(__file__).parents[2] / "evaluation" / "golden" / "synthetic"
_BOARD_ID = "11111111-1111-4111-8111-111111111111"
_DOCUMENT_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1"
_CHUNK_ID = "a1000000-0000-4000-8000-000000000001"


def _real_dataset(root: Path) -> Path:
    root.mkdir()
    source_path = root / "sources" / "guide.pdf"
    source_path.parent.mkdir()
    source_path.write_bytes(b"%PDF-1.7\nmanifest-only test fixture\n")
    snapshot = root / "corpus.jsonl"
    snapshot.write_text(
        json.dumps(
            {
                "document_id": _DOCUMENT_ID,
                "board_id": _BOARD_ID,
                "file_name": "guide.pdf",
                "storage_key": "synthetic://answer-test/guide.pdf",
                "status": "READY",
                "metadata": {
                    "model_id": "BAAI/bge-m3",
                    "model_revision": "test-revision",
                    "dimension": 1024,
                    "normalized": True,
                },
                "chunks": [
                    {
                        "chunk_id": _CHUNK_ID,
                        "chunk_index": 0,
                        "content": "Approved procedure for the test.",
                        "page_number": 1,
                        "start_offset": 0,
                        "end_offset": 33,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    queries = root / "queries.jsonl"
    queries.write_text(
        json.dumps(
            {
                "question": "What is the approved procedure?",
                "board_id": _BOARD_ID,
                "expected_chunk_ids": [_CHUNK_ID],
                "answerable": True,
                "expected_facts": ["approved procedure"],
                "expected_citation_ids": [_CHUNK_ID],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "real_document_quality",
                "version": "real-test-v1",
                "chunk_snapshot": snapshot.name,
                "chunk_snapshot_sha256": hashlib.sha256(
                    snapshot.read_bytes()
                ).hexdigest(),
                "queries": queries.name,
                "queries_sha256": hashlib.sha256(queries.read_bytes()).hexdigest(),
                "embedding": {"model_id": "BAAI/bge-m3", "revision": "test-revision"},
                "chunking": {"version": "test-chunking-v1"},
                "human_labels_reviewed": True,
                "source_documents": [
                    {
                        "document_id": _DOCUMENT_ID,
                        "path": "sources/guide.pdf",
                        "source_url": "https://example.test/guide.pdf",
                        "license": "CC-BY-4.0",
                        "provenance": "Manifest-only test fixture",
                        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


def test_runner_reports_citations_refusals_failures_and_latency(tmp_path: Path) -> None:
    corpus = load_canonical_source_chunks(_DATASET)

    def answer(case):
        if not case.answerable:
            return AnswerResult(
                text="I cannot answer from the available evidence.",
                citations=(),
                refused=True,
            )
        if case.question.startswith("How"):
            source = corpus[case.expected_citation_ids[0]]
            return AnswerResult(
                text="Use the project access settings.",
                citations=(
                    Citation(
                        chunk_id=source.chunk_id,
                        document_id=source.document_id,
                        file_name=source.file_name,
                        page_number=source.page_number,
                        start_offset=source.start_offset,
                        end_offset=source.end_offset,
                    ),
                ),
                refused=False,
            )
        raise RuntimeError("fake answer failure")

    report = run_answers(_DATASET, answer, output_dir=tmp_path)

    assert report.dataset_kind is DatasetKind.SYNTHETIC_CORRECTNESS
    assert report.answer_client_kind == "fake_correctness"
    assert report.is_real_document_quality is False
    assert report.citation_accuracy == 0.5
    assert report.failure_count == 1
    assert report.refusal_count == 1
    assert report.correct_unanswerable_refusals == 1
    assert report.latency["generation"].sample_count == 3
    assert report.latency["total"].sample_count == 3
    assert report.faithfulness is None
    assert (tmp_path / "answer-report.json").is_file()
    assert (tmp_path / "answer-report.md").is_file()

    payload = json.loads((tmp_path / "answer-report.json").read_text(encoding="utf-8"))
    assert payload["dataset_kind"] == "synthetic_correctness"
    assert payload["faithfulness"] is None
    assert "storage_key" not in json.dumps(payload)


def test_fake_answer_client_cannot_claim_real_document_quality(tmp_path: Path) -> None:
    dataset = _real_dataset(tmp_path / "real")
    corpus = load_canonical_source_chunks(dataset)

    def fake_answer(case):
        source = corpus[case.expected_citation_ids[0]]
        return AnswerResult(
            text="The approved procedure is in the guide.",
            citations=(
                Citation(
                    chunk_id=source.chunk_id,
                    document_id=source.document_id,
                    file_name=source.file_name,
                    page_number=source.page_number,
                    start_offset=source.start_offset,
                    end_offset=source.end_offset,
                ),
            ),
            refused=False,
        )

    report = run_answers(dataset, fake_answer, output_dir=tmp_path / "report")

    assert report.dataset_kind is DatasetKind.REAL_DOCUMENT_QUALITY
    assert report.answer_client_kind == "fake_correctness"
    assert report.is_real_document_quality is False
    markdown = (tmp_path / "report" / "answer-report.md").read_text(encoding="utf-8")
    assert "not real-document answer quality" in markdown.lower()


def test_runner_rejects_invalid_answer_outputs(tmp_path: Path) -> None:
    report = run_answers(
        _DATASET, lambda _case: {"answer": "not typed"}, output_dir=tmp_path
    )

    assert report.failure_count == 3
    assert report.citation_accuracy == 0.0
    assert all(row.failure is not None for row in report.case_results)


class _FakeJudge:
    model_id = "fake-judge-v1"

    def __init__(self, result: FaithfulnessResult | Exception) -> None:
        self.result = result
        self.calls: list[tuple[str, tuple[SourceChunk, ...]]] = []

    def judge(self, answer: str, evidence):
        self.calls.append((answer, tuple(evidence)))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _faithful_result() -> FaithfulnessResult:
    return FaithfulnessResult(
        score=1.0,
        claims=(
            FaithfulnessClaim("Approved procedure is documented.", True, (_CHUNK_ID,)),
        ),
    )


def test_runner_reports_claim_verdicts_and_best_effort_trace_id(tmp_path: Path) -> None:
    corpus = load_canonical_source_chunks(_DATASET)
    source = corpus[_CHUNK_ID]
    judge = _FakeJudge(_faithful_result())

    def answer(case):
        if not case.answerable:
            return AnswerResult(
                text="I cannot answer from the available evidence.",
                citations=(),
                refused=True,
            )
        return AnswerResult(
            text="The approved procedure is documented.",
            citations=tuple(
                Citation(
                    chunk_id=source.chunk_id,
                    document_id=source.document_id,
                    file_name=source.file_name,
                    page_number=source.page_number,
                    start_offset=source.start_offset,
                    end_offset=source.end_offset,
                )
                for _ in range(2)
            ),
            refused=False,
        )

    class _TraceSink:
        def __init__(self) -> None:
            self.calls = []

        def record_faithfulness(self, **kwargs):
            self.calls.append(kwargs)
            return "f" * 32

    trace = _TraceSink()
    report = run_answers(
        _DATASET,
        answer,
        output_dir=tmp_path,
        judge=judge,
        trace_client=trace,
    )

    assert report.faithfulness == 1.0
    assert report.faithfulness_judge_kind == "fake_correctness"
    assert report.case_results[0].faithfulness == _faithful_result()
    assert report.case_results[0].trace_id == "f" * 32
    assert len(judge.calls) == 2  # The refusal case is excluded.
    assert len(judge.calls[0][1]) == 1  # Duplicate citations do not duplicate evidence.
    assert len(trace.calls) == 2
    payload = json.loads((tmp_path / "answer-report.json").read_text(encoding="utf-8"))
    assert payload["faithfulness_status"] == "measured_fake_correctness"
    assert payload["cases"][0]["faithfulness"]["claims"][0]["claim"]
    assert payload["cases"][0]["trace_id"] == "f" * 32
    assert "storage_key" not in json.dumps(payload)


def test_runner_rejects_cross_board_evidence_before_judge_call(tmp_path: Path) -> None:
    corpus = load_canonical_source_chunks(_DATASET)
    source = corpus[_CHUNK_ID]
    answer_result = AnswerResult(
        text="An answer with a citation.",
        citations=(
            Citation(
                chunk_id=source.chunk_id,
                document_id=source.document_id,
                file_name=source.file_name,
                page_number=source.page_number,
                start_offset=source.start_offset,
                end_offset=source.end_offset,
            ),
        ),
        refused=False,
    )
    judge = _FakeJudge(_faithful_result())
    cross_board = SourceChunk(
        **{**source.__dict__, "board_id": "22222222-2222-4222-8222-222222222222"}
    )

    report = run_answers(
        _DATASET,
        lambda _case: answer_result,
        output_dir=tmp_path,
        judge=judge,
        evidence_for_answer=lambda _case, _answer: [cross_board],
    )

    assert report.case_results[0].faithfulness is None
    assert report.case_results[0].faithfulness_failure is not None
    assert "evaluation failed" in report.case_results[0].faithfulness_failure
    assert judge.calls == []


def test_judge_failure_does_not_reclassify_answer_failure(tmp_path: Path) -> None:
    corpus = load_canonical_source_chunks(_DATASET)
    source = corpus[_CHUNK_ID]
    answer_result = AnswerResult(
        text="An answer with a citation.",
        citations=(
            Citation(
                chunk_id=source.chunk_id,
                document_id=source.document_id,
                file_name=source.file_name,
                page_number=source.page_number,
                start_offset=source.start_offset,
                end_offset=source.end_offset,
            ),
        ),
        refused=False,
    )

    report = run_answers(
        _DATASET,
        lambda _case: answer_result,
        output_dir=tmp_path,
        judge=_FakeJudge(RuntimeError("sensitive provider details")),
    )

    row = report.case_results[0]
    assert row.result == answer_result
    assert row.failure is None
    assert row.faithfulness is None
    assert "sensitive provider details" not in (row.faithfulness_failure or "")
    assert report.failure_count == 0
