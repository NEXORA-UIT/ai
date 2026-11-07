"""Comparison runner correctness tests use only the synthetic golden corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.run_retrieval import (
    VARIANTS,
    RerankerProfile,
    RetrievalOutput,
    run_retrieval,
)
from evaluation.schemas import DatasetKind, RetrievalFailure

_DATASET = Path(__file__).parents[2] / "evaluation" / "golden" / "synthetic"
_RELEVANT = "a1000000-0000-4000-8000-000000000001"
_RELEVANT_VI = "a1000000-0000-4000-8000-000000000002"


def _variants(callback):
    return {name: callback for name in VARIANTS}


def test_variants_share_snapshot_query_order_and_final_top_k(tmp_path: Path) -> None:
    seen = {name: [] for name in VARIANTS}

    def callback(request):
        seen[request.variant].append(request)
        ids = (_RELEVANT, _RELEVANT_VI) if request.case.answerable else ()
        return RetrievalOutput(
            chunk_ids=ids,
            stage_latencies_ms={"dense": 1.5, "bm25": 2.0, "rrf": 0.2},
            bm25_corpus_size=12,
        )

    report = run_retrieval(
        _DATASET,
        _variants(callback),
        k=2,
        candidate_k=20,
        output_dir=tmp_path,
        reranker_profile=RerankerProfile("reranker", "revision", 5, 192, False),
    )

    assert set(seen) == set(VARIANTS)
    signatures = {
        name: [
            (
                request.dataset_version,
                request.chunk_snapshot_sha256,
                request.query_snapshot_sha256,
                request.embedding_fingerprint,
                request.chunking_fingerprint,
                request.case.question,
                request.k,
            )
            for request in requests
        ]
        for name, requests in seen.items()
    }
    assert signatures["A"] == signatures["B"] == signatures["C"]
    assert all(
        request.candidate_k == 20 for name in ("B", "C") for request in seen[name]
    )
    assert report.dataset_kind is DatasetKind.SYNTHETIC_CORRECTNESS
    assert report.is_real_document_quality is False
    assert report.variants["A"].metrics.recall_at_k == 1.0
    assert report.variants["A"].metrics.hit_rate_at_k == 1.0
    assert report.variants["A"].metrics.mrr == 0.75
    assert report.variants["A"].bm25_corpus_size_mean == 12
    assert (tmp_path / "retrieval-report.json").is_file()
    markdown = (tmp_path / "retrieval-report.md").read_text(encoding="utf-8")
    assert "synthetic correctness" in markdown.lower()
    assert "real-document quality" in markdown.lower()
    assert "**Rerank candidates:** 5 | **Max tokens:** 192" in markdown
    assert "**Explicit model warmup:** no" in markdown
    encoded = json.loads(
        (tmp_path / "retrieval-report.json").read_text(encoding="utf-8")
    )
    assert encoded["dataset_kind"] == "synthetic_correctness"
    assert encoded["reranker_profile"] == {
        "model_id": "reranker",
        "model_revision": "revision",
        "candidate_limit": 5,
        "max_length": 192,
        "explicit_warmup": False,
    }


def test_mrr_and_case_results_use_the_same_final_top_k(tmp_path: Path) -> None:
    def callback(request):
        if not request.case.answerable:
            return RetrievalOutput(chunk_ids=())
        top_k_decoys = tuple(f"decoy-{index}" for index in range(request.k))
        if request.variant == "B":
            # B internally materializes its larger RRF candidate pool; rank 6
            # must not count when the shared final cutoff is k=5.
            return RetrievalOutput(
                chunk_ids=top_k_decoys + (request.case.expected_chunk_ids[0],)
            )
        return RetrievalOutput(chunk_ids=top_k_decoys)

    report = run_retrieval(
        _DATASET,
        _variants(callback),
        k=5,
        candidate_k=20,
        output_dir=tmp_path,
    )

    for variant in report.variants.values():
        assert variant.metrics.recall_at_k == 0.0
        assert variant.metrics.hit_rate_at_k == 0.0
        assert variant.metrics.mrr == 0.0
        assert all(
            len(row.result) <= 5
            for row in variant.case_results
            if not isinstance(row.result, RetrievalFailure)
        )
    assert report.variants["B"].case_results[0].result == tuple(
        f"decoy-{index}" for index in range(5)
    )


def test_answerable_retrieval_failure_counts_zero_and_is_reported(
    tmp_path: Path,
) -> None:
    def fails_first(request):
        if request.case.question.startswith("How"):
            raise RuntimeError("synthetic retrieval failure")
        return RetrievalOutput(chunk_ids=(_RELEVANT_VI,))

    def other(request):
        return RetrievalOutput(chunk_ids=(_RELEVANT, _RELEVANT_VI))

    variants = {"A": fails_first, "B": other, "C": other}
    report = run_retrieval(_DATASET, variants, k=1, output_dir=tmp_path)
    dense = report.variants["A"]

    assert dense.metrics.recall_at_k == 0.5
    assert dense.metrics.hit_rate_at_k == 0.5
    assert dense.metrics.mrr == 0.5
    assert dense.failure_count == 1
    assert isinstance(dense.case_results[0].result, RetrievalFailure)


def test_missing_variant_and_candidate_pool_over_cap_fail_before_calls(
    tmp_path: Path,
) -> None:
    called = False

    def callback(request):
        nonlocal called
        called = True
        return RetrievalOutput(chunk_ids=())

    with pytest.raises(ValueError, match="exactly"):
        run_retrieval(_DATASET, {"A": callback}, output_dir=tmp_path)
    with pytest.raises(ValueError, match="20"):
        run_retrieval(
            _DATASET,
            _variants(callback),
            candidate_k=21,
            output_dir=tmp_path,
        )
    assert called is False


def test_variant_errors_are_zeroed_and_missing_case_rows_are_rejected(
    tmp_path: Path,
) -> None:
    def callback(request):
        return RetrievalOutput(chunk_ids=(_RELEVANT,))

    report = run_retrieval(_DATASET, _variants(callback), output_dir=tmp_path)
    assert all(len(variant.case_results) == 3 for variant in report.variants.values())

    with pytest.raises(ValueError, match="same cases"):
        report.validate_case_ids(("only-one",))


def test_bm25_corpus_size_validation_and_failure_rows(tmp_path: Path) -> None:
    def invalid(request):
        return RetrievalOutput(chunk_ids=(), bm25_corpus_size=-1)

    report = run_retrieval(_DATASET, _variants(invalid), output_dir=tmp_path)
    assert report.variants["B"].failure_count == 3
    assert all(
        isinstance(row.result, RetrievalFailure)
        for row in report.variants["B"].case_results
    )
