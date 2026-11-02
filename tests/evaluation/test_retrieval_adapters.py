"""The evaluation adapters delegate to the existing retrieval implementations."""

from __future__ import annotations

from types import SimpleNamespace

from evaluation import adapters
from evaluation.run_retrieval import RetrievalRequest
from evaluation.schemas import GoldenCase


class _Embedder:
    model_name = "BAAI/bge-m3"
    revision = "test-revision"
    dimension = 1024

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[1.0] + [0.0] * 1023]


def _request(variant: str, candidate_k: int) -> RetrievalRequest:
    return RetrievalRequest(
        variant=variant,
        case_index=0,
        case=GoldenCase(
            question="What does the guide say?",
            board_id="11111111-1111-4111-8111-111111111111",
            expected_chunk_ids=("chunk-1",),
            answerable=True,
            expected_facts=("guide",),
            expected_citation_ids=("chunk-1",),
        ),
        k=3,
        candidate_k=candidate_k,
        dataset_version="synthetic-v1",
        chunk_snapshot_sha256="snapshot",
        query_snapshot_sha256="queries",
        embedding_fingerprint="BAAI/bge-m3@test-revision",
        chunking_fingerprint="synthetic-v1",
    )


def test_adapters_use_dense_sparse_rrf_and_reranker_without_copying_ranking(
    monkeypatch,
) -> None:
    embedder = _Embedder()
    calls: dict[str, list[object]] = {
        "dense": [],
        "sparse": [],
        "rrf": [],
        "rerank": [],
    }
    dense_hit = SimpleNamespace(chunk_id="dense-1")
    sparse_hit = SimpleNamespace(source=SimpleNamespace(chunk_id="sparse-1"))
    fused_hit = SimpleNamespace(source=SimpleNamespace(chunk_id="fused-1"))
    reranked_hit = SimpleNamespace(source=SimpleNamespace(chunk_id="reranked-1"))

    def fake_dense(vector, board_id, **kwargs):
        calls["dense"].append((vector, board_id, kwargs["top_k"]))
        return [dense_hit]

    def fake_sparse(question, board_id, **kwargs):
        calls["sparse"].append((question, board_id, kwargs["top_k"]))
        return SimpleNamespace(hits=[sparse_hit], corpus_size=9)

    def fake_rrf(candidates, **kwargs):
        calls["rrf"].append((candidates, kwargs["top_k"]))
        return [fused_hit]

    def fake_rerank(question, candidates, **kwargs):
        calls["rerank"].append((question, candidates, kwargs["top_k"]))
        return [reranked_hit]

    monkeypatch.setattr(adapters, "search_by_vector", fake_dense)
    monkeypatch.setattr(adapters, "search_sparse_with_stats", fake_sparse)
    monkeypatch.setattr(adapters, "fuse_rrf", fake_rrf)
    monkeypatch.setattr(adapters, "rerank_chunks", fake_rerank)
    variants = adapters.build_retrieval_variants(embedder=embedder)

    dense = variants["A"](_request("A", 3))
    hybrid = variants["B"](_request("B", 20))
    reranked = variants["C"](_request("C", 20))

    assert dense.chunk_ids == ("dense-1",)
    assert hybrid.chunk_ids == ("fused-1",)
    assert reranked.chunk_ids == ("reranked-1",)
    assert hybrid.bm25_corpus_size == reranked.bm25_corpus_size == 9
    assert calls["dense"][0][2] == 3
    assert [item[2] for item in calls["dense"][1:]] == [20, 20]
    assert [item[2] for item in calls["sparse"]] == [20, 20]
    assert [item[1] for item in calls["rrf"]] == [20, 20]
    assert calls["rerank"][0][2] == 3
    assert len(embedder.calls) == 3
    assert set(dense.stage_latencies_ms) == {"embedding", "dense"}
    assert {"dense", "bm25", "rrf"}.issubset(hybrid.stage_latencies_ms)
    assert "rerank" in reranked.stage_latencies_ms
