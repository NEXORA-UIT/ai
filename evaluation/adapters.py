"""Thin benchmark adapters over the existing M2/M3/M4/M5 retrieval code."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import cast

from evaluation.run_retrieval import (
    RetrievalCallable,
    RetrievalOutput,
    RetrievalRequest,
)
from src.embedding.config import DEFAULT_NORMALIZE_EMBEDDINGS
from src.embedding.embedder import Embedder
from src.ingestion.persistence.storage import DocumentStorage
from src.reranking.cross_encoder import Reranker
from src.reranking.select import rerank_chunks
from src.retrieval.dense import get_default_embedder, search_by_vector
from src.retrieval.fusion import fuse_rrf
from src.retrieval.metadata import EmbeddingFingerprint
from src.retrieval.schemas import HybridCandidates, RetrievedChunk, SparseHit
from src.retrieval.sparse import search_sparse_with_stats


def build_retrieval_variants(
    *,
    embedder: Embedder | None = None,
    storage: DocumentStorage | None = None,
    reranker: Reranker | None = None,
) -> Mapping[str, RetrievalCallable]:
    """Build A/B/C adapters, reusing injected/default model instances.

    These adapters call the production dense, sparse, RRF and reranker
    functions directly. Timing is captured around those existing stages; this
    module contains no duplicate retrieval or ranking implementation.
    """
    selected_embedder = embedder

    def active_embedder() -> Embedder:
        nonlocal selected_embedder
        if selected_embedder is None:
            selected_embedder = get_default_embedder()
        return selected_embedder

    def embed(
        request: RetrievalRequest,
    ) -> tuple[list[float], EmbeddingFingerprint, float]:
        model = active_embedder()
        started = time.perf_counter()
        vectors = model.embed_texts([request.case.question])
        elapsed_ms = (time.perf_counter() - started) * 1000
        if len(vectors) != 1:
            raise ValueError("embedder must return one vector for one query")
        fingerprint = EmbeddingFingerprint(
            model_id=model.model_name,
            model_revision=model.revision,
            dimension=model.dimension,
            normalized=DEFAULT_NORMALIZE_EMBEDDINGS,
        )
        return [float(value) for value in vectors[0]], fingerprint, elapsed_ms

    def dense_only(request: RetrievalRequest) -> RetrievalOutput:
        vector, fingerprint, embedding_ms = embed(request)
        started = time.perf_counter()
        dense_hits = search_by_vector(
            vector,
            request.case.board_id,
            top_k=request.k,
            fingerprint=fingerprint,
            storage=storage,
        )
        dense_ms = (time.perf_counter() - started) * 1000
        return RetrievalOutput(
            chunk_ids=tuple(hit.chunk_id for hit in dense_hits),
            stage_latencies_ms={"embedding": embedding_ms, "dense": dense_ms},
        )

    def hybrid_candidates(request: RetrievalRequest):
        vector, fingerprint, embedding_ms = embed(request)
        started = time.perf_counter()
        dense_hits = search_by_vector(
            vector,
            request.case.board_id,
            top_k=request.candidate_k,
            fingerprint=fingerprint,
            storage=storage,
        )
        dense_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        sparse_result = search_sparse_with_stats(
            request.case.question,
            request.case.board_id,
            top_k=request.candidate_k,
            fingerprint=fingerprint,
            storage=storage,
        )
        bm25_ms = (time.perf_counter() - started) * 1000
        # Type narrowing is constrained by the protocols above and the concrete
        # records returned by the retrieval package.
        candidates = HybridCandidates(
            dense=cast(list[RetrievedChunk], dense_hits),
            sparse=cast(list[SparseHit], sparse_result.hits),
        )
        return (
            candidates,
            {
                "embedding": embedding_ms,
                "dense": dense_ms,
                "bm25": bm25_ms,
            },
            sparse_result.corpus_size,
        )

    def hybrid_rrf(request: RetrievalRequest) -> RetrievalOutput:
        candidates, timings, corpus_size = hybrid_candidates(request)
        started = time.perf_counter()
        fused = fuse_rrf(candidates, top_k=request.candidate_k)
        timings["rrf"] = (time.perf_counter() - started) * 1000
        return RetrievalOutput(
            chunk_ids=tuple(hit.source.chunk_id for hit in fused),
            stage_latencies_ms=timings,
            bm25_corpus_size=corpus_size,
        )

    def hybrid_rrf_reranked(request: RetrievalRequest) -> RetrievalOutput:
        candidates, timings, corpus_size = hybrid_candidates(request)
        started = time.perf_counter()
        fused = fuse_rrf(candidates, top_k=request.candidate_k)
        timings["rrf"] = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        reranked = rerank_chunks(
            request.case.question,
            fused,
            top_k=request.k,
            reranker=reranker,
        )
        timings["rerank"] = (time.perf_counter() - started) * 1000
        return RetrievalOutput(
            chunk_ids=tuple(hit.source.chunk_id for hit in reranked),
            stage_latencies_ms=timings,
            bm25_corpus_size=corpus_size,
        )

    return {"A": dense_only, "B": hybrid_rrf, "C": hybrid_rrf_reranked}
