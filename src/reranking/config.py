"""Pinned configuration for the multilingual CPU cross-encoder."""

from os import environ

from src.retrieval.config import MAX_CANDIDATE_K

DEFAULT_RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANKER_MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
DEFAULT_RERANKER_DEVICE = "cpu"
DEFAULT_RERANKER_BATCH_SIZE = 16
DEFAULT_RERANK_CANDIDATE_K = 5
def _read_int(name: str, default: int) -> int:
    try:
        return int(environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def get_rerank_candidate_limit(*, top_k: int) -> int:
    limit = _read_int("AI_RERANK_CANDIDATE_K", DEFAULT_RERANK_CANDIDATE_K)
    if not 1 <= limit <= MAX_CANDIDATE_K:
        raise ValueError(
            f"AI_RERANK_CANDIDATE_K must be between 1 and {MAX_CANDIDATE_K}"
        )
    return max(limit, top_k)


def get_rerank_max_length() -> int | None:
    limit = _read_int("AI_RERANK_MAX_LENGTH", 0)
    if limit != 0 and not 32 <= limit <= 8192:
        raise ValueError("AI_RERANK_MAX_LENGTH must be 0 or between 32 and 8192")
    return limit or None
