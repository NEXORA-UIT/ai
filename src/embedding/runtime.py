"""Process-local default embedding runtime shared by ingestion and retrieval."""

from functools import lru_cache
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.embedding.embedder import BgeM3Embedder

_initialization_lock = Lock()


def get_default_embedder() -> "BgeM3Embedder":
    with _initialization_lock:
        return _cached_embedder()


@lru_cache(maxsize=1)
def _cached_embedder() -> "BgeM3Embedder":
    from src.embedding.embedder import BgeM3Embedder

    return BgeM3Embedder()
