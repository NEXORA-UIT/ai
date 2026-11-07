"""Keep retrieval configuration imports independent of model runtimes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_retrieval_config_does_not_eagerly_import_embedding_models() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script = (
        "import sys; "
        "from src.retrieval.config import DEFAULT_CANDIDATE_K; "
        "assert DEFAULT_CANDIDATE_K == 20; "
        "assert 'src.retrieval.dense' not in sys.modules; "
        "assert 'src.embedding.embedder' not in sys.modules; "
        "assert 'torch' not in sys.modules"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_evaluation_import_preserves_callable_retrieval_export() -> None:
    script = (
        "import evaluation.adapters; "
        "from src.retrieval import search_documents; "
        "assert callable(search_documents)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_embedding_config_does_not_import_inference_runtime() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script = (
        "import sys; "
        "from src.embedding.config import DEFAULT_DENSE_DIMENSION; "
        "assert DEFAULT_DENSE_DIMENSION == 1024; "
        "assert 'src.embedding.embedder' not in sys.modules; "
        "assert 'torch' not in sys.modules"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_existing_embedding_and_retrieval_exports_remain_importable_lazily() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script = (
        "import sys; "
        "from src.embedding import DEFAULT_DENSE_DIMENSION; "
        "from src.retrieval import SearchResult, search_documents; "
        "assert DEFAULT_DENSE_DIMENSION == 1024; "
        "assert SearchResult.__name__ == 'SearchResult'; "
        "assert callable(search_documents); "
        "assert 'torch' not in sys.modules"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
