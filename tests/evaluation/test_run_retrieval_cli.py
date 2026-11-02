"""Safety checks for the opt-in real corpus benchmark command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evaluation.manifest import EvaluationDataset
from evaluation.run_retrieval_cli import (
    _verify_database_and_seed,
    validate_test_database_target,
)
from evaluation.schemas import DatasetKind


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "postgresql://u:p@localhost:5432/nexora_ai_retrieval_test",
            "nexora_ai_retrieval_test",
        ),
        ("postgresql://u:p@localhost:5432/nexora_ai_dev", "nexora_ai_retrieval_test"),
        ("", "nexora_ai_retrieval_test"),
        ("postgresql://u:p@localhost:5432/nexora_ai_retrieval_test", "nexora_ai_prod"),
    ],
)
def test_database_target_requires_exact_test_database(url: str, expected: str) -> None:
    if expected == "nexora_ai_retrieval_test" and url.endswith(expected):
        assert validate_test_database_target(url, expected) == expected
    else:
        with pytest.raises(ValueError):
            validate_test_database_target(url, expected)


def test_database_preflight_checks_connected_name_and_seeded_chunk_scope() -> None:
    cursor = MagicMock()
    cursor.fetchone.return_value = ("nexora_ai_retrieval_test",)
    chunk_id = "a1000000-0000-4000-8000-000000000001"
    board_id = "11111111-1111-4111-8111-111111111111"
    cursor.fetchall.return_value = [(chunk_id, board_id)]
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.cursor.return_value.__enter__.return_value = cursor
    storage = MagicMock()
    storage.get_connection.return_value = connection
    dataset = EvaluationDataset(
        root=Path("."),
        kind=DatasetKind.SYNTHETIC_CORRECTNESS,
        version="test-v1",
        chunk_snapshot_path=Path("snapshot.jsonl"),
        chunk_snapshot_sha256="snapshot",
        queries_path=Path("queries.jsonl"),
        query_snapshot_sha256="queries",
        embedding_model_id="model",
        embedding_revision="revision",
        chunking_version="chunking",
        cases=(),
        snapshot_chunk_boards={chunk_id: board_id},
        source_document_count=0,
    )

    _verify_database_and_seed(storage, "nexora_ai_retrieval_test", dataset)

    assert cursor.execute.call_count == 2
    connection.close.assert_called_once_with()
