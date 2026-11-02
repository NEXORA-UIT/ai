"""Safety tests for the explicitly opted-in synthetic corpus seeder."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evaluation.seed_corpus import UnsafeSeedTargetError, seed_corpus
from src.embedding.config import (
    DEFAULT_DENSE_DIMENSION,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
)

_TEST_URL = "postgresql://test_user:test_pass@127.0.0.1/nexora_ai_retrieval_test"
_DATASET = Path("evaluation/golden/synthetic/corpus.jsonl")


class _FixedEmbedder:
    model_name = DEFAULT_MODEL_ID
    revision = DEFAULT_MODEL_REVISION
    dimension = DEFAULT_DENSE_DIMENSION

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * (DEFAULT_DENSE_DIMENSION - 1) for _ in texts]


def _fake_storage(database_name: str):
    cursor = MagicMock()
    cursor.fetchone.return_value = (database_name,)
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.cursor.return_value.__enter__.return_value = cursor
    storage = MagicMock()
    storage.get_connection.return_value = connection
    return storage, connection, cursor


def test_seed_requires_explicit_opt_in_before_database_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_type = MagicMock()
    monkeypatch.setattr("evaluation.seed_corpus.DocumentStorage", storage_type)
    monkeypatch.setenv("TEST_DATABASE_URL", _TEST_URL)

    with pytest.raises(UnsafeSeedTargetError, match="explicit opt-in"):
        seed_corpus(
            test_database_url=_TEST_URL,
            expected_database_name="nexora_ai_retrieval_test",
            allow_test_seed=False,
            dataset_path=_DATASET,
            embedder=_FixedEmbedder(),
        )

    storage_type.assert_not_called()


def test_seed_requires_exact_test_database_url_and_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_type = MagicMock()
    monkeypatch.setattr("evaluation.seed_corpus.DocumentStorage", storage_type)
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", _TEST_URL)

    with pytest.raises(UnsafeSeedTargetError, match="TEST_DATABASE_URL"):
        seed_corpus(
            test_database_url="",
            expected_database_name="nexora_ai_retrieval_test",
            allow_test_seed=True,
            dataset_path=_DATASET,
            embedder=_FixedEmbedder(),
        )

    storage_type.assert_not_called()


@pytest.mark.parametrize(
    ("url", "expected_database_name"),
    [
        (
            _TEST_URL.replace("nexora_ai_retrieval_test", "nexora_ai_dev"),
            "nexora_ai_dev",
        ),
        (_TEST_URL, "nexora_ai_staging"),
    ],
    ids=["dev-target", "non-test-expected-name"],
)
def test_seed_rejects_non_test_target_before_database_access(
    monkeypatch: pytest.MonkeyPatch,
    url: str,
    expected_database_name: str,
) -> None:
    storage_type = MagicMock()
    monkeypatch.setattr("evaluation.seed_corpus.DocumentStorage", storage_type)
    monkeypatch.setenv("TEST_DATABASE_URL", url)

    with pytest.raises(UnsafeSeedTargetError, match="_test"):
        seed_corpus(
            test_database_url=url,
            expected_database_name=expected_database_name,
            allow_test_seed=True,
            dataset_path=_DATASET,
            embedder=_FixedEmbedder(),
        )

    storage_type.assert_not_called()


def test_seed_checks_connected_database_before_any_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, connection, cursor = _fake_storage("nexora_ai_staging")
    monkeypatch.setattr("evaluation.seed_corpus.DocumentStorage", lambda **_: storage)
    monkeypatch.setenv("TEST_DATABASE_URL", _TEST_URL)

    with pytest.raises(UnsafeSeedTargetError, match="connected database"):
        seed_corpus(
            test_database_url=_TEST_URL,
            expected_database_name="nexora_ai_retrieval_test",
            allow_test_seed=True,
            dataset_path=_DATASET,
            embedder=_FixedEmbedder(),
        )

    assert len(cursor.execute.call_args_list) == 1
    assert "current_database" in cursor.execute.call_args.args[0].casefold()
    connection.close.assert_called_once()


def test_seed_only_writes_fixed_dataset_ids_inside_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, connection, cursor = _fake_storage("nexora_ai_retrieval_test")
    monkeypatch.setattr("evaluation.seed_corpus.DocumentStorage", lambda **_: storage)
    monkeypatch.setenv("TEST_DATABASE_URL", _TEST_URL)

    seed_corpus(
        test_database_url=_TEST_URL,
        expected_database_name="nexora_ai_retrieval_test",
        allow_test_seed=True,
        dataset_path=_DATASET,
        embedder=_FixedEmbedder(),
    )

    calls = cursor.execute.call_args_list
    assert "current_database" in calls[0].args[0].casefold()
    assert all(
        not any(token in call.args[0].casefold() for token in ("truncate", "drop"))
        for call in calls
    )
    seeded_document_ids = {
        call.args[1]["id"]
        for call in calls
        if "insert into documents" in call.args[0].casefold()
    }
    corpus_documents = {
        json.loads(line)["document_id"]
        for line in _DATASET.read_text(encoding="utf-8").splitlines()
    }
    assert {str(item) for item in seeded_document_ids} == corpus_documents
    assert connection.close.call_count == 2


@pytest.mark.integration
@pytest.mark.database
def test_seed_corpus_round_trips_to_isolated_postgres(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import uuid

    import psycopg

    database_url = os.environ.get("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is required for the seeder integration test")
    dataset = [
        json.loads(line)
        for line in _DATASET.read_text(encoding="utf-8").splitlines()
    ]
    expected_ids = [uuid.UUID(document["document_id"]) for document in dataset]

    monkeypatch.setenv("TEST_DATABASE_URL", database_url)
    try:
        seed_corpus(
            test_database_url=database_url,
            expected_database_name="nexora_ai_retrieval_test",
            allow_test_seed=True,
            dataset_path=_DATASET,
            embedder=_FixedEmbedder(),
        )
        with psycopg.connect(database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT count(*) FROM documents WHERE id = ANY(%s::uuid[]);",
                    (expected_ids,),
                )
                assert cursor.fetchone()[0] == len(dataset)
    finally:
        with psycopg.connect(database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT current_database();")
                if cursor.fetchone()[0] != "nexora_ai_retrieval_test":
                    raise AssertionError("cleanup target is not the test database")
                cursor.execute(
                    "DELETE FROM documents WHERE id = ANY(%s::uuid[]);",
                    (expected_ids,),
                )
