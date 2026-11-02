"""Safely seed fixed evaluation corpora into an explicitly named test DB."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from urllib.parse import unquote, urlsplit
from uuid import UUID

from pgvector import Vector

from evaluation.manifest import load_evaluation_dataset
from src.embedding.config import DEFAULT_DENSE_DIMENSION
from src.embedding.embedder import Embedder
from src.ingestion.storage import DocumentStorage

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_GOLDEN_ROOT = _REPOSITORY_ROOT / "evaluation" / "golden"
_UPLOADED_BY = UUID("00000000-0000-4000-8000-000000000007")
_ALLOWED_STATUSES = {"PENDING", "PROCESSING", "READY", "FAILED"}
_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


class UnsafeSeedTargetError(RuntimeError):
    """Raised when the requested write target is not an explicit test DB."""


def _database_name_from_url(database_url: str) -> str:
    parsed = urlsplit(database_url)
    if parsed.scheme not in {"postgres", "postgresql"}:
        raise UnsafeSeedTargetError("TEST_DATABASE_URL must use PostgreSQL")
    database_name = unquote(parsed.path.lstrip("/"))
    if not database_name or "/" in database_name:
        raise UnsafeSeedTargetError("TEST_DATABASE_URL must name one database")
    return database_name


def _validate_target(
    *,
    test_database_url: str,
    expected_database_name: str,
    allow_test_seed: bool,
    dataset_path: Path,
) -> tuple[str, Path]:
    if allow_test_seed is not True:
        raise UnsafeSeedTargetError("explicit opt-in is required to seed a test DB")
    configured_url = os.environ.get("TEST_DATABASE_URL")
    if not test_database_url or configured_url != test_database_url:
        raise UnsafeSeedTargetError(
            "test_database_url must exactly match TEST_DATABASE_URL"
        )
    if not isinstance(
        expected_database_name, str
    ) or not expected_database_name.endswith("_test"):
        raise UnsafeSeedTargetError("expected database name must end with _test")
    url_database_name = _database_name_from_url(test_database_url)
    if url_database_name != expected_database_name:
        raise UnsafeSeedTargetError(
            "TEST_DATABASE_URL database name does not match expected test database"
        )

    resolved_path = Path(dataset_path).resolve()
    try:
        resolved_path.relative_to(_GOLDEN_ROOT.resolve())
    except ValueError as exc:
        raise UnsafeSeedTargetError(
            "seed dataset must live under evaluation/golden"
        ) from exc
    if not resolved_path.is_file():
        raise ValueError(f"seed dataset does not exist: {resolved_path}")
    return url_database_name, resolved_path


def _load_documents(dataset_path: Path) -> list[dict[str, object]]:
    documents: list[dict[str, object]] = []
    document_ids: set[UUID] = set()
    chunk_ids: set[UUID] = set()
    for line_number, line in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            document = json.loads(line)
            document_id = UUID(document["document_id"])
            UUID(document["board_id"])
            file_name = document["file_name"]
            storage_key = document["storage_key"]
            status = document["status"]
            metadata = document["metadata"]
            chunks = document["chunks"]
            if document_id in document_ids:
                raise ValueError("duplicate document ID")
            document_ids.add(document_id)
            if (
                not isinstance(file_name, str)
                or not file_name.strip()
                or len(file_name) > 255
            ):
                raise ValueError("invalid file name")
            if (
                not isinstance(storage_key, str)
                or not storage_key.strip()
                or len(storage_key) > 512
            ):
                raise ValueError("invalid storage key")
            if status not in _ALLOWED_STATUSES:
                raise ValueError("invalid document status")
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be an object")
            if not isinstance(chunks, list) or not chunks:
                raise ValueError("document must contain at least one chunk")
            chunk_indexes: set[int] = set()
            for chunk in chunks:
                chunk_id = UUID(chunk["chunk_id"])
                chunk_index = chunk["chunk_index"]
                content = chunk["content"]
                if chunk_id in chunk_ids:
                    raise ValueError("duplicate chunk ID")
                chunk_ids.add(chunk_id)
                if (
                    isinstance(chunk_index, bool)
                    or not isinstance(chunk_index, int)
                    or chunk_index < 0
                    or chunk_index in chunk_indexes
                ):
                    raise ValueError("invalid or duplicate chunk index")
                chunk_indexes.add(chunk_index)
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("chunk content must contain text")
                page_number = chunk.get("page_number")
                if page_number is not None and (
                    isinstance(page_number, bool)
                    or not isinstance(page_number, int)
                    or page_number < 1
                ):
                    raise ValueError("page_number must be a positive integer or null")
                start_offset = chunk.get("start_offset")
                end_offset = chunk.get("end_offset")
                if (start_offset is None) != (end_offset is None):
                    raise ValueError("chunk offsets must both be null or both present")
                if start_offset is not None and (
                    isinstance(start_offset, bool)
                    or isinstance(end_offset, bool)
                    or not isinstance(start_offset, int)
                    or not isinstance(end_offset, int)
                    or start_offset < 0
                    or end_offset <= start_offset
                ):
                    raise ValueError("chunk offsets are invalid")
            documents.append(document)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid seed document at {dataset_path}:{line_number}"
            ) from exc
    if not documents:
        raise ValueError(f"seed dataset is empty: {dataset_path}")
    return documents


def _validate_fingerprint(documents: list[dict[str, object]]) -> None:
    for document in documents:
        metadata = document["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        model_id = metadata.get("model_id")
        revision = metadata.get("model_revision")
        dimension = metadata.get("dimension")
        normalized = metadata.get("normalized")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("metadata.model_id must be nonempty")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("metadata.model_revision must be nonempty")
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension != DEFAULT_DENSE_DIMENSION
        ):
            raise ValueError("metadata.dimension must match pgvector dimensions")
        if not isinstance(normalized, bool):
            raise ValueError("metadata.normalized must be a boolean")


def _validate_embedder_fingerprint(dataset_path: Path, embedder: Embedder) -> None:
    """Require generated vectors to use the manifest's primary embedding model."""
    dataset = load_evaluation_dataset(dataset_path.parent)
    if dataset.chunk_snapshot_path.resolve() != dataset_path.resolve():
        raise ValueError("seed dataset must match the manifest chunk snapshot")
    if (
        embedder.model_name != dataset.embedding_model_id
        or embedder.revision != dataset.embedding_revision
        or embedder.dimension != DEFAULT_DENSE_DIMENSION
    ):
        raise ValueError(
            "seed embedder model, revision, and dimension must match the manifest"
        )


def _embed_documents(
    documents: list[dict[str, object]], embedder: Embedder
) -> list[list[float]]:
    chunks = [chunk for document in documents for chunk in document["chunks"]]
    texts = [chunk["content"] for chunk in chunks]
    vectors = embedder.embed_texts(texts)
    if len(vectors) != len(texts):
        raise ValueError("embedder must return one vector per chunk")
    validated: list[list[float]] = []
    for vector in vectors:
        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("embedding vector must contain numeric values") from exc
        if len(values) != DEFAULT_DENSE_DIMENSION:
            raise ValueError("embedding vector has an incompatible dimension")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("embedding vector must contain finite values")
        if not any(value != 0.0 for value in values):
            raise ValueError("embedding vector must not be the zero vector")
        validated.append(values)
    return validated


def _assert_connected_database(storage: DocumentStorage, expected_name: str) -> None:
    connection = storage.get_connection()
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT current_database();")
                row = cursor.fetchone()
                if (
                    row is None
                    or row[0] != expected_name
                    or not row[0].endswith("_test")
                ):
                    raise UnsafeSeedTargetError(
                        "connected database does not match the expected _test database"
                    )
    finally:
        connection.close()


def seed_corpus(
    *,
    test_database_url: str,
    expected_database_name: str,
    allow_test_seed: bool,
    dataset_path: Path,
    embedder: Embedder,
) -> None:
    """Upsert only the dataset's fixed IDs after verifying an explicit test DB."""
    database_name, resolved_path = _validate_target(
        test_database_url=test_database_url,
        expected_database_name=expected_database_name,
        allow_test_seed=allow_test_seed,
        dataset_path=dataset_path,
    )
    documents = _load_documents(resolved_path)
    _validate_fingerprint(documents)
    _validate_embedder_fingerprint(resolved_path, embedder)

    storage = DocumentStorage(database_url=test_database_url)
    _assert_connected_database(storage, database_name)
    vectors = _embed_documents(documents, embedder)
    vector_iter = iter(vectors)
    document_ids = [UUID(document["document_id"]) for document in documents]

    connection = storage.get_connection()
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT current_database();")
                row = cursor.fetchone()
                if (
                    row is None
                    or row[0] != database_name
                    or not row[0].endswith("_test")
                ):
                    raise UnsafeSeedTargetError(
                        "connected database changed before corpus write"
                    )
                cursor.execute(
                    "DELETE FROM documents WHERE id = ANY(%(document_ids)s::uuid[]);",
                    {"document_ids": document_ids},
                )
                for document in documents:
                    document_chunks = document["chunks"]
                    content_bytes = "\n".join(
                        chunk["content"] for chunk in document_chunks
                    ).encode("utf-8")
                    suffix = Path(document["file_name"]).suffix.lower()
                    mime_type = _MIME_TYPES.get(suffix, "text/plain")
                    cursor.execute(
                        """
                        INSERT INTO documents (
                            id, board_id, uploaded_by, file_name, storage_key,
                            mime_type, file_size, checksum, status, metadata
                        ) VALUES (
                            %(id)s, %(board_id)s, %(uploaded_by)s, %(file_name)s,
                            %(storage_key)s, %(mime_type)s, %(file_size)s,
                            %(checksum)s, %(status)s, %(metadata)s::jsonb
                        );
                        """,
                        {
                            "id": UUID(document["document_id"]),
                            "board_id": UUID(document["board_id"]),
                            "uploaded_by": _UPLOADED_BY,
                            "file_name": document["file_name"],
                            "storage_key": document["storage_key"],
                            "mime_type": mime_type,
                            "file_size": len(content_bytes),
                            "checksum": hashlib.sha256(content_bytes).hexdigest(),
                            "status": document["status"],
                            "metadata": json.dumps(document["metadata"]),
                        },
                    )
                    for chunk in document_chunks:
                        vector = next(vector_iter)
                        cursor.execute(
                            """
                            INSERT INTO document_chunks (
                                id, document_id, chunk_index, content, page_number,
                                start_offset, end_offset, embedding, metadata
                            ) VALUES (
                                %(id)s, %(document_id)s, %(chunk_index)s, %(content)s,
                                %(page_number)s, %(start_offset)s, %(end_offset)s,
                                %(embedding)s, '{}'::jsonb
                            );
                            """,
                            {
                                "id": UUID(chunk["chunk_id"]),
                                "document_id": UUID(document["document_id"]),
                                "chunk_index": chunk["chunk_index"],
                                "content": chunk["content"],
                                "page_number": chunk.get("page_number"),
                                "start_offset": chunk.get("start_offset"),
                                "end_offset": chunk.get("end_offset"),
                                "embedding": Vector(vector),
                            },
                        )
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-test-seed", action="store_true", required=True)
    parser.add_argument("--expected-database-name", required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_REPOSITORY_ROOT
        / "evaluation"
        / "golden"
        / "synthetic"
        / "corpus.jsonl",
    )
    args = parser.parse_args()

    from src.embedding.embedder import BgeM3Embedder

    database_url = os.environ.get("TEST_DATABASE_URL", "")
    try:
        documents = _load_documents(args.dataset.resolve())
        seed_corpus(
            test_database_url=database_url,
            expected_database_name=args.expected_database_name,
            allow_test_seed=args.allow_test_seed,
            dataset_path=args.dataset,
            embedder=BgeM3Embedder(),
        )
    except Exception as exc:
        parser.error(str(exc))
    chunk_count = sum(len(document["chunks"]) for document in documents)
    print(f"Seeded {len(documents)} evaluation documents and {chunk_count} chunks.")


if __name__ == "__main__":
    main()
