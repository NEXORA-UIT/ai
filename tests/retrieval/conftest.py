"""Fixtures for isolated PostgreSQL/pgvector retrieval tests."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Iterator

import pytest
from pgvector import Vector

from src.ingestion.storage import DocumentStorage


@pytest.fixture
def retrieval_database() -> Iterator[
    tuple[DocumentStorage, Callable[[str, str, str, list[float]], tuple[str, str]]]
]:
    """Yield test storage and a seeder that tracks only its own documents."""
    database_url = os.environ.get("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is required for pgvector integration tests")

    storage = DocumentStorage(database_url=database_url)
    document_ids: list[uuid.UUID] = []

    def seed_document(
        board_id: str, status: str, label: str, embedding: list[float]
    ) -> tuple[str, str]:
        document_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        uploaded_by = uuid.uuid4()
        content = f"retrieval test chunk for {label}"
        document_ids.append(document_id)

        connection = storage.get_connection()
        try:
            with connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO documents (
                            id, board_id, uploaded_by, file_name, storage_key,
                            mime_type, file_size, checksum, status, metadata
                        ) VALUES (
                            %(id)s, %(board_id)s, %(uploaded_by)s, %(file_name)s,
                            %(storage_key)s, 'text/plain', %(file_size)s,
                            %(checksum)s, %(status)s, '{}'::jsonb
                        );
                        """,
                        {
                            "id": document_id,
                            "board_id": uuid.UUID(board_id),
                            "uploaded_by": uploaded_by,
                            "file_name": f"{label}.txt",
                            "storage_key": f"local://retrieval-tests/{document_id}.txt",
                            "file_size": len(content),
                            "checksum": uuid.uuid4().hex,
                            "status": status,
                        },
                    )
                    cursor.execute(
                        """
                        INSERT INTO document_chunks (
                            id, document_id, chunk_index, content, page_number,
                            start_offset, end_offset, embedding, metadata
                        ) VALUES (
                            %(id)s, %(document_id)s, 0, %(content)s, NULL,
                            0, %(end_offset)s, %(embedding)s, '{}'::jsonb
                        );
                        """,
                        {
                            "id": chunk_id,
                            "document_id": document_id,
                            "content": content,
                            "end_offset": len(content),
                            "embedding": Vector(embedding),
                        },
                    )
        finally:
            connection.close()

        return str(document_id), str(chunk_id)

    try:
        yield storage, seed_document
    finally:
        if document_ids:
            connection = storage.get_connection()
            try:
                with connection:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            "DELETE FROM documents WHERE id = ANY(%s);",
                            (document_ids,),
                        )
            finally:
                connection.close()
