"""End-to-end smoke test verifying pipeline integration with real services."""

import os
import uuid

import pytest

from src.database import probe_pgvector
from src.ingestion.pipeline import ingest_file_to_storage
from src.ingestion.storage import DocumentStorage


@pytest.fixture(autouse=True)
def require_smoke_prerequisites() -> None:
    """Verify live smoke prerequisites at test execution time."""
    if os.environ.get("RUN_REAL_MODEL") != "1":
        pytest.skip("Smoke test requires RUN_REAL_MODEL=1")
    db_url = os.environ.get("TEST_DATABASE_URL")
    if not db_url:
        pytest.skip("Smoke test requires TEST_DATABASE_URL")
    probe_pgvector(db_url)


@pytest.mark.integration
@pytest.mark.real_model
def test_end_to_end_smoke_ingestion():
    fixture_path = "tests/fixtures/synthetic_smoke_doc.txt"
    assert os.path.exists(fixture_path)

    db_url = os.environ["TEST_DATABASE_URL"]
    storage = DocumentStorage(database_url=db_url)

    board_id = uuid.uuid4()
    user_id = uuid.uuid4()

    doc_id, emb_doc = ingest_file_to_storage(
        file_path=fixture_path,
        board_id=board_id,
        uploaded_by=user_id,
        storage=storage,
    )

    try:
        # Read back from database
        stored = storage.get_document_with_chunks(doc_id)
        assert stored is not None
        assert stored.document.status == "READY"
        assert stored.document.file_name == "synthetic_smoke_doc.txt"
        assert len(stored.chunks) == emb_doc.total_chunks
        assert stored.document.metadata.get("model_id") == "BAAI/bge-m3"

        first_chunk = stored.chunks[0]
        assert len(first_chunk.embedding) == 1024
        assert first_chunk.content.startswith("DỰ ÁN NEXORA")
    finally:
        # Cleanup test document
        conn = storage.get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM documents WHERE id = %s;", (doc_id,))
        finally:
            conn.close()
