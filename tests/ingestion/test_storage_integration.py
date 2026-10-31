"""Integration tests for PostgreSQL and pgvector storage.

Executed against a live test database when TEST_DATABASE_URL is provided.
When TEST_DATABASE_URL is not provided or lacks pgvector, tests are skipped.
When configured, any implementation or database failure results in a test FAIL.
"""

from __future__ import annotations

import math
import os
import threading
import uuid
from typing import Any

import pytest
from psycopg import errors

from src.database import probe_pgvector
from src.ingestion.exceptions import (
    BoardMismatchError,
    DatabaseConnectionError,
    DocumentProcessingConflictError,
    ProcessingOwnershipLostError,
)
from src.ingestion.schemas import (
    Chunk,
    EmbeddedChunk,
    EmbeddedDocument,
    FileValidationResult,
)
from src.ingestion.storage import DocumentStorage


def get_test_db_url() -> str | None:
    """Retrieve test database connection URL from environment."""
    return os.environ.get("TEST_DATABASE_URL")


pytestmark = [
    pytest.mark.integration,
    pytest.mark.database,
]


@pytest.fixture(autouse=True)
def require_live_database() -> None:
    """Verify test database connection and pgvector prerequisites at test time.

    Never connects to the database during pytest module collection.
    If TEST_DATABASE_URL is not set, skips live tests.
    If TEST_DATABASE_URL is set, failure to connect or missing pgvector FAILS the test.
    """
    url = get_test_db_url()
    if not url:
        pytest.skip("TEST_DATABASE_URL environment variable is not configured.")

    # Any connection failure or vector error raises and FAILS the test
    probe_pgvector(url)


@pytest.fixture
def storage() -> DocumentStorage:
    """Provide DocumentStorage configured with TEST_DATABASE_URL."""
    url = get_test_db_url()
    assert url is not None
    return DocumentStorage(database_url=url)


@pytest.fixture
def tracked_doc_ids(storage: DocumentStorage):
    """Scoped cleanup fixture: tracks created document IDs and cleans ONLY those IDs."""
    doc_ids: list[str] = []
    yield doc_ids

    if doc_ids:
        conn = storage.get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM documents WHERE id = ANY(%s);",
                        (doc_ids,),
                    )
        finally:
            conn.close()


def create_sample_embedded_doc(
    num_chunks: int,
    doc_id: str,
    doc_name: str = "integration_test.txt",
) -> EmbeddedDocument:
    """Helper to generate sample EmbeddedDocument with unit vectors."""
    chunks = []
    for i in range(num_chunks):
        vec = [0.0] * 1024
        vec[i % 1024] = 1.0
        c = Chunk(
            chunk_index=i,
            content=f"Integration chunk content text block {i}",
            token_count=15,
            start_offset=i * 100,
            end_offset=(i + 1) * 100,
            page_number=1,
            metadata={"tag": f"sample_{i}"},
        )
        chunks.append(EmbeddedChunk(chunk=c, dense_embedding=vec))

    return EmbeddedDocument(
        document_name=doc_name,
        total_chunks=len(chunks),
        total_tokens=len(chunks) * 15,
        chunks=chunks,
        model_name="BAAI/bge-m3",
        model_revision="5617a9f61b028005a4858fdac845db406aefb181",
        dimension=1024,
        normalized=True,
    )


class TestLiveStorageIntegration:
    """Live database tests for concurrency, transactions, and roundtrip fidelity."""

    def test_roundtrip_document_and_chunks(self, storage, tracked_doc_ids):
        board_id = uuid.uuid4()
        user_id = uuid.uuid4()
        file_val = FileValidationResult(
            file_name="roundtrip.txt",
            file_size_bytes=1024,
            sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            mime_type="text/plain",
        )

        doc_id, token = storage.claim_or_create_document(
            board_id=board_id,
            uploaded_by=user_id,
            file_name=file_val.file_name,
            storage_key="local://roundtrip.txt",
            mime_type=file_val.mime_type,
            file_size=file_val.file_size_bytes,
            checksum=file_val.sha256,
        )
        tracked_doc_ids.append(doc_id)

        emb_doc = create_sample_embedded_doc(num_chunks=3, doc_id=doc_id)
        storage.save_embedded_document(
            document_id=doc_id,
            processing_token=token,
            embedded_doc=emb_doc,
            file_validation=file_val,
            storage_key="local://roundtrip.txt",
        )

        stored = storage.get_document_with_chunks(doc_id)
        assert stored is not None
        assert stored.document.status == "READY"
        assert stored.document.processing_token is None
        assert stored.document.board_id == str(board_id)
        assert stored.document.uploaded_by == str(user_id)
        assert len(stored.chunks) == 3

        for i, chunk in enumerate(stored.chunks):
            assert chunk.chunk_index == i
            assert chunk.content == f"Integration chunk content text block {i}"
            assert chunk.start_offset == i * 100
            assert chunk.end_offset == (i + 1) * 100
            assert len(chunk.embedding) == 1024
            expected_vec = [0.0] * 1024
            expected_vec[i % 1024] = 1.0
            for actual, exp in zip(chunk.embedding, expected_vec, strict=True):
                assert math.isclose(actual, exp, abs_tol=1e-4)

    def test_reprocess_fewer_chunks_leaves_no_stale_chunks(
        self, storage, tracked_doc_ids
    ):
        board_id = uuid.uuid4()
        user_id = uuid.uuid4()
        file_val = FileValidationResult("reprocess.txt", 500, "sha256", "text/plain")

        # Initial run with 5 chunks
        doc_id, token1 = storage.claim_or_create_document(
            board_id=board_id,
            uploaded_by=user_id,
            file_name=file_val.file_name,
            storage_key="local://reprocess.txt",
            mime_type=file_val.mime_type,
            file_size=file_val.file_size_bytes,
            checksum=file_val.sha256,
        )
        tracked_doc_ids.append(doc_id)

        emb_5 = create_sample_embedded_doc(5, doc_id)
        storage.save_embedded_document(
            doc_id, token1, emb_5, file_val, "local://reprocess.txt"
        )

        stored = storage.get_document_with_chunks(doc_id)
        assert len(stored.chunks) == 5

        # Reprocess run with 2 chunks
        _, token2 = storage.claim_or_create_document(
            board_id=board_id,
            uploaded_by=user_id,
            file_name=file_val.file_name,
            storage_key="local://reprocess.txt",
            mime_type=file_val.mime_type,
            file_size=file_val.file_size_bytes,
            checksum=file_val.sha256,
            document_id=doc_id,
        )

        emb_2 = create_sample_embedded_doc(2, doc_id)
        storage.save_embedded_document(
            doc_id, token2, emb_2, file_val, "local://reprocess.txt"
        )

        stored_after = storage.get_document_with_chunks(doc_id)
        assert len(stored_after.chunks) == 2
        assert [c.chunk_index for c in stored_after.chunks] == [0, 1]

    def test_reprocess_midway_failure_rolls_back_and_preserves_old_chunks(
        self, storage, tracked_doc_ids
    ):
        """Prove error after deleting old chunks rolls back cleanly."""
        board_id = uuid.uuid4()
        user_id = uuid.uuid4()
        file_val = FileValidationResult("rollback.txt", 300, "sha256", "text/plain")

        # 1. Initial successful run with 3 chunks
        doc_id, token1 = storage.claim_or_create_document(
            board_id=board_id,
            uploaded_by=user_id,
            file_name=file_val.file_name,
            storage_key="local://rollback.txt",
            mime_type=file_val.mime_type,
            file_size=file_val.file_size_bytes,
            checksum=file_val.sha256,
        )
        tracked_doc_ids.append(doc_id)

        emb_old = create_sample_embedded_doc(3, doc_id)
        storage.save_embedded_document(
            doc_id, token1, emb_old, file_val, "local://rollback.txt"
        )

        stored_init = storage.get_document_with_chunks(doc_id)
        assert stored_init is not None
        assert stored_init.document.status == "READY"
        assert len(stored_init.chunks) == 3
        old_chunk_contents = [c.content for c in stored_init.chunks]

        # 2. Claim for re-processing
        _, token2 = storage.claim_or_create_document(
            board_id=board_id,
            uploaded_by=user_id,
            file_name=file_val.file_name,
            storage_key="local://rollback.txt",
            mime_type=file_val.mime_type,
            file_size=file_val.file_size_bytes,
            checksum=file_val.sha256,
            document_id=doc_id,
        )

        # 3. Inject mid-transaction failure during executemany (AFTER DELETE)
        emb_new = create_sample_embedded_doc(2, doc_id)

        import psycopg
        from pgvector.psycopg import register_vector

        class FailingCursor(psycopg.Cursor):
            def executemany(self, *args, **kwargs):
                raise errors.DiskFull("Simulated database failure during chunk insert")

        def failing_get_conn():
            conn = psycopg.connect(
                get_test_db_url(),
                connect_timeout=10,
                cursor_factory=FailingCursor,
            )
            register_vector(conn)
            return conn

        storage_failing = DocumentStorage(
            database_url=get_test_db_url(),
            connection_factory=failing_get_conn,
        )

        with pytest.raises(DatabaseConnectionError):
            storage_failing.save_embedded_document(
                doc_id, token2, emb_new, file_val, "local://rollback.txt"
            )

        # 4. Prove that old chunks are completely preserved in DB due to atomic rollback
        stored_after_fail = storage.get_document_with_chunks(doc_id)
        assert stored_after_fail is not None
        assert stored_after_fail.document.status == "PROCESSING"
        assert len(stored_after_fail.chunks) == 3
        assert [c.content for c in stored_after_fail.chunks] == old_chunk_contents

        # 5. Transition to FAILED via orchestrator failure handling
        marked = storage.mark_document_failed(
            document_id=doc_id,
            processing_token=token2,
            error_stage="storage_persist",
            error_type="DiskFull",
            error_message="Simulated database failure during chunk insert",
        )
        assert marked is True

        stored_final = storage.get_document_with_chunks(doc_id)
        assert stored_final is not None
        assert stored_final.document.status == "FAILED"
        assert len(stored_final.chunks) == 3
        assert stored_final.document.metadata.get("error_stage") == "storage_persist"

    def test_concurrent_claim_non_existent_document_only_one_succeeds(
        self, storage, tracked_doc_ids
    ):
        target_doc_id = str(uuid.uuid4())
        tracked_doc_ids.append(target_doc_id)
        board_id = uuid.uuid4()
        user_id = uuid.uuid4()
        file_val = FileValidationResult("concurrent.txt", 100, "sha256", "text/plain")

        results: list[tuple[str, Any]] = []
        barrier = threading.Barrier(2)

        def worker(worker_name: str) -> None:
            # Independent storage and connection per worker thread
            worker_storage = DocumentStorage(database_url=get_test_db_url())
            barrier.wait(timeout=15.0)  # Synchronize simultaneous release
            try:
                doc_id, token = worker_storage.claim_or_create_document(
                    board_id=board_id,
                    uploaded_by=user_id,
                    file_name=file_val.file_name,
                    storage_key="local://concurrent.txt",
                    mime_type=file_val.mime_type,
                    file_size=file_val.file_size_bytes,
                    checksum=file_val.sha256,
                    document_id=target_doc_id,
                )
                results.append((worker_name, ("SUCCESS", doc_id, token)))
            except Exception as exc:
                results.append((worker_name, ("ERROR", exc)))

        t1 = threading.Thread(target=worker, args=("W1",))
        t2 = threading.Thread(target=worker, args=("W2",))
        t1.start()
        t2.start()
        t1.join(timeout=20.0)
        t2.join(timeout=20.0)
        assert not t1.is_alive(), "Worker 1 deadlocked/timed out"
        assert not t2.is_alive(), "Worker 2 deadlocked/timed out"

        assert len(results) == 2
        successes = [r for r in results if r[1][0] == "SUCCESS"]
        errors = [r for r in results if r[1][0] == "ERROR"]

        assert len(successes) == 1
        assert len(errors) == 1
        assert isinstance(errors[0][1][1], DocumentProcessingConflictError)

    def test_concurrent_claim_existing_ready_document_only_one_succeeds(
        self, storage, tracked_doc_ids
    ):
        board_id = uuid.uuid4()
        user_id = uuid.uuid4()
        file_val = FileValidationResult("existing.txt", 100, "sha256", "text/plain")

        # Prepare document in READY status
        doc_id, token = storage.claim_or_create_document(
            board_id=board_id,
            uploaded_by=user_id,
            file_name=file_val.file_name,
            storage_key="local://existing.txt",
            mime_type=file_val.mime_type,
            file_size=file_val.file_size_bytes,
            checksum=file_val.sha256,
        )
        tracked_doc_ids.append(doc_id)
        emb = create_sample_embedded_doc(1, doc_id)
        storage.save_embedded_document(
            doc_id, token, emb, file_val, "local://existing.txt"
        )

        results: list[tuple[str, Any]] = []
        barrier = threading.Barrier(2)

        def worker(worker_name: str) -> None:
            worker_storage = DocumentStorage(database_url=get_test_db_url())
            barrier.wait(timeout=15.0)
            try:
                _, tok = worker_storage.claim_or_create_document(
                    board_id=board_id,
                    uploaded_by=user_id,
                    file_name=file_val.file_name,
                    storage_key="local://existing.txt",
                    mime_type=file_val.mime_type,
                    file_size=file_val.file_size_bytes,
                    checksum=file_val.sha256,
                    document_id=doc_id,
                    reclaim=False,
                )
                results.append((worker_name, ("SUCCESS", tok)))
            except Exception as exc:
                results.append((worker_name, ("ERROR", exc)))

        t1 = threading.Thread(target=worker, args=("W1",))
        t2 = threading.Thread(target=worker, args=("W2",))
        t1.start()
        t2.start()
        t1.join(timeout=20.0)
        t2.join(timeout=20.0)
        assert not t1.is_alive(), "Worker 1 deadlocked/timed out"
        assert not t2.is_alive(), "Worker 2 deadlocked/timed out"

        assert len(results) == 2
        successes = [r for r in results if r[1][0] == "SUCCESS"]
        errors = [r for r in results if r[1][0] == "ERROR"]

        assert len(successes) == 1
        assert len(errors) == 1
        assert isinstance(errors[0][1][1], DocumentProcessingConflictError)

    def test_concurrency_ownership_lost_prevents_late_write(
        self, storage, tracked_doc_ids
    ):
        storage_a = DocumentStorage(database_url=get_test_db_url())
        storage_b = DocumentStorage(database_url=get_test_db_url())

        board_id = uuid.uuid4()
        user_id = uuid.uuid4()
        file_val = FileValidationResult("race.txt", 100, "sha256", "text/plain")

        # Process A claims
        doc_id, token_a = storage_a.claim_or_create_document(
            board_id=board_id,
            uploaded_by=user_id,
            file_name=file_val.file_name,
            storage_key="local://race.txt",
            mime_type=file_val.mime_type,
            file_size=file_val.file_size_bytes,
            checksum=file_val.sha256,
        )
        tracked_doc_ids.append(doc_id)

        # Process B explicitly reclaims on independent connection
        _, token_b = storage_b.claim_or_create_document(
            board_id=board_id,
            uploaded_by=user_id,
            file_name=file_val.file_name,
            storage_key="local://race.txt",
            mime_type=file_val.mime_type,
            file_size=file_val.file_size_bytes,
            checksum=file_val.sha256,
            document_id=doc_id,
            reclaim=True,
        )
        assert token_a != token_b

        # Process A tries to persist with stale token -> Rejected before touching chunks
        emb_a = create_sample_embedded_doc(2, doc_id)
        with pytest.raises(ProcessingOwnershipLostError):
            storage_a.save_embedded_document(
                doc_id, token_a, emb_a, file_val, "local://race.txt"
            )

        # Process A tries to mark FAILED -> Returns False without altering state
        failed_res = storage_a.mark_document_failed(
            doc_id, token_a, "stage", "type", "error"
        )
        assert failed_res is False

        # Process B persists successfully
        emb_b = create_sample_embedded_doc(3, doc_id)
        storage_b.save_embedded_document(
            doc_id, token_b, emb_b, file_val, "local://race.txt"
        )

        final_stored = storage.get_document_with_chunks(doc_id)
        assert final_stored.document.status == "READY"
        assert len(final_stored.chunks) == 3

    def test_document_deleted_before_persist_not_recreated(
        self, storage, tracked_doc_ids
    ):
        board_id = uuid.uuid4()
        user_id = uuid.uuid4()
        file_val = FileValidationResult("deleted.txt", 100, "sha256", "text/plain")

        doc_id, token = storage.claim_or_create_document(
            board_id=board_id,
            uploaded_by=user_id,
            file_name=file_val.file_name,
            storage_key="local://deleted.txt",
            mime_type=file_val.mime_type,
            file_size=file_val.file_size_bytes,
            checksum=file_val.sha256,
        )
        tracked_doc_ids.append(doc_id)

        # Concurrently delete document from DB
        conn = storage.get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM documents WHERE id = %s;", (doc_id,))
        finally:
            conn.close()

        emb = create_sample_embedded_doc(1, doc_id)
        with pytest.raises(
            ProcessingOwnershipLostError, match="deleted before persist"
        ):
            storage.save_embedded_document(
                doc_id, token, emb, file_val, "local://deleted.txt"
            )

        # Confirm document was NOT silently recreated
        assert storage.get_document_with_chunks(doc_id) is None

    def test_reprocess_different_board_rejected(self, storage, tracked_doc_ids):
        board_a = uuid.uuid4()
        board_b = uuid.uuid4()
        user_id = uuid.uuid4()
        file_val = FileValidationResult("board.txt", 100, "sha256", "text/plain")

        doc_id, _ = storage.claim_or_create_document(
            board_id=board_a,
            uploaded_by=user_id,
            file_name=file_val.file_name,
            storage_key="local://board.txt",
            mime_type=file_val.mime_type,
            file_size=file_val.file_size_bytes,
            checksum=file_val.sha256,
        )
        tracked_doc_ids.append(doc_id)

        with pytest.raises(BoardMismatchError):
            storage.claim_or_create_document(
                board_id=board_b,
                uploaded_by=user_id,
                file_name=file_val.file_name,
                storage_key="local://board.txt",
                mime_type=file_val.mime_type,
                file_size=file_val.file_size_bytes,
                checksum=file_val.sha256,
                document_id=doc_id,
            )

    def test_db_check_constraint_offsets(self, storage, tracked_doc_ids):
        """Direct SQL verification of document_chunks offset CHECK constraint."""
        board_id = uuid.uuid4()
        user_id = uuid.uuid4()
        doc_id, _ = storage.claim_or_create_document(
            board_id=board_id,
            uploaded_by=user_id,
            file_name="offsets.txt",
            storage_key="local://offsets.txt",
            mime_type="text/plain",
            file_size=100,
            checksum="sha256",
        )
        tracked_doc_ids.append(doc_id)

        conn = storage.get_connection()
        try:
            # Case 1: (NULL, NULL) -> Accepted
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO document_chunks (
                            id, document_id, chunk_index, content,
                            start_offset, end_offset, embedding
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s);
                        """,
                        (str(uuid.uuid4()), doc_id, 0, "c0", None, None, [0.0] * 1024),
                    )

            # Case 2: (0, 10) -> Accepted
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO document_chunks (
                            id, document_id, chunk_index, content,
                            start_offset, end_offset, embedding
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s);
                        """,
                        (str(uuid.uuid4()), doc_id, 1, "c1", 0, 10, [0.0] * 1024),
                    )

            # Case 3: (0, NULL) -> Rejected by CHECK constraint
            with pytest.raises(errors.CheckViolation):
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO document_chunks (
                                id, document_id, chunk_index, content,
                                start_offset, end_offset, embedding
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s);
                            """,
                            (str(uuid.uuid4()), doc_id, 2, "c2", 0, None, [0.0] * 1024),
                        )

            # Case 4: (NULL, 10) -> Rejected by CHECK constraint
            with pytest.raises(errors.CheckViolation):
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO document_chunks (
                                id, document_id, chunk_index, content,
                                start_offset, end_offset, embedding
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s);
                            """,
                            (
                                str(uuid.uuid4()),
                                doc_id,
                                3,
                                "c3",
                                None,
                                10,
                                [0.0] * 1024,
                            ),
                        )

            # Case 5: (-1, 10) -> Rejected by CHECK constraint
            with pytest.raises(errors.CheckViolation):
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO document_chunks (
                                id, document_id, chunk_index, content,
                                start_offset, end_offset, embedding
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s);
                            """,
                            (str(uuid.uuid4()), doc_id, 4, "c4", -1, 10, [0.0] * 1024),
                        )

            # Case 6: (10, 5) -> Rejected by CHECK constraint
            with pytest.raises(errors.CheckViolation):
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO document_chunks (
                                id, document_id, chunk_index, content,
                                start_offset, end_offset, embedding
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s);
                            """,
                            (str(uuid.uuid4()), doc_id, 5, "c5", 10, 5, [0.0] * 1024),
                        )

            # Case 7: (10, 10) -> Rejected by CHECK constraint
            with pytest.raises(errors.CheckViolation):
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO document_chunks (
                                id, document_id, chunk_index, content,
                                start_offset, end_offset, embedding
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s);
                            """,
                            (str(uuid.uuid4()), doc_id, 6, "c6", 10, 10, [0.0] * 1024),
                        )
        finally:
            conn.close()
