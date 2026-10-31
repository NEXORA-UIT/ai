"""Unit tests for DocumentStorage and ingestion persistence logic using mocks."""

import uuid
from unittest.mock import MagicMock

import pytest

from src.ingestion.exceptions import (
    BoardMismatchError,
    DocumentProcessingConflictError,
    ProcessingOwnershipLostError,
    StorageValidationError,
)
from src.ingestion.schemas import (
    Chunk,
    EmbeddedChunk,
    EmbeddedDocument,
    FileValidationResult,
)
from src.ingestion.storage import DocumentStorage


def create_dummy_embedded_doc(
    num_chunks: int = 2,
    dim: int = 1024,
    corrupt_dim: int | None = None,
    corrupt_index: int | None = None,
    empty_text: bool = False,
    nan_vector: bool = False,
    all_zero_vector: bool = False,
    unnormalized: bool = False,
    corrupt_offsets: tuple[int, int] | None = None,
    invalid_page: int | None = None,
) -> EmbeddedDocument:
    """Helper to generate a dummy EmbeddedDocument for validation tests."""
    chunks = []
    for i in range(num_chunks):
        vec_dim = corrupt_dim if (corrupt_dim and i == 0) else dim
        if all_zero_vector:
            vec = [0.0] * vec_dim
        elif nan_vector:
            vec = [float("nan")] * vec_dim
        elif unnormalized:
            vec = [1.0] * vec_dim  # Norm will be sqrt(1024) = 32
        else:
            # 1.0 at index 0, rest 0.0 -> L2 norm = 1.0
            vec = [1.0] + [0.0] * (vec_dim - 1)

        idx = corrupt_index if (corrupt_index is not None and i == 0) else i
        text = "" if (empty_text and i == 0) else f"Sample chunk content text {i}"

        offsets = (
            corrupt_offsets if (corrupt_offsets and i == 0) else (i * 50, (i + 1) * 50)
        )
        page = invalid_page if (invalid_page is not None and i == 0) else 1

        c = Chunk(
            chunk_index=idx,
            content=text,
            token_count=10,
            start_offset=offsets[0],
            end_offset=offsets[1],
            page_number=page,
            metadata={"sample": "tag"},
        )
        chunks.append(EmbeddedChunk(chunk=c, dense_embedding=vec))

    return EmbeddedDocument(
        document_name="test.txt",
        total_chunks=len(chunks),
        total_tokens=len(chunks) * 10,
        chunks=chunks,
        model_name="BAAI/bge-m3",
        model_revision="5617a9f61b028005a4858fdac845db406aefb181",
        dimension=dim,
        normalized=True,
    )


class TestPreSaveValidation:
    """Tests for pre-save validation rules."""

    def test_validate_valid_document_passes(self):
        storage = DocumentStorage(connection_factory=lambda: MagicMock())
        doc = create_dummy_embedded_doc(2)
        # Should not raise
        storage.validate_pre_save(doc)

    def test_validate_empty_chunks_raises(self):
        storage = DocumentStorage(connection_factory=lambda: MagicMock())
        doc = EmbeddedDocument(
            document_name="test.txt",
            total_chunks=0,
            total_tokens=0,
            chunks=[],
            model_name="BAAI/bge-m3",
            model_revision="rev",
        )
        with pytest.raises(StorageValidationError, match="0 chunks"):
            storage.validate_pre_save(doc)

    def test_validate_chunk_count_mismatch_raises(self):
        storage = DocumentStorage(connection_factory=lambda: MagicMock())
        doc = create_dummy_embedded_doc(2)
        # Manually alter total_chunks to cause mismatch
        object.__setattr__(doc, "total_chunks", 5)
        with pytest.raises(StorageValidationError, match="Chunk count mismatch"):
            storage.validate_pre_save(doc)

    def test_validate_non_continuous_indices_raises(self):
        storage = DocumentStorage(connection_factory=lambda: MagicMock())
        doc = create_dummy_embedded_doc(2, corrupt_index=5)
        with pytest.raises(StorageValidationError, match="indices must be continuous"):
            storage.validate_pre_save(doc)

    def test_validate_empty_text_raises(self):
        storage = DocumentStorage(connection_factory=lambda: MagicMock())
        doc = create_dummy_embedded_doc(2, empty_text=True)
        with pytest.raises(StorageValidationError, match="empty content"):
            storage.validate_pre_save(doc)

    def test_validate_nan_vector_raises(self):
        storage = DocumentStorage(connection_factory=lambda: MagicMock())
        doc = create_dummy_embedded_doc(1, nan_vector=True)
        with pytest.raises(StorageValidationError, match="contains NaN or Inf"):
            storage.validate_pre_save(doc)

    def test_validate_all_zero_vector_raises(self):
        storage = DocumentStorage(connection_factory=lambda: MagicMock())
        doc = create_dummy_embedded_doc(1, all_zero_vector=True)
        with pytest.raises(StorageValidationError, match="all-zero vector"):
            storage.validate_pre_save(doc)

    def test_validate_unnormalized_vector_raises(self):
        storage = DocumentStorage(connection_factory=lambda: MagicMock())
        doc = create_dummy_embedded_doc(1, unnormalized=True)
        with pytest.raises(StorageValidationError, match="L2 norm"):
            storage.validate_pre_save(doc)

    def test_validate_none_offsets_accepted(self):
        storage = DocumentStorage(connection_factory=lambda: MagicMock())
        doc = create_dummy_embedded_doc(1)
        # Set both offsets to None
        object.__setattr__(doc.chunks[0].chunk, "start_offset", None)
        object.__setattr__(doc.chunks[0].chunk, "end_offset", None)
        # Should pass validation
        storage.validate_pre_save(doc)

    @pytest.mark.parametrize(
        ("start", "end"),
        [
            (0, None),
            (None, 10),
            (50, 50),
            (50, 40),
            (-1, 10),
        ],
    )
    def test_validate_invalid_offsets_raises(self, start, end):
        storage = DocumentStorage(connection_factory=lambda: MagicMock())
        doc = create_dummy_embedded_doc(1)
        object.__setattr__(doc.chunks[0].chunk, "start_offset", start)
        object.__setattr__(doc.chunks[0].chunk, "end_offset", end)
        with pytest.raises(StorageValidationError, match="offsets invalid"):
            storage.validate_pre_save(doc)

    def test_validate_invalid_page_number_raises(self):
        storage = DocumentStorage(connection_factory=lambda: MagicMock())
        doc = create_dummy_embedded_doc(1, invalid_page=0)
        with pytest.raises(StorageValidationError, match="page_number must be > 0"):
            storage.validate_pre_save(doc)


class TestClaimOrCreateDocument:
    """Tests for claim_or_create_document logic."""

    def test_claim_new_document_success(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        doc_id = str(uuid.uuid4())
        # ON CONFLICT DO NOTHING RETURNING id returns doc_id
        mock_cur.fetchone.return_value = (doc_id,)

        storage = DocumentStorage(connection_factory=lambda: mock_conn)
        board_id = uuid.uuid4()
        user_id = uuid.uuid4()

        persisted_id, token = storage.claim_or_create_document(
            board_id=board_id,
            uploaded_by=user_id,
            file_name="doc.txt",
            storage_key="local://doc.txt",
            mime_type="text/plain",
            file_size=100,
            checksum="abc123sha256",
            document_id=doc_id,
        )

        assert persisted_id == doc_id
        assert isinstance(token, uuid.UUID)
        assert mock_cur.execute.call_count >= 1
        mock_conn.close.assert_called_once()

    def test_claim_existing_document_board_mismatch_raises(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        doc_id = str(uuid.uuid4())
        board_original = str(uuid.uuid4())
        board_other = str(uuid.uuid4())

        # First insert returns None (conflict)
        # Second SELECT FOR UPDATE returns existing row with board_original
        mock_cur.fetchone.side_effect = [
            None,
            (doc_id, board_original, str(uuid.uuid4()), "READY", None),
        ]

        storage = DocumentStorage(connection_factory=lambda: mock_conn)
        with pytest.raises(BoardMismatchError, match="belongs to board"):
            storage.claim_or_create_document(
                board_id=board_other,
                uploaded_by=uuid.uuid4(),
                file_name="doc.txt",
                storage_key="local://doc.txt",
                mime_type="text/plain",
                file_size=100,
                checksum="abc123sha256",
                document_id=doc_id,
            )

    def test_claim_existing_processing_without_reclaim_raises(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        doc_id = str(uuid.uuid4())
        board_id = str(uuid.uuid4())

        # Conflict, then SELECT returns status='PROCESSING'
        mock_cur.fetchone.side_effect = [
            None,
            (doc_id, board_id, str(uuid.uuid4()), "PROCESSING", str(uuid.uuid4())),
        ]

        storage = DocumentStorage(connection_factory=lambda: mock_conn)
        with pytest.raises(
            DocumentProcessingConflictError, match="currently in PROCESSING"
        ):
            storage.claim_or_create_document(
                board_id=board_id,
                uploaded_by=uuid.uuid4(),
                file_name="doc.txt",
                storage_key="local://doc.txt",
                mime_type="text/plain",
                file_size=100,
                checksum="abc123sha256",
                document_id=doc_id,
                reclaim=False,
            )

    def test_claim_existing_processing_with_reclaim_succeeds(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        doc_id = str(uuid.uuid4())
        board_id = str(uuid.uuid4())

        mock_cur.fetchone.side_effect = [
            None,
            (doc_id, board_id, str(uuid.uuid4()), "PROCESSING", str(uuid.uuid4())),
        ]

        storage = DocumentStorage(connection_factory=lambda: mock_conn)
        persisted_id, token = storage.claim_or_create_document(
            board_id=board_id,
            uploaded_by=uuid.uuid4(),
            file_name="doc.txt",
            storage_key="local://doc.txt",
            mime_type="text/plain",
            file_size=100,
            checksum="abc123sha256",
            document_id=doc_id,
            reclaim=True,
        )

        assert persisted_id == doc_id
        assert isinstance(token, uuid.UUID)


class TestSaveAndMarkFailed:
    """Tests for save_embedded_document and mark_document_failed."""

    def test_save_ownership_lost_on_token_mismatch(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        doc_id = str(uuid.uuid4())
        my_token = uuid.uuid4()
        other_token = uuid.uuid4()

        # SELECT FOR UPDATE returns other_token
        mock_cur.fetchone.return_value = (doc_id, "PROCESSING", str(other_token), {})

        storage = DocumentStorage(connection_factory=lambda: mock_conn)
        doc = create_dummy_embedded_doc(2)
        val = FileValidationResult("test.txt", 100, "sha256", "text/plain")

        with pytest.raises(ProcessingOwnershipLostError, match="token mismatch"):
            storage.save_embedded_document(
                document_id=doc_id,
                processing_token=my_token,
                embedded_doc=doc,
                file_validation=val,
                storage_key="local://test.txt",
            )

    def test_save_success_deletes_old_and_inserts_new(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        doc_id = str(uuid.uuid4())
        token = uuid.uuid4()

        mock_cur.fetchone.return_value = (
            doc_id,
            "PROCESSING",
            str(token),
            {"custom": "value"},
        )

        storage = DocumentStorage(connection_factory=lambda: mock_conn)
        doc = create_dummy_embedded_doc(2)
        val = FileValidationResult("test.txt", 100, "sha256", "text/plain")

        storage.save_embedded_document(
            document_id=doc_id,
            processing_token=token,
            embedded_doc=doc,
            file_validation=val,
            storage_key="local://test.txt",
        )

        # Verify DELETE from document_chunks was executed
        delete_calls = [
            c
            for c in mock_cur.execute.call_args_list
            if "DELETE FROM document_chunks" in str(c)
        ]
        assert len(delete_calls) == 1

        # Verify executemany for document_chunks
        mock_cur.executemany.assert_called_once()
        args, _ = mock_cur.executemany.call_args
        assert len(args[1]) == 2  # 2 chunks inserted

    def test_mark_document_failed_token_mismatch_returns_false(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        doc_id = str(uuid.uuid4())
        mock_cur.fetchone.return_value = (doc_id, "PROCESSING", str(uuid.uuid4()), {})

        storage = DocumentStorage(connection_factory=lambda: mock_conn)
        success = storage.mark_document_failed(
            document_id=doc_id,
            processing_token=uuid.uuid4(),  # Different token
            error_stage="embedding",
            error_type="InferenceError",
            error_message="Error",
        )
        assert success is False
