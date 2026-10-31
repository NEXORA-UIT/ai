"""Unit tests for pipeline storage orchestration and CLI safety."""

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.cli import parse_safe_db_info
from src.ingestion.exceptions import (
    DatabaseConnectionError,
    TikaExtractionError,
)
from src.ingestion.pipeline import ingest_file_to_storage
from src.ingestion.schemas import (
    Chunk,
    ChunkedDocument,
    CleanedDocument,
    EmbeddedChunk,
    EmbeddedDocument,
    ExtractionResult,
)
from src.ingestion.storage import sanitize_error_message


def dummy_artifacts():
    ext = ExtractionResult(
        raw_text="content",
        metadata={},
        file_name="test.txt",
        mime_type="text/plain",
    )
    clean = CleanedDocument("content", "test.txt", "text/plain", {}, "1.0.0", None)
    c = Chunk(
        chunk_index=0,
        content="content",
        token_count=2,
        start_offset=0,
        end_offset=7,
        page_number=1,
        metadata={},
    )
    chunked = ChunkedDocument("test.txt", 1, 2, [c])
    ec = EmbeddedChunk(c, [1.0] + [0.0] * 1023)
    emb = EmbeddedDocument("test.txt", 1, 2, [ec], "model", "rev")
    return ext, clean, chunked, emb


class TestPipelineOrchestration:
    """Verify orchestration guarantees and transaction boundaries."""

    @patch("src.ingestion.pipeline.validate_file")
    @patch("src.ingestion.pipeline.extract_clean_chunk_and_embed")
    def test_pipeline_success_lifecycle(self, mock_extract, mock_validate):
        ext, clean, chunked, emb = dummy_artifacts()
        mock_validate.return_value = (
            Path("tests/fixtures/synthetic_smoke_doc.txt"),
            "text/plain",
            100,
        )
        mock_extract.return_value = (ext, clean, chunked, emb)

        mock_storage = MagicMock()
        target_token = uuid.uuid4()
        mock_storage.claim_or_create_document.return_value = (
            "doc-123",
            target_token,
        )

        board_id = uuid.uuid4()
        user_id = uuid.uuid4()

        doc_id, returned_emb = ingest_file_to_storage(
            file_path="tests/fixtures/synthetic_smoke_doc.txt",
            board_id=board_id,
            uploaded_by=user_id,
            storage=mock_storage,
        )

        assert doc_id == "doc-123"
        assert returned_emb == emb

        # Verify claim was called first
        mock_storage.claim_or_create_document.assert_called_once()
        # Verify save was called with token
        saved_call = mock_storage.save_embedded_document.call_args[1]
        assert saved_call["document_id"] == "doc-123"
        assert saved_call["processing_token"] == target_token
        assert saved_call["embedded_doc"] == emb
        mock_storage.mark_document_failed.assert_not_called()

    @patch("src.ingestion.pipeline.validate_file")
    @patch("src.ingestion.pipeline.extract_clean_chunk_and_embed")
    def test_pipeline_extraction_failure_marks_failed_and_preserves_error(
        self, mock_extract, mock_validate
    ):
        ext, _, _, _ = dummy_artifacts()
        mock_validate.return_value = (
            Path("tests/fixtures/synthetic_smoke_doc.txt"),
            "text/plain",
            100,
        )
        mock_extract.side_effect = TikaExtractionError("Extraction bombed")

        mock_storage = MagicMock()
        target_token = uuid.uuid4()
        mock_storage.claim_or_create_document.return_value = (
            "doc-123",
            target_token,
        )

        with pytest.raises(TikaExtractionError, match="Extraction bombed"):
            ingest_file_to_storage(
                file_path="tests/fixtures/synthetic_smoke_doc.txt",
                board_id=uuid.uuid4(),
                uploaded_by=uuid.uuid4(),
                storage=mock_storage,
            )

        mock_storage.mark_document_failed.assert_called_once_with(
            document_id="doc-123",
            processing_token=target_token,
            error_stage="extraction_or_embedding",
            error_type="TikaExtractionError",
            error_message="Extraction bombed",
        )
        mock_storage.save_embedded_document.assert_not_called()

    @patch("src.ingestion.pipeline.validate_file")
    @patch("src.ingestion.pipeline.extract_clean_chunk_and_embed")
    def test_pipeline_failure_logging_error_does_not_mask_original_error(
        self, mock_extract, mock_validate
    ):
        ext, _, _, _ = dummy_artifacts()
        mock_validate.return_value = (
            Path("tests/fixtures/synthetic_smoke_doc.txt"),
            "text/plain",
            100,
        )
        mock_extract.side_effect = TikaExtractionError("Original root error")

        mock_storage = MagicMock()
        mock_storage.claim_or_create_document.return_value = (
            "doc-123",
            uuid.uuid4(),
        )
        # DB connection drops during failure logging
        mock_storage.mark_document_failed.side_effect = DatabaseConnectionError(
            "DB disconnected"
        )

        with pytest.raises(TikaExtractionError, match="Original root error"):
            ingest_file_to_storage(
                file_path="tests/fixtures/synthetic_smoke_doc.txt",
                board_id=uuid.uuid4(),
                uploaded_by=uuid.uuid4(),
                storage=mock_storage,
            )


class TestCliSafeInfo:
    """Verify safe info parsing without credentials leakage."""

    def test_parse_safe_info_strips_credentials(self):
        url = "postgresql://my_secret_user:super_secret_pw@db.internal:5433/nexora_test"
        info = parse_safe_db_info(url)
        assert info["configured"] is True
        assert info["host"] == "db.internal"
        assert info["port"] == 5433
        assert info["database"] == "nexora_test"
        # Confirm credentials are NOT in dict
        assert "super_secret_pw" not in str(info)
        assert "my_secret_user" not in str(info)


class TestErrorSanitization:
    """Verify error sanitization protects against leaking secrets and raw SQL."""

    def test_sanitize_error_message_strips_database_credentials(self):
        err = "Failed to connect to postgresql://admin:secret123@db.example.com:5432/proddb"
        sanitized = sanitize_error_message(err)
        assert "secret123" not in sanitized
        assert "admin" not in sanitized
        assert "[REDACTED_DATABASE_URL]" in sanitized

    def test_sanitize_error_message_strips_embedded_sql(self):
        err = "Query failed: SELECT * FROM documents WHERE id = 'secret-uuid'"
        sanitized = sanitize_error_message(err)
        assert "secret-uuid" not in sanitized
        assert "[REDACTED_SQL]" in sanitized

    def test_sanitize_error_message_bounds_length(self):
        long_err = "A" * 600
        sanitized = sanitize_error_message(long_err, max_length=100)
        assert len(sanitized) == 100
        assert sanitized.endswith("...")
