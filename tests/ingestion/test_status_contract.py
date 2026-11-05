from typing import get_args
from uuid import UUID

from src.api.schemas import DocumentUploadResponse
from src.ingestion.status import DocumentStatus


def test_document_status_values_match_ingestion_database_contract() -> None:
    assert DocumentStatus.PROCESSING.value == "PROCESSING"
    assert DocumentStatus.READY.value == "READY"
    assert DocumentStatus.FAILED.value == "FAILED"


def test_ready_status_serializes_as_public_contract_string() -> None:
    response = DocumentUploadResponse(
        document_id=UUID("11111111-1111-4111-8111-111111111111"),
        status=DocumentStatus.READY,
        chunk_count=1,
        total_tokens=10,
    )

    assert response.model_dump()["status"] == "READY"
    status_type = DocumentUploadResponse.model_fields["status"].annotation
    assert get_args(status_type) == (DocumentStatus.READY,)
