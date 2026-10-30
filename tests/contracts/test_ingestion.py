from uuid import uuid4

import pytest
from pydantic import ValidationError

from nexora_ai.contracts.ingestion import IngestionJobV1


def valid_payload():
    return {
        "schemaVersion": 1,
        "correlationId": str(uuid4()),
        "documentId": str(uuid4()),
        "boardId": str(uuid4()),
        "processingRevision": 1,
    }


def test_ingestion_contract_serializes_as_camel_case():
    payload = valid_payload()
    assert (
        IngestionJobV1.model_validate(payload).model_dump(mode="json", by_alias=True)
        == payload
    )


@pytest.mark.parametrize(
    "change",
    [
        {"schemaVersion": 2},
        {"processingRevision": 0},
        {"processingRevision": -1},
        {"processingRevision": "1"},
        {"documentId": "not-a-uuid"},
        {"boardId": "not-a-uuid"},
        {"documentText": "must never be in the queue"},
    ],
)
def test_ingestion_rejects_invalid_or_sensitive_shape(change):
    with pytest.raises(ValidationError):
        IngestionJobV1.model_validate(valid_payload() | change)
