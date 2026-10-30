"""Future ingestion queue payload, without document text or file URLs."""

from typing import Annotated
from uuid import UUID

from pydantic import Field

from nexora_ai.contracts.base import SchemaVersion, WireModel


class IngestionJobV1(WireModel):
    schema_version: SchemaVersion
    correlation_id: UUID
    document_id: UUID
    board_id: UUID
    processing_revision: Annotated[int, Field(strict=True, gt=0)]
