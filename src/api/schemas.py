"""Public request and response models for the internal AI API."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """Summary of a document ingested successfully."""

    document_id: UUID
    status: Literal["READY"]
    chunk_count: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
