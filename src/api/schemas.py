"""Public request and response models for the internal AI API."""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from src.ingestion.status import DocumentStatus
from src.retrieval.config import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_SEARCH_TOP_K,
    MAX_CANDIDATE_K,
    MAX_SEARCH_TOP_K,
)
from src.validation import QUESTION_TEXT_REQUIRED


class DocumentUploadResponse(BaseModel):
    """Summary of a document ingested successfully."""

    document_id: UUID
    status: Literal[DocumentStatus.READY]
    chunk_count: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class SearchRequest(BaseModel):
    """Board-scoped retrieval query and bounded candidate limits."""

    question: str = Field(
        min_length=1,
        description="Question used to find the most relevant document chunks",
    )
    board_id: UUID = Field(description="UUID of the caller-authorized Board")
    top_k: Annotated[
        int,
        Field(
            default=DEFAULT_SEARCH_TOP_K,
            ge=1,
            le=MAX_SEARCH_TOP_K,
            strict=True,
            description="Number of final citation chunks to return",
        ),
    ]
    candidate_k: Annotated[
        int,
        Field(
            default=DEFAULT_CANDIDATE_K,
            ge=1,
            le=MAX_CANDIDATE_K,
            strict=True,
            description="Maximum dense and BM25 candidate pool before reranking",
        ),
    ]

    @field_validator("question")
    @classmethod
    def question_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(QUESTION_TEXT_REQUIRED)
        return value

    @model_validator(mode="after")
    def candidate_pool_must_cover_final_results(self) -> SearchRequest:
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        return self


class SearchChunkResponse(BaseModel):
    """Citation-ready chunk fields intentionally excluding storage internals."""

    chunk_id: str
    document_id: str
    chunk_index: int
    file_name: str
    content: str
    page_number: int | None
    start_offset: int | None
    end_offset: int | None
    rerank_score: float
    rrf_score: float


class SearchResponse(BaseModel):
    """Public retrieval response scoped to one Board."""

    question: str
    board_id: UUID
    chunks: list[SearchChunkResponse]


class CitationResponse(BaseModel):
    chunk_id: str
    document_id: str
    file_name: str
    page_number: int | None
    start_offset: int | None
    end_offset: int | None


class AnswerResponse(BaseModel):
    question: str
    board_id: UUID
    answer: str
    refused: bool
    citations: list[CitationResponse]
