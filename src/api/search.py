"""Swagger JSON endpoint for Board-scoped dense and lexical retrieval."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator, model_validator

from src.api.config import (
    API_V1_PREFIX,
    DATABASE_UNAVAILABLE_DETAIL,
    SEARCH_ROUTE_PATH,
)
from src.api.security import require_service_key
from src.ingestion.exceptions import DatabaseConnectionError
from src.retrieval.config import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_SEARCH_TOP_K,
    MAX_CANDIDATE_K,
    MAX_SEARCH_TOP_K,
)
from src.retrieval.schemas import SearchResult
from src.validation import QUESTION_TEXT_REQUIRED

router = APIRouter(prefix=API_V1_PREFIX, tags=["Retrieval"])


def search_documents(
    question: str,
    board_id: UUID,
    *,
    top_k: int,
    candidate_k: int,
) -> SearchResult:
    """Load the retrieval stack only when a search request is made."""
    from src.retrieval.search_documents import search_documents as retrieve

    return retrieve(
        question,
        board_id,
        top_k=top_k,
        candidate_k=candidate_k,
    )


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


@router.post(
    SEARCH_ROUTE_PATH,
    response_model=SearchResponse,
    summary="Search documents in one Board",
    description=(
        "Run Board-scoped dense and BM25 retrieval, fuse candidates with RRF, "
        "and rerank the final citation-ready chunks. The caller must already be "
        "authorized for the supplied Board."
    ),
    responses={
        status.HTTP_401_UNAUTHORIZED: {"description": "Missing or invalid service key"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "Invalid question, Board ID, or result limits"
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Search failed or returned an invalid Board scope"
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": DATABASE_UNAVAILABLE_DETAIL
        },
    },
    dependencies=[Depends(require_service_key)],
)
def search_board_documents(request: SearchRequest) -> SearchResponse:
    """Return bounded, reranked evidence without internal storage keys."""
    try:
        result: SearchResult = search_documents(
            request.question,
            request.board_id,
            top_k=request.top_k,
            candidate_k=request.candidate_k,
        )
    except DatabaseConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=DATABASE_UNAVAILABLE_DETAIL,
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed",
        ) from exc

    try:
        result_board_id = UUID(result.board_id)
    except (TypeError, ValueError, AttributeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search returned data outside the requested Board",
        ) from exc
    if result_board_id != request.board_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search returned data outside the requested Board",
        )

    return SearchResponse(
        question=request.question,
        board_id=request.board_id,
        chunks=[
            SearchChunkResponse(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                file_name=chunk.file_name,
                content=chunk.content,
                page_number=chunk.page_number,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                rerank_score=chunk.rerank_score,
                rrf_score=chunk.rrf_score,
            )
            for chunk in result.chunks
        ],
    )
