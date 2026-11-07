"""Swagger JSON endpoint for Board-scoped dense and lexical retrieval."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.config import (
    API_V1_PREFIX,
    DATABASE_UNAVAILABLE_DETAIL,
    SEARCH_ROUTE_PATH,
)
from src.api.schemas import SearchChunkResponse, SearchRequest, SearchResponse
from src.api.security import require_service_key
from src.ingestion.exceptions import DatabaseConnectionError
from src.retrieval.schemas import SearchResult

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
