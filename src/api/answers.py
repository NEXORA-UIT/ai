"""Internal Swagger endpoint for Board-scoped grounded answers."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from src.answering.answer import (
    AnswerProviderError,
    EvidenceScopeError,
)
from src.answering.service import (
    AnswerConfigurationError,
    AnswerGenerationError,
    AnswerRetrievalError,
    answer_question,
)
from src.api.config import (
    ANSWER_ROUTE_PATH,
    API_V1_PREFIX,
    DATABASE_UNAVAILABLE_DETAIL,
)
from src.api.schemas import AnswerResponse, CitationResponse, SearchRequest
from src.api.security import require_service_key
from src.persistence.errors import DatabaseConnectionError

router = APIRouter(prefix=API_V1_PREFIX, tags=["Answers"])
logger = logging.getLogger("uvicorn.error")


@router.post(
    ANSWER_ROUTE_PATH,
    response_model=AnswerResponse,
    summary="Answer a question from one Board's documents",
    description=(
        "Retrieve Board-scoped evidence, generate a grounded answer with DeepSeek, "
        "and return verified citations. The caller must already be authorized "
        "for the supplied Board."
    ),
    responses={
        401: {"description": "Missing or invalid service key"},
        422: {"description": "Invalid question, Board ID, or result limits"},
        502: {"description": "Answer provider failed"},
        503: {"description": "AI database or DeepSeek key is unavailable"},
    },
    dependencies=[Depends(require_service_key)],
)
def answer_board_question(request: SearchRequest) -> AnswerResponse:
    """Return a grounded answer or an explicit refusal without storage details."""
    try:
        result = answer_question(
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
    except AnswerConfigurationError as exc:
        raise HTTPException(
            status_code=503, detail="DeepSeek API key is not configured"
        ) from exc
    except EvidenceScopeError as exc:
        logger.exception("RAG answer evidence scope failed")
        raise HTTPException(
            status_code=500, detail="Answer evidence is outside the requested Board"
        ) from exc
    except AnswerProviderError as exc:
        logger.exception("RAG answer provider failed")
        raise HTTPException(status_code=502, detail="Answer provider failed") from exc
    except AnswerRetrievalError as exc:
        logger.exception("RAG answer retrieval failed")
        raise HTTPException(status_code=500, detail="Answer retrieval failed") from exc
    except AnswerGenerationError as exc:
        logger.exception("RAG answer generation failed")
        raise HTTPException(status_code=500, detail="Answer generation failed") from exc

    return AnswerResponse(
        question=request.question,
        board_id=request.board_id,
        answer=result.text,
        refused=result.refused,
        citations=[
            CitationResponse(
                chunk_id=citation.chunk_id,
                document_id=citation.document_id,
                file_name=citation.file_name,
                page_number=citation.page_number,
                start_offset=citation.start_offset,
                end_offset=citation.end_offset,
            )
            for citation in result.citations
        ],
    )
