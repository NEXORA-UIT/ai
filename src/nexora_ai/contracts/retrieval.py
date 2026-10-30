"""Future board-scoped retrieval and cited-answer wire formats."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, StringConstraints, field_validator

from nexora_ai.contracts.base import SchemaVersion, WireModel

MAX_QUESTION_LENGTH = 2000
MAX_CITATION_SNIPPET_LENGTH = 1000

Question = Annotated[
    str,
    StringConstraints(min_length=1, max_length=MAX_QUESTION_LENGTH, pattern=r"\S"),
]
AnswerText = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
Snippet = Annotated[
    str,
    StringConstraints(
        min_length=1, max_length=MAX_CITATION_SNIPPET_LENGTH, pattern=r"\S"
    ),
]
DocumentName = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
EventText = Annotated[str, StringConstraints(min_length=1)]
PublicErrorMessage = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
RagErrorCode = Literal["RAG_QUERY_FAILED", "MODEL_UNAVAILABLE", "REQUEST_TIMEOUT"]


class RagQueryRequestV1(WireModel):
    schema_version: SchemaVersion
    board_id: UUID
    correlation_id: UUID
    question: Question

    @field_validator("question")
    @classmethod
    def trim_question(cls, value: str) -> str:
        return value.strip()


class CitationV1(WireModel):
    document_id: UUID
    document_name: DocumentName
    chunk_id: UUID
    snippet: Snippet
    source_locator: str | None = None
    page_number: Annotated[int, Field(strict=True, gt=0)] | None = None

    @field_validator("snippet")
    @classmethod
    def trim_snippet(cls, value: str) -> str:
        return value.strip()

    @field_validator("document_name")
    @classmethod
    def trim_document_name(cls, value: str) -> str:
        return value.strip()


class RagAnswerV1(WireModel):
    schema_version: SchemaVersion
    correlation_id: UUID
    board_id: UUID
    answer: AnswerText
    citations: list[CitationV1]

    @field_validator("answer")
    @classmethod
    def trim_answer(cls, value: str) -> str:
        return value.strip()


class RagAnswerDeltaEventV1(WireModel):
    schema_version: SchemaVersion
    correlation_id: UUID
    delta: EventText


class RagCitationEventV1(WireModel):
    schema_version: SchemaVersion
    correlation_id: UUID
    citation: CitationV1


class RagAnswerDoneEventV1(WireModel):
    schema_version: SchemaVersion
    correlation_id: UUID
    answer: AnswerText
    citations: list[CitationV1]

    @field_validator("answer")
    @classmethod
    def trim_answer(cls, value: str) -> str:
        return value.strip()


class RagErrorEventV1(WireModel):
    schema_version: SchemaVersion
    correlation_id: UUID
    code: RagErrorCode
    message: PublicErrorMessage
    retryable: Annotated[bool, Field(strict=True)]

    @field_validator("message")
    @classmethod
    def trim_message(cls, value: str) -> str:
        return value.strip()
