from uuid import uuid4

import pytest
from pydantic import ValidationError

from nexora_ai.contracts.retrieval import (
    CitationV1,
    RagAnswerDeltaEventV1,
    RagAnswerDoneEventV1,
    RagAnswerV1,
    RagCitationEventV1,
    RagErrorEventV1,
    RagQueryRequestV1,
)


def test_query_trims_question_and_rejects_blank():
    payload = {
        "schemaVersion": 1,
        "boardId": str(uuid4()),
        "correlationId": str(uuid4()),
        "question": "  Nội dung tài liệu?  ",
    }
    query = RagQueryRequestV1.model_validate(payload)
    assert query.question == "Nội dung tài liệu?"
    with pytest.raises(ValidationError):
        RagQueryRequestV1.model_validate(payload | {"question": "   "})


def test_query_enforces_wire_length_before_trimming():
    payload = {
        "schemaVersion": 1,
        "boardId": str(uuid4()),
        "correlationId": str(uuid4()),
        "question": f" {'x' * 2000} ",
    }
    with pytest.raises(ValidationError):
        RagQueryRequestV1.model_validate(payload)


def test_citation_requires_document_name_and_supports_page_less_sources():
    payload = {
        "documentId": str(uuid4()),
        "documentName": "meeting-notes.txt",
        "chunkId": str(uuid4()),
        "snippet": "Decision recorded in paragraph two.",
        "sourceLocator": "paragraph:2",
    }

    citation = CitationV1.model_validate(payload)

    assert citation.document_name == "meeting-notes.txt"
    assert citation.page_number is None
    assert citation.source_locator == "paragraph:2"
    payload_without_name = payload.copy()
    payload_without_name.pop("documentName")
    with pytest.raises(ValidationError):
        CitationV1.model_validate(payload_without_name)


def test_answer_has_traceable_citation_ids_and_nullable_source_fields():
    payload = {
        "schemaVersion": 1,
        "correlationId": str(uuid4()),
        "boardId": str(uuid4()),
        "answer": "  Theo tài liệu...  ",
        "citations": [
            {
                "documentId": str(uuid4()),
                "documentName": "SRS.pdf",
                "chunkId": str(uuid4()),
                "snippet": "  Đoạn trích dẫn  ",
                "sourceLocator": None,
                "pageNumber": None,
            }
        ],
    }
    answer = RagAnswerV1.model_validate(payload)
    normalized_payload = payload | {
        "answer": "Theo tài liệu...",
        "citations": [payload["citations"][0] | {"snippet": "Đoạn trích dẫn"}],
    }
    assert answer.model_dump(mode="json", by_alias=True) == normalized_payload


def test_citation_snippet_enforces_wire_length_before_trimming():
    payload = {
        "schemaVersion": 1,
        "correlationId": str(uuid4()),
        "boardId": str(uuid4()),
        "answer": "ok",
        "citations": [
            {
                "documentId": str(uuid4()),
                "documentName": "SRS.pdf",
                "chunkId": str(uuid4()),
                "snippet": f" {'x' * 1000} ",
            }
        ],
    }
    with pytest.raises(ValidationError):
        RagAnswerV1.model_validate(payload)


def test_answer_rejects_bad_page_and_extra_fields():
    payload = {
        "schemaVersion": 1,
        "correlationId": str(uuid4()),
        "boardId": str(uuid4()),
        "answer": "ok",
        "citations": [
            {
                "documentId": str(uuid4()),
                "documentName": "SRS.pdf",
                "chunkId": str(uuid4()),
                "snippet": "text",
                "pageNumber": 0,
            }
        ],
    }
    with pytest.raises(ValidationError):
        RagAnswerV1.model_validate(payload)
    with pytest.raises(ValidationError):
        RagAnswerV1.model_validate(payload | {"userId": str(uuid4()), "citations": []})


def test_stream_event_models_validate_versioned_wire_payloads():
    correlation_id = str(uuid4())
    citation = {
        "documentId": str(uuid4()),
        "documentName": "meeting-notes.txt",
        "chunkId": str(uuid4()),
        "snippet": "Decision recorded in paragraph two.",
        "sourceLocator": "paragraph:2",
    }

    delta_event = RagAnswerDeltaEventV1.model_validate(
        {"schemaVersion": 1, "correlationId": correlation_id, "delta": " "}
    )
    citation_event = RagCitationEventV1.model_validate(
        {
            "schemaVersion": 1,
            "correlationId": correlation_id,
            "citation": citation,
        }
    )
    done_event = RagAnswerDoneEventV1.model_validate(
        {
            "schemaVersion": 1,
            "correlationId": correlation_id,
            "answer": "See paragraph two.",
            "citations": [citation],
        }
    )
    error_event = RagErrorEventV1.model_validate(
        {
            "schemaVersion": 1,
            "correlationId": correlation_id,
            "code": "MODEL_UNAVAILABLE",
            "message": "The answer service is temporarily unavailable.",
            "retryable": True,
        }
    )

    assert delta_event.delta == " "
    assert citation_event.citation.page_number is None
    assert done_event.citations[0].document_name == "meeting-notes.txt"
    assert error_event.code == "MODEL_UNAVAILABLE"
    with pytest.raises(ValidationError):
        RagErrorEventV1.model_validate(
            {
                "schemaVersion": 1,
                "correlationId": correlation_id,
                "code": "DATABASE_ERROR",
                "message": "Failed",
                "retryable": False,
            }
        )
