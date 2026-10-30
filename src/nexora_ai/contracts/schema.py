"""Canonical contract registry used to export schemas for other services."""

from nexora_ai.contracts.ingestion import IngestionJobV1
from nexora_ai.contracts.ping import PingJobV1, PingResultV1
from nexora_ai.contracts.retrieval import (
    CitationV1,
    RagAnswerDeltaEventV1,
    RagAnswerDoneEventV1,
    RagAnswerV1,
    RagCitationEventV1,
    RagErrorEventV1,
    RagQueryRequestV1,
)

SCHEMAS = {
    "ping-job": PingJobV1,
    "ping-result": PingResultV1,
    "ingestion-job": IngestionJobV1,
    "rag-query-request": RagQueryRequestV1,
    "rag-answer": RagAnswerV1,
    "citation": CitationV1,
    "rag-answer-delta-event": RagAnswerDeltaEventV1,
    "rag-citation-event": RagCitationEventV1,
    "rag-answer-done-event": RagAnswerDoneEventV1,
    "rag-error-event": RagErrorEventV1,
}
