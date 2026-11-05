"""Canonical lifecycle states persisted for ingested documents."""

from enum import StrEnum


class DocumentStatus(StrEnum):
    """Document lifecycle values shared by ingestion, retrieval, and the API."""

    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"
