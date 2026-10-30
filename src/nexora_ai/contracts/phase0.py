"""Phase 0 BullMQ probe payload and completed-job result."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BeforeValidator

from nexora_ai.contracts.base import SchemaVersion, WireModel


def _require_boolean(value: object) -> object:
    if type(value) is not bool:
        raise ValueError("must be a boolean")
    return value


def _require_integer(value: object) -> object:
    if type(value) is not int:
        raise ValueError("must be an integer")
    return value


Accepted = Annotated[Literal[True], BeforeValidator(_require_boolean)]
SuccessfulStatus = Annotated[Literal[200], BeforeValidator(_require_integer)]


class PingJobV1(WireModel):
    schema_version: SchemaVersion
    correlation_id: UUID


class PingResultV1(WireModel):
    schema_version: SchemaVersion
    correlation_id: UUID
    accepted: Accepted
    express_status: SuccessfulStatus
