from uuid import uuid4

import pytest
from pydantic import ValidationError

from nexora_ai.contracts.ping import PingJobV1, PingResultV1


def test_ping_payload_round_trips_in_node_json_shape():
    correlation_id = str(uuid4())
    job = PingJobV1.model_validate(
        {"schemaVersion": 1, "correlationId": correlation_id}
    )
    assert job.model_dump(mode="json", by_alias=True) == {
        "schemaVersion": 1,
        "correlationId": correlation_id,
    }
    result = PingResultV1.model_validate(
        {
            "schemaVersion": 1,
            "correlationId": correlation_id,
            "accepted": True,
            "expressStatus": 200,
        }
    )
    assert result.model_dump(mode="json", by_alias=True)["expressStatus"] == 200


@pytest.mark.parametrize(
    "change",
    [
        {"schemaVersion": 2},
        {"schemaVersion": True},
        {"correlationId": "not-a-uuid"},
        {"unexpected": "value"},
    ],
)
def test_ping_rejects_invalid_payload(change):
    payload = {"schemaVersion": 1, "correlationId": str(uuid4())} | change
    with pytest.raises(ValidationError):
        PingJobV1.model_validate(payload)


def test_result_rejects_non_success():
    with pytest.raises(ValidationError):
        PingResultV1.model_validate(
            {
                "schemaVersion": 1,
                "correlationId": str(uuid4()),
                "accepted": False,
                "expressStatus": 503,
            }
        )


@pytest.mark.parametrize("field", [{"accepted": 1}, {"expressStatus": True}])
def test_result_does_not_coerce_booleans_and_integers(field):
    payload = {
        "schemaVersion": 1,
        "correlationId": str(uuid4()),
        "accepted": True,
        "expressStatus": 200,
    }
    with pytest.raises(ValidationError):
        PingResultV1.model_validate(payload | field)
