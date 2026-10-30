from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from nexora_ai.worker.handlers.phase0 import PING_JOB_NAME, dispatch_job, handle_ping


@pytest.fixture
def anyio_backend():
    return "asyncio"


def ping_payload():
    return {"schemaVersion": 1, "correlationId": str(uuid4())}


@pytest.mark.anyio
async def test_ping_calls_express_and_returns_versioned_result():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"message": "Hello World!"})

    transport = httpx.MockTransport(respond)
    async with httpx.AsyncClient(transport=transport) as client:
        payload = ping_payload()
        result = await handle_ping(payload, "http://express.test", client=client)
    assert result.model_dump(mode="json", by_alias=True) == {
        "schemaVersion": 1,
        "correlationId": payload["correlationId"],
        "accepted": True,
        "expressStatus": 200,
    }
    assert str(requests[0].url) == "http://express.test/"


@pytest.mark.anyio
async def test_express_http_error_is_not_reported_as_completed():
    transport = httpx.MockTransport(lambda request: httpx.Response(503))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await handle_ping(ping_payload(), "http://express.test", client=client)


@pytest.mark.anyio
async def test_express_timeout_is_not_reported_as_completed():
    def timeout(request):
        raise httpx.ReadTimeout("timed out", request=request)

    transport = httpx.MockTransport(timeout)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.ReadTimeout):
            await handle_ping(ping_payload(), "http://express.test", client=client)


@pytest.mark.anyio
async def test_dispatch_rejects_unknown_job_and_invalid_contract():
    with pytest.raises(ValueError, match="Unsupported job"):
        await dispatch_job("unknown", {}, "http://express.test")
    with pytest.raises(ValidationError):
        await dispatch_job(
            PING_JOB_NAME, {"schemaVersion": True}, "http://express.test"
        )
