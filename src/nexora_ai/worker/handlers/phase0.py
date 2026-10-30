"""Safe transport-only probe job used to verify the shared BullMQ queue."""

import httpx

from nexora_ai.contracts.phase0 import PingJobV1, PingResultV1
from nexora_ai.contracts.queue import PHASE0_PING_JOB_NAME

PING_JOB_NAME = PHASE0_PING_JOB_NAME
EXPRESS_REQUEST_TIMEOUT_SECONDS = 5.0


async def handle_ping(
    payload: PingJobV1 | dict[str, object],
    express_base_url: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> PingResultV1:
    job = (
        payload if isinstance(payload, PingJobV1) else PingJobV1.model_validate(payload)
    )
    url = f"{express_base_url.rstrip('/')}/"

    if client is None:
        async with httpx.AsyncClient(
            timeout=EXPRESS_REQUEST_TIMEOUT_SECONDS
        ) as http_client:
            response = await http_client.get(url)
    else:
        response = await client.get(url)

    response.raise_for_status()
    return PingResultV1(
        schemaVersion=1,
        correlationId=job.correlation_id,
        accepted=True,
        expressStatus=response.status_code,
    )


async def dispatch_job(
    name: str,
    data: dict[str, object],
    express_base_url: str,
) -> dict[str, object]:
    if name != PING_JOB_NAME:
        raise ValueError(f"Unsupported job: {name}")
    result = await handle_ping(data, express_base_url)
    return result.model_dump(mode="json", by_alias=True)
