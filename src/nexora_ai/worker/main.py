"""Standalone BullMQ worker entrypoint."""

import asyncio
import logging
import signal
from collections.abc import Awaitable, Callable

from bullmq import Job, Worker
from redis.asyncio import Redis

from nexora_ai.contracts.queue import QUEUE_NAME
from nexora_ai.core.config import Settings
from nexora_ai.worker.handlers.phase0 import dispatch_job

logger = logging.getLogger("nexora_ai.worker")
JobProcessor = Callable[[Job, str | None], Awaitable[dict[str, object]]]
REDIS_CONNECT_TIMEOUT_SECONDS = 5
REDIS_STARTUP_TIMEOUT_SECONDS = 5


def create_processor(express_base_url: str) -> JobProcessor:
    async def process(job: Job, job_token: str | None = None) -> dict[str, object]:
        del job_token
        return await dispatch_job(job.name, job.data, express_base_url)

    return process


def create_worker(redis_url: str, processor: JobProcessor) -> Worker:
    return Worker(
        QUEUE_NAME,
        processor,
        {"connection": redis_url, "concurrency": 1, "autorun": False},
    )


def log_worker_error(error: Exception) -> None:
    logger.error("event=worker_error errorType=%s", type(error).__name__)


def log_failed_job(job: Job | None, error: Exception) -> None:
    job_data = job.data if job is not None and isinstance(job.data, dict) else {}
    correlation_id = job_data.get("correlationId")
    if not isinstance(correlation_id, str):
        correlation_id = None

    logger.error(
        "event=job_failed queue=%s jobId=%s jobName=%s correlationId=%s errorType=%s",
        QUEUE_NAME,
        job.id if job else None,
        job.name if job else None,
        correlation_id,
        type(error).__name__,
    )


async def run_worker(settings: Settings | None = None) -> None:
    settings = settings or Settings()
    redis_url, express_base_url = settings.require_worker()
    redis_client = Redis.from_url(
        redis_url, socket_connect_timeout=REDIS_CONNECT_TIMEOUT_SECONDS
    )
    try:
        await asyncio.wait_for(
            redis_client.ping(), timeout=REDIS_STARTUP_TIMEOUT_SECONDS
        )
    except Exception as error:
        raise RuntimeError("Python Worker could not connect to Redis") from error
    finally:
        await redis_client.aclose()

    processor = create_processor(express_base_url)
    worker = create_worker(redis_url, processor)
    worker.on("error", lambda error, *_: log_worker_error(error))
    worker.on("failed", log_failed_job)

    loop = asyncio.get_running_loop()
    stopping = asyncio.Event()
    for handled_signal in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(handled_signal, stopping.set)
        except NotImplementedError:
            signal.signal(
                handled_signal,
                lambda *_: loop.call_soon_threadsafe(stopping.set),
            )

    run_task = asyncio.create_task(worker.run())
    stop_task = asyncio.create_task(stopping.wait())
    try:
        done, _ = await asyncio.wait(
            (run_task, stop_task), return_when=asyncio.FIRST_COMPLETED
        )
        if run_task in done:
            run_task.result()
    finally:
        stop_task.cancel()
        await worker.close()
        if not run_task.done():
            await run_task


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
