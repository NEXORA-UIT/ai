import logging
from types import SimpleNamespace

from nexora_ai.worker import main as worker_main


def test_worker_disables_bullmq_constructor_autorun(monkeypatch):
    captured = {}

    class WorkerStub:
        def __init__(self, name, processor, options):
            captured["name"] = name
            captured["processor"] = processor
            captured["options"] = options

    monkeypatch.setattr(worker_main, "Worker", WorkerStub)

    async def processor(job, job_token=None):
        return {}

    worker = worker_main.create_worker("redis://localhost:6379", processor)

    assert isinstance(worker, WorkerStub)
    assert captured["name"] == worker_main.QUEUE_NAME
    assert captured["processor"] is processor
    assert captured["options"]["autorun"] is False


def test_failed_job_log_includes_correlation_id_without_raw_error(caplog):
    job = SimpleNamespace(
        id="job-1",
        name="phase0.ping",
        data={"correlationId": "f1bb7c6d-2ab8-4e99-95b7-5c3dc5a0a7bf"},
    )
    error = RuntimeError("redis://worker:secret@redis:6379 unavailable")

    with caplog.at_level(logging.ERROR, logger="nexora_ai.worker"):
        worker_main.log_failed_job(job, error)

    assert "correlationId=f1bb7c6d-2ab8-4e99-95b7-5c3dc5a0a7bf" in caplog.text
    assert "job-1" in caplog.text
    assert "event=job_failed" in caplog.text
    assert "secret" not in caplog.text
