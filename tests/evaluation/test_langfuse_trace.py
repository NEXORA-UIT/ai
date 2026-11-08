"""Langfuse trace tests use fake OTLP HTTP endpoints and no real credentials."""

from __future__ import annotations

import base64
import json

import httpx

from evaluation.faithfulness import FaithfulnessClaim, FaithfulnessResult
from evaluation.langfuse_trace import LangfuseTraceClient


def _result() -> FaithfulnessResult:
    return FaithfulnessResult(
        score=1.0,
        claims=(
            FaithfulnessClaim(
                "The answer is supported.",
                True,
                ("a1000000-0000-4000-8000-000000000001",),
            ),
        ),
    )


def test_langfuse_sends_safe_otlp_json_with_auth_and_trace_id(monkeypatch) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"partialSuccess": {}}, request=request)

    client = LangfuseTraceClient(
        public_key="public-test",
        secret_key="secret-test",
        base_url="https://langfuse.example.test",
        transport=httpx.MockTransport(handler),
    )
    try:
        trace_id = client.record_faithfulness(
            dataset_version="real-v1",
            case_index=3,
            model_id="deepseek-flash",
            result=_result(),
        )
    finally:
        client.close()

    assert trace_id is not None
    assert len(requests) == 1
    request = requests[0]
    assert request.url.path == "/api/public/otel/v1/traces"
    assert request.headers["x-langfuse-ingestion-version"] == "4"
    expected_auth = base64.b64encode(b"public-test:secret-test").decode()
    assert request.headers["authorization"] == f"Basic {expected_auth}"
    payload = json.loads(request.content)
    span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert span["traceId"] == trace_id
    attributes = {row["key"]: row["value"] for row in span["attributes"]}
    assert attributes["langfuse.observation.type"]["stringValue"] == "evaluator"
    assert (
        attributes["langfuse.observation.model.name"]["stringValue"] == "deepseek-flash"
    )
    assert attributes["langfuse.observation.output"]["stringValue"]
    assert "The answer is supported." not in request.content.decode()
    assert "secret-test" not in request.content.decode()


def test_missing_langfuse_keys_disables_tracing_without_request(monkeypatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    client = LangfuseTraceClient()

    assert client.enabled is False
    assert (
        client.record_faithfulness(
            dataset_version="synthetic-v1",
            case_index=1,
            model_id="deepseek-flash",
            result=_result(),
        )
        is None
    )
    client.close()


def test_langfuse_outage_is_best_effort_and_does_not_raise() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("secret-test trace transport failed", request=request)

    client = LangfuseTraceClient(
        public_key="public-test",
        secret_key="secret-test",
        transport=httpx.MockTransport(handler),
    )
    try:
        assert (
            client.record_faithfulness(
                dataset_version="real-v1",
                case_index=0,
                model_id="deepseek-flash",
                result=_result(),
            )
            is None
        )
    finally:
        client.close()
