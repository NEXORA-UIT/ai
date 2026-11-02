"""Best-effort Langfuse v4 trace export over OTLP/HTTP JSON."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from uuid import uuid4

import httpx

from evaluation.faithfulness import FaithfulnessResult

_DEFAULT_BASE_URL = "https://cloud.langfuse.com"
_TRACE_PATH = "/api/public/otel/v1/traces"


def _attribute(key: str, value: str | int | float | bool) -> dict[str, object]:
    if isinstance(value, bool):
        encoded: dict[str, object] = {"boolValue": value}
    elif isinstance(value, int):
        encoded = {"intValue": str(value)}
    elif isinstance(value, float):
        encoded = {"doubleValue": value}
    else:
        encoded = {"stringValue": value}
    return {"key": key, "value": encoded}


class LangfuseTraceClient:
    """Export aggregate evaluator telemetry without answers or source content."""

    def __init__(
        self,
        *,
        public_key: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        public = (
            public_key
            if public_key is not None
            else os.environ.get("LANGFUSE_PUBLIC_KEY")
        )
        secret = (
            secret_key
            if secret_key is not None
            else os.environ.get("LANGFUSE_SECRET_KEY")
        )
        configured = (
            isinstance(public, str)
            and bool(public.strip())
            and isinstance(secret, str)
            and bool(secret.strip())
        )
        self.enabled = configured
        self._auth = (public.strip(), secret.strip()) if configured else None
        selected_base = (
            base_url
            if base_url is not None
            else os.environ.get("LANGFUSE_BASE_URL", _DEFAULT_BASE_URL)
        )
        if not isinstance(selected_base, str) or not selected_base.strip():
            selected_base = _DEFAULT_BASE_URL
        self._base_url = selected_base.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
        )

    def record_faithfulness(
        self,
        *,
        dataset_version: str,
        case_index: int,
        model_id: str,
        result: FaithfulnessResult,
    ) -> str | None:
        """Send evaluator metadata and aggregate scores; never send claim text."""
        if not self.enabled or self._auth is None:
            return None
        if not isinstance(result, FaithfulnessResult):
            return None
        trace_id = uuid4().hex
        span_id = uuid4().hex[:16]
        now_ns = str(time.time_ns())
        score = result.score
        summary: Mapping[str, object] = {
            "score": score,
            "claim_count": len(result.claims),
            "supported_claim_count": sum(claim.supported for claim in result.claims),
            "unsupported_claim_count": sum(
                not claim.supported for claim in result.claims
            ),
        }
        attributes = [
            _attribute("langfuse.observation.type", "evaluator"),
            _attribute("langfuse.observation.name", "faithfulness-evaluation"),
            _attribute("langfuse.observation.model.name", model_id),
            _attribute(
                "langfuse.observation.input",
                json.dumps(
                    {
                        "dataset_version": dataset_version,
                        "case_index": case_index,
                    },
                    separators=(",", ":"),
                ),
            ),
            _attribute(
                "langfuse.observation.output",
                json.dumps(summary, separators=(",", ":")),
            ),
            _attribute(
                "langfuse.observation.metadata.dataset_version", dataset_version
            ),
            _attribute("langfuse.observation.metadata.case_index", case_index),
            _attribute("langfuse.observation.metadata.claim_count", len(result.claims)),
        ]
        if score is not None:
            attributes.append(_attribute("langfuse.observation.score", score))
        payload = {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            _attribute("service.name", "nexora-ai-evaluation")
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "nexora-ai-evaluation"},
                            "spans": [
                                {
                                    "traceId": trace_id,
                                    "spanId": span_id,
                                    "name": "faithfulness-evaluation",
                                    "kind": 1,
                                    "startTimeUnixNano": now_ns,
                                    "endTimeUnixNano": now_ns,
                                    "attributes": attributes,
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        try:
            response = self._client.post(
                _TRACE_PATH,
                auth=self._auth,
                headers={"x-langfuse-ingestion-version": "4"},
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            return None
        return trace_id

    def close(self) -> None:
        """Close the pooled HTTP client."""
        self._client.close()

    def __enter__(self) -> LangfuseTraceClient:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
