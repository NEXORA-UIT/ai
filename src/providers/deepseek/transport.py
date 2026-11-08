"""HTTPX client for the DeepSeek JSON chat completion envelope."""

import json
import math
import os
from collections.abc import Mapping

import httpx

DEEPSEEK_API_BASE_URL = "https://api.deepseek.com"


class DeepSeekRequestError(RuntimeError):
    """Transport or non-success HTTP status."""


class DeepSeekResponseError(RuntimeError):
    """Response does not contain a structured assistant message."""


class DeepSeekJsonTransport:
    def __init__(
        self,
        *,
        api_key: str | None,
        timeout_seconds: float,
        transport: httpx.BaseTransport | None,
    ) -> None:
        selected_key = (
            api_key if api_key is not None else os.environ.get("DEEPSEEK_API_KEY")
        )
        if not isinstance(selected_key, str) or not selected_key.strip():
            raise ValueError("DEEPSEEK_API_KEY must be configured")
        if isinstance(timeout_seconds, bool):
            raise ValueError("timeout_seconds must be a positive finite number")
        try:
            self.timeout_seconds = float(timeout_seconds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "timeout_seconds must be a positive finite number"
            ) from exc
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive finite number")

        self._client = httpx.Client(
            base_url=DEEPSEEK_API_BASE_URL,
            headers={"Authorization": f"Bearer {selected_key.strip()}"},
            timeout=httpx.Timeout(self.timeout_seconds),
            transport=transport,
        )

    def complete_json(self, payload: dict[str, object]) -> dict[str, object]:
        try:
            response = self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPError:
            raise DeepSeekRequestError("DeepSeek request failed") from None

        try:
            body = response.json()
            choices = body.get("choices") if isinstance(body, Mapping) else None
            if not isinstance(choices, list) or not choices:
                raise ValueError("response choices are missing")
            choice = choices[0]
            message = choice.get("message") if isinstance(choice, Mapping) else None
            content = message.get("content") if isinstance(message, Mapping) else None
            if not isinstance(content, str) or not content.strip():
                raise ValueError("assistant content is empty")
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("structured output must be an object")
            return parsed
        except (ValueError, TypeError):
            raise DeepSeekResponseError(
                "DeepSeek returned an invalid structured response"
            ) from None

    def close(self) -> None:
        self._client.close()
