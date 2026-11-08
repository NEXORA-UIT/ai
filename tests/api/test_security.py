"""Tests for the internal API service-key boundary."""

from collections.abc import Iterator

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.api.security import require_service_key


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    app = FastAPI()

    @app.get("/private", dependencies=[Depends(require_service_key)])
    def private_route() -> dict[str, str]:
        return {"status": "ok"}

    monkeypatch.setenv("AI_SERVICE_API_KEY", "demo-secret")
    with TestClient(app) as test_client:
        yield test_client


def test_service_key_auth_accepts_matching_bearer_token(client: TestClient) -> None:
    response = client.get("/private", headers={"Authorization": "Bearer demo-secret"})

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(
    "authorization",
    [None, "Basic demo-secret", "Bearer wrong-secret"],
)
def test_service_key_auth_rejects_missing_or_invalid_credentials(
    client: TestClient, authorization: str | None
) -> None:
    headers = {} if authorization is None else {"Authorization": authorization}

    response = client.get("/private", headers=headers)

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing service API key"}


def test_service_key_auth_fails_closed_when_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AI_SERVICE_API_KEY", raising=False)
    app = FastAPI()

    @app.get("/private", dependencies=[Depends(require_service_key)])
    def private_route() -> dict[str, str]:
        return {"status": "ok"}

    response = TestClient(app).get("/private", headers={"Authorization": "Bearer x"})

    assert response.status_code == 503
    assert response.json() == {"detail": "AI service API key is not configured"}


def test_service_key_auth_is_declared_in_openapi(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert schema["components"]["securitySchemes"]["AIServiceBearer"] == {
        "type": "http",
        "scheme": "bearer",
    }
    assert schema["paths"]["/private"]["get"]["security"] == [
        {"AIServiceBearer": []}
    ]
