import pytest
from httpx import ASGITransport, AsyncClient

from nexora_ai.api.main import create_app


def test_api_uses_ai_service_title():
    assert create_app().title == "Nexora AI Service"


@pytest.mark.anyio
async def test_health_is_independent_of_external_services(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1")
    monkeypatch.setenv("DATABASE_URL", "postgresql://bad:bad@127.0.0.1:1/bad")
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.fixture
def anyio_backend():
    return "asyncio"
